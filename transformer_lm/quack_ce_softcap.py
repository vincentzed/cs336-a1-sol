# Vendored + modified from quack 0.5.3 (quack/cross_entropy.py)
# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.  (BSD-3 / original quack license)
# Modification (CS336 SOL): logit softcap fused into the CE. FORM ENCODED IN SIGN:
#   softcap > 0: zc = cap*tanh(z/cap)                      (Gemma-2 form)
#   softcap < 0: zc = a*sigmoid((z+5)/7.5), a=|softcap|    (modded L1508, via
#                a*sig(u) = (a/2)(1+tanh(u/2)); b=5, c=7.5 fixed as in modded)
# Sign-encoding threads the form through every existing softcap plumbing site
# and JIT-cache key with zero extra parameters.
# fwd/bwd CuTeDSL kernels. Loss/lse computed in the CAPPED domain; backward chains
# dzc/dz = 1 - tanh^2(z/cap). The linear dgrad/wgrad GEMMs (torch autograd via
# F.linear) consume the softcap-chained dlogits and need no modification.
# Registered under sol:: op names so quack's own ops are untouched.

import math
from functools import partial
from typing import Optional, Type, Literal

import torch
import torch.nn.functional as F
from torch import Tensor

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Float32, Boolean, const_expr

import quack.utils as utils
import quack.copy_utils as copy_utils
import quack.layout_utils as layout_utils
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.dsl import cute_op
from quack.reduce import row_reduce, online_softmax_reduce
from quack.reduction_base import ReductionBase
from quack.cache import jit_cache
from quack.cute_dsl_utils import torch2cute_dtype_map
from cutlass.base_dsl.arch import Arch


class CrossEntropySoftcap(ReductionBase):
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        N: int,
        online_softmax: bool = True,
        softcap: float = 0.0,
    ):
        self.online_softmax = online_softmax
        self.softcap = float(softcap)  # Python float at trace time -> Constexpr branches
        super().__init__(
            dtype,
            N,
            stage=2 if not self.online_softmax else 1,
            reduction_dtype=Float32 if not self.online_softmax else Int64,
        )
        self.reload_from = None if N <= 16384 or self.online_softmax else "smem"

    def _threads_per_row(self):
        N = self.N
        for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128)]:
            if N <= limit:
                return threads
        return 256

    def _set_cluster_n(self):
        arch = cutlass.base_dsl.BaseDSL._get_dsl().get_arch_enum()
        if arch < Arch.sm_90:
            self.cluster_n = 1
            return
        max_cluster = 8 if arch.major == 12 else 16
        N = self.N
        if arch.major == 12 and const_expr(self.dtype.width >= 32):
            thresholds = [(16 * 1024, 1), (32 * 1024, 2), (64 * 1024, 4), (128 * 1024, 8)]
        elif const_expr(self.dtype.width == 16):
            thresholds = [(16 * 1024, 1), (32 * 1024, 2), (64 * 1024, 4), (128 * 1024, 8)]
        else:
            thresholds = [(16 * 1024, 1), (64 * 1024, 2), (128 * 1024, 4), (256 * 1024, 8)]
        for limit, cluster in thresholds:
            if N <= limit:
                self.cluster_n = cluster
                return
        self.cluster_n = max_cluster

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mTarget: cute.Tensor,
        mTargetLogit: Optional[cute.Tensor],
        mLoss: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mdX: Optional[cute.Tensor],
        mWeight: Optional[cute.Tensor],
        ignore_index: Int32,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        if const_expr(mTargetLogit is None):
            mTargetLogit = mX
        if const_expr(mdX is not None):
            assert mdX.element_type == self.dtype
        self._set_cluster_n()
        largest_dtype_width = const_expr(mX.element_type.width)
        if const_expr(mdX is not None):
            largest_dtype_width = const_expr(max(largest_dtype_width, mdX.element_type.width))
        vecsize = math.gcd(self.N, 128 // largest_dtype_width)
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        self.kernel(
            mX, mTarget, mTargetLogit, mLoss, mLSE, mdX, mWeight,
            ignore_index, tiler_mn, tiled_copy, threads_per_row,
        ).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if const_expr(self.cluster_n > 1) else None,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mTarget: cute.Tensor,
        mTargetLogit: cute.Tensor,
        mLoss: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mdX: Optional[cute.Tensor],
        mWeight: Optional[cute.Tensor],
        ignore_index: Int32,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        cluster_y = const_expr(0) if const_expr(self.cluster_n == 1) else cute.arch.block_idx()[1]
        tv_layout = tiled_copy.layout_tv_tiled

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        gX, cX = [cute.local_tile(mT, tiler_mn, (bidx, cluster_y)) for mT in (mX, idX)]

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16
        )
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(smem, tv_layout)

        thr_copy = tiled_copy.get_slice(tidx)

        tXgX = thr_copy.partition_S(gX)
        tXsX = thr_copy.partition_D(sX)
        tXcX = thr_copy.partition_S(cX)[(0, None), None, None]
        tXrX = cute.make_rmem_tensor_like(tXgX)

        is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tXpX = (
            None if is_even_N else copy_utils.predicate_k(thr_copy.partition_S(cX), limit=shape[1])
        )
        copy = partial(copy_utils.copy, pred=tXpX)

        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        row = tXcX[0][0]
        target = Int32.zero
        target_weight = Float32.zero
        if row < shape[0]:
            target = Int32(mTarget[row])
            if const_expr(mWeight is not None):
                if target != ignore_index:
                    target_weight = Float32(mWeight[target])
            else:
                target_weight = 1.0

        if row < shape[0]:
            copy(tXgX, tXsX, is_async=True)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        if const_expr(not is_even_N):
            utils.fill_oob(tXsX, tXpX, -tXsX.element_type.inf)
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(Float32)

        # ---- SOFTCAP (modification): cap logits before the softmax reduce ----
        if const_expr(self.softcap > 0.0):
            x = cute.math.tanh(x * (1.0 / self.softcap), fastmath=True) * self.softcap
        elif const_expr(self.softcap < 0.0):
            # sigmoid form: a*sig((z+5)/7.5) = (a/2)*(1 + tanh((z+5)/15))
            x = (cute.math.tanh((x + 5.0) * (1.0 / 15.0), fastmath=True) + 1.0) * (-self.softcap * 0.5)
        if const_expr(self.softcap != 0.0):
            if const_expr(not is_even_N):
                # both caps map the -inf OOB fill to a finite value (-cap or 0),
                # corrupting the reduce: restore -inf on OOB lanes.
                xf = cute.make_rmem_tensor_like(tXrX, Float32)
                xf.store(x)
                utils.fill_oob(xf, tXpX, -Float32.inf)
                x = xf.load()

        target_logit = Float32.zero
        should_ignore = Boolean(target == ignore_index)
        if row < shape[0] and tXcX[0][1] == 0 and not should_ignore:
            if const_expr(cute.rank(mTargetLogit.shape) == 2):
                target_logit = Float32(mTargetLogit[row, target])
            else:
                assert cute.rank(mTargetLogit.shape) == 1
                target_logit = Float32(mTargetLogit[row])
        # ---- SOFTCAP (modification): the target logit is read RAW from gmem ----
        if const_expr(self.softcap > 0.0):
            target_logit = (
                cute.math.tanh(target_logit * (1.0 / self.softcap), fastmath=True) * self.softcap
            )
        elif const_expr(self.softcap < 0.0):
            target_logit = (
                cute.math.tanh((target_logit + 5.0) * (1.0 / 15.0), fastmath=True) + 1.0
            ) * (-self.softcap * 0.5)

        if const_expr(not self.online_softmax):
            max_x = row_reduce(
                x,
                cute.ReductionOp.MAX,
                threads_per_row,
                reduction_buffer[None, None, 0],
                mbar_ptr + 0 if const_expr(self.cluster_n > 1) else None,
                init_val=-Float32.inf,
                hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
            )
            if const_expr(self.reload_from == "smem"):
                cute.autovec_copy(tXsX, tXrX)
                x = tXrX.load().to(Float32)
                # SOFTCAP: reloaded raw logits must be re-capped
                if const_expr(self.softcap > 0.0):
                    x = cute.math.tanh(x * (1.0 / self.softcap), fastmath=True) * self.softcap
                elif const_expr(self.softcap < 0.0):
                    x = (cute.math.tanh((x + 5.0) * (1.0 / 15.0), fastmath=True) + 1.0) * (-self.softcap * 0.5)
                if const_expr(self.softcap != 0.0):
                    if const_expr(not is_even_N):
                        xf = cute.make_rmem_tensor_like(tXrX, Float32)
                        xf.store(x)
                        utils.fill_oob(xf, tXpX, -Float32.inf)
                        x = xf.load()
            log2_e = math.log2(math.e)
            exp_x = cute.math.exp2(x * log2_e - (max_x * log2_e), fastmath=False)
            denom = row_reduce(
                exp_x,
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer[None, None, 1],
                mbar_ptr + 1 if const_expr(self.cluster_n > 1) else None,
                init_val=0.0,
            )
        else:
            max_x, denom, exp_x = online_softmax_reduce(
                x,
                threads_per_row,
                reduction_buffer[None, None, 0],
                mbar_ptr,
                hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
                return_exp_x=const_expr(mdX is not None),
            )

        if (
            tXcX[0][1] == 0
            and row < shape[0]
            and (self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0)
        ):
            lse = max_x + cute.math.log(denom, fastmath=True)
            loss_val = target_weight * (lse - target_logit) if not should_ignore else Float32.zero
            mLoss[row] = mLoss.element_type(loss_val)
            if const_expr(mLSE is not None):
                mLSE[row] = lse

        if const_expr(mdX is not None):
            denom_inv = (
                cute.arch.rcp_approx(denom)
                if not (denom == 0.0 or denom != denom or should_ignore)
                else Float32.zero
            )
            probs = exp_x * denom_inv
            gdX = cute.local_tile(mdX, tiler_mn, (bidx, cluster_y))
            tXgdX = thr_copy.partition_D(gdX)
            tXrdX = cute.make_rmem_tensor_like(tXgdX)
            tXcFull = thr_copy.partition_S(cX)
            tXrdX_f32 = cute.make_rmem_tensor_like(tXrX, Float32)
            tXrdX_f32.store(probs)
            if not should_ignore:
                for i in cutlass.range(cute.size(tXrX), unroll_full=True):
                    tXrdX_f32[i] = tXrdX_f32[i] if tXcFull[i][1] != target else tXrdX_f32[i] - 1.0
            if const_expr(mWeight is not None):
                tXrdX_f32.store(tXrdX_f32.load() * target_weight)
            # ---- SOFTCAP (modification): chain dzc/dz = 1 - tanh^2 = 1 - (zc/cap)^2 ----
            if const_expr(self.softcap > 0.0):
                t = x * (1.0 / self.softcap)  # x is already capped: tanh(z/cap) = x/cap
                tXrdX_f32.store(tXrdX_f32.load() * (1.0 - t * t))
            elif const_expr(self.softcap < 0.0):
                # sigmoid form: dzc/dz = zc*(1 - zc/a)/c, c=7.5; x is already capped (=zc)
                a_inv = 1.0 / (-self.softcap)
                tXrdX_f32.store(tXrdX_f32.load() * (x * (1.0 - x * a_inv) * (1.0 / 7.5)))
            tXrdX.store(tXrdX_f32.load().to(tXrdX.element_type))
            if row < shape[0]:
                copy(tXrdX, tXgdX)

    @staticmethod
    @jit_cache
    def compile(
        dtype, target_dtype, target_logit_dtype, N,
        has_lse, has_dx, weight_dtype, target_logit_ndim, softcap,
    ):
        batch_sym = cute.sym_int()
        div = math.gcd(128 // dtype.width, N)
        x_cute = fake_tensor(dtype, (batch_sym, N), div)
        dx_cute = fake_tensor(dtype, (batch_sym, N), div) if has_dx else None
        target_cute = fake_tensor(target_dtype, (batch_sym,))
        if target_logit_dtype is not None:
            if target_logit_ndim == 2:
                target_logit_cute = fake_tensor(target_logit_dtype, (batch_sym, cute.sym_int()), div)
            else:
                target_logit_cute = fake_tensor(target_logit_dtype, (batch_sym,))
        else:
            target_logit_cute = None
        loss_cute = fake_tensor(Float32, (batch_sym,))
        lse_cute = fake_tensor(Float32, (batch_sym,)) if has_lse else None
        weight_cute = fake_tensor(weight_dtype, (N,)) if weight_dtype is not None else None
        return cute.compile(
            CrossEntropySoftcap(dtype, N, online_softmax=not has_dx, softcap=softcap),
            x_cute, target_cute, target_logit_cute, loss_cute, lse_cute, dx_cute, weight_cute,
            Int32(0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )


@cute_op("sol::ce_softcap_fwd_out", mutates_args={"loss", "lse"})
def ce_softcap_fwd_out(
    x: Tensor,
    target: Tensor,
    loss: Tensor,
    lse: Tensor,
    softcap: float,
    ignore_index: int = -100,
) -> None:
    """Softcap CE forward: loss/lse in the capped domain. Always returns lse (needed
    for the softcap backward); no fused-dx variant (bwd kernel handles the chain)."""
    assert x.dim() == 2 and target.dim() == 1
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert target.dtype in [torch.int32, torch.int64]
    if x.size(0) == 0:
        return
    N = x.size(1)
    dtype = torch2cute_dtype_map[x.dtype]
    target_dtype = torch2cute_dtype_map[target.dtype]
    CrossEntropySoftcap.compile(
        dtype, target_dtype, None, N, True, False, None, None, float(softcap)
    )(x, target, None, loss, lse, None, None, Int32(ignore_index))


class CrossEntropyBackwardSoftcap:
    def __init__(self, dtype: Type[cutlass.Numeric], N: int, softcap: float = 0.0):
        self.dtype = dtype
        self.N = N
        self.softcap = float(softcap)
        self.vecsize = 128 // dtype.width

    def _threads_per_row(self):
        N = min(self.N, 16384)
        for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128)]:
            if N <= limit:
                return threads
        return 256

    def _get_tiled_copy(self, vecsize: int):
        assert self.N % vecsize == 0
        N = min(self.N, 16384)
        num_threads = 128 if N <= 16384 else 256
        threads_per_row = self._threads_per_row()
        cols_per_block = num_threads // threads_per_row
        num_blocks_N = cute.ceil_div(N // vecsize, threads_per_row)
        tiler_mn = (cols_per_block, vecsize * num_blocks_N * threads_per_row)
        tiled_copy = copy_utils.tiled_copy_2d(
            self.dtype, threads_per_row, num_threads, num_copy_elems=vecsize
        )
        return tiled_copy, tiler_mn, threads_per_row

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mTarget: cute.Tensor,
        mDLoss: cute.Tensor,
        mdX: cute.Tensor,
        mLSE: cute.Tensor,
        mWeight: Optional[cute.Tensor],
        ignore_index: Int32,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        assert mdX.element_type == self.dtype
        vecsize = math.gcd(self.N, 128 // self.dtype.width)
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        mDLoss, mTarget, mLSE = [
            layout_utils.expand(X, dim=1, size=self.N) for X in (mDLoss, mTarget, mLSE)
        ]
        self.kernel(
            mX, mTarget, mDLoss, mdX, mLSE, mWeight, ignore_index,
            mX.shape, tiler_mn, tiled_copy, threads_per_row,
        ).launch(
            grid=[
                cute.ceil_div(mX.shape[0], tiler_mn[0]),
                cute.ceil_div(mX.shape[1], tiler_mn[1]),
                1,
            ],
            block=[num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mTarget: cute.Tensor,
        mDLoss: cute.Tensor,
        mdX: cute.Tensor,
        mLSE: cute.Tensor,
        mWeight: Optional[cute.Tensor],
        ignore_index: Int32,
        shape: cute.Shape,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16
        )

        idX = cute.make_identity_tensor(shape)
        gX, gdX, cX = [cute.local_tile(mT, tiler_mn, (bidx, bidy)) for mT in (mX, mdX, idX)]

        thr_copy = tiled_copy.get_slice(tidx)

        tXgX = thr_copy.partition_S(gX)
        tXsX = thr_copy.partition_D(sX)
        tXcX = thr_copy.partition_S(cX)[(0, None), None, None]
        tXcFull = thr_copy.partition_S(cX)
        tXgdX = thr_copy.partition_D(gdX)
        tXrX, tXrdX = [cute.make_rmem_tensor_like(thr) for thr in (tXgX, tXgdX)]

        is_even_N = const_expr(shape[1] % tiler_mn[1] == 0)
        tXpX = (
            None if is_even_N else copy_utils.predicate_k(thr_copy.partition_S(cX), limit=shape[1])
        )
        copy = partial(copy_utils.copy, pred=tXpX)

        row = tXcX[0][0]
        target = Int32.zero
        target_weight = Float32.zero
        if row < shape[0]:
            target = Int32(mTarget[row])
            if const_expr(mWeight is not None):
                if target != ignore_index:
                    target_weight = Float32(mWeight[target])
            else:
                target_weight = 1.0

        if row < shape[0]:
            copy(tXgX, tXsX, is_async=True)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        if const_expr(not is_even_N):
            utils.fill_oob(tXsX, tXpX, -tXsX.element_type.inf)
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(Float32)

        # ---- SOFTCAP (modification): probs come from CAPPED logits; keep tanh for the chain ----
        t = Float32.zero  # tanh(z/cap) [tanh form] or tanh((z+5)/15) [sigmoid form]
        if const_expr(self.softcap > 0.0):
            t = cute.math.tanh(x * (1.0 / self.softcap), fastmath=True)
            x = t * self.softcap
        elif const_expr(self.softcap < 0.0):
            # sigmoid form: zc = (a/2)*(1 + tanh((z+5)/15)), a = |softcap|
            t = cute.math.tanh((x + 5.0) * (1.0 / 15.0), fastmath=True)
            x = (t + 1.0) * (-self.softcap * 0.5)

        dloss = Float32.zero
        lse = Float32.zero
        if row < shape[0]:
            should_ignore = Boolean(target == ignore_index)
            if not should_ignore:
                dloss = Float32(mDLoss[row])
            lse = Float32(mLSE[row])

        log2_e = math.log2(math.e)
        probs = cute.math.exp2(x * log2_e - (lse * log2_e), fastmath=True)
        prob_shifted = probs - 1.0
        mask = cute.make_rmem_tensor_like(tXrX, Boolean)
        for i in cutlass.range(cute.size(tXcFull), unroll_full=True):
            mask[i] = tXcFull[i][1] == target
        grad = cute.where(mask.load(), prob_shifted, probs)
        grad = grad * dloss * target_weight
        # ---- SOFTCAP (modification): chain rule ----
        if const_expr(self.softcap > 0.0):
            grad = grad * (1.0 - t * t)          # dzc/dz = 1 - tanh^2(z/cap)
        elif const_expr(self.softcap < 0.0):
            # dzc/dz = zc*(1-zc/a)/7.5 = a*(1-t^2)/30 with t = tanh((z+5)/15)
            grad = grad * ((1.0 - t * t) * (-self.softcap * (1.0 / 30.0)))

        tXrdX.store(grad.to(tXrdX.element_type))
        if row < shape[0]:
            copy(tXrdX, tXgdX)

    @staticmethod
    @jit_cache
    def compile(dtype, target_dtype, N, weight_dtype, softcap):
        batch_sym = cute.sym_int()
        div = math.gcd(128 // dtype.width, N)
        x_cute, dx_cute = [fake_tensor(dtype, (batch_sym, N), div)] * 2
        target_cute = fake_tensor(target_dtype, (batch_sym,))
        dloss_cute = cute.runtime.make_fake_tensor(Float32, (batch_sym,), stride=(cute.sym_int64(),))
        lse_cute = fake_tensor(Float32, (batch_sym,))
        weight_cute = fake_tensor(weight_dtype, (N,)) if weight_dtype is not None else None
        return cute.compile(
            CrossEntropyBackwardSoftcap(dtype, N, softcap=softcap),
            x_cute, target_cute, dloss_cute, dx_cute, lse_cute, weight_cute,
            Int32(0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )


def _ce_softcap_backward(x, target, dloss, lse, dx, softcap, ignore_index=-100):
    assert x.dim() == 2 and target.dim() == 1 and dloss.dim() == 1 and lse.dim() == 1
    if x.size(0) == 0:
        return
    N = x.size(1)
    dtype = torch2cute_dtype_map[x.dtype]
    target_dtype = torch2cute_dtype_map[target.dtype]
    CrossEntropyBackwardSoftcap.compile(dtype, target_dtype, N, None, float(softcap))(
        x, target, dloss, dx, lse, None, Int32(ignore_index)
    )


@cute_op("sol::ce_softcap_bwd_out", mutates_args={"dx"})
def ce_softcap_bwd_out(
    x: Tensor, target: Tensor, dloss: Tensor, lse: Tensor, dx: Tensor,
    softcap: float, ignore_index: int = -100,
) -> None:
    _ce_softcap_backward(x, target, dloss, lse, dx, softcap, ignore_index)


class SoftcapCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, target, softcap, ignore_index=-100, inplace_backward=True):
        M = x.size(0)
        loss = torch.empty(M, device=x.device, dtype=torch.float32)
        lse = torch.empty(M, device=x.device, dtype=torch.float32)
        ce_softcap_fwd_out(x, target, loss, lse, float(softcap), ignore_index)
        ctx.save_for_backward(x, target, lse)
        ctx.softcap = float(softcap)
        ctx.ignore_index = ignore_index
        ctx.inplace_backward = inplace_backward
        return loss

    @staticmethod
    def backward(ctx, dloss):
        x, target, lse = ctx.saved_tensors
        if ctx.inplace_backward and not torch.compiler.is_compiling():
            dx = x
            _ce_softcap_backward(x, target, dloss, lse, dx, ctx.softcap, ctx.ignore_index)
        else:
            dx = torch.empty_like(x)
            ce_softcap_bwd_out(x, target, dloss, lse, dx, ctx.softcap, ctx.ignore_index)
        return dx, None, None, None, None


def linear_cross_entropy_softcap_func(
    x: Tensor,           # (..., d)
    weight: Tensor,      # (V, d)
    target: Tensor,      # (...,)
    softcap: float,
    ignore_index: int = -100,
    reduction: Literal["none", "mean", "sum"] = "mean",
    inplace_backward: bool = True,
) -> Tensor:
    """F.linear + softcap CE (CuTeDSL). The capped-chained dlogits flow through
    torch autograd's linear backward (dgrad/wgrad GEMMs unmodified)."""
    assert softcap != 0.0, "use quack's plain CE for softcap == 0 (sign encodes form: >0 tanh, <0 sigmoid)"
    y = F.linear(x.reshape(-1, x.shape[-1]), weight)  # (M, V) raw logits
    t = target.reshape(-1)
    loss = SoftcapCrossEntropyFunction.apply(y, t, softcap, ignore_index, inplace_backward)
    if reduction == "mean":
        return loss.sum() / (t != ignore_index).sum().float()
    elif reduction == "sum":
        return loss.sum()
    return loss
