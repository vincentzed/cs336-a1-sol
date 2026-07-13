"""Route B: TransformerEngine MXFP8 bench on Modal B200 (separate app from modal_bench).

TE never built on the local box (ancient system NCCL headers vs ep.cpp). Modal's
Ubuntu-24.04 CUDA-13 image + apt libnccl-dev should have modern headers; try the
wheel/source chain and bench te.Linear MXFP8 vs bf16 at our exact GEMM mix.

Run: ~/.local/bin/modal run modal_te.py::bench_main
"""
import modal

app = modal.App("cs336-te-mxfp8")

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "clang")  # clang++: TE's ext build invokes it (attempt-4 failure); nccl already present
    .pip_install("torch==2.12.1", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("numpy", "einops", "packaging", "pydantic",
                 "wheel", "setuptools", "pybind11", "ninja", "cmake")  # TE build deps
    .pip_install("nvidia-cudnn-cu13==9.24.0.43")  # cudnn.h lives in the wheel, not the image
    # TE install chain: 2.17 wheel-or-source, else 2.16, with pytorch framework forced.
    # Fix for build attempt #3 ("fatal error: cudnn.h"): expose the pip-wheel cuDNN
    # headers/libs to the compiler via CPATH/LIBRARY_PATH.
    .run_commands(
        "CUDNN_ROOT=/usr/local/lib/python3.12/site-packages/nvidia/cudnn; "
        "export CPATH=$CUDNN_ROOT/include:${CPATH:-} "
        "LIBRARY_PATH=$CUDNN_ROOT/lib:${LIBRARY_PATH:-} "
        "LD_LIBRARY_PATH=$CUDNN_ROOT/lib:${LD_LIBRARY_PATH:-}; "
        "NVTE_FRAMEWORK=pytorch pip install --no-build-isolation 'transformer-engine[pytorch]==2.17' "
        "|| NVTE_FRAMEWORK=pytorch pip install --no-build-isolation 'transformer-engine[pytorch]==2.16' "
        "|| NVTE_FRAMEWORK=pytorch pip install --no-build-isolation 'transformer-engine[pytorch]'",
        gpu="B200",  # TE source build JIT-detects arch; give it a GPU
    )
)


@app.function(image=image, gpu="B200", timeout=1800)
def bench():
    import time

    import torch

    torch.manual_seed(0)
    dev = "cuda"
    M = 256 * 512  # batch*ctx = 131072 rows, our exact GEMM mix below
    SHAPES = [(1024, 3072), (1024, 1024), (1024, 4096), (4096, 1024)]  # qkv, wo, fc1, fc2
    LAYERS = 16
    results = {}

    def bench_stack(make_linear, autocast_ctx, label):
        blocks = []
        for _ in range(LAYERS):
            blocks.append([make_linear(k, n) for (k, n) in SHAPES])
        x0 = torch.randn(M, 1024, device=dev, dtype=torch.bfloat16, requires_grad=True)
        def step():
            x = x0
            with autocast_ctx():
                for lin_qkv, lin_wo, lin_fc1, lin_fc2 in blocks:
                    a = lin_qkv(x)
                    x = x + lin_wo(a[:, :1024])
                    h = torch.nn.functional.relu(lin_fc1(x)) ** 2
                    x = x + lin_fc2(h)
                loss = x.float().square().mean()
            loss.backward()
            x0.grad = None
            for row in blocks:
                for lin in row:
                    for p in lin.parameters():
                        p.grad = None
        for _ in range(5):
            step()
        torch.cuda.synchronize()
        ts = []
        for _ in range(20):
            t0 = time.perf_counter(); step(); torch.cuda.synchronize()
            ts.append(1000 * (time.perf_counter() - t0))
        ts.sort()
        results[label] = ts[len(ts) // 2]
        print(f"{label}: {results[label]:.1f} ms (median-20)", flush=True)
        del blocks
        torch.cuda.empty_cache()

    import contextlib

    # 1) plain torch bf16 (cuBLAS reference)
    bench_stack(
        lambda k, n: torch.nn.Linear(k, n, bias=False, device=dev, dtype=torch.bfloat16),
        contextlib.nullcontext, "torch_bf16",
    )

    import transformer_engine.pytorch as te
    print("TE version:", te.__version__ if hasattr(te, "__version__") else "?", flush=True)
    from transformer_engine.common import recipe as te_recipe
    rec = None
    for name in ("MXFP8BlockScaling", "Float8CurrentScaling", "DelayedScaling"):
        if hasattr(te_recipe, name):
            rec = getattr(te_recipe, name)()
            print("recipe:", name, flush=True)
            break

    # 2) te.Linear bf16 (isolates TE-layer overhead from fp8 gain)
    bench_stack(
        lambda k, n: te.Linear(k, n, bias=False, params_dtype=torch.bfloat16, device=dev),
        contextlib.nullcontext, "te_bf16",
    )
    # 3) te.Linear under MXFP8 autocast
    bench_stack(
        lambda k, n: te.Linear(k, n, bias=False, params_dtype=torch.bfloat16, device=dev),
        lambda: te.fp8_autocast(enabled=True, fp8_recipe=rec), "te_mxfp8",
    )
    return results


@app.local_entrypoint()
def bench_main():
    print("RESULTS:", bench.remote())
