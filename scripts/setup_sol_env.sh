#!/usr/bin/env bash
# Build the SOL environment. RUN THIS ONLY WHEN A GPU IS FREE: it compiles
# CuTeDSL kernels (quack / FA4) and needs a device to build+verify against.
#
# Target box: B300 (sm100), CUDA 13.3 / driver 610, Python 3.12, torch 2.12 (cu13).
#
# Dependency conflict handled here (pip cannot resolve it as one extra):
#   * gram-newton-schulz 0.1.6 hard-pins nvidia-cutlass-dsl==4.5.2 + quack==0.5.0
#   * we WANT nvidia-cutlass-dsl 4.6.0, and quack-kernels 0.6.1 requires ==4.6.0
#   => install cutlass-dsl 4.6.0 + quack 0.6.1 first, then GNS with --no-deps.
set -euo pipefail

TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"  # adjust to the cu13 wheel index
PY="${PY:-uv pip install}"        # installs into the uv project .venv
RUN="${RUN:-uv run --no-sync python}"  # run python IN that venv (no re-resolve); box has no bare `python`
FA4_FROM_SOURCE="${FA4_FROM_SOURCE:-auto}"   # auto|always|never

# --- 0. refuse to run unless a GPU is GENUINELY free (kernel builds need a device) --
# NB: util=0 with vram~100% is the sglang server idle-but-UP (autotuning paused) --
# NOT free. Key on VRAM actually released: >=200GB free means the server let go.
free=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
       | awk -F', ' '$3-$2 > 200000 {print $1}' | head -1 || true)
if [[ -z "$free" ]]; then
  echo "ERROR: no GPU with >200GB free -- box still busy (b8zhong's sglang server up)." >&2
  echo "       util=0/vram=100 does NOT count as free. Wait for VRAM to release." >&2
  exit 1
fi
echo "=== building SOL env; will verify on GPU $free ==="
export CUDA_VISIBLE_DEVICES="$free"

# --- 1. torch 2.12 (cu13) ------------------------------------------------------
$PY --upgrade "torch>=2.12,<2.13" --index-url "$TORCH_INDEX"
$RUN -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# --- 1b. base project runtime deps (uv sync would refetch torch/hit the FA4 prerelease) ---
$PY einops jaxtyping numpy psutil regex tqdm wandb

# --- 2. cuDNN (LATEST) + frontend ---------------------------------------------
# 9.24.0.43 is the latest; ABI-compatible over torch 2.12/2.13's bundled 9.20
# (same libcudnn.so.9). Decoupled from cutlass-dsl. Do NOT use frontend[cutedsl]
# (that extra pins cutlass-dsl==4.5.0, conflicting with our 4.6.0).
$PY "nvidia-cudnn-cu13==9.24.0.43" "nvidia-cudnn-frontend>=1.26"

# --- 3. TransformerEngine (OPTIONAL) -------------------------------------------
# Only needed for --rmsnorm te (default is torch) and a TE-MXFP8 route we don't use
# (MXFP8 goes via quack per the CuTeDSL study). TE 2.17's ep.cpp fails to compile
# against this box's older NCCL headers (ncclWindow_vidmem mismatch), so it is
# non-fatal and skippable: SKIP_TE=1 to skip the (slow, failing) build attempt.
if [[ "${SKIP_TE:-0}" != "1" ]]; then
  $PY --no-build-isolation "transformer-engine[pytorch]>=2.16" \
    || echo "WARN: TE build failed (NCCL header mismatch) -- skipping; not needed for bf16 SOL run."
fi

# --- 4. CuTeDSL chain: cutlass-dsl 4.6.0 -> quack 0.6.1 -> GNS (--no-deps) ------
$PY "nvidia-cutlass-dsl[cu13]==4.6.0"
$PY "quack-kernels==0.6.1"                       # requires cutlass-dsl==4.6.0 (satisfied)
$PY --no-deps "gram-newton-schulz==0.1.6"        # skip its cutlass==4.5.2 / quack==0.5.0 pins

# --- 5. FlashAttention-4 (CuTe-DSL sm100 fwd/bwd); wheel, else from source -----
if [[ "$FA4_FROM_SOURCE" != "always" ]] && $PY --prerelease=allow "flash-attn-4[cu13]" --no-build-isolation; then
  echo "FA4 installed from wheel."
elif [[ "$FA4_FROM_SOURCE" != "never" ]]; then
  echo "FA4 wheel unavailable -> building from source."
  SRC="${SRC:-/tmp/flash-attention-src}"
  rm -rf "$SRC" && git clone --depth 1 https://github.com/Dao-AILab/flash-attention "$SRC"
  $PY --no-build-isolation -e "$SRC/flash_attn/cute[dev,cu13]"
fi

# --- 6. verify -----------------------------------------------------------------
$RUN - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| sm", torch.cuda.get_device_capability())
try:
    import transformer_engine.pytorch as te; print("TE ok")
except Exception as e:
    print("TE unavailable (optional, ok):", type(e).__name__)
from gram_newton_schulz import Muon, GramNewtonSchulz, POLAR_EXPRESS_COEFFICIENTS; print("gram-newton-schulz ok")
from flash_attn.cute import flash_attn_func; print("FA4 flash_attn_func ok")
import cudnn; print("cudnn-frontend", cudnn.__version__)
PY
echo "=== SOL env ready. Now: bash scripts/bench_sol.sh ==="
