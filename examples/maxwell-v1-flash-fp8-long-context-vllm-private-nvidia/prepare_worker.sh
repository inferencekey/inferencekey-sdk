#!/usr/bin/env bash
# Prepare a self-hosted InferenceKey worker (Linux, NVIDIA, CUDA 13 driver) to
# serve Maxwell-V1.0-Flash with the stock `vllm` backend. Run it once, as the
# user the worker runs as, BEFORE `python main.py`.
#
# Why: the model's serving code lives in its authors' vLLM fork
# (github.com/maxwell-model/vllm, branch maxwell-v1-flash), which is not on
# PyPI. The fork is one commit of Python on top of upstream vLLM a9eafde5, so
# this script installs the upstream nightly wheel of that commit (CUDA kernels
# for sm_75 to sm_120, including Ada sm_89) and copies the fork's Python files
# over it. The authors' Docker image is built for sm_90 and sm_103 only, so it
# does not run on Ada or Ampere cards.
#
# It installs into the venv path the worker derives from the workload's
# vllm_version, so the worker finds vLLM already importable and skips its own
# pip install.
set -euo pipefail

VLLM_VERSION="0.30.1rc1.dev219+ga9eafde59"
UPSTREAM_COMMIT="a9eafde59cbd55182dc2265cc398b7186f0a0eaa"
FORK_COMMIT="4fb7bcae00e39a54bbc368e81fb4d16e11891426"
RUNTIME_DIR="${INFERENCEKEY_RUNTIME_DIR:-/opt/inferencekey}"
# InstallProfile key of {backend: vllm, version: VLLM_VERSION} (no torch or
# transformers pin, CUDA host). Change it only together with VLLM_VERSION.
VENV="$RUNTIME_DIR/venvs/vllm-26a731c5684ad61c"
WHEEL="vllm-${VLLM_VERSION}-cp38-abi3-manylinux_2_28_x86_64.whl"
FILES=(
  vllm/config/attention.py
  vllm/config/speculative.py
  vllm/cute_utils/__init__.py
  vllm/model_executor/layers/fused_moe/experts/trtllm_bf16_moe.py
  vllm/model_executor/models/config.py
  vllm/model_executor/models/registry.py
  vllm/model_executor/warmup/kernel_warmup.py
  vllm/model_executor/warmup/maxwell_v1_flash_warmup.py
  vllm/models/maxwell_v1_flash/__init__.py
  vllm/models/maxwell_v1_flash/common/__init__.py
  vllm/models/maxwell_v1_flash/common/indexer.py
  vllm/models/maxwell_v1_flash/common/linear.py
  vllm/models/maxwell_v1_flash/common/ops/__init__.py
  vllm/models/maxwell_v1_flash/common/ops/index_topk.py
  vllm/models/maxwell_v1_flash/common/ops/sparse_attn.py
  vllm/models/maxwell_v1_flash/envs.py
  vllm/models/maxwell_v1_flash/nvidia/__init__.py
  vllm/models/maxwell_v1_flash/nvidia/model.py
  vllm/models/maxwell_v1_flash/nvidia/mtp.py
  vllm/models/maxwell_v1_flash/nvidia/ops/__init__.py
  vllm/models/maxwell_v1_flash/nvidia/ops/fused_qk_norm_rope.py
  vllm/models/maxwell_v1_flash/nvidia/ops/index_decode_score.py
  vllm/models/maxwell_v1_flash/nvidia/ops/mega_mhc.py
  vllm/models/maxwell_v1_flash/nvidia/ops/mhc.py
  vllm/models/maxwell_v1_flash/nvidia/ops/mxfp6.py
  vllm/models/maxwell_v1_flash/nvidia/ops/split_bf16_gemm.py
  vllm/models/maxwell_v1_flash/nvidia/sparse_attention.py
  vllm/transformers_utils/config.py
  vllm/transformers_utils/configs/__init__.py
  vllm/transformers_utils/configs/maxwell_v1_flash.py
  vllm/transformers_utils/model_arch_config_convertor.py
)

PYTHON="${PYTHON:-python3.12}"
command -v "$PYTHON" >/dev/null || { echo "$PYTHON is required (set PYTHON=...)" >&2; exit 1; }
mkdir -p "$RUNTIME_DIR/venvs"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

"$PYTHON" -m venv "$VENV"
"$VENV/bin/pip" install -q uv
curl -fsSL -o "$TMP/$WHEEL" \
  "https://wheels.vllm.ai/$UPSTREAM_COMMIT/${WHEEL//+/%2B}"
"$VENV/bin/uv" pip install --python "$VENV/bin/python" "$TMP/$WHEEL" ninja \
  --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match

# Overlay the fork's Python files (the only files its commit changes under vllm/).
SITE=$("$VENV/bin/python" -c "import site; print(site.getsitepackages()[0])")
for f in "${FILES[@]}"; do
  mkdir -p "$SITE/$(dirname "$f")"
  curl -fsSL -o "$SITE/$f" "https://raw.githubusercontent.com/maxwell-model/vllm/$FORK_COMMIT/$f"
done
# vLLM JIT-builds a few kernels with ninja at warm-up.
ln -sf "$VENV/bin/ninja" /usr/local/bin/ninja 2>/dev/null || true

"$VENV/bin/python" -c "import vllm; assert vllm.__version__ == '$VLLM_VERSION', vllm.__version__; \
import vllm.models.maxwell_v1_flash.nvidia.model; print('ready:', vllm.__version__)"
echo "venv: $VENV"
