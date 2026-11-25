#!/usr/bin/env bash
set -eu

runtime="${DHD_JTORCH_RUNTIME:-$HOME/.local/share/dhd-jtorch}"
python_base="$HOME/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu"
if [[ ! -x "$runtime/env/bin/python" ]]; then
    printf 'Run bash install_jtorch.sh to install the DHD runtime.\n' >&2
    exit 1
fi
if [[ -r "$runtime/python-config-path" ]]; then
    read -r installed_config < "$runtime/python-config-path"
fi
if [[ -r "$runtime/cuda-path" ]]; then
    read -r installed_cuda < "$runtime/cuda-path"
fi
export python_config_path="${python_config_path:-${installed_config:-$python_base/bin/python3.11-config}}"
export CUDA_PATH="${CUDA_PATH:-${installed_cuda:-$runtime/cuda}}"
export nvcc_path="$CUDA_PATH/bin/nvcc"
nvrtc_lib="$runtime/env/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib"
export LD_LIBRARY_PATH="$CUDA_PATH/lib:$CUDA_PATH/lib64:$nvrtc_lib:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" cache_name="${cache_name:-dhd_cuda}" use_mkl=0
export NVIDIA_TF32_OVERRIDE="${NVIDIA_TF32_OVERRIDE:-0}"

if [[ "${1:-}" == --check ]]; then
    exec "$runtime/env/bin/python" -B -c '
import torch
import jittor
if torch.Tensor is not jittor.Var:
    raise RuntimeError("Expected JTorch to provide the torch module.")
jittor.flags.use_cuda = 1
if (torch.tensor([1.0]) + 2).item() != 3:
    raise RuntimeError("JTorch CUDA arithmetic check failed.")
print("JTorch CUDA check passed.")
'
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == --scene-io ]]; then
    shift
    export PYTHONPATH="$script_dir${PYTHONPATH:+:$PYTHONPATH}"
    exec "$runtime/env/bin/python" -B -m dhd.scene_io "$@"
fi
exec "$runtime/env/bin/python" -B "$script_dir/infer_single.py" run "$@" --device cuda:0
