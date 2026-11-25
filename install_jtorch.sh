#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runtime="${DHD_JTORCH_RUNTIME:-$HOME/.local/share/dhd-jtorch}"
mkdir -p -- "$runtime"
runtime="$(cd -- "$runtime" && pwd)"

fail() {
    printf '%s\n' "$*" >&2
    exit 1
}

if [[ "$#" != 0 && ( "$#" != 1 || "$1" != --with-scene-io ) ]]; then
    fail "Usage: bash install_jtorch.sh [--with-scene-io]"
fi

[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] ||
    fail "DHD JTorch setup requires x86-64 Linux or WSL2."
for command in curl tar xz g++ nvidia-smi; do
    command -v "$command" >/dev/null || fail "Install $command before running this setup."
done
nvidia-smi -L

venv="$runtime/env/bin/python"
managed_python="$HOME/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu/bin/python3.11"
if [[ -x "$venv" ]]; then
    python="$venv"
elif [[ -n "${DHD_PYTHON:-}" ]]; then
    python="$DHD_PYTHON"
elif [[ -x "$managed_python" ]]; then
    python="$managed_python"
elif command -v python3.11-config >/dev/null && command -v python3.11 >/dev/null; then
    python="$(command -v python3.11)"
else
    if command -v uv >/dev/null; then
        uv="$(command -v uv)"
    else
        uv="$runtime/bin/uv"
        if [[ ! -x "$uv" ]]; then
            curl -fLsS --retry 3 https://astral.sh/uv/0.13.7/install.sh -o "$runtime/install-uv.sh"
            UV_UNMANAGED_INSTALL="$runtime/bin" sh "$runtime/install-uv.sh"
        fi
    fi
    export UV_PYTHON_INSTALL_DIR="$runtime/python"
    export UV_PYTHON_BIN_DIR="$runtime/bin"
    "$uv" python install 3.11.18
    python="$("$uv" python find --managed-python 3.11.18)"
fi
"$python" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 11))' ||
    fail "Set DHD_PYTHON to a Python 3.11 interpreter."

config="${python_config_path:-$("$python" -c 'import sys; print(sys.base_prefix)')/bin/python3.11-config}"
if [[ ! -x "$config" ]]; then
    config="$(command -v python3.11-config || true)"
fi
[[ -x "$config" ]] || fail "Python 3.11 development headers and python3.11-config are required."
printf '%s\n' "$config" > "$runtime/python-config-path"

if [[ ! -x "$venv" ]]; then
    "$python" -m venv "$runtime/env"
fi
if ! "$venv" -m pip --version >/dev/null 2>&1; then
    "$venv" -m ensurepip --upgrade
fi
"$venv" -m pip install --disable-pip-version-check -r "$root/requirements-jtorch.txt"
"$venv" -m pip install --disable-pip-version-check --no-deps efficientnet-pytorch==0.7.0

cuda="${CUDA_PATH:-$runtime/cuda}"
if [[ ! -e "$cuda" && "$cuda" == "$runtime/cuda" ]]; then
    "$venv" "$root/scripts/install_cuda.py" "$cuda" "${DHD_CUDA_CACHE:-$runtime/downloads}"
fi
[[ -x "$cuda/bin/nvcc" ]] || fail "CUDA nvcc not found under $cuda. Set CUDA_PATH to a CUDA 12.8 toolkit."
cuda="$(cd -- "$cuda" && pwd)"
printf '%s\n' "$cuda" > "$runtime/cuda-path"

DHD_JTORCH_RUNTIME="$runtime" bash "$root/infer_jtorch_wsl.sh" --check
printf 'Setup complete. Run bash infer_jtorch_wsl.sh from the DHD repository.\n'
