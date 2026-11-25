#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'Usage: bash infer_scene_wsl.sh SCENE_DIR OUTPUT_DIR WEIGHTS_NPZ\n'
}
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    usage
    exit 0
fi
if [[ "$#" != 3 ]]; then
    usage >&2
    exit 2
fi

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
scene="$1"
output="$2"
weights="$3"
if [[ ! -f "$scene/scene.json" || ! -f "$weights" ]]; then
    printf 'Expected scene.json under %s and an inference weights file at %s.\n' "$scene" "$weights" >&2
    exit 2
fi
mkdir -p -- "$output"
stage="$(mktemp -d -- "$output/.dhd-scene.XXXXXXXX")"
files=(predictions.npz bev.npz tracks.json bev.png)
cleanup() {
    rm -f -- "$stage/input.npz"
    for name in "${files[@]}"; do
        rm -f -- "$stage/$name"
    done
    rmdir -- "$stage"
}
trap cleanup EXIT
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"

bash "$root/infer_jtorch_wsl.sh" --scene-io prepare "$scene" "$weights" "$stage/input.npz"
bash "$root/infer_jtorch_wsl.sh" "$weights" "$stage/predictions.npz" --input "$stage/input.npz"
bash "$root/infer_jtorch_wsl.sh" --scene-io finish "$stage/predictions.npz" "$stage"
for name in "${files[@]}"; do
    mv -fT -- "$stage/$name" "$output/$name"
done
printf 'Scene results saved in %s\n' "$output"
