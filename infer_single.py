"""Run one DHD FP32 inference batch with JTorch."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch


INPUT_KEYS = ("image", "intrinsics", "extrinsics", "future_egomotion")
OUTPUT_KEYS = ("segmentation", "instance_flow")
ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "ckpt/dhd-inference-weights.npz"
DEFAULT_OUTPUT = ROOT / "outputs/predictions.npz"


def synthetic_batch(cfg):
    frames = cfg.TIME_RECEPTIVE_FIELD
    cameras = len(cfg.IMAGE.NAMES)
    height, width = cfg.IMAGE.FINAL_DIM
    shape = (1, frames, cameras)
    image = np.random.default_rng(42).standard_normal(
        (*shape, 3, height, width), dtype=np.float32
    )
    camera_matrix = np.array(
        [[300, 0, width / 2], [0, 300, height / 2], [0, 0, 1]], dtype=np.float32
    )
    intrinsics = np.broadcast_to(camera_matrix, (*shape, 3, 3)).copy()
    extrinsics = np.broadcast_to(
        np.diag(np.array([1, -1, -1, 1], dtype=np.float32)), (*shape, 4, 4)
    ).copy()
    extrinsics[0, :, :, 0, 3] = np.arange(cameras, dtype=np.float32) * 6
    return image, intrinsics, extrinsics, np.zeros((1, frames, 6), dtype=np.float32)


def load_weights(model, archive):
    expected = model.state_dict()
    available = set(archive.files) - {"config"}
    missing = set(expected) - available
    unexpected = available - set(expected)
    # JTorch batch normalization has no PyTorch batch counter.
    unexpected = {key for key in unexpected if not key.endswith(".num_batches_tracked")}
    weights = {}
    mismatched = set()
    for key in set(expected) & available:
        weight = archive[key]
        shape = tuple(expected[key].shape)
        # JTorch represents scalar parameters as single-element vectors.
        if weight.ndim == 0 and shape == (1,):
            weight = weight.reshape(shape)
        if weight.shape != shape:
            mismatched.add(key)
        weights[key] = torch.tensor(weight)
    if missing or unexpected or mismatched:
        raise ValueError(
            f"Checkpoint mismatch. Missing: {sorted(missing)}. "
            f"Unexpected: {sorted(unexpected)}. Shape mismatch: {sorted(mismatched)}."
        )
    model.load_state_dict(weights)
    return len(expected)


def run_benchmark(args, model, inputs, synchronize, backend):
    def timed_forward():
        synchronize()
        start = perf_counter()
        output = model(*inputs, is_train=False)
        synchronize()
        forward_done = perf_counter()
        predictions = {key: output[key].detach().cpu().numpy() for key in OUTPUT_KEYS}
        return predictions, (forward_done - start) * 1000, (perf_counter() - start) * 1000

    _, first_forward, first_with_copy = timed_forward()
    for _ in range(args.warmup):
        timed_forward()

    forward_ms = []
    forward_with_copy_ms = []
    for _ in range(args.iterations):
        predictions, forward, with_copy = timed_forward()
        forward_ms.append(forward)
        forward_with_copy_ms.append(with_copy)

    results = {
        "backend": backend,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "device": args.device,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "first_forward_ms": first_forward,
        "first_forward_with_output_copy_ms": first_with_copy,
        "forward_ms": forward_ms,
        "forward_with_output_copy_ms": forward_with_copy_ms,
    }
    args.benchmark.parent.mkdir(parents=True, exist_ok=True)
    args.benchmark.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return predictions


def infer(args):
    jittor = sys.modules.get("jittor")
    using_jtorch = jittor is not None and torch.Tensor is jittor.Var
    if not using_jtorch:
        raise RuntimeError("This launcher requires JTorch. Run bash install_jtorch.sh first.")
    print(f"Backend: JTorch ({args.device})", flush=True)
    jittor.flags.use_cuda = int(args.device.startswith("cuda"))
    if jittor.flags.use_cuda:
        # Use cuDNN's heuristic rather than per-process timing to select convolution algorithms.
        # Small changes in encoder roundoff can move SISW scores across its hard Top-K cutoff.
        jittor.cudnn.set_algorithm_cache_size(0)
    # EfficientNet imports model_zoo even when built from checkpoint weights.
    if not hasattr(torch.utils, "model_zoo"):
        torch.utils.model_zoo = SimpleNamespace()
    if not hasattr(torch, "inverse"):
        from jittor.linalg import inv

        torch.inverse = inv
    functional = torch.nn.functional
    if not hasattr(functional, "adaptive_avg_pool2d"):
        def adaptive_avg_pool2d(tensor, output_size):
            return jittor.nn.AdaptiveAvgPool2d(output_size)(tensor)

        functional.adaptive_avg_pool2d = adaptive_avg_pool2d

    # EfficientNet passes its decoded strides as lists of length one or two.
    conv_init = torch.nn.Conv2d.__init__

    def init_conv(conv, in_channels, out_channels, kernel_size, stride=1, *args, **kwargs):
        if isinstance(stride, list):
            stride = stride[0] if len(stride) == 1 else tuple(stride)
        conv_init(conv, in_channels, out_channels, kernel_size, stride, *args, **kwargs)

    torch.nn.Conv2d.__init__ = init_conv

    def out_of_place(init):
        def initialize(module, *args, inplace=False, **kwargs):
            init(module, *args, **kwargs)

        return initialize

    for activation in (torch.nn.ReLU, torch.nn.LeakyReLU):
        activation.__init__ = out_of_place(activation.__init__)

    from dhd.config import get_cfg
    from dhd.models.jtorch_dhd import JTorchDHD

    with np.load(args.weights, allow_pickle=False) as archive:
        cfg = get_cfg(cfg_dict=json.loads(archive["config"].item()))
        model = JTorchDHD(cfg, pretrained_encoder=False)
        count = load_weights(model, archive)
    print(f"Loaded {count} model tensors", flush=True)
    model = model.to(args.device).eval()
    if args.input:
        print(f"Using input {args.input}", flush=True)
        with np.load(args.input, allow_pickle=False) as archive:
            arrays = tuple(archive[key] for key in INPUT_KEYS)
    else:
        print("Using synthetic images and camera geometry", flush=True)
        arrays = synthetic_batch(cfg)

    inputs = [torch.tensor(array, dtype=torch.float32).to(args.device) for array in arrays]
    with torch.no_grad():
        if args.benchmark:
            synchronize = lambda: jittor.sync_all(True)
            backend = f"JTorch / Jittor {jittor.__version__}"
            predictions = run_benchmark(args, model, inputs, synchronize, backend)
        else:
            output = model(*inputs, is_train=False)
            predictions = {key: output[key].detach().cpu().numpy() for key in OUTPUT_KEYS}
    for key, prediction in predictions.items():
        if not np.isfinite(prediction).all():
            raise ValueError(f"{key} contains non-finite values")
        print(f"{key}: {prediction.shape}, {prediction.dtype}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output:
        np.savez(output, **predictions)
    print(f"Saved predictions to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run DHD without Lightning or a dataset loader")
    run.add_argument("weights", type=Path, nargs="?", default=DEFAULT_WEIGHTS)
    run.add_argument("output", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    run.add_argument("--input", type=Path, help="NPZ containing one preprocessed input batch")
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--benchmark", type=Path, help="Save synchronized latency samples as JSON")
    run.add_argument("--warmup", type=int, default=5)
    run.add_argument("--iterations", type=int, default=15)
    args = parser.parse_args()
    if not args.weights.is_file():
        run.error(f"Missing weights {args.weights}. See the checkpoint section in README.md.")
    if args.weights.suffix.lower() != ".npz":
        run.error("Supply converted DHD inference weights in NPZ format.")
    if args.input and not args.input.is_file():
        run.error(f"Missing input {args.input}.")
    if args.output.resolve() in {path.resolve() for path in (args.weights, args.input) if path}:
        run.error("Choose an output path distinct from the weights and input files.")
    if args.benchmark and args.benchmark.resolve() in {
        path.resolve() for path in (args.weights, args.input, args.output) if path
    }:
        run.error("Choose a benchmark log path distinct from the weights, input and output files.")
    if args.benchmark and (args.warmup < 0 or args.iterations < 1):
        run.error("Benchmark requires nonnegative warmup and at least one iteration")
    infer(args)


if __name__ == "__main__":
    main()
