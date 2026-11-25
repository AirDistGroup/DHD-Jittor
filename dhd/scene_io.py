"""Prepare a raw scene and decode DHD predictions with the JTorch runtime."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch


def prepare(scene, weights, output):
    metadata = json.loads((scene / "scene.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise ValueError(f"Unsupported scene.json schema in {scene}")
    with np.load(weights, allow_pickle=False) as archive:
        config = json.loads(archive["config"].item())
    frames = metadata["frames"]
    if len(frames) < config["TIME_RECEPTIVE_FIELD"]:
        raise ValueError("The scene does not contain enough input frames for these weights.")

    from dhd.scene_input import export_window

    arrays, _ = export_window(
        scene, frames, metadata["keyframes"], metadata["calibration"], metadata["poses"],
        config["IMAGE"], next_lidar_pose=metadata.get("next_lidar_pose"),
    )
    np.savez(output, **arrays)


def save_bev(probability, target):
    width = 360
    canvas = Image.new("RGB", (2 * width + 24, width + 36), "#f4f6fa")
    label = ImageDraw.Draw(canvas)
    for index, (frame, title) in enumerate(((1, "Current vehicle probability"),
                                             (-1, "Last predicted vehicle probability"))):
        pixels = np.rint(probability[0, frame] * 255).astype(np.uint8)
        image = Image.fromarray(pixels).resize((width, width), Image.Resampling.NEAREST)
        x = 8 + index * (width + 8)
        label.text((x, 8), title, fill="#18233a")
        canvas.paste(image, (x, 28))
    canvas.save(target)


def finish(predictions, directory):
    from dhd.utils.instance import predict_instance_segmentation

    with np.load(predictions, allow_pickle=False) as archive:
        output = {name: torch.tensor(archive[name]) for name in ("segmentation", "instance_flow")}
    with torch.no_grad():
        probabilities = output["segmentation"].softmax(dim=2)[:, :, 1].numpy()
        semantic = output["segmentation"].argmax(dim=2).numpy().astype(np.uint8)
        instances, centers = predict_instance_segmentation(output, compute_matched_centers=True)
    instances = instances.numpy().astype(np.int32)
    tracks = []
    for instance, points in centers.items():
        present = np.flatnonzero(np.any(instances[0] == instance, axis=(1, 2)))
        tracks.append({"instance_id": int(instance), "points": [
            {"output_index": int(index), "column": float(col), "row": float(row)}
            for index, (col, row) in zip(present, points, strict=True)
        ]})
    np.savez_compressed(directory / "bev.npz", vehicle_probability=probabilities,
                        semantic_class=semantic, instance_id=instances)
    (directory / "tracks.json").write_text(
        json.dumps({"current_output_index": 1, "tracks": tracks}, indent=2) + "\n", encoding="utf-8"
    )
    save_bev(probabilities, directory / "bev.png")
    print(f"Decoded {len(tracks)} current-frame instances", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    before = commands.add_parser("prepare")
    before.add_argument("scene", type=Path)
    before.add_argument("weights", type=Path)
    before.add_argument("output", type=Path)
    after = commands.add_parser("finish")
    after.add_argument("predictions", type=Path)
    after.add_argument("directory", type=Path)
    args = parser.parse_args()
    import jittor

    if torch.Tensor is not jittor.Var:
        raise RuntimeError("Scene inference requires the JTorch runtime.")
    jittor.flags.use_cuda = 1
    if args.command == "prepare":
        prepare(args.scene, args.weights, args.output)
    else:
        finish(args.predictions, args.directory)


if __name__ == "__main__":
    main()
