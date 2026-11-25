"""Prepare a calibrated Air-Co-Pred window for DHD inference."""

import numpy as np
from PIL import Image
from pyquaternion import Quaternion
import torch

from dhd.utils.geometry import (
    convert_egopose_to_matrix_numpy,
    invert_matrix_egopose_numpy,
    mat2pose_vec,
    resize_and_crop_image,
    update_intrinsics,
)


def rigid_transform(rotation, translation):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def export_window(root, frames, keyframes, calibration, poses, config, next_lidar_pose=None):
    cameras = config["NAMES"]
    scale = config["RESIZE_SCALE"]
    height, width = config["FINAL_DIM"]
    resized = (int(config["ORIGINAL_WIDTH"] * scale), int(config["ORIGINAL_HEIGHT"] * scale))
    left, top = max(0, (resized[0] - width) // 2), config["TOP_CROP"]
    crop = (left, top, left + width, top + height)
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    lidar_poses = [poses[keyframes[frame["token"]]["LIDAR_TOP"]["ego_pose_token"]] for frame in frames]
    image, intrinsics, extrinsics, filenames = [], [], [], []
    for frame, lidar in zip(frames, lidar_poses):
        yaw = Quaternion(lidar["rotation"]).yaw_pitch_roll[0]
        flat_lidar = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)])
        lidar_to_world = rigid_transform(flat_lidar.rotation_matrix, lidar["translation"])
        frame_images, frame_intrinsics, frame_extrinsics, frame_files = [], [], [], []
        for camera in cameras:
            sample = keyframes[frame["token"]][camera]
            sensor = calibration[sample["calibrated_sensor_token"]]
            camera_to_ego = rigid_transform(Quaternion(sensor["rotation"]).rotation_matrix, sensor["translation"])
            pose = poses[sample["ego_pose_token"]]
            inverse_rotation = Quaternion(pose["rotation"]).inverse.rotation_matrix
            world_to_ego = rigid_transform(inverse_rotation, inverse_rotation @ -np.asarray(pose["translation"]))
            lidar_to_camera = np.linalg.inv(camera_to_ego) @ world_to_ego @ lidar_to_world
            frame_extrinsics.append(np.linalg.inv(lidar_to_camera).astype(np.float32))
            intrinsic = torch.tensor(sensor["camera_intrinsic"], dtype=torch.float32)
            frame_intrinsics.append(update_intrinsics(intrinsic, top, left, scale, scale).numpy())
            with Image.open(root / sample["filename"]) as original:
                rgb = resize_and_crop_image(original.convert("RGB"), resized, crop)
                pixels = torch.tensor(np.array(rgb)).permute(2, 0, 1).float() / 255.0
                frame_images.append(((pixels - mean) / std).numpy())
            frame_files.append(sample["filename"])
        image.append(frame_images)
        intrinsics.append(frame_intrinsics)
        extrinsics.append(frame_extrinsics)
        filenames.append(frame_files)
    matrices = [convert_egopose_to_matrix_numpy(pose) for pose in lidar_poses]
    motion = [invert_matrix_egopose_numpy(b) @ a for a, b in zip(matrices, matrices[1:])]
    motion.append(np.eye(4) if next_lidar_pose is None else
                  invert_matrix_egopose_numpy(convert_egopose_to_matrix_numpy(next_lidar_pose)) @ matrices[-1])
    motion = np.stack(motion).astype(np.float32)
    motion[:, 3, :] = [0, 0, 0, 1]
    return dict(
        image=np.asarray(image, dtype=np.float32)[None],
        intrinsics=np.asarray(intrinsics, dtype=np.float32)[None],
        extrinsics=np.asarray(extrinsics, dtype=np.float32)[None],
        future_egomotion=mat2pose_vec(torch.tensor(motion)).numpy()[None],
    ), filenames
