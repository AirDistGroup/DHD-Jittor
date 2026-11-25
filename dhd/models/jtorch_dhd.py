"""JTorch-only model adapter for the original DHD projection semantics."""

import numpy as np
import torch

from dhd.models.dhd import DHD
from dhd.utils.jtorch_voxel import ordered_prefix, project_points


class JTorchDHD(DHD):
    def gbg(self, intrinsics, extrinsics):
        rotation = extrinsics[..., :3, :3].cpu().numpy()
        inverse = np.linalg.inv(intrinsics.cpu().numpy())
        transform = torch.tensor(rotation @ inverse).to(intrinsics.device)
        points = project_points(self.frustum.reshape(-1, 3), transform.reshape(-1, 3, 3),
                                extrinsics[..., :3, 3].reshape(-1, 3).contiguous())
        return points.reshape(*intrinsics.shape[:-2], *self.frustum.shape)

    def projection_to_birds_eye_view(self, x, geometry):
        bev_x, bev_y, bev_z = self.bev_dimension.tolist()
        channels = x.shape[-1]
        output = torch.zeros((x.shape[0], channels, bev_x, bev_y), dtype=torch.float, device=x.device)
        start = self.bev_start_position - self.bev_resolution / 2.0
        for batch in range(x.shape[0]):
            bins = ((geometry[batch] - start) / self.bev_resolution).reshape(-1, 3).long()
            valid = ((bins >= 0) & (bins < self.bev_dimension)).all(dim=1)
            bins = bins[valid]
            count = bins.shape[0]
            if count == 0:
                continue
            values = x[batch].reshape(-1, channels)[valid]
            ranks = (bins[:, 0] * bev_y + bins[:, 1]) * bev_z + bins[:, 2]
            order = ranks.argsort()
            ranks = ranks[order]
            last = torch.ones(count, dtype=torch.bool, device=x.device)
            last[:-1] = ranks[1:] != ranks[:-1]
            prefix = ordered_prefix(values[order].contiguous())[last]
            sums = torch.cat((prefix[:1], prefix[1:] - prefix[:-1]))
            bins = bins[order][last]
            grid = torch.zeros((bev_z, bev_x, bev_y, channels), dtype=torch.float, device=x.device)
            grid[bins[:, 2], bins[:, 0], bins[:, 1]] = sums
            output[batch] = grid.permute(0, 3, 1, 2).squeeze(0)
        return output
