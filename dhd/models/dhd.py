# ------------------------------------------------------------------------
# DHD
# Copyright (c) 2024 Zhechao Wang. All Rights Reserved.
# ------------------------------------------------------------------------
import time

import torch
import torch.nn as nn
from dhd.models.encoder import Encoder
#from dhd.models.tiny_encoder import Encoder
from dhd.models.stconv import MultiBranchSTconv
from dhd.models.temporal_model import TemporalModelIdentity
from dhd.utils.geometry import (calculate_birds_eye_view_parameters,
                                cumulative_warp_features)
from dhd.utils.network import (pack_sequence_dim, set_bn_momentum,
                               unpack_sequence_dim)
import numpy as np
from dhd.models.fusion_model import *#collaborate_single_gru_UMC, single_gru_UMC, newUMC, gru_UMC, myUMC_temporal, maxUMC,avgUMC, V2VNetFusion, V2XViTFusion, Where2commFusion, When2commFusion, Who2commFusion, myWhere2commFusion, UMC, myUMC, myV2VNetFusion
class DistanceAwareDeformConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        from torchvision.ops import DeformConv2d

        super(DistanceAwareDeformConv2D, self).__init__()

        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size,
                                     kernel_size=kernel_size, stride=stride,
                                     padding=padding, dilation=dilation)

        self.deform_conv = DeformConv2d(in_channels, out_channels,
                                            kernel_size=kernel_size, stride=stride,
                                            padding=padding, dilation=dilation)

    def forward(self, x):
        offset = self.offset_conv(x)
        x = self.deform_conv(x, offset)

        return x
class DHD(nn.Module):
    def __init__(self, cfg, pretrained_encoder=True):
        super().__init__()
        self.cfg = cfg

        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            self.cfg.LIFT.X_BOUND, self.cfg.LIFT.Y_BOUND, self.cfg.LIFT.Z_BOUND
        )
        self.bev_resolution = nn.Parameter(bev_resolution, requires_grad=False)
        self.bev_start_position = nn.Parameter(bev_start_position, requires_grad=False)
        self.bev_dimension = nn.Parameter(bev_dimension, requires_grad=False)

        self.encoder_downsample = self.cfg.MODEL.ENCODER.DOWNSAMPLE
        self.encoder_out_channels = self.cfg.MODEL.ENCODER.OUT_CHANNELS
        self.frustum = self.create_frustum()
        self.depth_channels, _, _, _ = self.frustum.shape
        if self.cfg.TIME_RECEPTIVE_FIELD == 1:
            assert self.cfg.MODEL.TEMPORAL_MODEL.NAME == 'identity'

        # temporal block
        self.receptive_field = self.cfg.TIME_RECEPTIVE_FIELD
        self.n_future = self.cfg.N_FUTURE_FRAMES

        if self.cfg.MODEL.SUBSAMPLE:
            assert self.cfg.DATASET.NAME == 'lyft'
            self.receptive_field = 3
            self.n_future = 5

        # Spatial extent in bird's-eye view, in meters
        self.spatial_extent = (self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1])
        self.bev_size = (self.bev_dimension[0].item(), self.bev_dimension[1].item())

        # Encoder
        self.encoder = Encoder(cfg=self.cfg.MODEL.ENCODER, D=self.depth_channels,
                               pretrained=pretrained_encoder)
        # Temporal model
        temporal_in_channels = self.encoder_out_channels
        temporal_out_channels = temporal_in_channels    
        if self.n_future == 0:
            from dhd.models.decoder import Decoder

            self.temporal_model = TemporalModelIdentity(temporal_in_channels, self.receptive_field)
            # Decoder
            self.future_pred_in_channels = temporal_out_channels
            self.decoder = Decoder(
                in_channels=self.future_pred_in_channels,
                n_classes=len(self.cfg.SEMANTIC_SEG.WEIGHTS),
                predict_future_flow=self.cfg.INSTANCE_FLOW.ENABLED,
            )
        elif self.n_future > 0:
            if self.cfg.MODEL.STCONV.INPUT_EGOPOSE:
                temporal_in_channels += 6
            # Multi-branch STConv
            self.stconv = MultiBranchSTconv(cfg=self.cfg, in_channels = temporal_in_channels)
            if self.cfg.MODEL.NAME == 'SISW':
                self.fusion = SISW()
            elif self.cfg.MODEL.NAME == 'all':
                self.fusion = None
            else:
                raise NotImplementedError(f'Number of future frames {self.n_future}.')

        set_bn_momentum(self, self.cfg.MODEL.BN_MOMENTUM)

    def create_frustum(self):
        # Create grid in image plane
        h, w = self.cfg.IMAGE.FINAL_DIM
        downsampled_h, downsampled_w = h // self.encoder_downsample, w // self.encoder_downsample

        # Depth grid
        depth_grid = torch.arange(*self.cfg.LIFT.D_BOUND, dtype=torch.float)
        depth_grid = depth_grid.view(-1, 1, 1).expand(-1, downsampled_h, downsampled_w)
        n_depth_slices = depth_grid.shape[0]

        # x and y grids
        x_grid = torch.linspace(0, w - 1, downsampled_w, dtype=torch.float)
        x_grid = x_grid.view(1, 1, downsampled_w).expand(n_depth_slices, downsampled_h, downsampled_w)
        y_grid = torch.linspace(0, h - 1, downsampled_h, dtype=torch.float)
        y_grid = y_grid.view(1, downsampled_h, 1).expand(n_depth_slices, downsampled_h, downsampled_w)

        # Dimension (n_depth_slices, downsampled_h, downsampled_w, 3)
        # containing data points in the image: left-right, top-bottom, depth
        frustum = torch.stack((x_grid, y_grid, depth_grid), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def custom_linspace(self, start, end, num_points):
        # Generate linearly spaced values between 0 and 1
        linear_points = np.linspace(0, 1, num_points)

        # Apply a transformation to make lower values more dense
        # For example, using a square root function which will densify lower values
        transformed_points = np.cbrt(linear_points)

        # Scale back to the desired range [start, end]
        scaled_points = end - (end - start) * transformed_points
        scaled_points = sorted(scaled_points)
        scaled_points = torch.tensor(scaled_points, dtype=torch.float)
        return scaled_points
    def create_height_frustum(self):
        # Create grid in image plane
        h, w = self.cfg.IMAGE.FINAL_DIM
        downsampled_h, downsampled_w = h // self.encoder_downsample, w // self.encoder_downsample
        start, end, step = self.cfg.LIFT.D_BOUND
        # Depth grid
        height_grid = self.custom_linspace(start, end, int((end - start) / step))#torch.arange(*self.cfg.LIFT.D_BOUND, dtype=torch.float)
        height_grid = height_grid.view(-1, 1, 1).expand(-1, downsampled_h, downsampled_w)
        n_depth_slices = height_grid.shape[0]

        # x and y grids
        x_grid = torch.linspace(0, w - 1, downsampled_w, dtype=torch.float)
        x_grid = x_grid.view(1, 1, downsampled_w).expand(n_depth_slices, downsampled_h, downsampled_w)
        y_grid = torch.linspace(0, h - 1, downsampled_h, dtype=torch.float)
        y_grid = y_grid.view(1, downsampled_h, 1).expand(n_depth_slices, downsampled_h, downsampled_w)

        # Dimension (n_depth_slices, downsampled_h, downsampled_w, 3)
        # containing data points in the image: left-right, top-bottom, depth
        frustum = torch.stack((x_grid, y_grid, height_grid), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def forward(self, image, intrinsics, extrinsics, future_egomotion, future_distribution_inputs=None, noise=None, is_train=True):
        output = {}
        start_time = time.time()

        # Only process features from the past and present
        image = image[:, :self.receptive_field].contiguous()
        intrinsics = intrinsics[:, :self.receptive_field].contiguous()
        extrinsics = extrinsics[:, :self.receptive_field].contiguous()
        future_egomotion = future_egomotion[:, :self.receptive_field].contiguous()

        # Lifting features and project to bird's-eye view
        x, bandwidth = self.calculate_birds_eye_view_features(image, intrinsics, extrinsics, is_train)

        # Warp past features to the present's reference frame
        x = cumulative_warp_features(
            x.clone(), future_egomotion,
            mode='bilinear', spatial_extent=self.spatial_extent,
        )

        perception_time = time.time()
       
        # Temporal model
        if self.n_future == 0:
            states = self.temporal_model(x)
            bev_output = self.decoder(states[:, -1:])
        elif self.n_future > 0:
            if self.cfg.MODEL.STCONV.INPUT_EGOPOSE:
                b, s, c = future_egomotion.shape
                h, w = x.shape[-2:]
                future_egomotions_spatial = future_egomotion.view(b, s, c, 1, 1).expand(b, s, c, h, w)
                # at time 0, no egomotion so feed zero vector
                future_egomotions_spatial = torch.cat([torch.zeros_like(future_egomotions_spatial[:, :1]),
                                                    future_egomotions_spatial[:, :(self.receptive_field-1)]], dim=1)
                x = torch.cat([x, future_egomotions_spatial], dim=-3)

            bev_output = self.stconv(x)
        else:
            raise NotImplementedError(f'Number of future frames {self.n_future}.')

        prediction_time = time.time()
        output['band_width'] = bandwidth
        output['perception_time'] = perception_time - start_time
        output['prediction_time'] = prediction_time - perception_time
        output['total_time'] = output['perception_time'] + output['prediction_time']
        
        output = {**output, **bev_output}

        return output

    def gbg(self, intrinsics, extrinsics):
        """Calculate the (x, y, z) 3D position of the features.
        """
        # Small projection errors can send points to different voxels at cell boundaries.
        extrinsics = extrinsics.double()
        rotation, translation = extrinsics[..., :3, :3], extrinsics[..., :3, 3]
        BATCH, N, _ = translation.shape
        # Add batch, camera dimension, and a dummy dimension at the end
        points = self.frustum.double().unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        device = intrinsics.device
        inverse = torch.inverse(intrinsics.cpu().double())
        # The JTorch CUDA batched matmul wrapper rejects FP64 for this shape.
        A = (rotation.cpu().unsqueeze(-1) * inverse.unsqueeze(-3)).sum(dim=-2).to(device)
        B = translation.unsqueeze(3)
        H = -50
        points = points.repeat(BATCH, N, 1, 1, 1, 1, 1)
        a21 = A[:, :, 2, 0].view(BATCH, N, 1, 1, 1, 1)
        a22 = A[:, :, 2, 1].view(BATCH, N, 1, 1, 1, 1)
        a23 = A[:, :, 2, 2].view(BATCH, N, 1, 1, 1, 1)
        b2 = B[:, :, 2, 0].view(BATCH, N, 1, 1, 1, 1)
        x = points[:, :, :, :, :, 0]
        y = points[:, :, :, :, :, 1]
        z = points[:, :, :, :, :, 2]
        # Calculate new z values based on the given formula
        xiebian = (H - b2) / (a21 * x + a22 * y + a23)
        short_xiebian = z * xiebian / -H
        new_z = xiebian - short_xiebian
        # Assign the new z values back to the original tensor
        points[:, :, :, :, :, 2] = new_z#.unsqueeze(-1).unsqueeze(-1)
        # Camera to ego reference frame
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:3]), 5)
        points = (A.view(BATCH, N, 1, 1, 1, 3, 3) * points.transpose(-1, -2)).sum(dim=-1)
        points += translation.view(BATCH
                                   , N, 1, 1, 1, 3)
        # The 3 dimensions in the ego reference frame are: (forward, sides, height)
        return points.to(intrinsics.dtype)

    def encoder_forward(self, x):
        # batch, n_cameras, channels, height, width
        b, n, c, h, w = x.shape

        x = x.view(b * n, c, h, w)
        x = self.encoder(x)
        x = x.view(b, n, *x.shape[1:])
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x

    def projection_to_birds_eye_view(self, x, geometry):
        """ Adapted from https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/models.py#L200"""
        # batch, n_cameras, depth, height, width, channels
        batch, n, d, h, w, c = x.shape
        bev_x, bev_y, bev_z = self.bev_dimension.tolist()
        output = torch.zeros(
            (batch, c, bev_x, bev_y), dtype=torch.float, device=x.device
        )

        # Number of 3D points
        N = n * d * h * w
        for b in range(batch):
            # flatten x
            x_b = x[b].reshape(N, c)

            # Convert positions to integer indices
            geometry_b = ((geometry[b] - (self.bev_start_position - self.bev_resolution / 2.0)) / self.bev_resolution)
            geometry_b = geometry_b.view(N, 3).long()

            # Mask out points that are outside the considered spatial extent.
            mask = (
                    (geometry_b[:, 0] >= 0)
                    & (geometry_b[:, 0] < self.bev_dimension[0])
                    & (geometry_b[:, 1] >= 0)
                    & (geometry_b[:, 1] < self.bev_dimension[1])
                    & (geometry_b[:, 2] >= 0)
                    & (geometry_b[:, 2] < self.bev_dimension[2])
            )
            x_b = x_b[mask]
            geometry_b = geometry_b[mask]

            # Sum directly in each voxel, without subtracting global FP32 prefix sums.
            bev_feature = torch.zeros((bev_z * bev_x * bev_y, c), dtype=torch.float, device=x.device)
            if x_b.shape[0]:
                voxel_index = (geometry_b[:, 2] * bev_x + geometry_b[:, 0]) * bev_y + geometry_b[:, 1]
                indices = voxel_index.unsqueeze(1).expand(-1, c)
                if hasattr(bev_feature, "scatter_add_"):
                    bev_feature.scatter_add_(0, indices, x_b.float())
                else:
                    # JTorch index_add_ does not accumulate repeated CUDA indices reliably.
                    bev_feature = bev_feature.scatter(0, indices, x_b.float(), reduce="add")
            bev_feature = bev_feature.reshape(bev_z, bev_x, bev_y, c)

            # Put channel in second position and remove z dimension
            bev_feature = bev_feature.permute((0, 3, 1, 2))
            bev_feature = bev_feature.squeeze(0)

            output[b] = bev_feature

        return output

    def calculate_birds_eye_view_features(self, x, intrinsics, extrinsics, is_train):
        b, s, n, c, h, w = x.shape
        # Reshape
        x = pack_sequence_dim(x)
        intrinsics = pack_sequence_dim(intrinsics)
        extrinsics = pack_sequence_dim(extrinsics)

        geometry = self.gbg(intrinsics, extrinsics)#self.get_geometry_height_angle(intrinsics, extrinsics)
        x = self.encoder_forward(x)
        if self.fusion == None:
            x = self.projection_to_birds_eye_view(x, geometry)
            bandwidth = 1
        else:
            bev_features = self.generate_each_bev(features=x, geometry=geometry)
            res = self.fusion(bev_features, is_train)#istrain
            if len(res) == 2:
                x = res[0]
                bandwidth = res[1]
            else:
                x = res
                bandwidth = 1
        x = unpack_sequence_dim(x, b, s)

        return x, bandwidth
    def generate_each_bev(self, features, geometry):
        bev_results = []
        for i in range(features.size(1)):
            # Slice the current camera view.
            x_slice = features[:, i:i+1, ...]
            geometry_slice = geometry[:, i:i+1, ...]

            # Project this view into BEV.
            bev_result = self.projection_to_birds_eye_view(x_slice, geometry_slice)

            # Append the projection.
            bev_results.append(bev_result)
        # Stack the projected views.
        bev_tensor = torch.stack(bev_results, dim=1)
        return bev_tensor
