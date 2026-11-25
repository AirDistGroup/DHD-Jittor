"""Numerical compatibility for the upstream CUDA voxel prefix scan."""

import jittor as jt


def ordered_prefix(features):
    """Match PyTorch's row-ordered FP32 cumsum(0) on a matrix of voxel features."""
    return jt.code(
        features.shape, features.dtype, [features],
        cpu_src=r"""
        @PRECALC
        for (int channel = 0; channel < in0_shape1; ++channel) {
            float sum = 0.0f;
            for (int row = 0; row < in0_shape0; ++row) {
                sum += @in0(row, channel);
                @out(row, channel) = sum;
            }
        }
        """,
        cuda_src=r"""
        __global__ void scan_rows(@ARGS_DEF) {
            @PRECALC
            const int channel = blockIdx.x * blockDim.x + threadIdx.x;
            if (channel >= in0_shape1) return;
            float sum = 0.0f;
            for (int row = 0; row < in0_shape0; ++row) {
                sum = __fadd_rn(sum, @in0(row, channel));
                @out(row, channel) = sum;
            }
        }
        scan_rows<<<(in0_shape1 + 31) / 32, 32>>>(@ARGS);
        """,
    )


def project_points(frustum, transform, translation):
    """Project in FP32 with the original eager rounding boundaries."""
    if not jt.flags.use_cuda:
        x, y, depth = (frustum[:, i][None, :] for i in range(3))
        a, b, c = (transform[:, 2, i][:, None] for i in range(3))
        distance = (-50 - translation[:, 2:3]) / (a * x + b * y + c)
        z = distance - depth * distance * 0.02
        rays = jt.stack((x * z, y * z, z), dim=-1)
        return (rays.unsqueeze(-2) * transform.unsqueeze(1)).sum(-1) + translation.unsqueeze(1)

    return jt.code(
        (transform.shape[0], frustum.shape[0], 3), frustum.dtype,
        [frustum, transform, translation],
        cuda_src=r"""
        __global__ void project(@ARGS_DEF) {
            @PRECALC
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= in0_shape0 * in1_shape0) return;
            const int camera = index / in0_shape0;
            const int point = index % in0_shape0;
            const float x = @in0(point, 0), y = @in0(point, 1);
            const float ax = __fmul_rn(@in1(camera, 2, 0), x);
            const float ay = __fmul_rn(@in1(camera, 2, 1), y);
            const float denom = __fadd_rn(__fadd_rn(ax, ay), @in1(camera, 2, 2));
            const float distance = __fdiv_rn(__fsub_rn(-50.0f, @in2(camera, 2)), denom);
            const float short_distance = __fmul_rn(__fmul_rn(@in0(point, 2), distance), 0.02f);
            const float z = __fsub_rn(distance, short_distance);
            const float ray_x = __fmul_rn(x, z), ray_y = __fmul_rn(y, z);
            for (int axis = 0; axis < 3; ++axis) {
                const float dx = __fmul_rn(@in1(camera, axis, 0), ray_x);
                // Match the upstream CUDA 3x3 by 3x1 matmul rounding sequence.
                const float xy = __fmaf_rn(@in1(camera, axis, 1), ray_y, dx);
                const float dz = __fmul_rn(@in1(camera, axis, 2), z);
                @out(camera, point, axis) = __fadd_rn(__fadd_rn(xy, dz), @in2(camera, axis));
            }
        }
        project<<<(in0_shape0 * in1_shape0 + 255) / 256, 256>>>(@ARGS);
        """,
    )
