#include "evolution_common.cl"

__kernel
void init_geodesics(STANDARD_ARGS(), __global float* positions3, __global float* initial_dirs3, __global float* velocities3, int geodesic_count, float scale, int4 dim)
{
    int idx = get_global_id(0);

    if(idx >= geodesic_count)
        return;

    float px = positions3[idx * 3 + 0];
    float py = positions3[idx * 3 + 1];
    float pz = positions3[idx * 3 + 1];

    float dirx = initial_dirs3[idx * 3 + 0];
    float diry = initial_dirs3[idx * 3 + 1];
    float dirz = initial_dirs3[idx * 3 + 2];

    float vx = 0;
    float vy = 0;
    float vz = 0;

    {
        float TEMPORARIEStparticleinit;

        vx = OUT_VX;
        vy = OUT_VY;
        vz = OUT_VZ;
    }

    velocities3[idx * 3 + 0] = vx;
    velocities3[idx * 3 + 1] = vy;
    velocities3[idx * 3 + 2] = vz;
}

///this returns the change in X, which is not velocity
///its unfortunate that position, aka X, and the conformal factor are called the same thing here
///the reason why these functions use out parameters is to work around a significant optimisation failure in AMD's opencl compiler
void velocity_to_XDiff(float3* out, float3 Xpos, float3 vel, float scale, int4 dim, STANDARD_ARGS())
{
    float3 voxel_pos = world_to_voxel(Xpos, dim, scale);

    ///isn't this already handled internally?
    voxel_pos = clamp(voxel_pos, (float3)(BORDER_WIDTH,BORDER_WIDTH,BORDER_WIDTH), (float3)(dim.x, dim.y, dim.z) - BORDER_WIDTH - 1);

    float fx = voxel_pos.x;
    float fy = voxel_pos.y;
    float fz = voxel_pos.z;

    float V0 = vel.x;
    float V1 = vel.y;
    float V2 = vel.z;

    float TEMPORARIES6;

    float d0 = X0Diff;
    float d1 = X1Diff;
    float d2 = X2Diff;

    *out = (float3){d0, d1, d2};
}

void calculate_V_derivatives(float3* out, float3 Xpos, float3 vel, float scale, int4 dim, STANDARD_ARGS())
{
    float3 voxel_pos = world_to_voxel(Xpos, dim, scale);

    ///isn't this already handled internally?
    voxel_pos = clamp(voxel_pos, (float3)(BORDER_WIDTH,BORDER_WIDTH,BORDER_WIDTH), (float3)(dim.x, dim.y, dim.z) - BORDER_WIDTH - 1);

    float fx = voxel_pos.x;
    float fy = voxel_pos.y;
    float fz = voxel_pos.z;

    float V0 = vel.x;
    float V1 = vel.y;
    float V2 = vel.z;

    float TEMPORARIES6;

    float d0 = V0Diff;
    float d1 = V1Diff;
    float d2 = V2Diff;

    *out = (float3){d0, d1, d2};
}
