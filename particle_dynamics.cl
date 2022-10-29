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

__kernel
void trace_geodesics(__global float* positions_in, __global float* velocities_in, __global float* positions_out, __global float* velocities_out, int geodesic_count, STANDARD_ARGS(), float scale, int4 dim, float timestep)
{
    int idx = get_global_id(0);

    if(idx >= geodesic_count)
        return;

    float3 Xpos = {positions_in[idx * 3 + 0], positions_in[idx * 3 + 1], positions_in[idx * 3 + 2]};
    float3 vel = {velocities_in[idx * 3 + 0], velocities_in[idx * 3 + 1], velocities_in[idx * 3 + 2]};

    float3 accel;
    calculate_V_derivatives(&accel, Xpos, vel, scale, dim, GET_STANDARD_ARGS());

    float3 XDiff;
    velocity_to_XDiff(&XDiff, Xpos, vel, scale, dim, GET_STANDARD_ARGS());

    Xpos += XDiff * timestep;
    vel += accel * timestep;

    positions_out[idx * 3 + 0] = Xpos.x;
    positions_out[idx * 3 + 1] = Xpos.y;
    positions_out[idx * 3 + 2] = Xpos.z;

    velocities_out[idx * 3 + 0] = vel.x;
    velocities_out[idx * 3 + 1] = vel.y;
    velocities_out[idx * 3 + 2] = vel.z;
}

__kernel
void build_matter_sources(__global float* positions_in, __global float* velocities_in, int geodesic_count, STANDARD_ARGS(), float scale, int4 dim)
{
    int idx = get_global_index(0);
    int local_size = get_local_size(0);

    int num = ceil((float)geodesic_count / local_size);

    for(int i=0; i < num; i++)
    {
        int gidx = idx + i * local_size;

        float3 world_pos = (float3)(positions_in[gidx * 3 + 0], positions_in[gidx * 3 + 1], positions_in[gidx * 3 + 2]);
        float3 vel = (float3)(velocities_in[gidx * 3 + 0], velocities_in[gidx * 3 + 1], velocities_in[gidx * 3 + 2]);

        float3 voxel_pos = world_to_voxel(world_pos, dim, scale);

        int ix = round(voxel_pos.x);
        int iy = round(voxel_pos.y);
        int iz = round(voxel_pos.z);

        {
            float TEMPORARIESadmmatter;

            float vadm_S = OUT_ADM_S;
            float vadm_Si0 = OUT_ADM_SI0;
            float vadm_Si1 = OUT_ADM_SI1;
            float vadm_Si2 = OUT_ADM_SI2;
            float vadm_Sij0 = OUT_ADM_SIJ0;
            float vadm_Sij1 = OUT_ADM_SIJ1;
            float vadm_Sij2 = OUT_ADM_SIJ2;
            float vadm_Sij3 = OUT_ADM_SIJ3;
            float vadm_Sij4 = OUT_ADM_SIJ4;
            float vadm_Sij5 = OUT_ADM_SIJ5;
            float vadm_p = OUT_ADM_P;

            int index = IDX(ix,iy,iz);

            adm_S[index] += vadm_S;
            adm_Si0[index] += vadm_Si0;
            adm_Si1[index] += vadm_Si1;
            adm_Si2[index] += vadm_Si2;
            adm_Sij0[index] += vadm_Sij0;
            adm_Sij1[index] += vadm_Sij1;
            adm_Sij2[index] += vadm_Sij2;
            adm_Sij3[index] += vadm_Sij3;
            adm_Sij4[index] += vadm_Sij4;
            adm_Sij5[index] += vadm_Sij5;
            adm_p[index] += vadm_p;
        }
    }
}
