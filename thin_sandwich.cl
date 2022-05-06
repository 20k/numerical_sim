#include "common.cl"
#include "transform_position.cl"
#include "generic_laplace.cl"

__kernel
void u_to_phi(__global float* u_in, __global float* phi_out, float scale, int4 dim)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    float3 offset = transform_position(ix, iy, iz, dim, scale);

    float ox = offset.x;
    float oy = offset.y;
    float oz = offset.z;

    float phi = U_TO_PHI;

    phi_out[IDX(ix,iy,iz)] = phi;
}

__kernel
void gA_phi_to_gA(__global float* gA_phi, __global float* phi, __global float* gA_out, float scale, int4 dim)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    float3 offset = transform_position(ix, iy, iz, dim, scale);

    float ox = offset.x;
    float oy = offset.y;
    float oz = offset.z;

    gA_out[IDX(ix, iy, iz)] = gA_phi[IDX(ix, iy, iz)] / phi[IDX(ix, iy, iz)];
}

__kernel
void calculate_djbj(__global float* gB0_in, __global float* gB1_in, __global float* gB2_in,
                    __global float* djbj_out,
                    float scale, int4 dim, __constant int* last_still_going)
{
    if(*last_still_going == 0)
        return;

    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    float3 offset = transform_position(ix, iy, iz, dim, scale);

    float ox = offset.x;
    float oy = offset.y;
    float oz = offset.z;

    float v0 = BDJBJ;

    djbj_out[IDX(ix,iy,iz)] = v0;
}

__kernel
void iterative_sandwich(__global float* gB0_in, __global float* gB1_in, __global float* gB2_in,
                        __global float* gB0_out, __global float* gB1_out, __global float* gB2_out,
                        __global float* gA_phi_in,
                        __global float* gA_phi_out,
                        __global float* phi,
                        __global float* djbj,
                        float scale, int4 dim, __constant int* last_still_going, __global int* still_going, float etol)
{
    if(*last_still_going == 0)
        return;

    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    if(ix < 1 || iy < 1 || iz < 1 || ix >= dim.x - 1 || iy >= dim.y - 1 || iz >= dim.z - 1)
        return;

    float3 offset = transform_position(ix, iy, iz, dim, scale);

    float ox = offset.x;
    float oy = offset.y;
    float oz = offset.z;

    float gB0_RHS = B_gB0_RHS;
    float gB1_RHS = B_gB1_RHS;
    float gB2_RHS = B_gB2_RHS;

    float gA_PHI_RHS = B_gA_PHI_RHS;

    laplace_interior(gB0_in, gB0_out, scale * scale * gB0_RHS, ix, iy, iz, scale, dim, still_going, etol);
    laplace_interior(gB1_in, gB1_out, scale * scale * gB1_RHS, ix, iy, iz, scale, dim, still_going, etol);
    laplace_interior(gB2_in, gB2_out, scale * scale * gB2_RHS, ix, iy, iz, scale, dim, still_going, etol);
    laplace_interior(gA_phi_in, gA_phi_out, scale * scale * gA_PHI_RHS, ix, iy, iz, scale, dim, still_going, etol);
}
