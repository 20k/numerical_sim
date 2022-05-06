#include "common.cl"

__kernel
void u_to_phi(__global float* u_in, __global float* phi_out, int4 dim)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    float phi = U_TO_PHI;

    phi_out[IDX(ix,iy,iz)] = phi;
}

__kernel
void calculate_djbj(__global float* gB0_in, __global float* gB1_in, __global float* gB2_in,
                    __global float* djbj0_out, __global float* djbj1_out, __global float* djbj2_out,
                    float scale, int4 dim, __constant int* last_still_going)
{
    if(*last_still_going == 0)
        return;

    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    float v0 = DJBJ0;
    float v1 = DJBJ1;
    float v2 = DJBJ2;

    djbj0_out[IDX(ix,iy,iz)] = v0;
    djbj1_out[IDX(ix,iy,iz)] = v1;
    djbj2_out[IDX(ix,iy,iz)] = v2;
}

__kernel
void iterative_sandwich(__global float* gB0_in, __global float* gB1_in, __global float* gB2_in,
                        __global float* gB0_out, __global float* gB1_out, __global float* gB2_out,
                        __global float* gA_phi_in,
                        __global float* gA_phi_out,
                        __global float* u_arg,
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

    float gB0_RHS = D_gB0_RHS;
    float gB1_RHS = D_gB1_RHS;
    float gB2_RHS = D_gB2_RHS;

    float gA_PHI_RHS = D_gA_PHI_RHS;

    laplace_interior(gB0_in, gB0_out, scale * scale * gB0_RHS, ix, iy, iz, scale, dim, still_going, etol);
    laplace_interior(gB1_in, gB1_out, scale * scale * gB1_RHS, ix, iy, iz, scale, dim, still_going, etol);
    laplace_interior(gB2_in, gB2_out, scale * scale * gB2_RHS, ix, iy, iz, scale, dim, still_going, etol);
    laplace_interior(gA_phi_in, gA_phi_out, scale * scale * gA_PHI_RHS, ix, iy, iz, scale, dim, still_going, etol);
}
