#include "common.cl"
#include "transform_position.cl"
#include "generic_laplace.cl"

__kernel
void simple_tov_solver(__global float* phi_in,
                       __global float* phi_out,
                       __global float* gA_phi_in,
                       __global float* gA_phi_out,
                       float scale, int4 dim, __constant int* last_still_going, __global int* still_going, float etol,
                        __global ushort* order_ptr)
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

    int order = order_ptr[IDX(ix,iy,iz)];

    float gA_PHI_RHS = B_gA_PHI_RHS;
    float PHI_RHS = B_PHI_RHS;

    laplace_interior(phi_in, phi_out, scale * scale * PHI_RHS, ix, iy, iz, scale, dim, still_going, etol);
    laplace_interior(gA_phi_in, gA_phi_out, scale * scale * gA_PHI_RHS, ix, iy, iz, scale, dim, still_going, etol);

    gA_phi_out[IDX(ix,iy,iz)] = clamp(gA_phi_out[IDX(ix,iy,iz)], 0.f, phi_out[IDX(ix,iy,iz)]);
}
