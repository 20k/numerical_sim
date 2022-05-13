#include "common.cl"
#include "transform_position.cl"
#include "generic_laplace.cl"
#include "laplace_order.cl"

__kernel
void simple_tov_solver_phi(__global float* phi_in,
                       __global float* phi_out,
                       float scale, int4 dim, __constant int* last_still_going, __global int* still_going, float etol,
                        __global ushort* order_ptr)
{
    //if(*last_still_going == 0)
    //    return;

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

    float PHI_RHS = B_PHI_RHS;

    laplace_interior(phi_in, phi_out, scale * scale * PHI_RHS, ix, iy, iz, scale, dim, still_going, etol);

    if((ix == 98 || ix == 100 || ix == 102) && iy == dim.y/2 && iz == dim.z/2)
    {
        float rho = DBG_RHO;
        float press = DBG_PRESSURE;

        printf("Tovs %i %f %f rho %f press %f\n", ix, phi_in[IDX(ix,iy,iz)], phi_out[IDX(ix,iy,iz)], rho, press);
    }
}

__kernel
void simple_tov_solver_gA_phi(__global float* phi_in,
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

    laplace_interior(gA_phi_in, gA_phi_out, scale * scale * gA_PHI_RHS, ix, iy, iz, scale, dim, still_going, etol);

    gA_phi_out[IDX(ix,iy,iz)] = clamp(gA_phi_out[IDX(ix,iy,iz)], 0.f, phi_in[IDX(ix,iy,iz)]);

    /*if((ix == 98 || ix == 100 || ix == 102) && iy == dim.y/2 && iz == dim.z/2)
    {
        float rho = DBG_RHO;
        float press = DBG_PRESSURE;

        printf("Tovs %i %f %f %f rho %f press %f\n", ix, phi_in[IDX(ix,iy,iz)], phi_out[IDX(ix,iy,iz)], gA_phi_out[IDX(ix,iy,iz)], rho, press);
    }*/
}
