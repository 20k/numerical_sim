#include "common.cl"
#include "transform_position.cl"
#include "generic_laplace.cl"
#include "laplace_order.cl"

__kernel
void simple_tov_solver_phi(__global float* u_offset_in,
                           __global float* u_offset_out,
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

    if(SHOULD_NOT_USE_INTEGRATION_CONSTANT == 0)
    {
        u_offset_out[IDX(ix,iy,iz)] = INTEGRATION_CONSTANT;
    }
    else
    {
        float PHI_RHS = B_PHI_RHS;

        laplace_interior(u_offset_in, u_offset_out, scale * scale * PHI_RHS, ix, iy, iz, scale, dim, still_going, etol);
    }

    /*if(ix == 98 && iy == 125 && iz == 125)
    {
        printf("Hellothere2 %f\n", u_offset_out[IDX(ix,iy,iz)]);
    }*/

    /*if((ix == 97 || ix == 100 || ix == 102 || ix == 120 || ix == 160) && iy == dim.y/2 && iz == dim.z/2)
    {
        float rho = DBG_RHO;

        printf("Tovs %i %f %f rho %f\n", ix, u_offset_in[IDX(ix,iy,iz)], u_offset_out[IDX(ix,iy,iz)], rho);
    }*/
}

__kernel
void tov_u_to_phi(__global float* u_offset_in, __global float* phi_out, float scale, int4 dim)
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

    float phi = B_U_TO_PHI;

    phi_out[IDX(ix,iy,iz)] = phi;
}
