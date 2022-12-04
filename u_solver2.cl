#include "transform_position.cl"
#include "common.cl"
#include "generic_laplace.cl"

float calculate_phi(__global float* u_offset_in, int ix, int iy, int iz, float scale, int4 dim)
{
    float3 offset = transform_position(ix, iy, iz, dim, scale);

    float ox = offset.x;
    float oy = offset.y;
    float oz = offset.z;

    return GET_PHI;
}

float calculate_guess(int ix, int iy, int iz, float scale, int4 dim)
{
    float3 offset = transform_position(ix, iy, iz, dim, scale);

    float ox = offset.x;
    float oy = offset.y;
    float oz = offset.z;

    return GET_GUESS;
}

///1, -2, 1
///total 0 is -6

float calc_d2(int ix, int iy, int iz, float scale, int4 dim, int direction)
{
    if(direction == 0)
    {
        return      (calculate_guess(ix-1, iy, iz, scale, dim)
               - 2 * calculate_guess(ix  , iy, iz, scale, dim)
               +     calculate_guess(ix+1, iy, iz, scale, dim)) / pow(scale, 2.f);
    }

    if(direction == 1)
    {
        return      (calculate_guess(ix, iy-1, iz, scale, dim)
               - 2 * calculate_guess(ix, iy  , iz, scale, dim)
               +     calculate_guess(ix, iy+1, iz, scale, dim)) / pow(scale, 2.f);
    }

    if(direction == 2)
    {
        return      (calculate_guess(ix, iy, iz-1, scale, dim)
               - 2 * calculate_guess(ix, iy, iz  , scale, dim)
               +     calculate_guess(ix, iy, iz+1, scale, dim)) / pow(scale, 2.f);
    }

    return NAN;

    /*if(direction == 1)
    {
        return (u_offset_in[IDX(ix,iy-1,iz)] - 2 * u_offset_in[IDX(ix,iy,iz)] + u_offset_in[IDX(ix,iy+1,iz)]) / pow(scale, 2.f);
    }

    if(direction == 2)
    {
        return (u_offset_in[IDX(ix,iy,iz-1)] - 2 * u_offset_in[IDX(ix,iy,iz)] + u_offset_in[IDX(ix,iy,iz+1)]) / pow(scale, 2.f);
    }*/
}

float laplace(int ix, int iy, int iz, float scale, int4 dim)
{
    return calc_d2(ix, iy, iz, scale, dim, 0) +
           calc_d2(ix, iy, iz, scale, dim, 1) +
           calc_d2(ix, iy, iz, scale, dim, 2);
}

__kernel
void iterative_u_solve(__global float* u_offset_in, __global float* u_offset_out, __global float* cached_aij_aIJ, __global float* cached_ppw2p, __global float* nonconformal_pH,
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

    float phi = calculate_phi(u_offset_in, ix, iy, iz, scale, dim);

    int index = IDX(ix,iy,iz);

    ///https://arxiv.org/pdf/1606.04881.pdf I think I need to do (85)
    ///ok no: I think what it is is that they're solving for ph in ToV, which uses tov's conformally flat variable
    ///whereas I'm getting values directly out of an analytic solution
    ///the latter term comes from phi^5 * X^(3/2) == phi^5 * phi^-6, == phi^-1

    float p1 = (-1.f/8.f) * cached_aij_aIJ[index] * pow(phi, -7.f);
    float p2 = -2 * M_PI * cached_ppw2p[index] * pow(phi, -3.f);
    float p3 = -2 * M_PI * pow(phi, -1.f) * nonconformal_pH[index];
    float p4 = - laplace(ix, iy, iz, scale, dim);

    //printf("Blah blah blabh\n");

    if(ix == 127 && iz == 127 && (iy == 126 || iy == 127 || iy == 128))
    {
        printf("P4 %.23f Guess %.23f %i\n", p4, calculate_guess(ix,iy,iz, scale, dim), iy);
    }

    float RHS = p1 + p2 + p3 + p4;

    float h2f0 = scale * scale * RHS;

    laplace_interior(u_offset_in, u_offset_out, h2f0, ix, iy, iz, scale, dim, still_going, etol);
}
