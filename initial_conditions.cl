#include "common.cl"
#include "transform_position.cl"

#ifdef INITIAL_BCAIJ
__kernel
void calculate_bcAij(__global float* tov_phi,
                     __global float* bcAij0,  __global float* bcAij1,  __global float* bcAij2,  __global float* bcAij3,  __global float* bcAij4,  __global float* bcAij5,
                     float scale, int4 dim)
{
    float TEMPORARIESbcaij;

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

    bcAij0[IDX(ix,iy,iz)] = B_BCAIJ_0;
    bcAij1[IDX(ix,iy,iz)] = B_BCAIJ_1;
    bcAij2[IDX(ix,iy,iz)] = B_BCAIJ_2;
    bcAij3[IDX(ix,iy,iz)] = B_BCAIJ_3;
    bcAij4[IDX(ix,iy,iz)] = B_BCAIJ_4;
    bcAij5[IDX(ix,iy,iz)] = B_BCAIJ_5;
}
#endif // INITIAL_BCAIJ

#ifdef INITIAL_PPW2P
__kernel
void calculate_ppw2p(__global float* tov_phi,
                     __global float* ppw2p,
                     float scale, int4 dim)
{
    float TEMPORARIESppw2p;

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

    ppw2p[IDX(ix,iy,iz)] = B_PPW2P;
#endif // INITIAL_PPW2P
