#include "common.cl"
#include "transform_position.cl"

__kernel
void accum(__global float* tov_phi, __global float* aij_aIJ, __global float* ppw2p, float scale, int4 dim)
{
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

    float TEMPORARIESaccum;

    float out_aij_aIJ = D_AIJ_AIJ;
    float out_ppw2p = D_PPW2P;

    aij_aIJ[IDX(ix,iy,iz)] += out_aij_aIJ;
    ppw2p[IDX(ix,iy,iz)] += out_ppw2p;
}

