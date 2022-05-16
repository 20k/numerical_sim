#include "common.cl"
#include "transform_position.cl"

__kernel
void accum(__global float* tov_phi, __global float* aij_aIJ, __global float* ppw2p,
           __global float* bcAij0, __global float* bcAij1, __global float* bcAij2, __global float* bcAij3, __global float* bcAij4, __global float* bcAij5,
           __global float* superimposed_tov_phi,
           float scale, int4 dim)
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

    float out_bcAij0 = D_BCAIJ0;
    float out_bcAij1 = D_BCAIJ1;
    float out_bcAij2 = D_BCAIJ2;
    float out_bcAij3 = D_BCAIJ3;
    float out_bcAij4 = D_BCAIJ4;
    float out_bcAij5 = D_BCAIJ5;

    aij_aIJ[IDX(ix,iy,iz)] += out_aij_aIJ;
    ppw2p[IDX(ix,iy,iz)] += out_ppw2p;

    bcAij0[IDX(ix,iy,iz)] += out_bcAij0;
    bcAij1[IDX(ix,iy,iz)] += out_bcAij1;
    bcAij2[IDX(ix,iy,iz)] += out_bcAij2;
    bcAij3[IDX(ix,iy,iz)] += out_bcAij3;
    bcAij4[IDX(ix,iy,iz)] += out_bcAij4;
    bcAij5[IDX(ix,iy,iz)] += out_bcAij5;

    float super_tov_phi = D_TOV_PHI;

    /*if(ix == 98 && iy == 125 && iz == 125)
    {
        printf("Hello %f\n", super_tov_phi);
    }*/

    if(super_tov_phi != 0)
        superimposed_tov_phi[IDX(ix,iy,iz)] = super_tov_phi;
}

