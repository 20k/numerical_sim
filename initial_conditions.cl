#include "common.cl"
#include "transform_position.cl"

struct matter_data
{
    float4 position;
    float4 linear_momentum;
    float4 angular_momentum;
    float4 colour;
    float mass;
    float compactness;
};

#ifdef INITIAL_BCAIJ
__kernel
void calculate_bcAij(__global float* tov_phi,
                     __global float* bcAij0,  __global float* bcAij1,  __global float* bcAij2,  __global float* bcAij3,  __global float* bcAij4,  __global float* bcAij5,
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

    float TEMPORARIESbcaij;

    bcAij0[IDX(ix,iy,iz)] = B_BCAIJ_0;
    bcAij1[IDX(ix,iy,iz)] = B_BCAIJ_1;
    bcAij2[IDX(ix,iy,iz)] = B_BCAIJ_2;
    bcAij3[IDX(ix,iy,iz)] = B_BCAIJ_3;
    bcAij4[IDX(ix,iy,iz)] = B_BCAIJ_4;
    bcAij5[IDX(ix,iy,iz)] = B_BCAIJ_5;
}
#endif // INITIAL_BCAIJ

#ifdef INITIAL_BCAIJ_2
__kernel
void calculate_bcAij(__global struct matter_data* data,
                     __global float* tov_phi,
                     __global float* bcAij0,  __global float* bcAij1,  __global float* bcAij2,  __global float* bcAij3,  __global float* bcAij4,  __global float* bcAij5,
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

    float TEMPORARIESbcaij;

    bcAij0[IDX(ix,iy,iz)] = B_BCAIJ_0;
    bcAij1[IDX(ix,iy,iz)] = B_BCAIJ_1;
    bcAij2[IDX(ix,iy,iz)] = B_BCAIJ_2;
    bcAij3[IDX(ix,iy,iz)] = B_BCAIJ_3;
    bcAij4[IDX(ix,iy,iz)] = B_BCAIJ_4;
    bcAij5[IDX(ix,iy,iz)] = B_BCAIJ_5;
}
#endif // INITIAL_BCAIJ_2

#ifdef INITIAL_PPW2P_2
__kernel
void calculate_ppw2p(__global struct matter_data* data,
                     __global float* tov_phi,
                     __global float* ppw2p,
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

    float TEMPORARIESppw2p;

    ppw2p[IDX(ix,iy,iz)] = B_PPW2P;
}
#endif // INITIAL_PPW2P_2

#ifdef ACCUM_MATTER_VARIABLES
__kernel
void accum_matter_variables(__global float* tov_phi, ///named so because of convention for code gen
                            __global float* bcAij0_in,  __global float* bcAij1_in,  __global float* bcAij2_in,  __global float* bcAij3_in,  __global float* bcAij4_in,  __global float* bcAij5_in,
                            __global float* ppw2p_in,

                            __global float* tov_phi_accum,
                            __global float* bcAij0,  __global float* bcAij1,  __global float* bcAij2,  __global float* bcAij3,  __global float* bcAij4,  __global float* bcAij5,
                            __global float* ppw2p,
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

    bcAij0[IDX(ix,iy,iz)] += bcAij0_in[IDX(ix,iy,iz)];
    bcAij1[IDX(ix,iy,iz)] += bcAij1_in[IDX(ix,iy,iz)];
    bcAij2[IDX(ix,iy,iz)] += bcAij2_in[IDX(ix,iy,iz)];
    bcAij3[IDX(ix,iy,iz)] += bcAij3_in[IDX(ix,iy,iz)];
    bcAij4[IDX(ix,iy,iz)] += bcAij4_in[IDX(ix,iy,iz)];
    bcAij5[IDX(ix,iy,iz)] += bcAij5_in[IDX(ix,iy,iz)];
    ppw2p[IDX(ix,iy,iz)] += ppw2p_in[IDX(ix,iy,iz)];

    float super_tov_phi = B_TOV_PHI;

    if(super_tov_phi != 0)
        tov_phi_accum[IDX(ix,iy,iz)] = super_tov_phi;
}
#endif

#ifdef ACCUM_BLACK_HOLE_VARIABLES
__kernel
void accum_black_hole_variables(
                            __global float* bcAij0_in,  __global float* bcAij1_in,  __global float* bcAij2_in,  __global float* bcAij3_in,  __global float* bcAij4_in,  __global float* bcAij5_in,
                            __global float* bcAij0,  __global float* bcAij1,  __global float* bcAij2,  __global float* bcAij3,  __global float* bcAij4,  __global float* bcAij5,
                            float scale, int4 dim)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    if(ix < 1 || iy < 1 || iz < 1 || ix >= dim.x - 1 || iy >= dim.y - 1 || iz >= dim.z - 1)
        return;

    float TEMPORARIESaccum;

    bcAij0[IDX(ix,iy,iz)] += bcAij0_in[IDX(ix,iy,iz)];
    bcAij1[IDX(ix,iy,iz)] += bcAij1_in[IDX(ix,iy,iz)];
    bcAij2[IDX(ix,iy,iz)] += bcAij2_in[IDX(ix,iy,iz)];
    bcAij3[IDX(ix,iy,iz)] += bcAij3_in[IDX(ix,iy,iz)];
    bcAij4[IDX(ix,iy,iz)] += bcAij4_in[IDX(ix,iy,iz)];
    bcAij5[IDX(ix,iy,iz)] += bcAij5_in[IDX(ix,iy,iz)];
}
#endif

#ifdef CALCULATE_AIJ_AIJ
__kernel
void calculate_aij_aIJ(__global float* bcAij0, __global float* bcAij1, __global float* bcAij2, __global float* bcAij3, __global float* bcAij4, __global float* bcAij5,
                       __global float* aij_aIJ,
                       float scale, int4 dim)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    if(ix < 1 || iy < 1 || iz < 1 || ix >= dim.x - 1 || iy >= dim.y - 1 || iz >= dim.z - 1)
        return;

    aij_aIJ[IDX(ix,iy,iz)] = B_AIJ_AIJ;
}
#endif // CALCULATE_AIJ_AIJ

#ifdef ALL_MATTER_VARIABLES
__kernel
void multi_accumulate(__global struct matter_data* data,
                      __global float* pressure, __global float* rho, __global float* rhoH, __global float* p0,
                      __global float* Si0, __global float* Si1, __global float* Si2, __global float* colour0, __global float* colour1, __global float* colour2,
                      __global float* u_value, __global float* tov_phi, float scale, int4 dim)
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

    float TEMPORARIESmultiaccumulate;

    int index = IDX(ix,iy,iz);

    pressure[index] += ACCUM_PRESSURE;
    rho[index] += ACCUM_RHO;
    rhoH[index] += ACCUM_RHOH;
    p0[index] += ACCUM_P0;
    Si0[index] += ACCUM_SI0;
    Si1[index] += ACCUM_SI1;
    Si2[index] += ACCUM_SI2;
    colour0[index] += ACCUM_COLOUR0;
    colour1[index] += ACCUM_COLOUR1;
    colour2[index] += ACCUM_COLOUR2;
}
#endif // ALL_MATTER_VARIABLES
