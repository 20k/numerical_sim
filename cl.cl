///https://arxiv.org/pdf/1404.6523.pdf
///Gauge evolution equations

//#define SYMMETRY_BOUNDARY
//#define USE_GBB

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#include "transform_position.cl"
#include "common.cl"
#include "evolve_points.cl"

///because we need to cutoff slightly before the real edge due to various factors
#define RENDERING_CUTOFF_MULT 0.95f

float srgb_to_lin_single(float in)
{
    if(in < 0.04045f)
        return in / 12.92f;
    else
        return pow((in + 0.055f) / 1.055f, 2.4f);
}

float3 srgb_to_lin(float3 in)
{
    return (float3)(srgb_to_lin_single(in.x), srgb_to_lin_single(in.y), srgb_to_lin_single(in.z));
}

float buffer_indexh(__global const DERIV_PRECISION* const buffer, int x, int y, int z, int4 dim)
{
    return buffer[z * dim.x * dim.y + y * dim.x + x];
}

__kernel
void trapezoidal_accumulate(__global ushort4* points, int point_count, int4 dim, __global float* yn, __global float* fn, __global float* fnp1, float timestep)
{
    int local_idx = get_global_id(0);

    if(local_idx >= point_count)
        return;

    int ix = points[local_idx].x;
    int iy = points[local_idx].y;
    int iz = points[local_idx].z;

    int index = IDX(ix, iy, iz);

    float next = yn[index] + 0.5f * timestep * (fn[index] + fnp1[index]);

    fnp1[index] = next;
}

__kernel
void accumulate_rk4(__global ushort4* points, int point_count, int4 dim, __global float* accum, __global float* yn, float factor)
{
    int local_idx = get_global_id(0);

    if(local_idx >= point_count)
        return;

    int ix = points[local_idx].x;
    int iy = points[local_idx].y;
    int iz = points[local_idx].z;

    int index = IDX(ix, iy, iz);

    accum[index] += factor * yn[index];
}

__kernel
void copy_buffer(__global float* in, __global float* out, int max_size)
{
    int idx = get_global_id(0);

    if(idx >= max_size)
        return;

    out[idx] = in[idx];
}

__kernel
void copy_valid(__global ushort4* points, int point_count, __global float* in, __global float* out, int4 dim)
{
    int local_idx = get_global_id(0);

    if(local_idx >= point_count)
        return;

    int ix = points[local_idx].x;
    int iy = points[local_idx].y;
    int iz = points[local_idx].z;

    int index = IDX(ix, iy, iz);

    out[index] = in[index];
}

__kernel
void calculate_rk4_val(__global ushort4* points, int point_count, int4 dim, __global float* yn_inout, __global float* xn, float factor)
{
    int local_idx = get_global_id(0);

    if(local_idx >= point_count)
        return;

    int ix = points[local_idx].x;
    int iy = points[local_idx].y;
    int iz = points[local_idx].z;

    int index = IDX(ix, iy, iz);

    float yn = yn_inout[index];

    yn_inout[index] = xn[index] + factor * yn;
}

void buffer_write(__global float* buffer, int3 position, int4 dim, float value)
{
    buffer[position.z * dim.x * dim.y + position.y * dim.x + position.x] = value;
}

float3 voxel_to_world(float3 in, int4 dim, float scale)
{
    return transform_position(in.x, in.y, in.z, dim, scale);
}

float get_distance(int x1, int y1, int z1, int x2, int y2, int z2, int4 dim, float scale)
{
    float3 d1 = transform_position(x1, y1, z1, dim, scale);
    float3 d2 = transform_position(x2, y2, z2, dim, scale);

    return fast_length(d2 - d1);
}

#ifndef USE_GBB

    #define GET_ARGLIST(a, p) a p##cY0, a p##cY1, a p##cY2, a p##cY3, a p##cY4, a p##cY5, \
                a p##cA0, a p##cA1, a p##cA2, a p##cA3, a p##cA4, a p##cA5, \
                a p##cGi0, a p##cGi1, a p##cGi2, a p##K, a p##X, a p##gA, a p##gB0, a p##gB1, a p##gB2, \
                a p##Dp_star, a p##De_star, a p##DcS0, a p##DcS1, a p##DcS2

    #define GET_DERIVLIST(a, p) a p##dcYij0, a p##dcYij1, a p##dcYij2, a p##dcYij3, a p##dcYij4, a p##dcYij5, a p##dcYij6, a p##dcYij7, a p##dcYij8, a p##dcYij9, a p##dcYij10, a p##dcYij11, a p##dcYij12, a p##dcYij13, a p##dcYij14, a p##dcYij15, a p##dcYij16, a p##dcYij17, \
                        a p##digA0, a p##digA1, a p##digA2, \
                        a p##digB0, a p##digB1, a p##digB2, a p##digB3, a p##digB4, a p##digB5, a p##digB6, a p##digB7, a p##digB8, \
                        a p##dX0, a p##dX1, a p##dX2

    #define STANDARD_ARGS(p) GET_ARGLIST(__global float*, p)
    #define STANDARD_DERIVS(p) GET_DERIVLIST(__global DERIV_PRECISION*, p)

#else
    #define STANDARD_ARGS(p) __global float* p##cY0, __global float* p##cY1, __global float* p##cY2, __global float* p##cY3, __global float* p##cY4, __global float* p##cY5, \
                __global float* p##cA0, __global float* p##cA1, __global float* p##cA2, __global float* p##cA3, __global float* p##cA4, __global float* p##cA5, \
                __global float* p##cGi0, __global float* p##cGi1, __global float* p##cGi2, __global float* p##K, __global float* p##X, __global float* p##gA, __global float* p##gB0, __global float* p##gB1, __global float* p##gB2, \
                __global float* p##gBB0, __global float* p##gBB1, __global float* p##gBB2
#endif

#define ALL_ARGS(p) GET_ARGLIST(, p), GET_DERIVLIST(, p)

__kernel
void calculate_initial_conditions(STANDARD_ARGS(),
                                  __global float* u_value,
                                  __global float* bcAij0,
                                  __global float* bcAij1,
                                  __global float* bcAij2,
                                  __global float* bcAij3,
                                  __global float* bcAij4,
                                  __global float* bcAij5,
                                  float scale, int4 dim)
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

    float TEMPORARIES0;

    int index = IDX(ix, iy, iz);

    cY0[index] = init_cY0;
    cY1[index] = init_cY1;
    cY2[index] = init_cY2;
    cY3[index] = init_cY3;
    cY4[index] = init_cY4;
    cY5[index] = init_cY5;

    cA0[index] = init_cA0;
    cA1[index] = init_cA1;
    cA2[index] = init_cA2;
    cA3[index] = init_cA3;
    cA4[index] = init_cA4;
    cA5[index] = init_cA5;

    cGi0[index] = init_cGi0;
    cGi1[index] = init_cGi1;
    cGi2[index] = init_cGi2;

    K[index] = init_K;
    X[index] = init_X;

    gA[index] = init_gA;
    gB0[index] = init_gB0;
    gB1[index] = init_gB1;
    gB2[index] = init_gB2;

    #ifdef USE_GBB
    gBB0[index] = init_gBB0;
    gBB1[index] = init_gBB1;
    gBB2[index] = init_gBB2;
    #endif // USE_GBB

    ///nanananananananana
    //#define NANIFY
    #ifdef NANIFY
    float nan = NAN;

    ///87 152 16

    if(ix == 87 && iy == 152 && iz == 16)
    {
        printf("hi my type %i", is_low_order_evolved_point(ix, iy, iz, scale, dim));
    }
    if(ix == 86 && iy == 152 && iz == 16)
    {
        printf("hi my type left %i %i", is_low_order_evolved_point(ix, iy, iz, scale, dim), is_deep_boundary_point(ix, iy, iz, scale, dim));
    }

    if(is_deep_boundary_point(ix, iy, iz, scale, dim))
    {
        cY0[index] = nan;
        cY1[index] = nan;
        cY2[index] = nan;
        cY3[index] = nan;
        cY4[index] = nan;
        cY5[index] = nan;

        cA0[index] = nan;
        cA1[index] = nan;
        cA2[index] = nan;
        cA3[index] = nan;
        cA4[index] = nan;
        cA5[index] = nan;

        cGi0[index] = nan;
        cGi1[index] = nan;
        cGi2[index] = nan;

        K[index] = nan;
        X[index] = nan;

        gA[index] = nan;
        gB0[index] = nan;
        gB1[index] = nan;
        gB2[index] = nan;
    }

    #endif // NANIFY

    if(ix == (dim.x-1)/2 && iz == (dim.z-1)/2)
    {
        int hy = (dim.y-1)/2;

        if(iy == hy-1 || iy == hy || iy == hy+1)
        {
            printf("U: %.9f %i\n", u_value[IDX(ix,iy,iz)], iy);
        }
    }

    /*if(x == 50 && y == 50 && z == 50)
    {
        printf("gTEST0 %f\n", f->cY0);
        printf("TEST1 %f\n", f->cY1);
        printf("TEST2 %f\n", f->cY2);
        printf("TEST3 %f\n", f->cY3);
        printf("TEST4 %f\n", f->cY4);
        printf("TEST5 %f\n", f->cY5);
        printf("TESTb0 %f\n", f->gB0);
        printf("TESTX %f\n", f->X);
        printf("TESTgA %f\n", f->gA);
        printf("TESTK %f\n", f->K);
    }*/
}

#define IS_DEGENERATE(x) (isnan(x) || !isfinite(x))

#define NANCHECK_IMPL(w) if(IS_DEGENERATE(w[index])){printf("NAN " #w " %i %i %i %f\n", ix, iy, iz, w[index]);}
#define LNANCHECK_IMPL(w)  if(IS_DEGENERATE(w)){printf("NAN " #w " %i %i %i %f\n", ix, iy, iz, w);}
#define NNANCHECK_IMPL(w, name) if(IS_DEGENERATE(w)){printf("NAN " name " %i %i %i %f\n", ix, iy, iz, w);}


//#define DEBUGGING
#ifdef DEBUGGING
#define NANCHECK(w) NANCHECK_IMPL(w)
#define LNANCHECK(w) LNANCHECK_IMPL(w)
#define NNANCHECK(w, name) NNANCHECK_IMPL(w, name)
#else
#define NANCHECK(w)
#define LNANCHECK(w)
#define NNANCHECK(w, name)
#endif

__kernel
void nan_checker(__global ushort4* points, int point_count, __global float* arg, float scale, int4 dim)
{
    int idx = get_global_id(0);

    if(idx >= point_count)
        return;

    int ix = points[idx].x;
    int iy = points[idx].y;
    int iz = points[idx].z;

    int index = IDX(ix,iy,iz);

    NANCHECK_IMPL(arg);
}

__kernel
void calculate_hydrodynamic_initial_conditions(STANDARD_ARGS(),
                                               __global float* u_value,
                                               __global float* tov_phi,
                                               float scale, int4 dim)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    int index = IDX(ix,iy,iz);

    float3 offset = transform_position(ix, iy, iz, dim, scale);

    float ox = offset.x;
    float oy = offset.y;
    float oz = offset.z;

    float TEMPORARIEShydroconvert;

    float dp_val = build_p_star;
    float de_val = build_e_star;

    float cS0 = build_sk0;
    float cS1 = build_sk1;
    float cS2 = build_sk2;

    //printf("w2 %f %i %i %i\n", debug_w2, ix, iy,iz);

    ///dp_val and cS are both regular
    if(dp_val < 0.00001f)
    {
        de_val = 0;
    }

    dp_val = max(dp_val, 0.f);
    de_val = max(de_val, 0.f);

    /*LNANCHECK(D_eps_p0);
    LNANCHECK(D_p0);
    LNANCHECK(D_h);
    LNANCHECK(D_pressure);
    LNANCHECK(D_gA_u0);

    if(D_eps_p0 < 0)
    {
        printf("ep0 %f h %f p0 %f press %f p* %f rho %f rhoH %f conf press %f p0 %f phi %f W2 %f littlee %f eps %f\n", D_eps_p0, D_h, D_p0, D_pressure, D_p_star, D_rho, D_rhoH, D_conformal_pressure, D_conformal_rest_mass, D_phi, D_W2, D_littlee, D_eps);
    }*/

    //printf("%f %f\n", D_enthalpy, D_gA_u0);

    /*if(ix == 107 && iy == 125 && iz == 125)
    {
        printf("Dtc %f %f %f %f %f %f %f %f\n", D_p0, D_gA, D_u0, D_chip, D_X, D_phi, D_u, D_DYN);
    }*/


    /*if((ix == 98 || ix == 99) && iy == 125 && iz == 125)
    {
        printf("Debugging yay %f %f %f\n", p0D, pD, phiasdf);
    }*/

    Dp_star[index] = dp_val;
    De_star[index] = de_val;

    DcS0[index] = cS0;
    DcS1[index] = cS1;
    DcS2[index] = cS2;

    NANCHECK(Dp_star);
    NANCHECK(De_star);
    NANCHECK(DcS0);
    NANCHECK(DcS1);
    NANCHECK(DcS2);

    /*Dp_star[index] = 0;
    De_star[index] = 0;
    DcS0[index] = 0;
    DcS1[index] = 0;
    DcS2[index] = 0;*/
}

__kernel
void enforce_algebraic_constraints(__global ushort4* points, int point_count,
                                   STANDARD_ARGS(),
                                   float scale, int4 dim)
{
    int idx = get_global_id(0);

    if(idx >= point_count)
        return;

    int ix = points[idx].x;
    int iy = points[idx].y;
    int iz = points[idx].z;

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    int index = IDX(ix, iy, iz);

    #ifndef NO_CAIJYY

    float found_det = CY_DET;

    float tol = 1e-6;

    if(found_det <= 1 + tol && found_det >= 1 - tol)
        return;

    ///if this fires, its probably matter falling into a black hole
    NNANCHECK(CY_DET, "CY_DET");

    float fixed_cY0 = fix_cY0;
    float fixed_cY1 = fix_cY1;
    float fixed_cY2 = fix_cY2;
    float fixed_cY3 = fix_cY3;
    float fixed_cY4 = fix_cY4;
    float fixed_cY5 = fix_cY5;

    float fixed_cA0 = fix_cA0;
    float fixed_cA1 = fix_cA1;
    float fixed_cA2 = fix_cA2;
    float fixed_cA3 = fix_cA3;
    float fixed_cA4 = fix_cA4;
    float fixed_cA5 = fix_cA5;

    cY0[index] = fixed_cY0;
    cY1[index] = fixed_cY1;
    cY2[index] = fixed_cY2;
    cY3[index] = fixed_cY3;
    cY4[index] = fixed_cY4;
    cY5[index] = fixed_cY5;

    cA0[index] = fixed_cA0;
    cA1[index] = fixed_cA1;
    cA2[index] = fixed_cA2;
    cA3[index] = fixed_cA3;
    cA4[index] = fixed_cA4;
    cA5[index] = fixed_cA5;
    #else
    float fixed_cA3 = fix_cA3;

    cA3[index] = fixed_cA3;
    #endif // NO_CAIJYY
}

__kernel
void calculate_intermediate_data_thin(__global ushort4* points, int point_count,
                                      __global float* buffer, __global DERIV_PRECISION* buffer_out_1, __global DERIV_PRECISION* buffer_out_2, __global DERIV_PRECISION* buffer_out_3,
                                      float scale, int4 dim, __global ushort* order_ptr)
{
    int local_idx = get_global_id(0);

    if(local_idx >= point_count)
        return;

    int ix = points[local_idx].x;
    int iy = points[local_idx].y;
    int iz = points[local_idx].z;

    int order = order_ptr[IDX(ix,iy,iz)];

    if((order & D_FULL) > 0 || ((order & D_LOW) > 0))
    {
        float TEMPORARIES10;

        buffer_out_1[IDX(ix,iy,iz)] = init_buffer_intermediate0;
        buffer_out_2[IDX(ix,iy,iz)] = init_buffer_intermediate1;
        buffer_out_3[IDX(ix,iy,iz)] = init_buffer_intermediate2;
    }
    else
    {
        float TEMPORARIESdirectional;

        buffer_out_1[IDX(ix,iy,iz)] = init_buffer_intermediate0_directional;
        buffer_out_2[IDX(ix,iy,iz)] = init_buffer_intermediate1_directional;
        buffer_out_3[IDX(ix,iy,iz)] = init_buffer_intermediate2_directional;
    }
}

#if 0
__kernel
void calculate_intermediate_data_thin_cY5(__global ushort4* points, int point_count,
                                          __global float* cY0, __global float* cY1, __global float* cY2, __global float* cY3, __global float* cY4,
                                          __global DERIV_PRECISION* buffer_out_1, __global DERIV_PRECISION* buffer_out_2, __global DERIV_PRECISION* buffer_out_3,
                                         float scale, int4 dim)
{
    int local_idx = get_global_id(0);

    if(local_idx >= point_count)
        return;

    int ix = points[local_idx].x;
    int iy = points[local_idx].y;
    int iz = points[local_idx].z;

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    #ifndef SYMMETRY_BOUNDARY
    if(invalid_first(ix, iy, iz, dim))
        return;
    #endif // SYMMETRY_BOUNDARY

    float TEMPORARIES11;

    float i1 = init_cY5_intermediate0;
    float i2 = init_cY5_intermediate1;
    float i3 = init_cY5_intermediate2;

    buffer_out_1[IDX(ix,iy,iz)] = i1;
    buffer_out_2[IDX(ix,iy,iz)] = i2;
    buffer_out_3[IDX(ix,iy,iz)] = i3;
}
#endif // 0

__kernel
void calculate_momentum_constraint(__global ushort4* points, int point_count,
                                   STANDARD_ARGS(),
                                   __global float* momentum0, __global float* momentum1, __global float* momentum2,
                                   float scale, int4 dim)
{
    int local_idx = get_global_id(0);

    if(local_idx >= point_count)
        return;

    int ix = points[local_idx].x;
    int iy = points[local_idx].y;
    int iz = points[local_idx].z;

    float TEMPORARIES12;

    float m1 = init_momentum0;
    float m2 = init_momentum1;
    float m3 = init_momentum2;

    momentum0[IDX(ix,iy,iz)] = m1;
    momentum1[IDX(ix,iy,iz)] = m2;
    momentum2[IDX(ix,iy,iz)] = m3;
}

__kernel
void generate_sponge_points(__global ushort4* points, __global int* point_count, float scale, int4 dim)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    float sponge_factor = sponge_damp_coeff(ix, iy, iz, scale, dim);

    if(sponge_factor <= 0)
        return;

    if(is_deep_boundary_point(ix, iy, iz, scale, dim))
        return;

    int idx = atomic_inc(point_count);

    points[idx].xyz = (ushort3)(ix, iy, iz);
}

__kernel
void clean_data_thin(__global ushort4* points, int point_count,
                __global float* input,
                __global float* base,
                __global float* out,
                __global ushort* order_ptr,
                float scale, int4 dim,
                float timestep,
                float asym, float speed)
{
    int idx = get_global_id(0);

    if(idx >= point_count)
        return;

    int ix = points[idx].x;
    int iy = points[idx].y;
    int iz = points[idx].z;

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    float3 offset = transform_position(ix, iy, iz, dim, scale);

    float ox = offset.x;
    float oy = offset.y;
    float oz = offset.z;

    int index = IDX(ix, iy, iz);
    int order = order_ptr[index];

    float TEMPORARIESsommerthin;

    float sommer_dtc = sommer_thin_out;

    out[index] = sommer_dtc * timestep + base[index];
}

///https://cds.cern.ch/record/517706/files/0106072.pdf
///boundary conditions
///todo: damp to schwarzschild, not initial conditions?
__kernel
void clean_data(__global ushort4* points, int point_count,
                STANDARD_ARGS(),
                STANDARD_ARGS(base_),
                STANDARD_ARGS(o),
                __global ushort* order_ptr,
                float scale, int4 dim,
                float timestep)
{
    int idx = get_global_id(0);

    if(idx >= point_count)
        return;

    int ix = points[idx].x;
    int iy = points[idx].y;
    int iz = points[idx].z;

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    float3 offset = transform_position(ix, iy, iz, dim, scale);

    float ox = offset.x;
    float oy = offset.y;
    float oz = offset.z;

    int index = IDX(ix, iy, iz);
    int order = order_ptr[index];

    {
        float TEMPORARIESsommerfeld;

        float s_dtcY0 = sommer_dtcY0;
        float s_dtcY1 = sommer_dtcY1;
        float s_dtcY2 = sommer_dtcY2;
        float s_dtcY3 = sommer_dtcY3;
        float s_dtcY4 = sommer_dtcY4;
        float s_dtcY5 = sommer_dtcY5;

        float s_dtcA0 = sommer_dtcA0;
        float s_dtcA1 = sommer_dtcA1;
        float s_dtcA2 = sommer_dtcA2;
        float s_dtcA3 = sommer_dtcA3;
        float s_dtcA4 = sommer_dtcA4;
        float s_dtcA5 = sommer_dtcA5;

        float s_dtK = sommer_dtK;
        float s_dtX = sommer_dtX;

        float s_dtgA = sommer_dtgA;
        float s_dtgB0 = sommer_dtgB0;
        float s_dtgB1 = sommer_dtgB1;
        float s_dtgB2 = sommer_dtgB2;

        float s_dtcGi0 = sommer_dtcGi0;
        float s_dtcGi1 = sommer_dtcGi1;
        float s_dtcGi2 = sommer_dtcGi2;

        #ifdef SOMMER_MATTER
        float s_dtcp_star = sommer_dtcp_star;
        float s_dtce_star = sommer_dtce_star;

        float s_dtcS0 = sommer_dtcS0;
        float s_dtcS1 = sommer_dtcS1;
        float s_dtcS2 = sommer_dtcS2;
        #endif // SOMMER_MATTER

        ocY0[index] = s_dtcY0 * timestep + base_cY0[index];
        ocY1[index] = s_dtcY1 * timestep + base_cY1[index];
        ocY2[index] = s_dtcY2 * timestep + base_cY2[index];
        ocY3[index] = s_dtcY3 * timestep + base_cY3[index];
        ocY4[index] = s_dtcY4 * timestep + base_cY4[index];
        ocY5[index] = s_dtcY5 * timestep + base_cY5[index];

        ocA0[index] = s_dtcA0 * timestep + base_cA0[index];
        ocA1[index] = s_dtcA1 * timestep + base_cA1[index];
        ocA2[index] = s_dtcA2 * timestep + base_cA2[index];
        ocA3[index] = s_dtcA3 * timestep + base_cA3[index];
        ocA4[index] = s_dtcA4 * timestep + base_cA4[index];
        ocA5[index] = s_dtcA5 * timestep + base_cA5[index];

        oK[index] = s_dtK * timestep + base_K[index];
        oX[index] = s_dtX * timestep + base_X[index];

        ogA[index] = s_dtgA * timestep + base_gA[index];
        ogB0[index] = s_dtgB0 * timestep + base_gB0[index];
        ogB1[index] = s_dtgB1 * timestep + base_gB1[index];
        ogB2[index] = s_dtgB2 * timestep + base_gB2[index];

        ocGi0[index] = s_dtcGi0 * timestep + base_cGi0[index];
        ocGi1[index] = s_dtcGi1 * timestep + base_cGi1[index];
        ocGi2[index] = s_dtcGi2 * timestep + base_cGi2[index];

        #ifdef SOMMER_MATTER
        oDp_star[index] = s_dtcp_star * timestep + base_Dp_star[index];
        oDe_star[index] = s_dtce_star * timestep + base_De_star[index];

        oDcS0[index] = s_dtcS0 * timestep + base_DcS0[index];
        oDcS1[index] = s_dtcS1 * timestep + base_DcS1[index];
        oDcS2[index] = s_dtcS2 * timestep + base_DcS2[index];
        #endif // SOMMER_MATTER
    }

    #if 0
    ///https://authors.library.caltech.edu/8284/1/RINcqg07.pdf (34)
    float y_r = sponge_factor;

    //#define EVOLVE_CY_AT_BOUNDARY
    #ifndef EVOLVE_CY_AT_BOUNDARY
    cY0[index] += -y_r * (cY0[index] - initial_cY0) * timestep;
    cY1[index] += -y_r * (cY1[index] - initial_cY1) * timestep;
    cY2[index] += -y_r * (cY2[index] - initial_cY2) * timestep;
    cY3[index] += -y_r * (cY3[index] - initial_cY3) * timestep;
    cY4[index] += -y_r * (cY4[index] - initial_cY4) * timestep;
    cY5[index] += -y_r * (cY5[index] - initial_cY5) * timestep;
    #endif // EVOLVE_CY_AT_BOUNDARY

    cA0[index] += -y_r * (cA0[index] - initial_cA0) * timestep;
    cA1[index] += -y_r * (cA1[index] - initial_cA1) * timestep;
    cA2[index] += -y_r * (cA2[index] - initial_cA2) * timestep;
    #ifndef NO_CAIJYY
    cA3[index] += -y_r * (cA3[index] - initial_cA3) * timestep;
    #endif // NO_CAIJYY
    cA4[index] += -y_r * (cA4[index] - initial_cA4) * timestep;
    cA5[index] += -y_r * (cA5[index] - initial_cA5) * timestep;

    cGi0[index] += -y_r * (cGi0[index] - init_cGi0) * timestep;
    cGi1[index] += -y_r * (cGi1[index] - init_cGi1) * timestep;
    cGi2[index] += -y_r * (cGi2[index] - init_cGi2) * timestep;

    K[index] += -y_r * (K[index] - init_K) * timestep;
    X[index] += -y_r * (X[index] - initial_X) * timestep;

    gA[index] += -y_r * (gA[index] - fin_gA) * timestep;
    gB0[index] += -y_r * (gB0[index] - fin_gB0) * timestep;
    gB1[index] += -y_r * (gB1[index] - fin_gB1) * timestep;
    gB2[index] += -y_r * (gB2[index] - fin_gB2) * timestep;

    #ifdef USE_GBB
    gBB0[index] += -y_r * (gBB0[index] - fin_gBB0) * timestep;
    gBB1[index] += -y_r * (gBB1[index] - fin_gBB1) * timestep;
    gBB2[index] += -y_r * (gBB2[index] - fin_gBB2) * timestep;
    #endif // USE_GBB
    #endif // 0
}

#define DISSB 0.1f

__kernel
void evolve_cY(__global ushort4* points, int point_count,
            STANDARD_ARGS(),
            STANDARD_ARGS(o),
            STANDARD_ARGS(base_),
            __global float* momentum0, __global float* momentum1, __global float* momentum2,
            __global DERIV_PRECISION* dcYij0, __global DERIV_PRECISION* dcYij1, __global DERIV_PRECISION* dcYij2, __global DERIV_PRECISION* dcYij3, __global DERIV_PRECISION* dcYij4, __global DERIV_PRECISION* dcYij5, __global DERIV_PRECISION* dcYij6, __global DERIV_PRECISION* dcYij7, __global DERIV_PRECISION* dcYij8, __global DERIV_PRECISION* dcYij9, __global DERIV_PRECISION* dcYij10, __global DERIV_PRECISION* dcYij11, __global DERIV_PRECISION* dcYij12, __global DERIV_PRECISION* dcYij13, __global DERIV_PRECISION* dcYij14, __global DERIV_PRECISION* dcYij15, __global DERIV_PRECISION* dcYij16, __global DERIV_PRECISION* dcYij17,
            __global DERIV_PRECISION* digA0, __global DERIV_PRECISION* digA1, __global DERIV_PRECISION* digA2,
            __global DERIV_PRECISION* digB0, __global DERIV_PRECISION* digB1, __global DERIV_PRECISION* digB2, __global DERIV_PRECISION* digB3, __global DERIV_PRECISION* digB4, __global DERIV_PRECISION* digB5, __global DERIV_PRECISION* digB6, __global DERIV_PRECISION* digB7, __global DERIV_PRECISION* digB8,
            __global DERIV_PRECISION* dX0, __global DERIV_PRECISION* dX1, __global DERIV_PRECISION* dX2,
            float scale, int4 dim, float timestep, __global ushort* order_ptr)
{
    int local_idx = get_global_id(0);

    if(local_idx >= point_count)
        return;

    int ix = points[local_idx].x;
    int iy = points[local_idx].y;
    int iz = points[local_idx].z;

    int index = IDX(ix, iy, iz);
    int order = order_ptr[index];

    if((order & D_FULL) == 0 && ((order & D_LOW) == 0))
    {
        ocY0[index] = cY0[index];
        ocY1[index] = cY1[index];
        ocY2[index] = cY2[index];
        ocY3[index] = cY3[index];
        ocY4[index] = cY4[index];
        ocY5[index] = cY5[index];
        return;
    }

    float TEMPORARIEStcy;

    float f_dtcYij0 = dtcYij0;
    float f_dtcYij1 = dtcYij1;
    float f_dtcYij2 = dtcYij2;
    float f_dtcYij3 = dtcYij3;
    float f_dtcYij4 = dtcYij4;
    float f_dtcYij5 = dtcYij5;

    float b0 = base_cY0[index];
    float b1 = base_cY1[index];
    float b2 = base_cY2[index];
    float b3 = base_cY3[index];
    float b4 = base_cY4[index];
    float b5 = base_cY5[index];

    ocY0[index] = f_dtcYij0 * timestep + b0;
    ocY1[index] = f_dtcYij1 * timestep + b1;
    ocY2[index] = f_dtcYij2 * timestep + b2;
    ocY3[index] = f_dtcYij3 * timestep + b3;
    ocY4[index] = f_dtcYij4 * timestep + b4;
    ocY5[index] = f_dtcYij5 * timestep + b5;

    /*if(X[index] < DISSB)
    {
        ocY0[index] += (1 - ocY0[index]) * timestep;
        ocY1[index] += (0 - ocY1[index]) * timestep;
        ocY2[index] += (0 - ocY2[index]) * timestep;
        ocY3[index] += (1 - ocY3[index]) * timestep;
        ocY4[index] += (0 - ocY4[index]) * timestep;
        ocY5[index] += (1 - ocY5[index]) * timestep;
    }*/

    NANCHECK(ocY0);
    NANCHECK(ocY1);
    NANCHECK(ocY2);
    NANCHECK(ocY3);
    NANCHECK(ocY4);
    NANCHECK(ocY5);
}

__kernel
void evolve_cA(__global ushort4* points, int point_count,
            STANDARD_ARGS(),
            STANDARD_ARGS(o),
            STANDARD_ARGS(base_),
            __global float* momentum0, __global float* momentum1, __global float* momentum2,
            __global DERIV_PRECISION* dcYij0, __global DERIV_PRECISION* dcYij1, __global DERIV_PRECISION* dcYij2, __global DERIV_PRECISION* dcYij3, __global DERIV_PRECISION* dcYij4, __global DERIV_PRECISION* dcYij5, __global DERIV_PRECISION* dcYij6, __global DERIV_PRECISION* dcYij7, __global DERIV_PRECISION* dcYij8, __global DERIV_PRECISION* dcYij9, __global DERIV_PRECISION* dcYij10, __global DERIV_PRECISION* dcYij11, __global DERIV_PRECISION* dcYij12, __global DERIV_PRECISION* dcYij13, __global DERIV_PRECISION* dcYij14, __global DERIV_PRECISION* dcYij15, __global DERIV_PRECISION* dcYij16, __global DERIV_PRECISION* dcYij17,
            __global DERIV_PRECISION* digA0, __global DERIV_PRECISION* digA1, __global DERIV_PRECISION* digA2,
            __global DERIV_PRECISION* digB0, __global DERIV_PRECISION* digB1, __global DERIV_PRECISION* digB2, __global DERIV_PRECISION* digB3, __global DERIV_PRECISION* digB4, __global DERIV_PRECISION* digB5, __global DERIV_PRECISION* digB6, __global DERIV_PRECISION* digB7, __global DERIV_PRECISION* digB8,
            __global DERIV_PRECISION* dX0, __global DERIV_PRECISION* dX1, __global DERIV_PRECISION* dX2,
            float scale, int4 dim, float timestep, __global ushort* order_ptr)
{
    int local_idx = get_global_id(0);

    if(local_idx >= point_count)
        return;

    int ix = points[local_idx].x;
    int iy = points[local_idx].y;
    int iz = points[local_idx].z;

    int index = IDX(ix, iy, iz);
    int order = order_ptr[index];

    if((order & D_FULL) == 0 && ((order & D_LOW) == 0))
    {
        ocA0[index] = cA0[index];
        ocA1[index] = cA1[index];
        ocA2[index] = cA2[index];
        ocA3[index] = cA3[index];
        ocA4[index] = cA4[index];
        ocA5[index] = cA5[index];
        return;
    }

    float TEMPORARIEStca;

    float f_dtcAij0 = dtcAij0;
    float f_dtcAij1 = dtcAij1;
    float f_dtcAij2 = dtcAij2;
    float f_dtcAij3 = dtcAij3;
    float f_dtcAij4 = dtcAij4;
    float f_dtcAij5 = dtcAij5;

    float b0 = base_cA0[index];
    float b1 = base_cA1[index];
    float b2 = base_cA2[index];
    float b3 = base_cA3[index];
    float b4 = base_cA4[index];
    float b5 = base_cA5[index];

    ocA0[index] = f_dtcAij0 * timestep + b0;
    ocA1[index] = f_dtcAij1 * timestep + b1;
    ocA2[index] = f_dtcAij2 * timestep + b2;
    #ifndef NO_CAIJYY
    ocA3[index] = f_dtcAij3 * timestep + b3;
    #endif // NO_CAIJYY
    ocA4[index] = f_dtcAij4 * timestep + b4;
    ocA5[index] = f_dtcAij5 * timestep + b5;

    ///NAN ocA0 107 125 125

    /*if(X[index] < DISSB)
    {
        ocA0[index] += (0 - ocA0[index]) * timestep;
        ocA1[index] += (0 - ocA1[index]) * timestep;
        ocA2[index] += (0 - ocA2[index]) * timestep;
        ocA3[index] += (0 - ocA3[index]) * timestep;
        ocA4[index] += (0 - ocA4[index]) * timestep;
        ocA5[index] += (0 - ocA5[index]) * timestep;
    }*/

    NANCHECK(ocA0);
    NANCHECK(ocA1);
    NANCHECK(ocA2);
    NANCHECK(ocA3);
    NANCHECK(ocA4);
    NANCHECK(ocA5);

    /*if(ix == 97 && iy == 124 && iz == 124)
    {
        printf("Here we go again xsij %f %f %f cS0 %f\n", DBGXGA, cA0[index], cY0[index], Debug_cS0);
    }*/
}

__kernel
void evolve_cGi(__global ushort4* points, int point_count,
            STANDARD_ARGS(),
            STANDARD_ARGS(o),
            STANDARD_ARGS(base_),
            __global float* momentum0, __global float* momentum1, __global float* momentum2,
            __global DERIV_PRECISION* dcYij0, __global DERIV_PRECISION* dcYij1, __global DERIV_PRECISION* dcYij2, __global DERIV_PRECISION* dcYij3, __global DERIV_PRECISION* dcYij4, __global DERIV_PRECISION* dcYij5, __global DERIV_PRECISION* dcYij6, __global DERIV_PRECISION* dcYij7, __global DERIV_PRECISION* dcYij8, __global DERIV_PRECISION* dcYij9, __global DERIV_PRECISION* dcYij10, __global DERIV_PRECISION* dcYij11, __global DERIV_PRECISION* dcYij12, __global DERIV_PRECISION* dcYij13, __global DERIV_PRECISION* dcYij14, __global DERIV_PRECISION* dcYij15, __global DERIV_PRECISION* dcYij16, __global DERIV_PRECISION* dcYij17,
            __global DERIV_PRECISION* digA0, __global DERIV_PRECISION* digA1, __global DERIV_PRECISION* digA2,
            __global DERIV_PRECISION* digB0, __global DERIV_PRECISION* digB1, __global DERIV_PRECISION* digB2, __global DERIV_PRECISION* digB3, __global DERIV_PRECISION* digB4, __global DERIV_PRECISION* digB5, __global DERIV_PRECISION* digB6, __global DERIV_PRECISION* digB7, __global DERIV_PRECISION* digB8,
            __global DERIV_PRECISION* dX0, __global DERIV_PRECISION* dX1, __global DERIV_PRECISION* dX2,
            float scale, int4 dim, float timestep, __global ushort* order_ptr)
{
    int local_idx = get_global_id(0);

    if(local_idx >= point_count)
        return;

    int ix = points[local_idx].x;
    int iy = points[local_idx].y;
    int iz = points[local_idx].z;

    int index = IDX(ix, iy, iz);
    int order = order_ptr[index];

    if((order & D_FULL) == 0 && ((order & D_LOW) == 0))
    {
        ocGi0[index] = cGi0[index];
        ocGi1[index] = cGi1[index];
        ocGi2[index] = cGi2[index];
        return;
    }

    float TEMPORARIEStcgi;

    float f_dtcGi0 = dtcGi0;
    float f_dtcGi1 = dtcGi1;
    float f_dtcGi2 = dtcGi2;

    #ifdef USE_GBB
    float f_gBB0 = dtgBB0;
    float f_gBB1 = dtgBB1;
    float f_gBB2 = dtgBB2;
    #endif // USE_GBB

    {
        float b0 = base_cGi0[index];
        float b1 = base_cGi1[index];
        float b2 = base_cGi2[index];

        ocGi0[index] = f_dtcGi0 * timestep + b0;
        ocGi1[index] = f_dtcGi1 * timestep + b1;
        ocGi2[index] = f_dtcGi2 * timestep + b2;
    }

    /*if(X[index] < DISSB)
    {
        ocGi0[index] += (0 - ocGi0[index]) * timestep;
        ocGi1[index] += (0 - ocGi1[index]) * timestep;
        ocGi2[index] += (0 - ocGi2[index]) * timestep;
    }*/

    NANCHECK(ocGi0);
    NANCHECK(ocGi1);
    NANCHECK(ocGi2);

    #ifdef USE_GBB
    {
        float b0 = base_gBB0[index];
        float b1 = base_gBB1[index];
        float b2 = base_gBB2[index];

        ogBB0[index] = f_gBB0 * timestep + b0;
        ogBB1[index] = f_gBB1 * timestep + b1;
        ogBB2[index] = f_gBB2 * timestep + b2;
    }
    #endif // USE_GBB
}


__kernel
void evolve_K(__global ushort4* points, int point_count,
            STANDARD_ARGS(),
            STANDARD_ARGS(o),
            STANDARD_ARGS(base_),
            __global float* momentum0, __global float* momentum1, __global float* momentum2,
            __global DERIV_PRECISION* dcYij0, __global DERIV_PRECISION* dcYij1, __global DERIV_PRECISION* dcYij2, __global DERIV_PRECISION* dcYij3, __global DERIV_PRECISION* dcYij4, __global DERIV_PRECISION* dcYij5, __global DERIV_PRECISION* dcYij6, __global DERIV_PRECISION* dcYij7, __global DERIV_PRECISION* dcYij8, __global DERIV_PRECISION* dcYij9, __global DERIV_PRECISION* dcYij10, __global DERIV_PRECISION* dcYij11, __global DERIV_PRECISION* dcYij12, __global DERIV_PRECISION* dcYij13, __global DERIV_PRECISION* dcYij14, __global DERIV_PRECISION* dcYij15, __global DERIV_PRECISION* dcYij16, __global DERIV_PRECISION* dcYij17,
            __global DERIV_PRECISION* digA0, __global DERIV_PRECISION* digA1, __global DERIV_PRECISION* digA2,
            __global DERIV_PRECISION* digB0, __global DERIV_PRECISION* digB1, __global DERIV_PRECISION* digB2, __global DERIV_PRECISION* digB3, __global DERIV_PRECISION* digB4, __global DERIV_PRECISION* digB5, __global DERIV_PRECISION* digB6, __global DERIV_PRECISION* digB7, __global DERIV_PRECISION* digB8,
            __global DERIV_PRECISION* dX0, __global DERIV_PRECISION* dX1, __global DERIV_PRECISION* dX2,
            float scale, int4 dim, float timestep, __global ushort* order_ptr)
{
    int local_idx = get_global_id(0);

    if(local_idx >= point_count)
        return;

    int ix = points[local_idx].x;
    int iy = points[local_idx].y;
    int iz = points[local_idx].z;

    int index = IDX(ix, iy, iz);
    int order = order_ptr[index];

    if((order & D_FULL) == 0 && ((order & D_LOW) == 0))
    {
        oK[index] = K[index];
        return;
    }

    float TEMPORARIEStk;

    float f_dtK = dtK;

    float b0 = base_K[index];

    oK[index] = f_dtK * timestep + b0;

    /*if(X[index] < DISSB)
    {
        oK[index] += (0 - oK[index]) * timestep;
    }*/

    NANCHECK(oK);

    /*if(ix == 109 && iy == 125 && iz == 125)
    {
        printf("K s %f p %f h %f em %f p0 %f eps %f %f\n", Dbg_matter_s, Dbg_matter_p, Dbg_h, Dbg_em, Dbg_p0, Dbg_eps, Dbg_X);
    }*/
}


__kernel
void evolve_X(__global ushort4* points, int point_count,
            STANDARD_ARGS(),
            STANDARD_ARGS(o),
            STANDARD_ARGS(base_),
            __global float* momentum0, __global float* momentum1, __global float* momentum2,
            __global DERIV_PRECISION* dcYij0, __global DERIV_PRECISION* dcYij1, __global DERIV_PRECISION* dcYij2, __global DERIV_PRECISION* dcYij3, __global DERIV_PRECISION* dcYij4, __global DERIV_PRECISION* dcYij5, __global DERIV_PRECISION* dcYij6, __global DERIV_PRECISION* dcYij7, __global DERIV_PRECISION* dcYij8, __global DERIV_PRECISION* dcYij9, __global DERIV_PRECISION* dcYij10, __global DERIV_PRECISION* dcYij11, __global DERIV_PRECISION* dcYij12, __global DERIV_PRECISION* dcYij13, __global DERIV_PRECISION* dcYij14, __global DERIV_PRECISION* dcYij15, __global DERIV_PRECISION* dcYij16, __global DERIV_PRECISION* dcYij17,
            __global DERIV_PRECISION* digA0, __global DERIV_PRECISION* digA1, __global DERIV_PRECISION* digA2,
            __global DERIV_PRECISION* digB0, __global DERIV_PRECISION* digB1, __global DERIV_PRECISION* digB2, __global DERIV_PRECISION* digB3, __global DERIV_PRECISION* digB4, __global DERIV_PRECISION* digB5, __global DERIV_PRECISION* digB6, __global DERIV_PRECISION* digB7, __global DERIV_PRECISION* digB8,
            __global DERIV_PRECISION* dX0, __global DERIV_PRECISION* dX1, __global DERIV_PRECISION* dX2,
            float scale, int4 dim, float timestep, __global ushort* order_ptr)
{
    int local_idx = get_global_id(0);

    if(local_idx >= point_count)
        return;

    int ix = points[local_idx].x;
    int iy = points[local_idx].y;
    int iz = points[local_idx].z;

    int index = IDX(ix, iy, iz);
    int order = order_ptr[index];

    if((order & D_FULL) == 0 && ((order & D_LOW) == 0))
    {
        oX[index] = X[index];
        return;
    }

    float TEMPORARIEStx;

    float f_dtX = dtX;

    float b0 = base_X[index];

    oX[index] = max(f_dtX * timestep + b0, 0.f);

    /**/

    NANCHECK(oX);
}

__kernel
void evolve_gA(__global ushort4* points, int point_count,
            STANDARD_ARGS(),
            STANDARD_ARGS(o),
            STANDARD_ARGS(base_),
            __global float* momentum0, __global float* momentum1, __global float* momentum2,
            __global DERIV_PRECISION* dcYij0, __global DERIV_PRECISION* dcYij1, __global DERIV_PRECISION* dcYij2, __global DERIV_PRECISION* dcYij3, __global DERIV_PRECISION* dcYij4, __global DERIV_PRECISION* dcYij5, __global DERIV_PRECISION* dcYij6, __global DERIV_PRECISION* dcYij7, __global DERIV_PRECISION* dcYij8, __global DERIV_PRECISION* dcYij9, __global DERIV_PRECISION* dcYij10, __global DERIV_PRECISION* dcYij11, __global DERIV_PRECISION* dcYij12, __global DERIV_PRECISION* dcYij13, __global DERIV_PRECISION* dcYij14, __global DERIV_PRECISION* dcYij15, __global DERIV_PRECISION* dcYij16, __global DERIV_PRECISION* dcYij17,
            __global DERIV_PRECISION* digA0, __global DERIV_PRECISION* digA1, __global DERIV_PRECISION* digA2,
            __global DERIV_PRECISION* digB0, __global DERIV_PRECISION* digB1, __global DERIV_PRECISION* digB2, __global DERIV_PRECISION* digB3, __global DERIV_PRECISION* digB4, __global DERIV_PRECISION* digB5, __global DERIV_PRECISION* digB6, __global DERIV_PRECISION* digB7, __global DERIV_PRECISION* digB8,
            __global DERIV_PRECISION* dX0, __global DERIV_PRECISION* dX1, __global DERIV_PRECISION* dX2,
            float scale, int4 dim, float timestep, __global ushort* order_ptr)
{
    int local_idx = get_global_id(0);

    if(local_idx >= point_count)
        return;

    int ix = points[local_idx].x;
    int iy = points[local_idx].y;
    int iz = points[local_idx].z;

    int index = IDX(ix, iy, iz);
    int order = order_ptr[index];

    if((order & D_FULL) == 0 && ((order & D_LOW) == 0))
    {
        ogA[index] = gA[index];
        return;
    }

    float TEMPORARIEStga;

    float f_dtgA = dtgA;

    float b0 = base_gA[index];

    ogA[index] = max(f_dtgA * timestep + b0, 0.f);

    NANCHECK(ogA);
}


__kernel
void evolve_gB(__global ushort4* points, int point_count,
            STANDARD_ARGS(),
            STANDARD_ARGS(o),
            STANDARD_ARGS(base_),
            __global float* momentum0, __global float* momentum1, __global float* momentum2,
            __global DERIV_PRECISION* dcYij0, __global DERIV_PRECISION* dcYij1, __global DERIV_PRECISION* dcYij2, __global DERIV_PRECISION* dcYij3, __global DERIV_PRECISION* dcYij4, __global DERIV_PRECISION* dcYij5, __global DERIV_PRECISION* dcYij6, __global DERIV_PRECISION* dcYij7, __global DERIV_PRECISION* dcYij8, __global DERIV_PRECISION* dcYij9, __global DERIV_PRECISION* dcYij10, __global DERIV_PRECISION* dcYij11, __global DERIV_PRECISION* dcYij12, __global DERIV_PRECISION* dcYij13, __global DERIV_PRECISION* dcYij14, __global DERIV_PRECISION* dcYij15, __global DERIV_PRECISION* dcYij16, __global DERIV_PRECISION* dcYij17,
            __global DERIV_PRECISION* digA0, __global DERIV_PRECISION* digA1, __global DERIV_PRECISION* digA2,
            __global DERIV_PRECISION* digB0, __global DERIV_PRECISION* digB1, __global DERIV_PRECISION* digB2, __global DERIV_PRECISION* digB3, __global DERIV_PRECISION* digB4, __global DERIV_PRECISION* digB5, __global DERIV_PRECISION* digB6, __global DERIV_PRECISION* digB7, __global DERIV_PRECISION* digB8,
            __global DERIV_PRECISION* dX0, __global DERIV_PRECISION* dX1, __global DERIV_PRECISION* dX2,
            float scale, int4 dim, float timestep, __global ushort* order_ptr)
{
    int local_idx = get_global_id(0);

    if(local_idx >= point_count)
        return;

    int ix = points[local_idx].x;
    int iy = points[local_idx].y;
    int iz = points[local_idx].z;

    int index = IDX(ix, iy, iz);
    int order = order_ptr[index];

    if((order & D_FULL) == 0 && ((order & D_LOW) == 0))
    {
        ogB0[index] = gB0[index];
        ogB1[index] = gB1[index];
        ogB2[index] = gB2[index];
        return;
    }

    float TEMPORARIEStgb;

    float f_dtgB0 = dtgB0;
    float f_dtgB1 = dtgB1;
    float f_dtgB2 = dtgB2;

    float b0 = base_gB0[index];
    float b1 = base_gB1[index];
    float b2 = base_gB2[index];

    ogB0[index] = f_dtgB0 * timestep + b0;
    ogB1[index] = f_dtgB1 * timestep + b1;
    ogB2[index] = f_dtgB2 * timestep + b2;

    NANCHECK(ogB0);
    NANCHECK(ogB1);
    NANCHECK(ogB2);
}

#define MIN_P_STAR 1e-5f

#define HYDRO_ORDER 2

///this is incorrect due to intermediates needing to be 0 ?
__kernel
void calculate_hydro_evolved(__global ushort4* points, int point_count,
                             STANDARD_ARGS(),
                             float scale, int4 dim, __global ushort* order_ptr,
                             __global char* restrict should_evolve)
{
    int local_idx = get_global_id(0);

    if(local_idx >= point_count)
        return;

    int ix = points[local_idx].x;
    int iy = points[local_idx].y;
    int iz = points[local_idx].z;

    int index = IDX(ix, iy, iz);
    int order = order_ptr[index];

    if((order & D_FULL) == 0 && (order & D_LOW) == 0)
    {
        should_evolve[index] = false;
        return;
    }

    float P_count = 0;

    #pragma unroll
    for(int i=-HYDRO_ORDER; i <= HYDRO_ORDER; i++)
    {
        P_count += max(Dp_star[IDX(ix + i, iy, iz)], 0.f);
    }

    #pragma unroll
    for(int i=-HYDRO_ORDER; i <= HYDRO_ORDER; i++)
    {
        if(i == 0)
            continue;

        P_count += max(Dp_star[IDX(ix, iy + i, iz)], 0.f);
    }

    #pragma unroll
    for(int i=-HYDRO_ORDER; i <= HYDRO_ORDER; i++)
    {
        if(i == 0)
            continue;

        P_count += max(Dp_star[IDX(ix, iy, iz + i)], 0.f);
    }

    should_evolve[index] = P_count > 0;
}

///does not use any derivatives
__kernel
void calculate_hydro_intermediates(__global ushort4* points, int point_count,
                                   STANDARD_ARGS(),
                                   __global float* pressure,
                                   __global float* hW,
                                   float scale, int4 dim, __global ushort* order_ptr, __global char* restrict should_evolve)
{
    int local_idx = get_global_id(0);

    if(local_idx >= point_count)
        return;

    int ix = points[local_idx].x;
    int iy = points[local_idx].y;
    int iz = points[local_idx].z;

    int index = IDX(ix, iy, iz);
    int order = order_ptr[index];

    int any_valid = 0;

    #pragma unroll
    for(int i=-HYDRO_ORDER; i <= HYDRO_ORDER; i++)
    {
        any_valid += should_evolve[IDX(ix + i, iy, iz)];
    }

    #pragma unroll
    for(int i=-HYDRO_ORDER; i <= HYDRO_ORDER; i++)
    {
        if(i == 0)
            continue;

        any_valid += should_evolve[IDX(ix, iy + i, iz)];
    }

    #pragma unroll
    for(int i=-HYDRO_ORDER; i <= HYDRO_ORDER; i++)
    {
        if(i == 0)
            continue;

        any_valid += should_evolve[IDX(ix, iy, iz + i)];
    }

    if(any_valid == 0)
        return;

    if(Dp_star[index] < MIN_P_STAR)
    {
        pressure[index] = 0;
        hW[index] = 0;
        return;
    }

    float TEMPORARIEShydrointermediates;

    float cpress = init_pressure;
    float W_var = init_W;

    pressure[index] = cpress;
    hW[index] = W_var;

    NANCHECK(pressure);
    NANCHECK(hW);
}

__kernel
void add_hydro_artificial_viscosity(__global ushort4* points, int point_count,
                                    STANDARD_ARGS(),
                                    __global float* pressure,
                                    __global float* hW,
                                    float scale, int4 dim, __global ushort* order_ptr, __global char* restrict should_evolve)
{

    int local_idx = get_global_id(0);

    if(local_idx >= point_count)
        return;

    int ix = points[local_idx].x;
    int iy = points[local_idx].y;
    int iz = points[local_idx].z;

    int index = IDX(ix, iy, iz);
    int order = order_ptr[index];

    if(((order & D_FULL) == 0 && ((order & D_LOW) == 0)) || should_evolve[index] == 0)
        return;

    float TEMPORARIEShydroviscosity;

    float added = init_artificial_viscosity;

    if(added == 0)
        return;

    pressure[IDX(ix,iy,iz)] += added;
}

///does use derivatives
__kernel
void evolve_hydro_all(__global ushort4* points, int point_count,
                      STANDARD_ARGS(),
                      STANDARD_ARGS(o),
                      STANDARD_ARGS(base_),
                      __global float* pressure,
                      __global float* hW,
                      float scale, int4 dim, __global ushort* order_ptr, __global char* restrict should_evolve, float timestep)
{
    int local_idx = get_global_id(0);

    if(local_idx >= point_count)
        return;

    int ix = points[local_idx].x;
    int iy = points[local_idx].y;
    int iz = points[local_idx].z;

    int index = IDX(ix, iy, iz);
    int order = order_ptr[index];

    ///we're copying over base. Is that correct? Because sommerfeld
    if(((order & D_FULL) == 0 && ((order & D_LOW) == 0)) || should_evolve[index] == 0)
    {
        oDp_star[index] = Dp_star[index];
        oDe_star[index] = De_star[index];

        oDcS0[index] = DcS0[index];
        oDcS1[index] = DcS1[index];
        oDcS2[index] = DcS2[index];
        return;
    }

    float f_dtp_star = init_dtp_star;

    float base_p_star =  base_Dp_star[index];
    float fin_p_star = f_dtp_star * timestep + base_p_star;

    LNANCHECK(base_p_star);

    float TEMPORARIEShydrofinal;

    if(fin_p_star <= MIN_P_STAR)
    {
        oDp_star[index] = 0;
        oDe_star[index] = 0;
        oDcS0[index] = 0;
        oDcS1[index] = 0;
        oDcS2[index] = 0;

        return;
    }

    float f_dte_star = init_dte_star;
    float f_dtSk0 = init_dtSk0;
    float f_dtSk1 = init_dtSk1;
    float f_dtSk2 = init_dtSk2;

    float base_e_star = base_De_star[index];
    float base_cS0 = base_DcS0[index];
    float base_cS1 = base_DcS1[index];
    float base_cS2 = base_DcS2[index];

    LNANCHECK(base_e_star);
    LNANCHECK(base_cS0);
    LNANCHECK(base_cS1);
    LNANCHECK(base_cS2);

    float fin_e_star = f_dte_star * timestep + base_e_star;
    float fin_cS0 = f_dtSk0 * timestep + base_cS0;
    float fin_cS1 = f_dtSk1 * timestep + base_cS1;
    float fin_cS2 = f_dtSk2 * timestep + base_cS2;

    ///clamping to 0.05 this fixes some issues
    ///this makes a big difference to stability around collisions
    fin_cS0 = clamp(fin_cS0, -0.05f, 0.05f);
    fin_cS1 = clamp(fin_cS1, -0.05f, 0.05f);
    fin_cS2 = clamp(fin_cS2, -0.05f, 0.05f);

    /*fin_cS0 = clamp(fin_cS0, -1.f, 1.f);
    fin_cS1 = clamp(fin_cS1, -1.f, 1.f);
    fin_cS2 = clamp(fin_cS2, -1.f, 1.f);*/

    ///?
    if(fin_p_star < 1e-5 * p_star_max)
    {
        fin_e_star = min(fin_e_star, 10 * fin_p_star);
    }

    /*if(fin_p_star > 1)
    {
        fin_p_star = 1;
        fin_e_star = min(fin_e_star, 10 * fin_p_star);
    }*/

    ///this *does* seem to help
    /*if(X[index] < 0.1)
    {
        fin_p_star = 0;
        fin_e_star = 0;

        fin_cS0 = 0;
        fin_cS1 = 0;
        fin_cS2 = 0;
    }*/

    float area_half_width = scale * max(max(dim.x, dim.y), dim.z) / 2.f;

    float3 offset = transform_position(ix, iy, iz, dim, scale);

    float my_radius = fast_length(offset);

    float diss = 4;

    ///either interior to the black hole, or near the border. The latter is kind of hacky
    if(gA[index] < 0.45f || my_radius >= area_half_width * 0.85f)
    {
        fin_p_star += (0 - fin_p_star) * timestep * diss;
        fin_e_star += (0 - fin_e_star) * timestep * diss;

        fin_cS0 += (0 - fin_cS0) * timestep * diss;
        fin_cS1 += (0 - fin_cS1) * timestep * diss;
        fin_cS2 += (0 - fin_cS2) * timestep * diss;
    }

    fin_p_star = max(fin_p_star, 0.f);
    fin_e_star = max(fin_e_star, 0.f);

    oDp_star[index] = fin_p_star;
    oDe_star[index] = fin_e_star;

    oDcS0[index] = fin_cS0;
    oDcS1[index] = fin_cS1;
    oDcS2[index] = fin_cS2;

    NANCHECK(oDp_star);
    NANCHECK(oDe_star);
    NANCHECK(oDcS0);
    NANCHECK(oDcS1);
    NANCHECK(oDcS2);

    //if(ix == 97 && iy == 124 && iz == 124)
    /*if(ix == 94 && iy == 123 && iz == 125)
    {
        printf("McSigh p* %f e* %f cS0 %f cS1 %f cS2 %f lhs %f rhs %f fulldt %f\n", oDp_star[index], oDe_star[index], oDcS0[index], oDcS1[index], oDcS2[index], lhs_dtsk0, rhs_dtsk0, f_dtSk0);
    }*/


    /*if(ix == 98 && iy == 125 && iz == 125)
    {
        printf("Base? %f\n", base_e_star);

        printf("dg1 %.24f %f %f\n", DINTERIOR, DP1, DP2);

        printf("AVAL %f pq %f irhs %f p0eps %f rhsdte %f lhs %f estar %f %f %f\n", DBG_A, DBG_PQVIS, DBG_IRHS, DBG_p0eps, DBG_RHS_DTESTAR, DBG_LHS_DTESTAR, DBG_ESTARVI0, DBG_ESTARVI1, DBG_ESTARVI2);
    }*/
}

__kernel
void dissipate_single(__global ushort4* points, int point_count,
                      __global float* buffer, __global float* obuffer,
                      float coefficient,
                      float scale, int4 dim, float timestep, __global ushort* order_ptr)
{
    int local_idx = get_global_id(0);

    if(local_idx >= point_count)
        return;

    int ix = points[local_idx].x;
    int iy = points[local_idx].y;
    int iz = points[local_idx].z;

    int index = IDX(ix, iy, iz);
    int order = order_ptr[index];

    if((order & D_FULL) == 0)
        return;

    //#define VARIABLE_DAMP
    #ifdef VARIABLE_DAMP
    float3 offset = transform_position(ix, iy, iz, dim, scale);

    float radius = length(offset);

    float inner_radius = 9.f;
    float outer_radius = 15.f;

    float inner_diss = 1.f;
    float outer_diss = 2.f;

    float clamped = clamp(radius, inner_radius, outer_radius);

    float frac = (clamped - inner_radius) / (outer_radius - inner_radius);

    float damp = frac * (outer_diss - inner_diss) + inner_diss;

    //if(clamped < 14)
    //printf("Rad frac %f %f %i %i %i\n", clamped, damp, ix, iy, iz);
    #else
    float damp = 1;
    #endif

    /*float3 offset = transform_position(ix, iy, iz, dim, scale);

    float radius = length(offset);

    if(radius > 20)
        damp = 2.f;*/

    float TEMPORARIES9;

    float dissipate_single = KREISS_DISSIPATE_SINGULAR;

    float tol = 1e-6;

    if(dissipate_single <= tol && dissipate_single >= -tol)
        return;

    obuffer[index] += damp * dissipate_single * timestep;
}

__kernel
void dissipate_single_unidir(__global ushort4* points, int point_count,
                             __global float* buffer, __global float* obuffer,
                             float coefficient,
                             float scale, int4 dim, float timestep, __global ushort* order_ptr)
{
    int local_idx = get_global_id(0);

    if(local_idx >= point_count)
        return;

    int ix = points[local_idx].x;
    int iy = points[local_idx].y;
    int iz = points[local_idx].z;

    int index = IDX(ix, iy, iz);
    int order = order_ptr[index];

    if((order & D_FULL) == 0)
    {
        obuffer[index] = buffer[index];
        return;
    }

    float damp = 1;

    float TEMPORARIES9;

    float dissipate_single = KREISS_DISSIPATE_SINGULAR;

    obuffer[index] = buffer[index] + damp * dissipate_single * timestep;
}

__kernel
void render(STANDARD_ARGS(),
            __global DERIV_PRECISION* dcYij0, __global DERIV_PRECISION* dcYij1, __global DERIV_PRECISION* dcYij2, __global DERIV_PRECISION* dcYij3, __global DERIV_PRECISION* dcYij4, __global DERIV_PRECISION* dcYij5, __global DERIV_PRECISION* dcYij6, __global DERIV_PRECISION* dcYij7, __global DERIV_PRECISION* dcYij8, __global DERIV_PRECISION* dcYij9, __global DERIV_PRECISION* dcYij10, __global DERIV_PRECISION* dcYij11, __global DERIV_PRECISION* dcYij12, __global DERIV_PRECISION* dcYij13, __global DERIV_PRECISION* dcYij14, __global DERIV_PRECISION* dcYij15, __global DERIV_PRECISION* dcYij16, __global DERIV_PRECISION* dcYij17,
            __global DERIV_PRECISION* digA0, __global DERIV_PRECISION* digA1, __global DERIV_PRECISION* digA2,
            __global DERIV_PRECISION* digB0, __global DERIV_PRECISION* digB1, __global DERIV_PRECISION* digB2, __global DERIV_PRECISION* digB3, __global DERIV_PRECISION* digB4, __global DERIV_PRECISION* digB5, __global DERIV_PRECISION* digB6, __global DERIV_PRECISION* digB7, __global DERIV_PRECISION* digB8,
            __global DERIV_PRECISION* dX0, __global DERIV_PRECISION* dX1, __global DERIV_PRECISION* dX2,
            float scale, int4 dim, __write_only image2d_t screen)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    //int iz = dim.z/2;
    //int iy = (dim.y - 1)/2;
    //int iz = get_global_id(1);
    int iz = (dim.z - 1)/2;

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    if(ix <= 4 || ix >= dim.x - 5 || iy <= 4 || iy >= dim.y - 5 || iz <= 4 || iz >= dim.z - 5)
        return;

    float3 offset = transform_position(ix, iy, iz, dim, scale);

    int index_table[3][3] = {{0, 1, 2},
                             {1, 3, 4},
                             {2, 4, 5}};

    float3 max_scalar = 0;

    #ifdef RENDER_MATTER
    float matter = Dp_star[IDX(ix,iy,iz)];
    #else
    float matter = 0;
    #endif

    //for(int z = 20; z < dim.z-20; z++)

    {

        float sponge_factor = sponge_damp_coeff(ix, iy, iz, scale, dim);

        #define SOMMER_RENDER
        #ifndef SOMMER_RENDER
        if(sponge_factor > 0)
        #else
        if(sponge_factor == 1)
        #endif
        {
            float3 sponge_col = {sponge_factor, 0, 0};

            write_imagef(screen, (int2){get_global_id(0), get_global_id(1)}, (float4)(srgb_to_lin(sponge_col), 1));
            return;
        }

        int index = IDX(ix, iy, iz);

        #define RENDER_METRIC
        #ifdef RENDER_METRIC
        float Yxx = cY0[index];
        float Yxy = cY1[index];
        float Yxz = cY2[index];
        float Yyy = cY3[index];
        float Yyz = cY4[index];
        float cX = X[index];

        float Yzz = (1 + Yyy * Yxz * Yxz - 2 * Yxy * Yyz * Yxz + Yxx * Yyz * Yyz) / (Yxx * Yyy - Yxy * Yxy);

        float curvature = fabs(Yxx / cX) +
                          fabs(Yxy / cX) +
                          fabs(Yxz / cX) +
                          fabs(Yyy / cX) +
                          fabs(Yyz / cX) +
                          fabs(Yzz / cX);

        float ascalar = fabs(curvature / 1000.f);
        #endif // RENDER_METRIC

        //#define RENDER_CY
        #ifdef RENDER_CY
        float Yxx = cA0[index];
        float Yxy = cA1[index];
        float Yxz = cA2[index];
        float Yyy = cA3[index];
        float Yyz = cA4[index];
        float Yzz = cA5[index];

        float curvature = fabs(Yxx) +
                          fabs(Yxy) +
                          fabs(Yxz) +
                          fabs(Yyy) +
                          fabs(Yyz) +
                          fabs(Yzz);

        float ascalar = fabs(curvature / 1.f);
        #endif // RENDER_CY

        //#define RENDER_AIJ
        #ifdef RENDER_AIJ
        float Axx = cA0[index];
        float Axy = cA1[index];
        float Axz = cA2[index];
        float Ayy = cA3[index];
        float Ayz = cA4[index];
        float Azz = cA5[index];

        float curvature = fabs(Axx) +
                          fabs(Axy) +
                          fabs(Axz) +
                          fabs(Ayy) +
                          fabs(Ayz) +
                          fabs(Azz);

        float ascalar = fabs(curvature / 1.f);
        #endif // RENDER_AIJ

        //#define RENDER_K
        #ifdef RENDER_K
        float ascalar = fabs(K[index] * 40);
        #endif // RENDER_K

        //#define RENDER_X
        #ifdef RENDER_X
        float ascalar = fabs(X[index] / 50);
        #endif // RENDER_X

        //#define RENDER_CGI
        #ifdef RENDER_CGI
        float ascalar = fabs(cGi0[index]) + fabs(cGi1[index]) + fabs(cGi2[index]);
        #endif // RENDER_CGI

        //#define RENDER_GA
        #ifdef RENDER_GA
        float ascalar = fabs(gA[index] / 50);
        #endif // RENDER_GA

        //#define RENDER_GB
        #ifdef RENDER_GB
        float3 avec = (float3)(gB0[index], gB1[index], gB2[index]) * 4;

        avec = fabs(avec);
        #endif // RENDER_GB

        /*if(cX < 0.7)
            ascalar = 1;
        else
            ascalar = 0;*/

        //#define RENDER_DCY
        #ifdef RENDER_DCY
        float ascalar = fabs(dcYij0[index]) +
                        fabs(dcYij1[index]) +
                        fabs(dcYij2[index]) +
                        fabs(dcYij3[index]) +
                        fabs(dcYij4[index]) +
                        fabs(dcYij5[index]) +
                        fabs(dcYij6[index]) +
                        fabs(dcYij7[index]) +
                        fabs(dcYij8[index]) +
                        fabs(dcYij9[index]) +
                        fabs(dcYij10[index]) +
                        fabs(dcYij11[index]) +
                        fabs(dcYij12[index]) +
                        fabs(dcYij13[index]) +
                        fabs(dcYij14[index]) +
                        fabs(dcYij15[index]) +
                        fabs(dcYij16[index]) +
                        fabs(dcYij17[index]);

        ascalar *= 0.2f;
        #endif // RENDER_DCY

        #ifndef RENDER_GB
        max_scalar = max(ascalar, max_scalar);
        #else
        max_scalar = max(avec, max_scalar);
        #endif
    }

    float real = 0;

    #define RENDER_WAVES
    #ifdef RENDER_WAVES
    {
        float TEMPORARIES4;

        real = w4_real;

        real = fabs(real) * 1000.f;

        real = clamp(real, 0.f, 1.f);
    }
    #endif // RENDER_WAVES

    /*if(ix == 125 && iy == 125)
    {
        printf("scalar %f\n", max_scalar);
    }*/

    matter = clamp(matter * 10, 0.f, 1.f);

    max_scalar = clamp(max_scalar * 40, 0.f, 1.f);

    float3 col = {matter + real, matter, matter};

    col += max_scalar;

    col = clamp(col, 0.f, 1.f);

    float3 lin_col = srgb_to_lin(col);

    write_imagef(screen, (int2){get_global_id(0), get_global_id(1)}, (float4)(lin_col.xyz, 1));
    //write_imagef(screen, (int2){ix, iy}, (float4){max_sca1lar, max_scalar, max_scalar, 1});
}

#if 1
__kernel
void extract_waveform(__global ushort4* points, int point_count,
                      STANDARD_ARGS(),
                      __global DERIV_PRECISION* dcYij0, __global DERIV_PRECISION* dcYij1, __global DERIV_PRECISION* dcYij2, __global DERIV_PRECISION* dcYij3, __global DERIV_PRECISION* dcYij4, __global DERIV_PRECISION* dcYij5, __global DERIV_PRECISION* dcYij6, __global DERIV_PRECISION* dcYij7, __global DERIV_PRECISION* dcYij8, __global DERIV_PRECISION* dcYij9, __global DERIV_PRECISION* dcYij10, __global DERIV_PRECISION* dcYij11, __global DERIV_PRECISION* dcYij12, __global DERIV_PRECISION* dcYij13, __global DERIV_PRECISION* dcYij14, __global DERIV_PRECISION* dcYij15, __global DERIV_PRECISION* dcYij16, __global DERIV_PRECISION* dcYij17,
                      __global DERIV_PRECISION* digA0, __global DERIV_PRECISION* digA1, __global DERIV_PRECISION* digA2,
                      __global DERIV_PRECISION* digB0, __global DERIV_PRECISION* digB1, __global DERIV_PRECISION* digB2, __global DERIV_PRECISION* digB3, __global DERIV_PRECISION* digB4, __global DERIV_PRECISION* digB5, __global DERIV_PRECISION* digB6, __global DERIV_PRECISION* digB7, __global DERIV_PRECISION* digB8,
                      __global DERIV_PRECISION* dX0, __global DERIV_PRECISION* dX1, __global DERIV_PRECISION* dX2,
                      float scale, int4 dim, __global float2* waveform_out, __write_only image2d_t screen)
{
    int local_idx = get_global_id(0);

    if(local_idx >= point_count)
        return;

    int ix = points[local_idx].x;
    int iy = points[local_idx].y;
    int iz = points[local_idx].z;

    float3 offset = transform_position(ix, iy, iz, dim, scale);

    float TEMPORARIES4;

    waveform_out[local_idx].x = w4_real;
    waveform_out[local_idx].y = w4_complex;

    float zfrac = (float)iz / dim.z;

    #ifdef WAVE_DBG1
    float dbg_val = WAVE_DBG1;

    printf("Dval %f\n", dbg_val);
    #endif // WAVE_DBG1

    if(iz == (dim.z-1)/2)
        write_imagef(screen, (int2){ix, iy}, (float4)(zfrac, zfrac, zfrac, 1));

    #ifdef w4_debugr
    printf("Debugw4r %f\n", w4_debugr);
    #endif // w4_debug

    #ifdef w4_debugi
    printf("Debugw4i %f\n", w4_debugi);
    #endif // w4_debug
}
#endif // 0

/*struct lightray
{
    float3 x;
    float3 V;
    float T;
};

__kernel
void init_rays(__global float* cY0, __global float* cY1, __global float* cY2, __global float* cY3, __global float* cY4, __global float* cY5,
               __global float* cA0, __global float* cA1, __global float* cA2, __global float* cA3, __global float* cA4, __global float* cA5,
               __global float* cGi0, __global float* cGi1, __global float* cGi2, __global float* K, __global float* X, __global float* gA, __global float* gB0, __global float* gB1, __global float* gB2,
            float scale, __global struct intermediate_bssnok_data* temp_in, __global struct lightray* rays, float3 camera_pos, float4 camera_quat,
            float width, float height, int4 dim)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x >= width)
        return;

    if(y >= height)
        return;

    ///ray location

    float3 pos = camera_pos - (float3){dim.x, dim.y, dim.z}/2.f;

    pos = clamp(pos, (float3)(0,0,0), (float3)(dim.x, dim.y, dim.z) - 1);

    ///temporary while i don't do interpolation
    float3 fipos = round(pos);

    int ix = fipos.x;
    int iy = fipos.y;
    int iz = fipos.z;
}*/

struct lightray_simple
{
    float lp1;
    float lp2;
    float lp3;

    float V0;
    float V1;
    float V2;

    int x, y;

    float iter_frac;
    int hit_type;
    float density;
};

enum ds_result
{
    DS_NONE,
    DS_SKIP,
    DS_RETURN,
};

int calculate_ds_error(float err, float current_ds, float3 next_acceleration, float* next_ds_out)
{
    #define MIN_STEP 0.5f
    #define MAX_STEP 2.f

    float next_ds = err * 1/fast_length(next_acceleration);

    ///produces strictly worse results for kerr
    //next_ds = 0.99f * current_ds * clamp(next_ds / current_ds, 0.1f, 4.f);

    next_ds = clamp(next_ds, MIN_STEP, MAX_STEP);

    *next_ds_out = next_ds;

    //if(next_ds == MIN_STEP)
    //    return DS_RETURN;

    if(next_ds < current_ds/1.2f)
        return DS_SKIP;

    return DS_NONE;
}

///https://www.ccs.neu.edu/home/fell/CS4300/Lectures/Ray-TracingFormulas.pdf
float3 fix_ray_position(float3 cartesian_pos, float3 cartesian_velocity, float sphere_radius)
{
    cartesian_velocity = fast_normalize(cartesian_velocity);

    float3 C = (float3){0,0,0};

    float a = 1;
    float b = 2 * dot(cartesian_velocity, (cartesian_pos - C));
    float c = dot(C, C) + dot(cartesian_pos, cartesian_pos) - 2 * (dot(cartesian_pos, C)) - sphere_radius * sphere_radius;

    float discrim = b*b - 4 * a * c;

    if(discrim < 0)
        return cartesian_pos;

    float t0 = (-b - native_sqrt(discrim)) / (2 * a);
    float t1 = (-b + native_sqrt(discrim)) / (2 * a);

    float my_t = 0;

    if(fabs(t0) < fabs(t1))
        my_t = t0;
    else
        my_t = t1;

    return cartesian_pos + my_t * cartesian_velocity;
}

///this returns the change in X, which is not velocity
///its unfortunate that position, aka X, and the conformal factor are called the same thing here
///the reason why these functions use out parameters is to work around a significant optimisation failure in AMD's opencl compiler
void velocity_to_XDiff(float3* out, float3 Xpos, float3 vel, float scale, int4 dim, STANDARD_ARGS(), STANDARD_DERIVS())
{
    float3 voxel_pos = world_to_voxel(Xpos, dim, scale);

    ///isn't this already handled internally?
    voxel_pos = clamp(voxel_pos, (float3)(BORDER_WIDTH,BORDER_WIDTH,BORDER_WIDTH), (float3)(dim.x, dim.y, dim.z) - BORDER_WIDTH - 1);

    float fx = voxel_pos.x;
    float fy = voxel_pos.y;
    float fz = voxel_pos.z;

    float V0 = vel.x;
    float V1 = vel.y;
    float V2 = vel.z;

    float TEMPORARIES6;

    float d0 = X0Diff;
    float d1 = X1Diff;
    float d2 = X2Diff;

    *out = (float3){d0, d1, d2};
}

void calculate_V_derivatives(float3* out, float3 Xpos, float3 vel, float scale, int4 dim, STANDARD_ARGS(), STANDARD_DERIVS())
{
    float3 voxel_pos = world_to_voxel(Xpos, dim, scale);

    ///isn't this already handled internally?
    voxel_pos = clamp(voxel_pos, (float3)(BORDER_WIDTH,BORDER_WIDTH,BORDER_WIDTH), (float3)(dim.x, dim.y, dim.z) - BORDER_WIDTH - 1);

    float fx = voxel_pos.x;
    float fy = voxel_pos.y;
    float fz = voxel_pos.z;

    float V0 = vel.x;
    float V1 = vel.y;
    float V2 = vel.z;

    float TEMPORARIES6;

    float d0 = V0Diff;
    float d1 = V1Diff;
    float d2 = V2Diff;

    *out = (float3){d0, d1, d2};
}

__kernel
void calculate_adm_texture_coordinates(__global struct lightray_simple* finished_rays, __global float2* texture_coordinates, int width, int height,
                                       float3 camera_pos, float4 camera_quat,
                                       STANDARD_ARGS(), STANDARD_DERIVS(), float scale, int4 dim)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x >= width || y >= height)
        return;

    __global struct lightray_simple* ray = &finished_rays[y * width + x];

    float3 cpos = {ray->lp1, ray->lp2, ray->lp3};
    float3 cvel = {ray->V0, ray->V1, ray->V2};

    float3 XDiff;
    velocity_to_XDiff(&XDiff, cpos, cvel, scale, dim, ALL_ARGS());

    float uni_size = universe_size;

    cpos = fix_ray_position(cpos, XDiff, uni_size * RENDERING_CUTOFF_MULT);

    float fr = fast_length(cpos);
    float theta = acos(cpos.z / fr);
    float phi = atan2(cpos.y, cpos.x);

    float3 npolar = (float3)(fr, theta, phi);

    float thetaf = fmod(npolar.y, 2 * M_PI);
    float phif = npolar.z;

    if(thetaf >= M_PI)
    {
        phif += M_PI;
        thetaf -= M_PI;
    }

    phif = fmod(phif, 2 * M_PI);

    float sxf = (phif) / (2 * M_PI);
    float syf = thetaf / M_PI;

    sxf += 0.5f;

    texture_coordinates[y * width + x] = (float2)(sxf, syf);
}

__kernel
void init_rays(__global struct lightray_simple* rays, __global int* ray_count0,
                STANDARD_ARGS(),
                __global DERIV_PRECISION* dcYij0, __global DERIV_PRECISION* dcYij1, __global DERIV_PRECISION* dcYij2, __global DERIV_PRECISION* dcYij3, __global DERIV_PRECISION* dcYij4, __global DERIV_PRECISION* dcYij5, __global DERIV_PRECISION* dcYij6, __global DERIV_PRECISION* dcYij7, __global DERIV_PRECISION* dcYij8, __global DERIV_PRECISION* dcYij9, __global DERIV_PRECISION* dcYij10, __global DERIV_PRECISION* dcYij11, __global DERIV_PRECISION* dcYij12, __global DERIV_PRECISION* dcYij13, __global DERIV_PRECISION* dcYij14, __global DERIV_PRECISION* dcYij15, __global DERIV_PRECISION* dcYij16, __global DERIV_PRECISION* dcYij17,
                __global DERIV_PRECISION* digA0, __global DERIV_PRECISION* digA1, __global DERIV_PRECISION* digA2,
                __global DERIV_PRECISION* digB0, __global DERIV_PRECISION* digB1, __global DERIV_PRECISION* digB2, __global DERIV_PRECISION* digB3, __global DERIV_PRECISION* digB4, __global DERIV_PRECISION* digB5, __global DERIV_PRECISION* digB6, __global DERIV_PRECISION* digB7, __global DERIV_PRECISION* digB8,
                __global DERIV_PRECISION* dX0, __global DERIV_PRECISION* dX1, __global DERIV_PRECISION* dX2,
                float scale, float3 camera_pos, float4 camera_quat,
                int4 dim, int width, int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x >= width)
        return;

    if(y >= height)
        return;

    float lp0;
    float lp1;
    float lp2;
    float lp3;

    float V0;
    float V1;
    float V2;

    {
        float3 world_pos = camera_pos;

        float3 voxel_pos = world_to_voxel(world_pos, dim, scale);

        float fx = voxel_pos.x;
        float fy = voxel_pos.y;
        float fz = voxel_pos.z;

        float TEMPORARIES5;

        lp0 = lp0_d;
        lp1 = lp1_d;
        lp2 = lp2_d;
        lp3 = lp3_d;

        V0 = V0_d;
        V1 = V1_d;
        V2 = V2_d;
    }

    struct lightray_simple out;
    out.lp1 = lp1;
    out.lp2 = lp2;
    out.lp3 = lp3;

    out.V0 = V0;
    out.V1 = V1;
    out.V2 = V2;

    out.x = x;
    out.y = y;
    out.iter_frac = 0;
    out.hit_type = 0;

    rays[y * width + x] = out;

    if(x == 0 && y == 0)
        *ray_count0 = width * height;
}

float length_sq(float3 in)
{
    return dot(in, in);
}

float get_static_verlet_ds(float3 Xpos, __global float* X, float scale, int4 dim)
{
    float X_far = 0.9f;
    float X_near = 0.6f;

    float3 voxel_pos = world_to_voxel(Xpos, dim, scale);
    voxel_pos = clamp(voxel_pos, (float3)(BORDER_WIDTH,BORDER_WIDTH,BORDER_WIDTH), (float3)(dim.x, dim.y, dim.z) - BORDER_WIDTH - 1);

    float BH_X = buffer_read_linear(X, voxel_pos, dim);

    float my_fraction = (clamp(BH_X, X_near, X_far) - X_near) / (X_far - X_near);

    my_fraction = clamp(my_fraction, 0.f, 1.f);

    #ifdef RENDER_MATTER
    return mix(0.4f, 4.f, my_fraction) * 0.1f;
    #else
    return mix(0.4f, 4.f, my_fraction);
    #endif
}

#define SOLID_DENSITY 0.1

__kernel
void trace_rays(__global struct lightray_simple* rays_in, __global struct lightray_simple* rays_terminated,
                STANDARD_ARGS(),
                STANDARD_DERIVS(),
                float scale, int4 dim, int width, int height, float err_in)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x >= width || y >= height)
        return;

    struct lightray_simple ray_in = rays_in[y * width + x];

    float3 Xpos = {ray_in.lp1, ray_in.lp2, ray_in.lp3};
    float3 vel = {ray_in.V0, ray_in.V1, ray_in.V2};
    float3 Xpos_last = Xpos;

    int hit_type = 1;

    float u_sq = (universe_size * RENDERING_CUTOFF_MULT) * (universe_size * RENDERING_CUTOFF_MULT);

    float3 VHalf = (float3)(0,0,0);
    float3 VFull_approx = (float3)(0,0,0);

    float ds = 0;

    ///so: this performs the first two iterations of verlet early
    ///this means that the main verlet loop does not contain separate memory reads, resulting in a 40ms -> 28ms speedup due to
    ///optimisation
    #define VERLET_2
    #ifdef VERLET_2
    {
        ds = get_static_verlet_ds(Xpos, X, scale, dim);

        float3 ABase;
        calculate_V_derivatives(&ABase, Xpos, vel, scale, dim, ALL_ARGS());

        VHalf = vel + 0.5f * ABase * ds;

        VFull_approx = vel + ABase * ds;

        float3 XDiff;
        velocity_to_XDiff(&XDiff, Xpos, VHalf, scale, dim, ALL_ARGS());

        float3 XFull = Xpos + XDiff * ds;

        Xpos = XFull;
    }
    #endif // VERLET_2

    float density = 0;

    //#pragma unroll(16)
    for(int iteration=0; iteration < 512; iteration++)
    {
        #ifdef VERLET_2
        ///finish previous iteration
        {
            float3 AFull_approx;
            calculate_V_derivatives(&AFull_approx, Xpos, VFull_approx, scale, dim, ALL_ARGS());

            float3 VFull = VHalf + 0.5f * AFull_approx * ds;

            vel = VFull;
        }

        ///next iteration
        ds = get_static_verlet_ds(Xpos, X, scale, dim);

        float3 XDiff;

        {
            float3 ABase;
            calculate_V_derivatives(&ABase, Xpos, vel, scale, dim, ALL_ARGS());

            VHalf = vel + 0.5f * ABase * ds;

            VFull_approx = vel + ABase * ds;

            velocity_to_XDiff(&XDiff, Xpos, VHalf, scale, dim, ALL_ARGS());

            float3 XFull = Xpos + XDiff * ds;

            Xpos_last = Xpos;
            Xpos = XFull;
        }

        if(length_sq(Xpos) >= u_sq)
        {
            hit_type = 0;
            break;
        }

        #ifdef RENDER_MATTER
        float3 voxel_pos = world_to_voxel(Xpos, dim, scale);
        voxel_pos = clamp(voxel_pos, (float3)(BORDER_WIDTH,BORDER_WIDTH,BORDER_WIDTH), (float3)(dim.x, dim.y, dim.z) - BORDER_WIDTH - 1);

        float pstar_val = buffer_read_linear(Dp_star, voxel_pos, dim);

        /*if(pstar_val > 0.001f)
        {
            Xpos_last = Xpos;
            hit_type = 2;
            break;
        }*/

        density += pstar_val;

        if(density > SOLID_DENSITY)
        {
            Xpos_last = Xpos;
            hit_type = 2;
            break;
        }

        #endif // RENDER_MATTER

        #endif // VERLET_2

        //#define EULER
        #ifdef EULER
        float ds = mix(0.1f, 2.f, my_fraction);

        float3 accel;
        calculate_V_derivatives(&accel, Xpos, vel, scale, dim, ALL_ARGS());

        ///uncomment the accel*ds to get symplectic euler
        float3 XDiff;
        velocity_to_XDiff(&XDiff, Xpos, vel /*+ accel * ds*/, scale, dim, ALL_ARGS());

        Xpos += XDiff * ds;

        if(length_sq(Xpos) >= u_sq)
        {
            break;
        }

        vel += accel * ds;
        #endif // EULER

        /*if(x == (int)width/2 && y == (int)height/2)
        {
            printf("%f %f %f  %f %f %f\n", V0, V1, V2, lp1, lp2, lp3);
        }*/

        if(length_sq(XDiff) < 0.2f * 0.2f)
        {
            hit_type = 1;
            break;
        }
    }

    struct lightray_simple ray_out;
    ray_out.x = x;
    ray_out.y = y;

    ray_out.lp1 = Xpos_last.x;
    ray_out.lp2 = Xpos_last.y;
    ray_out.lp3 = Xpos_last.z;

    ray_out.V0 = vel.x;
    ray_out.V1 = vel.y;
    ray_out.V2 = vel.z;

    ray_out.iter_frac = 0;
    ray_out.hit_type = hit_type;
    ray_out.density = density;

    rays_terminated[y * width + x] = ray_out;
}

float4 read_mipmap(image2d_t mipmap1, sampler_t sam, float2 pos, float lod)
{
    return read_imagef(mipmap1, sam, pos, lod);
}

float circular_diff(float f1, float f2)
{
    float a1 = f1 * M_PI * 2;
    float a2 = f2 * M_PI * 2;

    float2 v1 = {cos(a1), sin(a1)};
    float2 v2 = {cos(a2), sin(a2)};

    return atan2(v1.x * v2.y - v1.y * v2.x, v1.x * v2.x + v1.y * v2.y) / (2 * M_PI);
}

float2 circular_diff2(float2 f1, float2 f2)
{
    return (float2)(circular_diff(f1.x, f2.x), circular_diff(f1.y, f2.y));
}

#define MIPMAP_CONDITIONAL(x) (x(mip_background))

__kernel void render_rays(__global struct lightray_simple* rays_in, __global int* ray_count, __write_only image2d_t screen,
                          STANDARD_ARGS(),
                          STANDARD_DERIVS(),
                          float scale, int4 dim, int width, int height,
                          __read_only image2d_t mip_background,
                          __global float2* texture_coordinates, sampler_t sam, float3 camera_pos)
{
    int idx = get_global_id(0);

    if(idx >= width * height)
        return;

    struct lightray_simple ray_in = rays_in[idx];

    float lp1 = ray_in.lp1;
    float lp2 = ray_in.lp2;
    float lp3 = ray_in.lp3;

    float V0 = ray_in.V0;
    float V1 = ray_in.V1;
    float V2 = ray_in.V2;

    int x = ray_in.x;
    int y = ray_in.y;

    float3 cpos = {lp1, lp2, lp3};
    float3 cvel = {V0, V1, V2};

    float3 XDiff;
    velocity_to_XDiff(&XDiff, cpos, cvel, scale, dim, ALL_ARGS());

    float density_frac = clamp(ray_in.density / SOLID_DENSITY, 0.f, 1.f);

    float3 density_col = (float3)(1,1,1) * density_frac;

    float uni_size = universe_size;

    if(ray_in.hit_type == 0)
    {
        cpos = fix_ray_position(cpos, XDiff, uni_size * RENDERING_CUTOFF_MULT);

        float sxf = texture_coordinates[y * width + x].x;
        float syf = texture_coordinates[y * width + x].y;

        #if 0
        float fr = fast_length(cpos);
        float theta = acos(cpos.z / fr);
        float phi = atan2(cpos.y, cpos.x);

        float sxf = (phi + M_PI) / (2 * M_PI);
        float syf = theta / M_PI;

        float4 val = (float4)(0,0,0,1);

        int x_half = fabs(fmod((sxf + 1) * 10.f, 1.f)) > 0.5 ? 1 : 0;
        int y_half = fabs(fmod((syf + 1) * 10.f, 1.f)) > 0.5 ? 1 : 0;

        val.x = x_half;
        val.y = y_half;

        if(syf < 0.1 || syf >= 0.9)
        {
            val.x = 0;
            val.y = 0;
            val.z = 1;
        }

        //val.xyz = clamp(ray_in.iter_frac, 0.f, 1.f);

        write_imagef(screen, (int2){x, y}, val);
        #endif // 0

        #define MIPMAPPING
        #ifdef MIPMAPPING
        int dx = 1;
        int dy = 1;

        if(x == width-1)
            dx = -1;

        if(y == height-1)
            dy = -1;

        float2 tl = texture_coordinates[y * width + x];
        float2 tr = texture_coordinates[y * width + x + dx];
        float2 bl = texture_coordinates[(y + dy) * width + x];

        ///higher = sharper
        float bias_frac = 1.3;

        //TL x 0.435143 TR 0.434950 TD -0.000149, aka (tr.x - tl.x) / 1.3
        float2 dx_vtc = circular_diff2(tl, tr) / bias_frac;
        float2 dy_vtc = circular_diff2(tl, bl) / bias_frac;

        if(dx == -1)
        {
            dx_vtc = -dx_vtc;
        }

        if(dy == -1)
        {
            dy_vtc = -dy_vtc;
        }

        //#define TRILINEAR
        #ifdef TRILINEAR
        dx_vtc.x *= MIPMAP_CONDITIONAL(get_image_width);
        dy_vtc.x *= MIPMAP_CONDITIONAL(get_image_width);

        dx_vtc.y *= MIPMAP_CONDITIONAL(get_image_height);
        dy_vtc.y *= MIPMAP_CONDITIONAL(get_image_height);

        //dx_vtc.x /= 10.f;
        //dy_vtc.x /= 10.f;

        dx_vtc /= 2.f;
        dy_vtc /= 2.f;

        float delta_max_sqr = max(dot(dx_vtc, dx_vtc), dot(dy_vtc, dy_vtc));

        float mip_level = 0.5 * log2(delta_max_sqr);

        //mip_level -= 0.5;

        float mip_clamped = clamp(mip_level, 0.f, 5.f);

        float4 end_result = MIPMAP_CONDITIONAL_READ(read_imagef, sam, ((float2){sxf, syf}), mip_clamped);
        #else

        dx_vtc.x *= MIPMAP_CONDITIONAL(get_image_width);
        dy_vtc.x *= MIPMAP_CONDITIONAL(get_image_width);

        dx_vtc.y *= MIPMAP_CONDITIONAL(get_image_height);
        dy_vtc.y *= MIPMAP_CONDITIONAL(get_image_height);

        ///http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1002.1336&rep=rep1&type=pdf
        float dv_dx = dx_vtc.y;
        float dv_dy = dy_vtc.y;

        float du_dx = dx_vtc.x;
        float du_dy = dy_vtc.x;

        float Ann = dv_dx * dv_dx + dv_dy * dv_dy;
        float Bnn = -2 * (du_dx * dv_dx + du_dy * dv_dy);
        float Cnn = du_dx * du_dx + du_dy * du_dy; ///only tells lies

        ///hecc
        #define HECKBERT
        #ifdef HECKBERT
        Ann = dv_dx * dv_dx + dv_dy * dv_dy + 1;
        Cnn = du_dx * du_dx + du_dy * du_dy + 1;
        #endif // HECKBERT

        float F = Ann * Cnn - Bnn * Bnn / 4;
        float A = Ann / F;
        float B = Bnn / F;
        float C = Cnn / F;

        float root = sqrt((A - C) * (A - C) + B*B);
        float a_prime = (A + C - root) / 2;
        float c_prime = (A + C + root) / 2;

        float majorRadius = native_rsqrt(a_prime);
        float minorRadius = native_rsqrt(c_prime);

        float theta = atan2(B, (A - C)/2);

        majorRadius = max(majorRadius, 1.f);
        minorRadius = max(minorRadius, 1.f);

        majorRadius = max(majorRadius, minorRadius);

        float fProbes = 2 * (majorRadius / minorRadius) - 1;
        int iProbes = floor(fProbes + 0.5f);

        int maxProbes = 8;

        iProbes = min(iProbes, maxProbes);

        if(iProbes < fProbes)
            minorRadius = 2 * majorRadius / (iProbes + 1);

        float levelofdetail = log2(minorRadius);

        int maxLod = MIPMAP_CONDITIONAL(get_image_num_mip_levels) - 1;

        if(levelofdetail > maxLod)
        {
            levelofdetail = maxLod;
            iProbes = 1;
        }

        float4 end_result = 0;

        if(iProbes == 1 || iProbes <= 1)
        {
            if(iProbes < 1)
                levelofdetail = maxLod;

            end_result = read_mipmap(mip_background, sam, (float2){sxf, syf}, levelofdetail);
        }
        else
        {
            float lineLength = 2 * (majorRadius - minorRadius);
            float du = cos(theta) * lineLength / (iProbes - 1);
            float dv = sin(theta) * lineLength / (iProbes - 1);

            float4 totalWeight = 0;
            float accumulatedProbes = 0;

            int startN = 0;

            ///odd probes
            if((iProbes % 2) == 1)
            {
                int probeArm = (iProbes - 1) / 2;

                startN = -2 * probeArm;
            }
            else
            {
                int probeArm = (iProbes / 2);

                startN = -2 * probeArm - 1;
            }

            int currentN = startN;
            float alpha = 2;

            float sU = du / MIPMAP_CONDITIONAL(get_image_width);
            float sV = dv / MIPMAP_CONDITIONAL(get_image_height);

            for(int cnt = 0; cnt < iProbes; cnt++)
            {
                float d_2 = (currentN * currentN / 4.f) * (du * du + dv * dv) / (majorRadius * majorRadius);

                ///not a performance issue
                float relativeWeight = native_exp(-alpha * d_2);

                float centreu = sxf;
                float centrev = syf;

                float cu = centreu + (currentN / 2.f) * sU;
                float cv = centrev + (currentN / 2.f) * sV;

                float4 fval = read_mipmap(mip_background, sam, (float2){cu, cv}, levelofdetail);

                totalWeight += relativeWeight * fval;
                accumulatedProbes += relativeWeight;

                currentN += 2;
            }

            end_result = totalWeight / accumulatedProbes;
        }

        #endif // TRILINEAR
        #endif // MIPMAPPING

        float3 with_density = clamp(srgb_to_lin(end_result.xyz) + density_col, 0.f, 1.f);

        write_imagef(screen, (int2){x, y}, (float4)(with_density, 1.f));
    }
    else if(ray_in.hit_type == 1)
    {
        float3 val = (float3)(0,0,0);

        val = density_col;

        //val.xyz = clamp(ray_in.iter_frac, 0.f, 1.f);

        write_imagef(screen, (int2){x, y}, (float4)(val.xyz,1));
    }
    else if(ray_in.hit_type == 2)
    {
        float3 val = (float3)(0.5f,0.5f,0.5f);

        float3 Xpos = (float3)(ray_in.lp1, ray_in.lp2, ray_in.lp3);

        float3 voxel_pos = world_to_voxel(Xpos, dim, scale);
        voxel_pos = clamp(voxel_pos, (float3)(BORDER_WIDTH,BORDER_WIDTH,BORDER_WIDTH), (float3)(dim.x, dim.y, dim.z) - BORDER_WIDTH - 1);

        float3 px = (float3)(1, 0, 0);
        float3 nx = (float3)(-1, 0, 0);
        float3 py = (float3)(0, 1, 0);
        float3 ny = (float3)(0, -1, 0);
        float3 pz = (float3)(0, 0, 1);
        float3 nz = (float3)(0, 0, -1);

        float pxv = buffer_read_linear(Dp_star, voxel_pos + px, dim);
        float nxv = buffer_read_linear(Dp_star, voxel_pos + nx, dim);
        float pyv = buffer_read_linear(Dp_star, voxel_pos + py, dim);
        float nyv = buffer_read_linear(Dp_star, voxel_pos + ny, dim);
        float pzv = buffer_read_linear(Dp_star, voxel_pos + pz, dim);
        float nzv = buffer_read_linear(Dp_star, voxel_pos + nz, dim);

        float3 normal = {pxv - nxv, pyv - nyv, pzv - nzv};

        normal = normalize(normal);

        //float pstar_val = buffer_read_linear(Dp_star, voxel_pos, dim);

        float3 to_point = camera_pos - Xpos;

        //printf("Normal %f %f %f\n", normal.x, normal.y, normal.z);

        //printf("Cam %f %f %f\n", camera_voxel_pos.x, camera_voxel_pos.y, camera_voxel_pos.z);

        float light = dot(normalize(to_point), normal);

        light = clamp(fabs(light), 0.f, 1.f);

        float3 col = (float3){1,1,1} * light + density_col;

        col = clamp(col, 0.f, 1.f);

        //col = fabs(normal);

        write_imagef(screen, (int2){x, y}, (float4)(col.xyz, 1));
    }
}

#if 0
struct lightray
{
    float4 pos;
    float4 vel;
    int x, y;
};

///todo: unify this with the above
///the memory overhead is extremely minimal for a huge performance boost
__kernel
void init_accurate_rays(STANDARD_ARGS(),
                        float scale, float3 camera_pos, float4 camera_quat,
                        int4 dim, __write_only image2d_t screen,
                        __global struct lightray* ray)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x >= get_image_width(screen))
        return;

    if(y >= get_image_height(screen))
        return;

    float width = get_image_width(screen);
    float height = get_image_height(screen);

    ///ray location

    float3 pos = camera_pos;

    pos = clamp(pos, (float3)(BORDER_WIDTH,BORDER_WIDTH,BORDER_WIDTH), (float3)(dim.x, dim.y, dim.z) - BORDER_WIDTH - 1);

    ///temporary while i don't do interpolation
    float lp0;
    float lp1;
    float lp2;
    float lp3;

    float V0;
    float V1;
    float V2;

    {
        float3 world_pos = camera_pos;

        float3 voxel_pos = world_to_voxel(world_pos, dim, scale);

        float fx = voxel_pos.x;
        float fy = voxel_pos.y;
        float fz = voxel_pos.z;

        float TEMPORARIES5;

        lp0 = lp0_d;
        lp1 = lp1_d;
        lp2 = lp2_d;
        lp3 = lp3_d;

        V0 = V0_d;
        V1 = V1_d;
        V2 = V2_d;
    }

    int ray_idx = y * (int)width + x;

    ray[ray_idx].pos = (float4){lp0, lp1, lp2, lp3};
    ray[ray_idx].vel = (float4){V0, V1, V2, 0};
    ray[ray_idx].x = x;
    ray[ray_idx].y = y;
}

__kernel
void step_accurate_rays(STANDARD_ARGS(),
                        float scale, float3 camera_pos, float4 camera_quat,
                        int4 dim, __write_only image2d_t screen,
                        __global struct lightray* ray, float timestep)
{
    float width = get_image_width(screen);
    float height = get_image_height(screen);

    int ray_idx = get_global_id(0);
    int ray_count = (int)width * (int)height;

    if(ray_idx >= ray_count)
        return;

    float lp1 = ray[ray_idx].pos.y;
    float lp2 = ray[ray_idx].pos.z;
    float lp3 = ray[ray_idx].pos.w;

    int x = ray[ray_idx].x;
    int y = ray[ray_idx].y;

    float3 cpos = {lp1, lp2, lp3};

    float3 voxel_pos = world_to_voxel(cpos, dim, scale);

    voxel_pos = clamp(voxel_pos, (float3)(BORDER_WIDTH,BORDER_WIDTH,BORDER_WIDTH), (float3)(dim.x, dim.y, dim.z) - BORDER_WIDTH - 1);

    float fx = voxel_pos.x;
    float fy = voxel_pos.y;
    float fz = voxel_pos.z;

    float TEMPORARIES6;

    float terminate_length = fast_length(cpos);

    if(terminate_length >= universe_size / 5.01f)
    {
        float fr = fast_length(cpos);
        float theta = acos(cpos.z / fr);
        float phi = atan2(cpos.y, cpos.x);

        float sxf = (phi + M_PI) / (2 * M_PI);
        float syf = theta / M_PI;

        float4 val = (float4)(0,0,0,1);

        int x_half = fabs(fmod((sxf + 1) * 10.f, 1.f)) > 0.5 ? 1 : 0;
        int y_half = fabs(fmod((syf + 1) * 10.f, 1.f)) > 0.5 ? 1 : 0;

        val.x = x_half;
        val.y = y_half;

        if(syf < 0.1 || syf >= 0.9)
        {
            val.x = 0;
            val.y = 0;
            val.z = 1;
        }

        write_imagef(screen, (int2){x, y}, val);
        return;
    }

    float V0 = ray[ray_idx].vel.x;
    float V1 = ray[ray_idx].vel.y;
    float V2 = ray[ray_idx].vel.z;

    float dX0 = X0Diff;
    float dX1 = X1Diff;
    float dX2 = X2Diff;

    float dV0 = V0Diff;
    float dV1 = V1Diff;
    float dV2 = V2Diff;

    V0 += dV0 * timestep;
    V1 += dV1 * timestep;
    V2 += dV2 * timestep;

    lp1 += dX0 * timestep;
    lp2 += dX1 * timestep;
    lp3 += dX2 * timestep;

    //if(x == (int)width/2 && y == (int)height/2)
    //printf("Pos %f %f %f\n", lp1, lp2, lp3);

    ray[ray_idx].vel.xyz = (float3)(V0, V1, V2);
    ray[ray_idx].pos.yzw = (float3)(lp1, lp2, lp3);

    //if(fast_length((float3){dX0, dX1, dX2}) < 0.01f)
    {
        write_imagef(screen, (int2){x, y}, (float4)(0, 0, 0, 1));
    }
}
#endif // 0

float3 rot_quat(const float3 point, float4 quat)
{
    quat = fast_normalize(quat);

    float3 t = 2.f * cross(quat.xyz, point);

    return point + quat.w * t + cross(quat.xyz, t);
}

__kernel
void trace_metric(STANDARD_ARGS(),
                  float scale, float3 camera_pos, float4 camera_quat,
                  int4 dim, __write_only image2d_t screen)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x >= get_image_width(screen))
        return;

    if(y >= get_image_height(screen))
        return;

    float width = get_image_width(screen);
    float height = get_image_height(screen);

    ///ray location
    float3 pos = world_to_voxel(camera_pos, dim, scale);

    pos = clamp(pos, (float3)(BORDER_WIDTH,BORDER_WIDTH,BORDER_WIDTH), (float3)(dim.x, dim.y, dim.z) - BORDER_WIDTH - 1);

    ///temporary while i don't do interpolation
    float p0 = pos.x;
    float p1 = pos.y;
    float p2 = pos.z;

    float FOV = 90;

    float fov_rad = (FOV / 360.f) * 2 * M_PI;

    float nonphysical_plane_half_width = width/2;
    float nonphysical_f_stop = nonphysical_plane_half_width / tan(fov_rad/2);

    float3 pixel_direction = {x - width/2, y - height/2, nonphysical_f_stop};

    pixel_direction = rot_quat(normalize(pixel_direction), camera_quat);

    float max_scalar = 0;

    for(int iteration=0; iteration < 8000; iteration++)
    {
        if(p0 < BORDER_WIDTH || p0 >= dim.x - BORDER_WIDTH - 1 || p1 < BORDER_WIDTH || p1 >= dim.y - BORDER_WIDTH - 1 || p2 < BORDER_WIDTH || p2 >= dim.z - BORDER_WIDTH - 1)
            break;

        #define TRACE_CONFORMAL
        #ifdef TRACE_CONFORMAL
        float Yxx = buffer_read_linear(cY0, (float3)(p0,p1,p2), dim);
        float Yxy = buffer_read_linear(cY1, (float3)(p0,p1,p2), dim);
        float Yxz = buffer_read_linear(cY2, (float3)(p0,p1,p2), dim);
        float Yyy = buffer_read_linear(cY3, (float3)(p0,p1,p2), dim);
        float Yyz = buffer_read_linear(cY4, (float3)(p0,p1,p2), dim);
        float cX = buffer_read_linear(X, (float3)(p0,p1,p2), dim);

        float Yzz = (1 + Yyy * Yxz * Yxz - 2 * Yxy * Yyz * Yxz + Yxx * Yyz * Yyz) / (Yxx * Yyy - Yxy * Yxy);

        float curvature = fabs(Yxx / cX) +
                          fabs(Yxy / cX) +
                          fabs(Yxz / cX) +
                          fabs(Yyy / cX) +
                          fabs(Yyz / cX) +
                          fabs(Yzz / cX);

        float ascalar = curvature / 1000.f;
        #endif // TRACE_CONFORMAL

        //#define TRACE_EXTRINSIC
        #ifdef TRACE_EXTRINSIC
        float cAxx = buffer_read_linear(cA0, (float3)(p0,p1,p2), dim);
        float cAxy = buffer_read_linear(cA1, (float3)(p0,p1,p2), dim);
        float cAxz = buffer_read_linear(cA2, (float3)(p0,p1,p2), dim);
        float cAyy = buffer_read_linear(cA3, (float3)(p0,p1,p2), dim);
        float cAyz = buffer_read_linear(cA4, (float3)(p0,p1,p2), dim);
        float cAzz = buffer_read_linear(cA5, (float3)(p0,p1,p2), dim);

        float curvature = fabs(cAxx) +
                          fabs(cAxy) +
                          fabs(cAxz) +
                          fabs(cAyy) +
                          fabs(cAyz) +
                          fabs(cAzz);

        float ascalar = curvature / 100.f;
        #endif // TRACE_EXTRINSIC

        //#define TRACE_K
        #ifdef TRACE_K
        float lK = buffer_read_linear(K,  (float3)(p0,p1,p2), dim);

        float ascalar = fabs(lK / 2.f);
        #endif // TRACE_K

        max_scalar = max(ascalar, max_scalar);

        p0 += pixel_direction.x;
        p1 += pixel_direction.y;
        p2 += pixel_direction.z;
    }

    max_scalar = max_scalar * 40;

    max_scalar = clamp(max_scalar, 0.f, 1.f);

    write_imagef(screen, (int2)(x, y), (float4)(max_scalar, max_scalar, max_scalar, 1));
}
