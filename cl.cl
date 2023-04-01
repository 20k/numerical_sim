///https://arxiv.org/pdf/1404.6523.pdf
///Gauge evolution equations

//#define SYMMETRY_BOUNDARY
//#define USE_GBB

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#include "transform_position.cl"
#include "common.cl"
#include "evolution_common.cl"
#include "evolve_points.cl"

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

__kernel
void do_rk4_accumulate(__global ushort4* points, int point_count, int4 dim, __global float* accum, __global float* base, __global float* Q, float factor)
{
    int local_idx = get_global_id(0);

    if(local_idx >= point_count)
        return;

    int ix = points[local_idx].x;
    int iy = points[local_idx].y;
    int iz = points[local_idx].z;

    int index = IDX(ix, iy, iz);

    accum[index] += (Q[index] - base[index]) * factor;
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
void enforce_algebraic_constraints(__global ushort4* points, int point_count,
                                   STANDARD_ARGS(),
                                   float scale, int4 dim)
{
    #if defined(DAMPED_CONSTRAINTS) && defined(X_IS_ACTUALLY_W)
    return;
    #endif

    int idx = get_global_id(0);

    if(idx >= point_count)
        return;

    int ix = points[idx].x;
    int iy = points[idx].y;
    int iz = points[idx].z;

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    int index = IDX(ix, iy, iz);

    if(gA[index] < 0)
        gA[index] = 0;

    if(gA[index] > 1)
        gA[index] = 1;

    #ifndef X_IS_ACTUALLY_W
    if(X[index] < 0)
        X[index] = 0;
    #endif

    #ifndef NO_CAIJYY

    float found_det = CY_DET;

    float tol = 1e-6;

    if(found_det <= 1 + tol && found_det >= 1 - tol)
        return;

    #ifndef DAMPED_CONSTRAINTS
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
    #endif
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
                                   STANDARD_UTILITY(),
                                   float scale, int4 dim,
                                   __global ushort* order_ptr)
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
        momentum0[index] = 0;
        momentum1[index] = 0;
        momentum2[index] = 0;
        return;
    }

    float TEMPORARIES12;

    float m1 = init_momentum0;
    float m2 = init_momentum1;
    float m3 = init_momentum2;

    momentum0[index] = m1;
    momentum1[index] = m2;
    momentum2[index] = m3;
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

///https://cds.cern.ch/record/517706/files/0106072.pdf
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

#define DISSB 0.1f

__kernel
void evolve_cY(__global ushort4* points, int point_count,
            STANDARD_ARGS(),
            STANDARD_ARGS(o),
            STANDARD_ARGS(base_),
            __global float* momentum0, __global float* momentum1, __global float* momentum2,
            STANDARD_DERIVS(),
            STANDARD_UTILITY(),
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
            STANDARD_DERIVS(),
            STANDARD_UTILITY(),
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
            STANDARD_DERIVS(),
            STANDARD_UTILITY(),
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
            STANDARD_DERIVS(),
            STANDARD_UTILITY(),
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
            STANDARD_DERIVS(),
            STANDARD_UTILITY(),
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
            STANDARD_DERIVS(),
            STANDARD_UTILITY(),
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
            STANDARD_DERIVS(),
            STANDARD_UTILITY(),
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

__kernel void evaluate_secant_impl(__global ushort4* points, int point_count,
                                   __global float* yn, __global float* xnm1, __global float* xnm2,
                                   __global float* s_xnm1, __global float* s_xnm2,
                                   int4 dim)
{
    int local_idx = get_global_id(0);

    if(local_idx >= point_count)
        return;

    int ix = points[local_idx].x;
    int iy = points[local_idx].y;
    int iz = points[local_idx].z;

    int index = IDX(ix, iy, iz);

    float bottom = (-s_xnm1[index] + s_xnm2[index]);

    if(fabs(bottom) < 0.000001f)
    {
        s_xnm2[index] = s_xnm1[index];
        return;
    }

    float next = xnm1[index] - (yn[index] - s_xnm1[index]) * (xnm1[index] - xnm2[index]) / bottom;

    s_xnm2[index] = next;
}

__kernel
void render(STANDARD_ARGS(),
            STANDARD_DERIVS(),
            STANDARD_UTILITY(),
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

        float ascalar = 0;

        #ifdef RENDER_MATTER_P
        ascalar = fabs(adm_p[index]) * 2000000;
        #endif // RENDER_MATTER_P

        //#define RENDER_METRIC
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

        ascalar = fabs(curvature / 1000.f);
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

        ascalar = fabs(curvature / 1.f);
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

        ascalar = fabs(curvature / 1.f);
        #endif // RENDER_AIJ

        //#define RENDER_K
        #ifdef RENDER_K
        ascalar = fabs(K[index] * 40);
        #endif // RENDER_K

        //#define RENDER_X
        #ifdef RENDER_X
        ascalar = fabs(X[index] / 50);
        #endif // RENDER_X

        //#define RENDER_CGI
        #ifdef RENDER_CGI
        ascalar = fabs(cGi0[index]) + fabs(cGi1[index]) + fabs(cGi2[index]);
        #endif // RENDER_CGI

        //#define RENDER_GA
        #ifdef RENDER_GA
        ascalar = fabs(gA[index] / 50);
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
        ascalar =       fabs(dcYij0[index]) +
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

        //#define RENDER_MOMENTUM
        #ifdef RENDER_MOMENTUM
        int order = D_FULL;

        float M0 = init_momentum0;
        float M1 = init_momentum1;
        float M2 = init_momentum2;

        float M = (fabs(M0) + fabs(M1) + fabs(M2)) / 3.f;

        ascalar = M * 1000 / 40.f;
        #endif // RENDER_MOMENTUM

        //#define RENDER_HAMILTONIAN
        #ifdef RENDER_HAMILTONIAN
        int order = D_FULL;

        float TEMPORARIEShamiltonian;

        float H0 = init_hamiltonian;

        ascalar = fabs(H0) * 100;

        /*if(ix == 100 && iy == 100)
        {
            printf("H0 %f\n", H0);
        }*/

        #endif // RENDER_HAMILTONIAN

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

        real = fabs(real) * 30.f;

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
                      STANDARD_DERIVS(),
                      float scale, int4 dim, __global float2* waveform_out/*, __write_only image2d_t screen*/)
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

    //if(iz == (dim.z-1)/2)
    //    write_imagef(screen, (int2){ix, iy}, (float4)(zfrac, zfrac, zfrac, 1));

    #ifdef w4_debugr
    printf("Debugw4r %f\n", w4_debugr);
    #endif // w4_debug

    #ifdef w4_debugi
    printf("Debugw4i %f\n", w4_debugi);
    #endif // w4_debug
}
#endif // 0
