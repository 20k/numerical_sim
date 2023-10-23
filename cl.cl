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

    float local_time = 0.f;

    float3 offset = transform_position(ix, iy, iz, dim, scale);

    float ox = offset.x;
    float oy = offset.y;
    float oz = offset.z;

    float TEMPORARIES0;

    int index = IDX(ix, iy, iz);

    cY0[index] = init_cY0 - 1;
    cY1[index] = init_cY1;
    cY2[index] = init_cY2;
    cY3[index] = init_cY3 - 1;
    cY4[index] = init_cY4;
    cY5[index] = init_cY5 - 1;

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
    X[index] = init_X - 1;

    gA[index] = init_gA - 1;
    gB0[index] = init_gB0;
    gB1[index] = init_gB1;
    gB2[index] = init_gB2;

    NANCHECK(cY0);
    NANCHECK(cY1);
    NANCHECK(cY2);
    NANCHECK(cY3);
    NANCHECK(cY4);
    NANCHECK(cY5);

    NANCHECK(cA0);
    NANCHECK(cA1);
    NANCHECK(cA2);
    NANCHECK(cA3);
    NANCHECK(cA4);
    NANCHECK(cA5);


    NANCHECK(cGi0);
    NANCHECK(cGi1);
    NANCHECK(cGi2);

    NANCHECK(K);
    NANCHECK(X);

    NANCHECK(gA);
    NANCHECK(gB0);
    NANCHECK(gB1);
    NANCHECK(gB2);


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

    //if(gA[index] > 1)
    //    gA[index] = 1;

    #ifndef X_IS_ACTUALLY_W
    if(X[index] < 0)
        X[index] = 0;
    #endif

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
    #endif
}

__kernel
void calculate_intermediate_data_thin(__global const ushort4* points, int point_count,
                                      __global const float* buffer, __global DERIV_PRECISION* buffer_out_1, __global DERIV_PRECISION* buffer_out_2, __global DERIV_PRECISION* buffer_out_3,
                                      float scale, int4 dim, __global const ushort* order_ptr)
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
void clean_data_thin(__global const ushort4* points, int point_count,
                __global const float* input,
                __global const float* base,
                __global float* out,
                __global const ushort* order_ptr,
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

    float sommer_fin = sommer_fin_out;

    out[index] = sommer_fin;
}

#define DISSB 0.1f

__kernel
void finish_midpoint_impl(__global ushort4* points, int point_count,
                     __global float* summed, __global float* znm1,
                     __global float* out, int4 dim)
{
    int local_idx = get_global_id(0);

    if(local_idx >= point_count)
        return;

    int ix = points[local_idx].x;
    int iy = points[local_idx].y;
    int iz = points[local_idx].z;

    int index = IDX(ix, iy, iz);

    out[index] = 0.5f * (summed[index] + znm1[index]);
}

__kernel
void construct_guess_impl(__global ushort4* points, int point_count,
                     __global float* a, __global float* b, int4 dim)
{
    int local_idx = get_global_id(0);

    if(local_idx >= point_count)
        return;

    int ix = points[local_idx].x;
    int iy = points[local_idx].y;
    int iz = points[local_idx].z;

    int index = IDX(ix, iy, iz);

    a[index] = 0.5f * (a[index] + b[index]);
}

__kernel
void midpoint_guess_impl(__global ushort4* points, int point_count,
                     __global float* a, __global float* b, int4 dim)
{
    int local_idx = get_global_id(0);

    if(local_idx >= point_count)
        return;

    int ix = points[local_idx].x;
    int iy = points[local_idx].y;
    int iz = points[local_idx].z;

    int index = IDX(ix, iy, iz);

    a[index] = 2 * a[index] - b[index];
}

__kernel
void finish_heun_impl(__global ushort4* points, int point_count,
                      __global float* yip1, __global float* yip2, int4 dim)
{
    int local_idx = get_global_id(0);

    if(local_idx >= point_count)
        return;

    int ix = points[local_idx].x;
    int iy = points[local_idx].y;
    int iz = points[local_idx].z;

    int index = IDX(ix, iy, iz);

    yip2[index] = 0.5f * (yip1[index] + yip2[index]);
}

__kernel
void bdf_sum_impl(__global ushort4* points, int point_count,
                  __global float* yp1, __global float* yp, int4 dim)
{
    int local_idx = get_global_id(0);

    if(local_idx >= point_count)
        return;

    int ix = points[local_idx].x;
    int iy = points[local_idx].y;
    int iz = points[local_idx].z;

    int index = IDX(ix, iy, iz);

    yp[index] = (4.f/3.f) * yp1[index] - (1.f/3.f) * yp[index];
}

__kernel
void finish_bs_impl(__global ushort4* points, int point_count,
                     __global float* high, __global float* low, int4 dim)
{
    int local_idx = get_global_id(0);

    if(local_idx >= point_count)
        return;

    int ix = points[local_idx].x;
    int iy = points[local_idx].y;
    int iz = points[local_idx].z;

    int index = IDX(ix, iy, iz);

    high[index] = (4 * high[index] - low[index]) / 3.f;
}

__kernel
void multiply_add_impl(__global ushort4* points, int point_count,
                       __global float* left, __global float* right, float cst1, float cst2, int which_buf, int4 dim)
{
    int local_idx = get_global_id(0);

    if(local_idx >= point_count)
        return;

    int ix = points[local_idx].x;
    int iy = points[local_idx].y;
    int iz = points[local_idx].z;

    int index = IDX(ix, iy, iz);

    float in_left = left[index];

    left[index] = left[index] * cst1 + right[index] * cst2;
}

__kernel
void render(STANDARD_CONST_ARGS(),
            STANDARD_CONST_DERIVS(),
            STANDARD_UTILITY(),
            __global const ushort* order_ptr,
            float scale, int4 dim, __write_only image2d_t screen, int debug_x, int debug_y)
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

    int index = IDX(ix, iy, iz);

    if(ix == debug_x && iy == debug_y)
    {
        printf("X %f\n", X[index]);
        printf("K %f\n", K[index]);
        printf("gA %f\n", gA[index]);
        printf("gB0 %f\n", gB0[index]);
        printf("gB1 %f\n", gB1[index]);
        printf("gB2 %f\n", gB2[index]);
        printf("CY %.24f %.24f %.24f %.24f %.24f %.24f\n", cY0[index], cY1[index], cY2[index], cY3[index], cY4[index], cY5[index]);
        //printf("Ps %.24f\n", Dp_star[index]);
    }

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
        ascalar = pow(fabs(X[index]), 2.f) / 50;
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

    if(X[index] * X[index] < 0.03f)
        col = (float3)(0, 0, 1);

    /*if(order_ptr[index] & D_GTE_WIDTH_4)
        col = (float3){0,0,1};
    else
        col = (float3){0,1,0};*/

    float3 lin_col = srgb_to_lin(col);

    write_imagef(screen, (int2){get_global_id(0), get_global_id(1)}, (float4)(lin_col.xyz, 1));
    //write_imagef(screen, (int2){ix, iy}, (float4){max_sca1lar, max_scalar, max_scalar, 1});
}

#if 1
__kernel
void extract_waveform(__global const ushort4* points, int point_count,
                      STANDARD_CONST_ARGS(),
                      STANDARD_CONST_DERIVS(),
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
