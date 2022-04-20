///https://arxiv.org/pdf/1404.6523.pdf
///Gauge evolution equations

//#define SYMMETRY_BOUNDARY
//#define USE_GBB

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#include "transform_position.cl"
#include "common.cl"

bool invalid_first(int ix, int iy, int iz, int4 dim)
{
    return ix < BORDER_WIDTH || iy < BORDER_WIDTH || iz < BORDER_WIDTH || ix >= dim.x - BORDER_WIDTH || iy >= dim.y - BORDER_WIDTH || iz >= dim.z - BORDER_WIDTH;
}

bool invalid_second(int ix, int iy, int iz, int4 dim)
{
    return ix < BORDER_WIDTH * 2 || iy < BORDER_WIDTH * 2 || iz < BORDER_WIDTH * 2 || ix >= dim.x - BORDER_WIDTH * 2 || iy >= dim.y - BORDER_WIDTH * 2 || iz >= dim.z - BORDER_WIDTH * 2;
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

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    #ifndef SYMMETRY_BOUNDARY
    if(invalid_second(ix, iy, iz, dim))
        return;
    #endif // SYMMETRY_BOUNDARY

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

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    #ifndef SYMMETRY_BOUNDARY
    if(invalid_second(ix, iy, iz, dim))
        return;
    #endif // SYMMETRY_BOUNDARY

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

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    #ifndef SYMMETRY_BOUNDARY
    if(invalid_second(ix, iy, iz, dim))
        return;
    #endif // SYMMETRY_BOUNDARY

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

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    #ifndef SYMMETRY_BOUNDARY
    if(invalid_second(ix, iy, iz, dim))
        return;
    #endif // SYMMETRY_BOUNDARY

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

float3 world_to_voxel(float3 world_pos, int4 dim, float scale)
{
    float3 centre = {(dim.x - 1)/2, (dim.y - 1)/2, (dim.z - 1)/2};

    return (world_pos / scale) + centre;
}

float get_distance(int x1, int y1, int z1, int x2, int y2, int z2, int4 dim, float scale)
{
    float3 d1 = transform_position(x1, y1, z1, dim, scale);
    float3 d2 = transform_position(x2, y2, z2, dim, scale);

    return fast_length(d2 - d1);
}

#ifndef USE_GBB
#define STANDARD_ARGS(p) __global float* p##cY0, __global float* p##cY1, __global float* p##cY2, __global float* p##cY3, __global float* p##cY4, __global float* p##cY5, \
            __global float* p##cA0, __global float* p##cA1, __global float* p##cA2, __global float* p##cA3, __global float* p##cA4, __global float* p##cA5, \
            __global float* p##cGi0, __global float* p##cGi1, __global float* p##cGi2, __global float* p##K, __global float* p##X, __global float* p##gA, __global float* p##gB0, __global float* p##gB1, __global float* p##gB2
#else
#define STANDARD_ARGS(p) __global float* p##cY0, __global float* p##cY1, __global float* p##cY2, __global float* p##cY3, __global float* p##cY4, __global float* p##cY5, \
            __global float* p##cA0, __global float* p##cA1, __global float* p##cA2, __global float* p##cA3, __global float* p##cA4, __global float* p##cA5, \
            __global float* p##cGi0, __global float* p##cGi1, __global float* p##cGi2, __global float* p##K, __global float* p##X, __global float* p##gA, __global float* p##gB0, __global float* p##gB1, __global float* p##gB2, \
            __global float* p##gBB0, __global float* p##gBB1, __global float* p##gBB2
#endif

__kernel
void calculate_initial_conditions(STANDARD_ARGS(),
                                  __global float* u_value,
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

    float bl_conformal = init_BL_val;

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

    float TEMPORARIES10;

    buffer_out_1[IDX(ix,iy,iz)] = init_buffer_intermediate0;
    buffer_out_2[IDX(ix,iy,iz)] = init_buffer_intermediate1;
    buffer_out_3[IDX(ix,iy,iz)] = init_buffer_intermediate2;
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

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;
    #ifndef SYMMETRY_BOUNDARY
    if(invalid_second(ix, iy, iz, dim))
        return;
    #endif // SYMMETRY_BOUNDARY

    float TEMPORARIES12;

    float m1 = init_momentum0;
    float m2 = init_momentum1;
    float m3 = init_momentum2;

    momentum0[IDX(ix,iy,iz)] = m1;
    momentum1[IDX(ix,iy,iz)] = m2;
    momentum2[IDX(ix,iy,iz)] = m3;
}

float sponge_damp_coeff(float x, float y, float z, float scale, int4 dim)
{
    float edge_half = scale * ((dim.x - 2)/2.f);

    //float sponge_r0 = scale * ((dim.x/2) - 48);
    float sponge_r0 = scale * (((dim.x - 1)/2.f) - 48);
    //float sponge_r0 = scale * ((dim.x/2) - 32);
    //float sponge_r0 = edge_half/2;
    float sponge_r1 = scale * (((dim.x - 1)/2.f) - 6);

    float3 fdim = ((float3)(dim.x, dim.y, dim.z) - 1)/2.f;

    float3 diff = ((float3){x, y, z} - fdim) * scale;

    #ifdef MANHATTEN_SPONGE
    float r = max(fabs(diff.x), max(fabs(diff.y), fabs(diff.z)));
    #else
    float r = fast_length(diff);
    #endif // MANHATTEN_SPONGE

    if(r <= sponge_r0)
        return 0.f;

    if(r >= sponge_r1)
        return 1.f;

    r = clamp(r, sponge_r0, sponge_r1);

    //return (r - sponge_r0) / (sponge_r1 - sponge_r0);

    float sigma = (sponge_r1 - sponge_r0) / 2;
    return native_exp(-pow((r - sponge_r1) / sigma, 2));
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

    bool all_high = true;

    for(int i=-1; i <= 1; i++)
    {
        for(int j=-1; j <= 1; j++)
        {
            for(int k=-1; k <= 1; k++)
            {
                if(sponge_damp_coeff(ix + i, iy + j, iz + k, scale, dim) < 1)
                {
                    all_high = false;
                }
            }
        }
    }

    if(all_high)
        return;

    int idx = atomic_inc(point_count);

    points[idx].xyz = (ushort3)(ix, iy, iz);
}

///could try this function also where it only calculates derivatives for dynamic fields
///aka do not calculate derivatives across the maximal sponge point
bool valid_first_derivative_point(int3 pos, float scale, int4 dim)
{
    ///one of the points would lie outside the boundary
    if(invalid_first(pos.x, pos.y, pos.z, dim))
        return false;

    int width = BORDER_WIDTH;

    for(int z=-width; z <= width; z++)
    {
        for(int y=-width; y <= width; y++)
        {
            for(int x=-width; x <= width; x++)
            {
                int3 combo = (int3)(x, y, z) + pos;

                float sponge_local = sponge_damp_coeff(combo.x, combo.y, combo.z, scale, dim);

                ///one of the points is non sponged, so we should calculate first derivatives
                if(sponge_local < 1)
                    return true;
            }
        }
    }

    ///no point is unsponged, therefore do not process
    return false;
}

///if any point is not a valid first derivative point, we can't evolve this correctly
bool valid_second_derivative_point(int3 pos, float scale, int4 dim)
{
    ///out of boundaries for the below loop
    if(invalid_first(pos.x, pos.y, pos.z, dim))
        return false;

    int width = BORDER_WIDTH;

    for(int z=-width; z <= width; z++)
    {
        for(int y=-width; y <= width; y++)
        {
            for(int x=-width; x <= width; x++)
            {
                int3 combo = (int3)(x, y, z) + pos;

                ///one of the underlying first derivatives would be invalid
                if(!valid_first_derivative_point(combo, scale, dim))
                    return false;
            }
        }
    }

    return true;
}

__kernel
void generate_evolution_points(__global ushort4* points_1st, __global int* point_count_1st,
                               __global ushort4* points_2nd, __global int* point_count_2nd,
                               float scale, int4 dim)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    int3 pos = (int3)(ix, iy, iz);

    if(valid_first_derivative_point(pos, scale, dim))
    {
        int idx = atomic_inc(point_count_1st);

        points_1st[idx].xyz = (ushort3)(ix, iy, iz);
    }

    if(valid_second_derivative_point(pos, scale, dim))
    {
        int idx = atomic_inc(point_count_2nd);

        points_2nd[idx].xyz = (ushort3)(ix, iy, iz);
    }
}

///https://cds.cern.ch/record/517706/files/0106072.pdf
///boundary conditions
///todo: damp to schwarzschild, not initial conditions?
__kernel
void clean_data(__global ushort4* points, int point_count,
                STANDARD_ARGS(),
                __global float* u_value,
                float scale, int4 dim,
                float timestep)
{
    int idx = get_global_id(0);

    if(idx >= point_count)
        return;

    int ix = points[idx].x;
    int iy = points[idx].y;
    int iz = points[idx].z;

    #ifdef SYMMETRY_BOUNDARY
    return;
    #endif // SYMMETRY_BOUNDARY

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    float sponge_factor = sponge_damp_coeff(ix, iy, iz, scale, dim);

    if(sponge_factor <= 0)
        return;

    float3 offset = transform_position(ix, iy, iz, dim, scale);

    float ox = offset.x;
    float oy = offset.y;
    float oz = offset.z;

    float bl_conformal = init_BL_val;

    float TEMPORARIES0;

    float initial_cY0 = init_cY0;
    float initial_cY1 = init_cY1;
    float initial_cY2 = init_cY2;
    float initial_cY3 = init_cY3;
    float initial_cY4 = init_cY4;
    float initial_cY5 = init_cY5;

    float initial_cA0 = init_cA0;
    float initial_cA1 = init_cA1;
    float initial_cA2 = init_cA2;
    float initial_cA3 = init_cA3;
    float initial_cA4 = init_cA4;
    float initial_cA5 = init_cA5;

    float initial_X = init_X;

    float fin_gA = init_gA;
    float fin_gB0 = init_gB0;
    float fin_gB1 = init_gB1;
    float fin_gB2 = init_gB2;

    #ifdef USE_GBB
    float fin_gBB0 = init_gBB0;
    float fin_gBB1 = init_gBB1;
    float fin_gBB2 = init_gBB2;
    #endif // USE_GBB

    int index = IDX(ix, iy, iz);

    ///todo: investigate if 2 full orbits is possible on the non radiative condition
    ///woooo
    //#define RADIATIVE
    #ifdef RADIATIVE
    fin_gA = 1;
    fin_gB0 = 0;
    fin_gB1 = 0;
    fin_gB2 = 0;

    initial_cY0 = 1;
    initial_cY1 = 0;
    initial_cY2 = 0;
    initial_cY3 = 1;
    initial_cY4 = 0;

    initial_cA0 = 0;
    initial_cA1 = 0;
    initial_cA2 = 0;
    initial_cA3 = 0;
    initial_cA4 = 0;
    initial_cA5 = 0;

    initial_X = 1;
    #endif // RADIATIVE

    ///https://authors.library.caltech.edu/8284/1/RINcqg07.pdf (34)
    float y_r = sponge_factor;

    #define EVOLVE_CY_AT_BOUNDARY
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
}

float3 srgb_to_lin(float3 C_srgb)
{
    return  0.012522878f * C_srgb +
            0.682171111f * C_srgb * C_srgb +
            0.305306011f * C_srgb * C_srgb * C_srgb;
}

#define NANCHECK(w) if(isnan(w[index])){printf("NAN " #w " %i %i %i\n", ix, iy, iz); debug = true;}

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
            float scale, int4 dim, float timestep)
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
    if(invalid_second(ix, iy, iz, dim))
        return;
    #endif // SYMMETRY_BOUNDARY

    int index = IDX(ix, iy, iz);

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
            float scale, int4 dim, float timestep)
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
    if(invalid_second(ix, iy, iz, dim))
        return;
    #endif // SYMMETRY_BOUNDARY

    int index = IDX(ix, iy, iz);

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
            float scale, int4 dim, float timestep)
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
    if(invalid_second(ix, iy, iz, dim))
        return;
    #endif // SYMMETRY_BOUNDARY

    int index = IDX(ix, iy, iz);

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
            float scale, int4 dim, float timestep)
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
    if(invalid_second(ix, iy, iz, dim))
        return;
    #endif // SYMMETRY_BOUNDARY

    int index = IDX(ix, iy, iz);

    float TEMPORARIEStk;

    float f_dtK = dtK;

    float b0 = base_K[index];

    oK[index] = f_dtK * timestep + b0;
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
            float scale, int4 dim, float timestep)
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
    if(invalid_second(ix, iy, iz, dim))
        return;
    #endif // SYMMETRY_BOUNDARY

    int index = IDX(ix, iy, iz);

    float TEMPORARIEStx;

    float f_dtX = dtX;

    float b0 = base_X[index];

    oX[index] = f_dtX * timestep + b0;
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
            float scale, int4 dim, float timestep)
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
    if(invalid_second(ix, iy, iz, dim))
        return;
    #endif // SYMMETRY_BOUNDARY

    int index = IDX(ix, iy, iz);

    float TEMPORARIEStga;

    float f_dtgA = dtgA;

    float b0 = base_gA[index];

    ogA[index] = f_dtgA * timestep + b0;
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
            float scale, int4 dim, float timestep)
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
    if(invalid_second(ix, iy, iz, dim))
        return;
    #endif // SYMMETRY_BOUNDARY

    int index = IDX(ix, iy, iz);

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
}

__kernel
void dissipate_single(__global ushort4* points, int point_count,
                      __global float* buffer, __global float* obuffer,
                      float coefficient,
                      float scale, int4 dim, float timestep)
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
    if(invalid_second(ix, iy, iz, dim))
        return;
    #endif // SYMMETRY_BOUNDARY

    int index = IDX(ix, iy, iz);

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

    float TEMPORARIES9;

    float dissipate_single = KREISS_DISSIPATE_SINGULAR;

    obuffer[index] += damp * dissipate_single * timestep;
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

    float max_scalar = 0;

    //for(int z = 20; z < dim.z-20; z++)

    {
        float sponge_factor = sponge_damp_coeff(ix, iy, iz, scale, dim);

        if(sponge_factor > 0)
        {
            float3 sponge_col = {sponge_factor, 0, 0};

            write_imagef(screen, (int2){get_global_id(0), get_global_id(1)}, (float4)(srgb_to_lin(sponge_col), 1));
            return;
        }

        int index = IDX(ix, iy, iz);

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

        /*if(cX < 0.7)
            ascalar = 1;
        else
            ascalar = 0;*/

        max_scalar = max(ascalar, max_scalar);
    }

    float real = 0;

    {
        float TEMPORARIES4;

        real = w4_real;

        real = fabs(real) * 1000.f;

        real = clamp(real, 0.f, 1.f);
    }

    /*if(ix == 125 && iy == 125)
    {
        printf("scalar %f\n", max_scalar);
    }*/

    max_scalar = max_scalar * 40;

    max_scalar = clamp(max_scalar, 0.f, 1.f);

    float3 col = {real, max_scalar, max_scalar};

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

float buffer_read_nearest_clamph(__global const DERIV_PRECISION* const buffer, int3 position, int4 dim)
{
    position = clamp(position, (int3)(0,0,0), dim.xyz - 1);

    return buffer[position.z * dim.x * dim.y + position.y * dim.x + position.x];
}

float buffer_read_linearh(__global const DERIV_PRECISION* const buffer, float3 position, int4 dim)
{
    float3 floored = floor(position);

    int3 ipos = (int3)(floored.x, floored.y, floored.z);

    float c000 = buffer_read_nearest_clamph(buffer, ipos + (int3)(0,0,0), dim);
    float c100 = buffer_read_nearest_clamph(buffer, ipos + (int3)(1,0,0), dim);

    float c010 = buffer_read_nearest_clamph(buffer, ipos + (int3)(0,1,0), dim);
    float c110 = buffer_read_nearest_clamph(buffer, ipos + (int3)(1,1,0), dim);

    float c001 = buffer_read_nearest_clamph(buffer, ipos + (int3)(0,0,1), dim);
    float c101 = buffer_read_nearest_clamph(buffer, ipos + (int3)(1,0,1), dim);

    float c011 = buffer_read_nearest_clamph(buffer, ipos + (int3)(0,1,1), dim);
    float c111 = buffer_read_nearest_clamph(buffer, ipos + (int3)(1,1,1), dim);

    float3 frac = position - floored;

    ///numerically symmetric across the centre of dim
    float c00 = c000 - frac.x * (c000 - c100);
    float c01 = c001 - frac.x * (c001 - c101);

    float c10 = c010 - frac.x * (c010 - c110);
    float c11 = c011 - frac.x * (c011 - c111);

    float c0 = c00 - frac.y * (c00 - c10);
    float c1 = c01 - frac.y * (c01 - c11);

    return c0 - frac.z * (c0 - c1);
}

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
    int early_terminate;
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

int should_early_terminate(int x, int y, int width, int height, __global int* termination_buffer)
{
    x = clamp(x, 0, width-1);
    y = clamp(y, 0, height-1);

    return termination_buffer[y * width + x] == 1;
}

__kernel
void calculate_singularities(__global struct lightray_simple* finished_rays, __global int* finished_count, __global int* termination_buffer, int width, int height)
{
    int id = get_global_id(0);

    if(id >= *finished_count)
        return;

    struct lightray_simple ray = finished_rays[id];

    int x = ray.x;
    int y = ray.y;

    termination_buffer[y * width + x] = ray.early_terminate;
}

__kernel
void init_rays(__global struct lightray_simple* rays, __global int* ray_count0, __global int* ray_count1,
                STANDARD_ARGS(),
                __global DERIV_PRECISION* dcYij0, __global DERIV_PRECISION* dcYij1, __global DERIV_PRECISION* dcYij2, __global DERIV_PRECISION* dcYij3, __global DERIV_PRECISION* dcYij4, __global DERIV_PRECISION* dcYij5, __global DERIV_PRECISION* dcYij6, __global DERIV_PRECISION* dcYij7, __global DERIV_PRECISION* dcYij8, __global DERIV_PRECISION* dcYij9, __global DERIV_PRECISION* dcYij10, __global DERIV_PRECISION* dcYij11, __global DERIV_PRECISION* dcYij12, __global DERIV_PRECISION* dcYij13, __global DERIV_PRECISION* dcYij14, __global DERIV_PRECISION* dcYij15, __global DERIV_PRECISION* dcYij16, __global DERIV_PRECISION* dcYij17,
                __global DERIV_PRECISION* digA0, __global DERIV_PRECISION* digA1, __global DERIV_PRECISION* digA2,
                __global DERIV_PRECISION* digB0, __global DERIV_PRECISION* digB1, __global DERIV_PRECISION* digB2, __global DERIV_PRECISION* digB3, __global DERIV_PRECISION* digB4, __global DERIV_PRECISION* digB5, __global DERIV_PRECISION* digB6, __global DERIV_PRECISION* digB7, __global DERIV_PRECISION* digB8,
                __global DERIV_PRECISION* dX0, __global DERIV_PRECISION* dX1, __global DERIV_PRECISION* dX2,
                float scale, float3 camera_pos, float4 camera_quat,
                int4 dim, int width, int height,
                __global int* termination_buffer,
                int prepass_width, int prepass_height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x >= width || y >= height)
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
    out.early_terminate = 0;

    #define USE_PREPASS
    #ifdef USE_PREPASS
    if(prepass_width != width && prepass_height != height)
    {
        float fx = (float)x / width;
        float fy = (float)y / height;

        int lx = round(fx * prepass_width);
        int ly = round(fy * prepass_height);

        if(should_early_terminate(lx-1, ly, prepass_width, prepass_height, termination_buffer) &&
           should_early_terminate(lx, ly, prepass_width, prepass_height, termination_buffer) &&
           should_early_terminate(lx+1, ly, prepass_width, prepass_height, termination_buffer) &&
           should_early_terminate(lx, ly-1, prepass_width, prepass_height, termination_buffer) &&
           should_early_terminate(lx, ly+1, prepass_width, prepass_height, termination_buffer))
        {
            out.early_terminate = 2;
        }
    }
    #endif // USE_PREPASS

    rays[y * width + x] = out;

    *ray_count0 = width * height;
    *ray_count1 = 0;
}

__kernel
void trace_rays(__global struct lightray_simple* rays_in, __global struct lightray_simple* rays_out, __global struct lightray_simple* rays_terminated,
                __global int* ray_count_in, __global int* ray_count_out, __global int* ray_count_terminated,
                STANDARD_ARGS(),
                __global DERIV_PRECISION* dcYij0, __global DERIV_PRECISION* dcYij1, __global DERIV_PRECISION* dcYij2, __global DERIV_PRECISION* dcYij3, __global DERIV_PRECISION* dcYij4, __global DERIV_PRECISION* dcYij5, __global DERIV_PRECISION* dcYij6, __global DERIV_PRECISION* dcYij7, __global DERIV_PRECISION* dcYij8, __global DERIV_PRECISION* dcYij9, __global DERIV_PRECISION* dcYij10, __global DERIV_PRECISION* dcYij11, __global DERIV_PRECISION* dcYij12, __global DERIV_PRECISION* dcYij13, __global DERIV_PRECISION* dcYij14, __global DERIV_PRECISION* dcYij15, __global DERIV_PRECISION* dcYij16, __global DERIV_PRECISION* dcYij17,
                __global DERIV_PRECISION* digA0, __global DERIV_PRECISION* digA1, __global DERIV_PRECISION* digA2,
                __global DERIV_PRECISION* digB0, __global DERIV_PRECISION* digB1, __global DERIV_PRECISION* digB2, __global DERIV_PRECISION* digB3, __global DERIV_PRECISION* digB4, __global DERIV_PRECISION* digB5, __global DERIV_PRECISION* digB6, __global DERIV_PRECISION* digB7, __global DERIV_PRECISION* digB8,
                __global DERIV_PRECISION* dX0, __global DERIV_PRECISION* dX1, __global DERIV_PRECISION* dX2,
                float scale, int4 dim, int width, int height, float err_in)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x >= width || y >= height)
        return;

    struct lightray_simple ray_in = rays_in[y * width + x];

    if(ray_in.early_terminate)
    {
        int next_idx = atomic_inc(ray_count_terminated);
        rays_terminated[next_idx] = ray_in;
        return;
    }

    float lp1 = ray_in.lp1;
    float lp2 = ray_in.lp2;
    float lp3 = ray_in.lp3;

    float V0 = ray_in.V0;
    float V1 = ray_in.V1;
    float V2 = ray_in.V2;

    float final_dX0 = 0;
    float final_dX1 = 0;
    float final_dX2 = 0;

    ///ray location
    ///temporary while i don't do interpolation
    float next_ds = 0.1f;

    bool deliberate_termination = false;
    bool last_skipped = false;
    bool hit_singularity = false;

    int iteration = 0;

    for(iteration=0; iteration < 65000; iteration++)
    {
        float3 cpos = {lp1, lp2, lp3};

        float3 voxel_pos = world_to_voxel(cpos, dim, scale);

        voxel_pos = clamp(voxel_pos, (float3)(BORDER_WIDTH,BORDER_WIDTH,BORDER_WIDTH), (float3)(dim.x, dim.y, dim.z) - BORDER_WIDTH - 1);

        float BH_X = buffer_read_linear(X, voxel_pos, dim);

        float fx = voxel_pos.x;
        float fy = voxel_pos.y;
        float fz = voxel_pos.z;

        /*int ix = floor(fx);
        int iy = floor(fy);
        int iz = floor(fz);*/

        float TEMPORARIES6;

        float ldX0 = X0Diff;
        float ldX1 = X1Diff;
        float ldX2 = X2Diff;

        float terminate_length = fast_length(cpos);

        if(terminate_length >= universe_size / 1.01f)
        {
            final_dX0 = ldX0;
            final_dX1 = ldX1;
            final_dX2 = ldX2;

            deliberate_termination = true;
            break;
        }

        float ds = next_ds;

        float dV0 = V0Diff;
        float dV1 = V1Diff;
        float dV2 = V2Diff;

        float3 next_acceleration = {dV0, dV1, dV2};

        int res = calculate_ds_error(err_in, ds, next_acceleration, &next_ds);

        /*if(BH_X < 0.8f)
        {
            ds = min(ds, 0.5f);
        }*/

        float X_far = 0.9f;
        float X_near = 0.6f;

        float my_fraction = (clamp(BH_X, X_near, X_far) - X_near) / (X_far - X_near);

        my_fraction = clamp(my_fraction, 0.f, 1.f);

        ds = mix(0.1f, 2.f, my_fraction);

        if(res == DS_RETURN)
        {
            deliberate_termination = true;
            break;
        }

        if(res == DS_SKIP)
        {
            last_skipped = true;
            continue;
        }

        last_skipped = false;

        /*ds = 0.025f * 1/fast_length(next_acceleration);
        ds = min(ds, 2.f);
        ds = max(ds, 0.5f);*/

        V0 += dV0 * ds;
        V1 += dV1 * ds;
        V2 += dV2 * ds;

        lp1 += ldX0 * ds;
        lp2 += ldX1 * ds;
        lp3 += ldX2 * ds;

        /*if(x == (int)width/2 && y == (int)height/2)
        {
            printf("%f %f %f  %f %f %f\n", V0, V1, V2, lp1, lp2, lp3);
        }*/

        if(fast_length((float3){ldX0, ldX1, ldX2}) < 0.2f)
        {
            hit_singularity = true;
            deliberate_termination = true;
            break;
        }
    }

    struct lightray_simple ray_out;
    ray_out.x = ray_in.x;
    ray_out.y = ray_in.y;

    ray_out.lp1 = lp1;
    ray_out.lp2 = lp2;
    ray_out.lp3 = lp3;

    ray_out.V0 = final_dX0;
    ray_out.V1 = final_dX1;
    ray_out.V2 = final_dX2;

    ray_out.iter_frac = iteration / 128.f;
    ray_out.early_terminate = hit_singularity;

    /*if(ray_out.early_terminate && width == 800)
    {
        printf("hit\n");
    }*/

    if(!deliberate_termination)
    {
        int next_idx = atomic_inc(ray_count_out);
        rays_out[next_idx] = ray_out;
    }
    else
    {
        int next_idx = atomic_inc(ray_count_terminated);
        rays_terminated[next_idx] = ray_out;
    }
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

__kernel void render_rays(__global struct lightray_simple* rays_in, __global int* ray_count, __write_only image2d_t screen, float scale)
{
    int idx = get_global_id(0);

    if(idx >= *ray_count)
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

    float uni_size = universe_size;

    float terminate_length = length(cpos);

    if(ray_in.early_terminate == 0)
    {
        cpos = fix_ray_position(cpos, cvel, uni_size * 1.01f);

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
    }
    else if(ray_in.early_terminate == 1)
    {
        float3 val = (float3)(0,0,0);

        //val.xyz = clamp(ray_in.iter_frac, 0.f, 1.f);

        write_imagef(screen, (int2){x, y}, (float4)(val.xyz,1));
    }
    else
    {
        float3 val = (float3)(0,1,0);

        //val.xyz = clamp(ray_in.iter_frac, 0.f, 1.f);

        write_imagef(screen, (int2){x, y}, (float4)(val.xyz,1));
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
