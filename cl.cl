///https://arxiv.org/pdf/1404.6523.pdf
///Gauge evolution equations

//#define SYMMETRY_BOUNDARY
//#define USE_GBB

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define IDX(i, j, k) ((k) * dim.x * dim.y + (j) * dim.x + (i))

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

float buffer_read_nearest(__global const float* const buffer, int3 position, int4 dim)
{
    return buffer[position.z * dim.x * dim.y + position.y * dim.x + position.x];
}

float buffer_read_linear(__global const float* const buffer, float3 position, int4 dim)
{
    /*position = round(position);

    int3 ipos = (int3)(position.x, position.y, position.z);

    return buffer[ipos.z * dim.x * dim.y + ipos.y * dim.x + ipos.x];*/

    float3 floored = floor(position);

    int3 ipos = (int3)(floored.x, floored.y, floored.z);

    float c000 = buffer_read_nearest(buffer, ipos + (int3)(0,0,0), dim);
    float c100 = buffer_read_nearest(buffer, ipos + (int3)(1,0,0), dim);

    float c010 = buffer_read_nearest(buffer, ipos + (int3)(0,1,0), dim);
    float c110 = buffer_read_nearest(buffer, ipos + (int3)(1,1,0), dim);


    float c001 = buffer_read_nearest(buffer, ipos + (int3)(0,0,1), dim);
    float c101 = buffer_read_nearest(buffer, ipos + (int3)(1,0,1), dim);

    float c011 = buffer_read_nearest(buffer, ipos + (int3)(0,1,1), dim);
    float c111 = buffer_read_nearest(buffer, ipos + (int3)(1,1,1), dim);

    float3 frac = position - floored;

    float c00 = c000 * (1 - frac.x) + c100 * frac.x;
    float c01 = c001 * (1 - frac.x) + c101 * frac.x;

    float c10 = c010 * (1 - frac.x) + c110 * frac.x;
    float c11 = c011 * (1 - frac.x) + c111 * frac.x;

    float c0 = c00 * (1 - frac.y) + c10 * frac.y;
    float c1 = c01 * (1 - frac.y) + c11 * frac.y;

    return c0 * (1 - frac.z) + c1 * frac.z;
}

void buffer_write(__global float* buffer, int3 position, int4 dim, float value)
{
    buffer[position.z * dim.x * dim.y + position.y * dim.x + position.x] = value;
}

float r_to_phys(float r)
{
    float a = 3;
    float r0 = 5.5f * 0.5f;
    float s = 1.2f * 0.5f;

    ///https://arxiv.org/pdf/gr-qc/0505055.pdf 5.5
    float R_r = (s / (2 * r * tanh(r0/s))) * log(cosh((r + r0)/s)/cosh((r - r0)/s));

    float r_phys = r * (a + (1 - a) * R_r);

    return r_phys;
}

/*float3 transform_position(int x, int y, int z, int4 dim, float scale)
{
    float3 centre = {dim.x/2, dim.y/2, dim.z/2};
    float3 pos = {x, y, z};

    float3 diff = pos - centre;

    float coordinate_r = fast_length(diff);
    coordinate_r = max(coordinate_r, 0.001f);

    float physical_r = r_to_phys(coordinate_r);

    float3 scaled_offset = diff * (physical_r / coordinate_r);
    //float3 scaled_offset = diff * (physical_r / coordinate_r);

    float3 unscaled = scaled_offset / 3;

    if(z == 125)
    {
        printf("%f %f\n", coordinate_r, physical_r);
    }

    return unscaled;
}*/

float polynomial(float x)
{
    return (1 + (-3 + 6 * (-1 + x)) * (-1 + x)) * x * x * x;
}

#define BULGE_AMOUNT 1

float3 transform_position(int x, int y, int z, int4 dim, float scale)
{
    float3 centre = {(dim.x - 1)/2.f, (dim.y - 1)/2.f, (dim.z - 1)/2.f};
    float3 pos = {x, y, z};

    float3 diff = pos - centre;

    diff = round(diff * 2) / 2.f;

    return diff * scale;

    float len = length(diff);

    if(len == 0)
        return diff;

    //if(len <= 0.0001)
    //    len = 0.0001;

    float real_len = len * scale;

    float edge = max(max(dim.x, dim.y), dim.z) * scale / 2.0f;

    float real_distance_r1 = 10.f;

    float r1 = real_distance_r1;
    float r2 = edge - 16 * scale;
    float r3 = edge;

    float bulge_amount = BULGE_AMOUNT;

    float r1b = r1 * bulge_amount;
    float r2b = r2;
    float r3b = r3;

    float rad = 0;

    /*if(real_len < r1)
    {
        rad = (real_len * r1b / r1) / scale;
    }
    else if(real_len < r2)
    {
        float frac = (real_len - r1) / (r2 - r1);

        float polynomial_frac = polynomial(clamp(frac, 0.f, 1.f));

        rad = (r1b + (polynomial_frac * (r2b - r1b))) / scale;
    }
    else
    {
        rad = real_len / scale;
    }

    return diff * rad / len;*/

    float3 norm = normalize(diff);

    if(real_len < r1b)
    {
        return norm * (real_len * r1 / r1b);
    }

    else if(real_len < r2b)
    {
        float frac = (real_len - r1b) / (r2b - r1b);

        float polynomial_frac = polynomial(clamp(frac, 0.f, 1.f));

        float next_len = r1 + polynomial_frac * (r2 - r1);

        return norm * next_len;
    }
    else
    {
        return norm * real_len;
    }
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

__kernel
void setup_u_offset(__global float* u_offset,
                    int4 dim)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    u_offset[IDX(ix, iy, iz)] = 1;
}

///https://learn.lboro.ac.uk/archive/olmp/olmp_resources/pages/workbooks_1_50_jan2008/Workbook33/33_2_elliptic_pde.pdf
///https://arxiv.org/pdf/1205.5111v1.pdf 78
///https://arxiv.org/pdf/gr-qc/0007085.pdf 76?
__kernel
void iterative_u_solve(__global float* u_offset_in, __global float* u_offset_out,
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

    float bl_s = init_BL_val;

    float aij_aIJ = init_aij_aIJ;

    float u = u_offset_in[IDX(ix, iy, iz)];

    //float X = 1/init_BL_val;

    //float B = (1.f/8.f) * pow(X, 7.f) * aij_aIJ;
    //float RHS = -B * pow(1 + X * u, -7);

    float RHS = -(1/8.f) * aij_aIJ * pow(bl_s + u, -7);

    float h2f0 = scale * scale * RHS;

    float uxm1 = u_offset_in[IDX(ix-1, iy, iz)];
    float uxp1 = u_offset_in[IDX(ix+1, iy, iz)];
    float uym1 = u_offset_in[IDX(ix, iy-1, iz)];
    float uyp1 = u_offset_in[IDX(ix, iy+1, iz)];
    float uzm1 = u_offset_in[IDX(ix, iy, iz-1)];
    float uzp1 = u_offset_in[IDX(ix, iy, iz+1)];

    ///so, floating point maths isn't associative
    ///which means that if we're on the other side of a symmetric boundary about the central plane
    ///the order of operations will be different
    ///the if statements correct this, which makes this method numerically symmetric, and implicitly
    ///converges to a symmetric solution if available
    float Xs = uxm1 + uxp1;

    if(ix > (dim.x - 1)/2)
        Xs = uxp1 + uxm1;

    float Ys = uyp1 + uym1;

    if(iy > (dim.y - 1)/2)
        Ys = uym1 + uyp1;

    float Zs = uzp1 + uzm1;

    if(iz > (dim.z - 1)/2)
        Zs = uzm1 + uzp1;

    ///-6u0 + the rest of the terms = h^2 f0
    float u0n1 = (1/6.f) * (Xs + Ys + Zs - h2f0);

    u_offset_out[IDX(ix, iy, iz)] = u0n1;
}

__kernel
void calculate_initial_conditions(__global float* cY0, __global float* cY1, __global float* cY2, __global float* cY3, __global float* cY4,
                                  __global float* cA0, __global float* cA1, __global float* cA2, __global float* cA3, __global float* cA4, __global float* cA5,
                                  __global float* cGi0, __global float* cGi1, __global float* cGi2, __global float* K, __global float* X, __global float* gA, __global float* gB0, __global float* gB1, __global float* gB2,
                                  #ifdef USE_gBB0
                                  __global float* gBB0, __global float* gBB1, __global float* gBB2,
                                  #endif // USE_gBB0
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

    if(ix == (250/2) && iz == (250/2))
    {
        if(iy == 124 || iy == 125 || iy == 126)
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
                                   __global float* cY0, __global float* cY1, __global float* cY2, __global float* cY3, __global float* cY4,
                                   __global float* cA0, __global float* cA1, __global float* cA2, __global float* cA3, __global float* cA4, __global float* cA5,
                                   __global float* cGi0, __global float* cGi1, __global float* cGi2, __global float* K, __global float* X, __global float* gA, __global float* gB0, __global float* gB1, __global float* gB2,
                                   #ifdef USE_gBB0
                                   __global float* gBB0, __global float* gBB1, __global float* gBB2,
                                   #endif // USE_gBB0
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
    float fixed_cA0 = fix_cA0;
    float fixed_cA1 = fix_cA1;
    float fixed_cA2 = fix_cA2;
    float fixed_cA3 = fix_cA3;
    float fixed_cA4 = fix_cA4;
    float fixed_cA5 = fix_cA5;

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

__kernel
void calculate_momentum_constraint(__global ushort4* points, int point_count,
                                   __global float* cY0, __global float* cY1, __global float* cY2, __global float* cY3, __global float* cY4,
            __global float* cA0, __global float* cA1, __global float* cA2, __global float* cA3, __global float* cA4, __global float* cA5,
            __global float* cGi0, __global float* cGi1, __global float* cGi2, __global float* K, __global float* X, __global float* gA, __global float* gB0, __global float* gB1, __global float* gB2,
            __global float* momentum0, __global float* momentum1, __global float* momentum2,
            float scale, int4 dim, float time)
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

float sponge_damp_coeff(float x, float y, float z, float scale, int4 dim, float time)
{
    float edge_half = scale * ((dim.x - 2)/2.f);

    //float sponge_r0 = scale * ((dim.x/2) - 48);
    float sponge_r0 = scale * (((dim.x - 1)/2.f) - 32);
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

    return (r - sponge_r0) / (sponge_r1 - sponge_r0);

    /*float sigma = (sponge_r1 - sponge_r0) / 6;
    return native_exp(-pow((r - sponge_r1) / sigma, 2));*/
}

__kernel
void generate_sponge_points(__global ushort4* points, __global int* point_count, float scale, int4 dim)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    float sponge_factor = sponge_damp_coeff(ix, iy, iz, scale, dim, 0);

    if(sponge_factor <= 0)
        return;

    bool all_high = true;

    for(int i=-1; i <= 1; i++)
    {
        for(int j=-1; j <= 1; j++)
        {
            for(int k=-1; k <= 1; k++)
            {
                if(sponge_damp_coeff(ix + i, iy + j, iz + k, scale, dim, 0) < 1)
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

__kernel
void generate_evolution_points(__global ushort4* points, __global int* point_count, __global ushort4* non_evolved_points, __global int* non_evolved_points_count, float scale, int4 dim)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    float sponge_factor = sponge_damp_coeff(ix, iy, iz, scale, dim, 0);

    if(sponge_factor >= 1)
    {
        int idx = atomic_inc(non_evolved_points_count);

        non_evolved_points[idx].xyz = (ushort3)(ix, iy, iz);
    }
    else
    {
        int idx = atomic_inc(point_count);

        points[idx].xyz = (ushort3)(ix, iy, iz);
    }
}

///https://cds.cern.ch/record/517706/files/0106072.pdf
///boundary conditions
///todo: damp to schwarzschild, not initial conditions?
__kernel
void clean_data(__global ushort4* points, int point_count,
                __global float* cY0, __global float* cY1, __global float* cY2, __global float* cY3, __global float* cY4,
                __global float* cA0, __global float* cA1, __global float* cA2, __global float* cA3, __global float* cA4, __global float* cA5,
                __global float* cGi0, __global float* cGi1, __global float* cGi2, __global float* K, __global float* X, __global float* gA, __global float* gB0, __global float* gB1, __global float* gB2,
                #ifdef USE_gBB0
                __global float* gBB0, __global float* gBB1, __global float* gBB2,
                #endif // USE_gBB0
                __global float* u_value,
                float scale, int4 dim, float time,
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

    float sponge_factor = sponge_damp_coeff(ix, iy, iz, scale, dim, time);

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
    gBB0[index] = mix(gBB0[index], init_gBB0, sponge_factor);
    gBB1[index] = mix(gBB1[index], init_gBB1, sponge_factor);
    gBB2[index] = mix(gBB2[index], init_gBB2, sponge_factor);
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
            __global float* cY0, __global float* cY1, __global float* cY2, __global float* cY3, __global float* cY4,
            __global float* cA0, __global float* cA1, __global float* cA2, __global float* cA3, __global float* cA4, __global float* cA5,
            __global float* cGi0, __global float* cGi1, __global float* cGi2, __global float* K, __global float* X, __global float* gA, __global float* gB0, __global float* gB1, __global float* gB2,
            #ifdef USE_gBB0
            __global float* gBB0, __global float* gBB1, __global float* gBB2,
            #endif // USE_gBB0
            __global float* ocY0, __global float* ocY1, __global float* ocY2, __global float* ocY3, __global float* ocY4,
            __global float* ocA0, __global float* ocA1, __global float* ocA2, __global float* ocA3, __global float* ocA4, __global float* ocA5,
            __global float* ocGi0, __global float* ocGi1, __global float* ocGi2, __global float* oK, __global float* oX, __global float* ogA, __global float* ogB0, __global float* ogB1, __global float* ogB2,
            #ifdef USE_gBB0
            __global float* ogBB0, __global float* ogBB1, __global float* ogBB2,
            #endif // USE_gBB0
            __global float* momentum0, __global float* momentum1, __global float* momentum2,
            __global DERIV_PRECISION* dcYij0, __global DERIV_PRECISION* dcYij1, __global DERIV_PRECISION* dcYij2, __global DERIV_PRECISION* dcYij3, __global DERIV_PRECISION* dcYij4, __global DERIV_PRECISION* dcYij5, __global DERIV_PRECISION* dcYij6, __global DERIV_PRECISION* dcYij7, __global DERIV_PRECISION* dcYij8, __global DERIV_PRECISION* dcYij9, __global DERIV_PRECISION* dcYij10, __global DERIV_PRECISION* dcYij11, __global DERIV_PRECISION* dcYij12, __global DERIV_PRECISION* dcYij13, __global DERIV_PRECISION* dcYij14, __global DERIV_PRECISION* dcYij15, __global DERIV_PRECISION* dcYij16, __global DERIV_PRECISION* dcYij17,
            __global DERIV_PRECISION* digA0, __global DERIV_PRECISION* digA1, __global DERIV_PRECISION* digA2,
            __global DERIV_PRECISION* digB0, __global DERIV_PRECISION* digB1, __global DERIV_PRECISION* digB2, __global DERIV_PRECISION* digB3, __global DERIV_PRECISION* digB4, __global DERIV_PRECISION* digB5, __global DERIV_PRECISION* digB6, __global DERIV_PRECISION* digB7, __global DERIV_PRECISION* digB8,
            __global DERIV_PRECISION* dX0, __global DERIV_PRECISION* dX1, __global DERIV_PRECISION* dX2,
            float scale, int4 dim, float timestep, float time, int current_simulation_boundary)
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
    //float f_dtcYij5 = dtcYij5;

    ocY0[index] = (f_dtcYij0);
    ocY1[index] = (f_dtcYij1);
    ocY2[index] = (f_dtcYij2);
    ocY3[index] = (f_dtcYij3);
    ocY4[index] = (f_dtcYij4);
}


__kernel
void evolve_cA(__global ushort4* points, int point_count,
            __global float* cY0, __global float* cY1, __global float* cY2, __global float* cY3, __global float* cY4,
            __global float* cA0, __global float* cA1, __global float* cA2, __global float* cA3, __global float* cA4, __global float* cA5,
            __global float* cGi0, __global float* cGi1, __global float* cGi2, __global float* K, __global float* X, __global float* gA, __global float* gB0, __global float* gB1, __global float* gB2,
            #ifdef USE_gBB0
            __global float* gBB0, __global float* gBB1, __global float* gBB2,
            #endif // USE_gBB0
            __global float* ocY0, __global float* ocY1, __global float* ocY2, __global float* ocY3, __global float* ocY4,
            __global float* ocA0, __global float* ocA1, __global float* ocA2, __global float* ocA3, __global float* ocA4, __global float* ocA5,
            __global float* ocGi0, __global float* ocGi1, __global float* ocGi2, __global float* oK, __global float* oX, __global float* ogA, __global float* ogB0, __global float* ogB1, __global float* ogB2,
            #ifdef USE_gBB0
            __global float* ogBB0, __global float* ogBB1, __global float* ogBB2,
            #endif // USE_gBB0
            __global float* momentum0, __global float* momentum1, __global float* momentum2,
            __global DERIV_PRECISION* dcYij0, __global DERIV_PRECISION* dcYij1, __global DERIV_PRECISION* dcYij2, __global DERIV_PRECISION* dcYij3, __global DERIV_PRECISION* dcYij4, __global DERIV_PRECISION* dcYij5, __global DERIV_PRECISION* dcYij6, __global DERIV_PRECISION* dcYij7, __global DERIV_PRECISION* dcYij8, __global DERIV_PRECISION* dcYij9, __global DERIV_PRECISION* dcYij10, __global DERIV_PRECISION* dcYij11, __global DERIV_PRECISION* dcYij12, __global DERIV_PRECISION* dcYij13, __global DERIV_PRECISION* dcYij14, __global DERIV_PRECISION* dcYij15, __global DERIV_PRECISION* dcYij16, __global DERIV_PRECISION* dcYij17,
            __global DERIV_PRECISION* digA0, __global DERIV_PRECISION* digA1, __global DERIV_PRECISION* digA2,
            __global DERIV_PRECISION* digB0, __global DERIV_PRECISION* digB1, __global DERIV_PRECISION* digB2, __global DERIV_PRECISION* digB3, __global DERIV_PRECISION* digB4, __global DERIV_PRECISION* digB5, __global DERIV_PRECISION* digB6, __global DERIV_PRECISION* digB7, __global DERIV_PRECISION* digB8,
            __global DERIV_PRECISION* dX0, __global DERIV_PRECISION* dX1, __global DERIV_PRECISION* dX2,
            float scale, int4 dim, float timestep, float time, int current_simulation_boundary)
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

    ocA0[index] = (f_dtcAij0);
    ocA1[index] = (f_dtcAij1);
    ocA2[index] = (f_dtcAij2);
    #ifndef NO_CAIJYY
    ocA3[index] = (f_dtcAij3);
    #endif // NO_CAIJYY
    ocA4[index] = (f_dtcAij4);
    ocA5[index] = (f_dtcAij5);
}

__kernel
void evolve_cGi(__global ushort4* points, int point_count,
            __global float* cY0, __global float* cY1, __global float* cY2, __global float* cY3, __global float* cY4,
            __global float* cA0, __global float* cA1, __global float* cA2, __global float* cA3, __global float* cA4, __global float* cA5,
            __global float* cGi0, __global float* cGi1, __global float* cGi2, __global float* K, __global float* X, __global float* gA, __global float* gB0, __global float* gB1, __global float* gB2,
            #ifdef USE_gBB0
            __global float* gBB0, __global float* gBB1, __global float* gBB2,
            #endif // USE_gBB0
            __global float* ocY0, __global float* ocY1, __global float* ocY2, __global float* ocY3, __global float* ocY4,
            __global float* ocA0, __global float* ocA1, __global float* ocA2, __global float* ocA3, __global float* ocA4, __global float* ocA5,
            __global float* ocGi0, __global float* ocGi1, __global float* ocGi2, __global float* oK, __global float* oX, __global float* ogA, __global float* ogB0, __global float* ogB1, __global float* ogB2,
            #ifdef USE_gBB0
            __global float* ogBB0, __global float* ogBB1, __global float* ogBB2,
            #endif // USE_gBB0
            __global float* momentum0, __global float* momentum1, __global float* momentum2,
            __global DERIV_PRECISION* dcYij0, __global DERIV_PRECISION* dcYij1, __global DERIV_PRECISION* dcYij2, __global DERIV_PRECISION* dcYij3, __global DERIV_PRECISION* dcYij4, __global DERIV_PRECISION* dcYij5, __global DERIV_PRECISION* dcYij6, __global DERIV_PRECISION* dcYij7, __global DERIV_PRECISION* dcYij8, __global DERIV_PRECISION* dcYij9, __global DERIV_PRECISION* dcYij10, __global DERIV_PRECISION* dcYij11, __global DERIV_PRECISION* dcYij12, __global DERIV_PRECISION* dcYij13, __global DERIV_PRECISION* dcYij14, __global DERIV_PRECISION* dcYij15, __global DERIV_PRECISION* dcYij16, __global DERIV_PRECISION* dcYij17,
            __global DERIV_PRECISION* digA0, __global DERIV_PRECISION* digA1, __global DERIV_PRECISION* digA2,
            __global DERIV_PRECISION* digB0, __global DERIV_PRECISION* digB1, __global DERIV_PRECISION* digB2, __global DERIV_PRECISION* digB3, __global DERIV_PRECISION* digB4, __global DERIV_PRECISION* digB5, __global DERIV_PRECISION* digB6, __global DERIV_PRECISION* digB7, __global DERIV_PRECISION* digB8,
            __global DERIV_PRECISION* dX0, __global DERIV_PRECISION* dX1, __global DERIV_PRECISION* dX2,
            float scale, int4 dim, float timestep, float time, int current_simulation_boundary)
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

    ocGi0[index] = (f_dtcGi0);
    ocGi1[index] = (f_dtcGi1);
    ocGi2[index] = (f_dtcGi2);
}


__kernel
void evolve_K(__global ushort4* points, int point_count,
            __global float* cY0, __global float* cY1, __global float* cY2, __global float* cY3, __global float* cY4,
            __global float* cA0, __global float* cA1, __global float* cA2, __global float* cA3, __global float* cA4, __global float* cA5,
            __global float* cGi0, __global float* cGi1, __global float* cGi2, __global float* K, __global float* X, __global float* gA, __global float* gB0, __global float* gB1, __global float* gB2,
            #ifdef USE_gBB0
            __global float* gBB0, __global float* gBB1, __global float* gBB2,
            #endif // USE_gBB0
            __global float* ocY0, __global float* ocY1, __global float* ocY2, __global float* ocY3, __global float* ocY4,
            __global float* ocA0, __global float* ocA1, __global float* ocA2, __global float* ocA3, __global float* ocA4, __global float* ocA5,
            __global float* ocGi0, __global float* ocGi1, __global float* ocGi2, __global float* oK, __global float* oX, __global float* ogA, __global float* ogB0, __global float* ogB1, __global float* ogB2,
            #ifdef USE_gBB0
            __global float* ogBB0, __global float* ogBB1, __global float* ogBB2,
            #endif // USE_gBB0
            __global float* momentum0, __global float* momentum1, __global float* momentum2,
            __global DERIV_PRECISION* dcYij0, __global DERIV_PRECISION* dcYij1, __global DERIV_PRECISION* dcYij2, __global DERIV_PRECISION* dcYij3, __global DERIV_PRECISION* dcYij4, __global DERIV_PRECISION* dcYij5, __global DERIV_PRECISION* dcYij6, __global DERIV_PRECISION* dcYij7, __global DERIV_PRECISION* dcYij8, __global DERIV_PRECISION* dcYij9, __global DERIV_PRECISION* dcYij10, __global DERIV_PRECISION* dcYij11, __global DERIV_PRECISION* dcYij12, __global DERIV_PRECISION* dcYij13, __global DERIV_PRECISION* dcYij14, __global DERIV_PRECISION* dcYij15, __global DERIV_PRECISION* dcYij16, __global DERIV_PRECISION* dcYij17,
            __global DERIV_PRECISION* digA0, __global DERIV_PRECISION* digA1, __global DERIV_PRECISION* digA2,
            __global DERIV_PRECISION* digB0, __global DERIV_PRECISION* digB1, __global DERIV_PRECISION* digB2, __global DERIV_PRECISION* digB3, __global DERIV_PRECISION* digB4, __global DERIV_PRECISION* digB5, __global DERIV_PRECISION* digB6, __global DERIV_PRECISION* digB7, __global DERIV_PRECISION* digB8,
            __global DERIV_PRECISION* dX0, __global DERIV_PRECISION* dX1, __global DERIV_PRECISION* dX2,
            float scale, int4 dim, float timestep, float time, int current_simulation_boundary)
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

    oK[index] = (f_dtK);
}


__kernel
void evolve_X(__global ushort4* points, int point_count,
            __global float* cY0, __global float* cY1, __global float* cY2, __global float* cY3, __global float* cY4,
            __global float* cA0, __global float* cA1, __global float* cA2, __global float* cA3, __global float* cA4, __global float* cA5,
            __global float* cGi0, __global float* cGi1, __global float* cGi2, __global float* K, __global float* X, __global float* gA, __global float* gB0, __global float* gB1, __global float* gB2,
            #ifdef USE_gBB0
            __global float* gBB0, __global float* gBB1, __global float* gBB2,
            #endif // USE_gBB0
            __global float* ocY0, __global float* ocY1, __global float* ocY2, __global float* ocY3, __global float* ocY4,
            __global float* ocA0, __global float* ocA1, __global float* ocA2, __global float* ocA3, __global float* ocA4, __global float* ocA5,
            __global float* ocGi0, __global float* ocGi1, __global float* ocGi2, __global float* oK, __global float* oX, __global float* ogA, __global float* ogB0, __global float* ogB1, __global float* ogB2,
            #ifdef USE_gBB0
            __global float* ogBB0, __global float* ogBB1, __global float* ogBB2,
            #endif // USE_gBB0
            __global float* momentum0, __global float* momentum1, __global float* momentum2,
            __global DERIV_PRECISION* dcYij0, __global DERIV_PRECISION* dcYij1, __global DERIV_PRECISION* dcYij2, __global DERIV_PRECISION* dcYij3, __global DERIV_PRECISION* dcYij4, __global DERIV_PRECISION* dcYij5, __global DERIV_PRECISION* dcYij6, __global DERIV_PRECISION* dcYij7, __global DERIV_PRECISION* dcYij8, __global DERIV_PRECISION* dcYij9, __global DERIV_PRECISION* dcYij10, __global DERIV_PRECISION* dcYij11, __global DERIV_PRECISION* dcYij12, __global DERIV_PRECISION* dcYij13, __global DERIV_PRECISION* dcYij14, __global DERIV_PRECISION* dcYij15, __global DERIV_PRECISION* dcYij16, __global DERIV_PRECISION* dcYij17,
            __global DERIV_PRECISION* digA0, __global DERIV_PRECISION* digA1, __global DERIV_PRECISION* digA2,
            __global DERIV_PRECISION* digB0, __global DERIV_PRECISION* digB1, __global DERIV_PRECISION* digB2, __global DERIV_PRECISION* digB3, __global DERIV_PRECISION* digB4, __global DERIV_PRECISION* digB5, __global DERIV_PRECISION* digB6, __global DERIV_PRECISION* digB7, __global DERIV_PRECISION* digB8,
            __global DERIV_PRECISION* dX0, __global DERIV_PRECISION* dX1, __global DERIV_PRECISION* dX2,
            float scale, int4 dim, float timestep, float time, int current_simulation_boundary)
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

    oX[index] =  (f_dtX);
}

__kernel
void evolve_gA(__global ushort4* points, int point_count,
            __global float* cY0, __global float* cY1, __global float* cY2, __global float* cY3, __global float* cY4,
            __global float* cA0, __global float* cA1, __global float* cA2, __global float* cA3, __global float* cA4, __global float* cA5,
            __global float* cGi0, __global float* cGi1, __global float* cGi2, __global float* K, __global float* X, __global float* gA, __global float* gB0, __global float* gB1, __global float* gB2,
            #ifdef USE_gBB0
            __global float* gBB0, __global float* gBB1, __global float* gBB2,
            #endif // USE_gBB0
            __global float* ocY0, __global float* ocY1, __global float* ocY2, __global float* ocY3, __global float* ocY4,
            __global float* ocA0, __global float* ocA1, __global float* ocA2, __global float* ocA3, __global float* ocA4, __global float* ocA5,
            __global float* ocGi0, __global float* ocGi1, __global float* ocGi2, __global float* oK, __global float* oX, __global float* ogA, __global float* ogB0, __global float* ogB1, __global float* ogB2,
            #ifdef USE_gBB0
            __global float* ogBB0, __global float* ogBB1, __global float* ogBB2,
            #endif // USE_gBB0
            __global float* momentum0, __global float* momentum1, __global float* momentum2,
            __global DERIV_PRECISION* dcYij0, __global DERIV_PRECISION* dcYij1, __global DERIV_PRECISION* dcYij2, __global DERIV_PRECISION* dcYij3, __global DERIV_PRECISION* dcYij4, __global DERIV_PRECISION* dcYij5, __global DERIV_PRECISION* dcYij6, __global DERIV_PRECISION* dcYij7, __global DERIV_PRECISION* dcYij8, __global DERIV_PRECISION* dcYij9, __global DERIV_PRECISION* dcYij10, __global DERIV_PRECISION* dcYij11, __global DERIV_PRECISION* dcYij12, __global DERIV_PRECISION* dcYij13, __global DERIV_PRECISION* dcYij14, __global DERIV_PRECISION* dcYij15, __global DERIV_PRECISION* dcYij16, __global DERIV_PRECISION* dcYij17,
            __global DERIV_PRECISION* digA0, __global DERIV_PRECISION* digA1, __global DERIV_PRECISION* digA2,
            __global DERIV_PRECISION* digB0, __global DERIV_PRECISION* digB1, __global DERIV_PRECISION* digB2, __global DERIV_PRECISION* digB3, __global DERIV_PRECISION* digB4, __global DERIV_PRECISION* digB5, __global DERIV_PRECISION* digB6, __global DERIV_PRECISION* digB7, __global DERIV_PRECISION* digB8,
            __global DERIV_PRECISION* dX0, __global DERIV_PRECISION* dX1, __global DERIV_PRECISION* dX2,
            float scale, int4 dim, float timestep, float time, int current_simulation_boundary)
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

    ogA[index] = (f_dtgA);
}


__kernel
void evolve_gB(__global ushort4* points, int point_count,
            __global float* cY0, __global float* cY1, __global float* cY2, __global float* cY3, __global float* cY4,
            __global float* cA0, __global float* cA1, __global float* cA2, __global float* cA3, __global float* cA4, __global float* cA5,
            __global float* cGi0, __global float* cGi1, __global float* cGi2, __global float* K, __global float* X, __global float* gA, __global float* gB0, __global float* gB1, __global float* gB2,
            #ifdef USE_gBB0
            __global float* gBB0, __global float* gBB1, __global float* gBB2,
            #endif // USE_gBB0
            __global float* ocY0, __global float* ocY1, __global float* ocY2, __global float* ocY3, __global float* ocY4,
            __global float* ocA0, __global float* ocA1, __global float* ocA2, __global float* ocA3, __global float* ocA4, __global float* ocA5,
            __global float* ocGi0, __global float* ocGi1, __global float* ocGi2, __global float* oK, __global float* oX, __global float* ogA, __global float* ogB0, __global float* ogB1, __global float* ogB2,
            #ifdef USE_gBB0
            __global float* ogBB0, __global float* ogBB1, __global float* ogBB2,
            #endif // USE_gBB0
            __global float* momentum0, __global float* momentum1, __global float* momentum2,
            __global DERIV_PRECISION* dcYij0, __global DERIV_PRECISION* dcYij1, __global DERIV_PRECISION* dcYij2, __global DERIV_PRECISION* dcYij3, __global DERIV_PRECISION* dcYij4, __global DERIV_PRECISION* dcYij5, __global DERIV_PRECISION* dcYij6, __global DERIV_PRECISION* dcYij7, __global DERIV_PRECISION* dcYij8, __global DERIV_PRECISION* dcYij9, __global DERIV_PRECISION* dcYij10, __global DERIV_PRECISION* dcYij11, __global DERIV_PRECISION* dcYij12, __global DERIV_PRECISION* dcYij13, __global DERIV_PRECISION* dcYij14, __global DERIV_PRECISION* dcYij15, __global DERIV_PRECISION* dcYij16, __global DERIV_PRECISION* dcYij17,
            __global DERIV_PRECISION* digA0, __global DERIV_PRECISION* digA1, __global DERIV_PRECISION* digA2,
            __global DERIV_PRECISION* digB0, __global DERIV_PRECISION* digB1, __global DERIV_PRECISION* digB2, __global DERIV_PRECISION* digB3, __global DERIV_PRECISION* digB4, __global DERIV_PRECISION* digB5, __global DERIV_PRECISION* digB6, __global DERIV_PRECISION* digB7, __global DERIV_PRECISION* digB8,
            __global DERIV_PRECISION* dX0, __global DERIV_PRECISION* dX1, __global DERIV_PRECISION* dX2,
            float scale, int4 dim, float timestep, float time, int current_simulation_boundary)
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

    ogB0[index] = (f_dtgB0);
    ogB1[index] = (f_dtgB1);
    ogB2[index] = (f_dtgB2);
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

    float TEMPORARIES9;

    float dissipate_single = KREISS_DISSIPATE_SINGULAR;

    obuffer[index] += dissipate_single * timestep;
}

__kernel
void render(__global float* cY0, __global float* cY1, __global float* cY2, __global float* cY3, __global float* cY4,
            __global float* cA0, __global float* cA1, __global float* cA2, __global float* cA3, __global float* cA4, __global float* cA5,
            __global float* cGi0, __global float* cGi1, __global float* cGi2, __global float* K, __global float* X, __global float* gA, __global float* gB0, __global float* gB1, __global float* gB2,
            #ifdef USE_gBB0
            __global float* ogBB0, __global float* ogBB1, __global float* ogBB2,
            #endif // USE_gBB0
            __global DERIV_PRECISION* dcYij0, __global DERIV_PRECISION* dcYij1, __global DERIV_PRECISION* dcYij2, __global DERIV_PRECISION* dcYij3, __global DERIV_PRECISION* dcYij4, __global DERIV_PRECISION* dcYij5, __global DERIV_PRECISION* dcYij6, __global DERIV_PRECISION* dcYij7, __global DERIV_PRECISION* dcYij8, __global DERIV_PRECISION* dcYij9, __global DERIV_PRECISION* dcYij10, __global DERIV_PRECISION* dcYij11, __global DERIV_PRECISION* dcYij12, __global DERIV_PRECISION* dcYij13, __global DERIV_PRECISION* dcYij14, __global DERIV_PRECISION* dcYij15, __global DERIV_PRECISION* dcYij16, __global DERIV_PRECISION* dcYij17,
            __global DERIV_PRECISION* digA0, __global DERIV_PRECISION* digA1, __global DERIV_PRECISION* digA2,
            __global DERIV_PRECISION* digB0, __global DERIV_PRECISION* digB1, __global DERIV_PRECISION* digB2, __global DERIV_PRECISION* digB3, __global DERIV_PRECISION* digB4, __global DERIV_PRECISION* digB5, __global DERIV_PRECISION* digB6, __global DERIV_PRECISION* digB7, __global DERIV_PRECISION* digB8,
            __global DERIV_PRECISION* dX0, __global DERIV_PRECISION* dX1, __global DERIV_PRECISION* dX2,
            float scale, int4 dim, __write_only image2d_t screen, float time)
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
        float sponge_factor = sponge_damp_coeff(ix, iy, iz, scale, dim, time);

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
    //write_imagef(screen, (int2){ix, iy}, (float4){max_scalar, max_scalar, max_scalar, 1});
}

#if 1
__kernel
void extract_waveform(__global float* cY0, __global float* cY1, __global float* cY2, __global float* cY3, __global float* cY4,
                      __global float* cA0, __global float* cA1, __global float* cA2, __global float* cA3, __global float* cA4, __global float* cA5,
                      __global float* cGi0, __global float* cGi1, __global float* cGi2, __global float* K, __global float* X, __global float* gA, __global float* gB0, __global float* gB1, __global float* gB2,
                      #ifdef USE_gBB0
                      __global float* gBB0, __global float* gBB1, __global float* gBB2,
                      #endif // USE_gBB0
                      __global DERIV_PRECISION* dcYij0, __global DERIV_PRECISION* dcYij1, __global DERIV_PRECISION* dcYij2, __global DERIV_PRECISION* dcYij3, __global DERIV_PRECISION* dcYij4, __global DERIV_PRECISION* dcYij5, __global DERIV_PRECISION* dcYij6, __global DERIV_PRECISION* dcYij7, __global DERIV_PRECISION* dcYij8, __global DERIV_PRECISION* dcYij9, __global DERIV_PRECISION* dcYij10, __global DERIV_PRECISION* dcYij11, __global DERIV_PRECISION* dcYij12, __global DERIV_PRECISION* dcYij13, __global DERIV_PRECISION* dcYij14, __global DERIV_PRECISION* dcYij15, __global DERIV_PRECISION* dcYij16, __global DERIV_PRECISION* dcYij17,
                      __global DERIV_PRECISION* digA0, __global DERIV_PRECISION* digA1, __global DERIV_PRECISION* digA2,
                      __global DERIV_PRECISION* digB0, __global DERIV_PRECISION* digB1, __global DERIV_PRECISION* digB2, __global DERIV_PRECISION* digB3, __global DERIV_PRECISION* digB4, __global DERIV_PRECISION* digB5, __global DERIV_PRECISION* digB6, __global DERIV_PRECISION* digB7, __global DERIV_PRECISION* digB8,
                      __global DERIV_PRECISION* dX0, __global DERIV_PRECISION* dX1, __global DERIV_PRECISION* dX2,
                      float scale, int4 dim, int4 pos, int4 waveform_dim, __global float2* waveform_out)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    //if(ix != pos.x || iy != pos.y || iz != pos.z)
    //    return;

    int4 tl = pos - waveform_dim/2;
    int4 br = pos + waveform_dim/2;

    if(ix < tl.x || iy < tl.y || iz < tl.z)
        return;

    if(ix >= br.x || iy >= br.y || iz >= br.z)
        return;

    int4 local_coordinate = (int4)(ix, iy, iz, 0) - tl;

    float3 offset = transform_position(ix, iy, iz, dim, scale);

    float TEMPORARIES4;

    /*for(int i=0; i < TEMP_COUNT4; i++)
    {
        if(!isfinite(pv[i]))
        {
            printf("%i idx is not finite %f\n", i, pv[i]);
        }
    }*/

    //int index = IDX(ix, iy, iz);

    //printf("Scale %f\n", scale);

    //printf("X %f\n", X[index]);

    /*float v0 = dbgv0;
    float v1 = dbgv1;
    float v2 = dbgv2;

    float v3 = dbgv3;
    float v4 = dbgv4;
    float v5 = dbgv5;*/

    /*float debug = dbgw2;
    //printf("debug %f %f %f %f %f %f %f %i %i %i\n", debug, v0, v1, v2, v3, v4, v5, ix, iy, iz);
    printf("debug %f %i %i %i %f %f %f\n", debug, ix, iy, iz, offset.x, offset.y, offset.z);*/

    int wave_idx = local_coordinate.z * waveform_dim.x * waveform_dim.y + local_coordinate.y * waveform_dim.x + local_coordinate.x;

    waveform_out[wave_idx].x = w4_real;
    waveform_out[wave_idx].y = w4_complex;

    //printf("WAV %f\n", waveform_out[0].x);

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

enum ds_result
{
    DS_NONE,
    DS_SKIP,
    DS_RETURN,
};

int calculate_ds_error(float current_ds, float3 next_acceleration, float* next_ds_out)
{
    #define MIN_STEP 0.5f

    float next_ds = 0.01f * 1/fast_length(next_acceleration);

    ///produces strictly worse results for kerr
    next_ds = 0.99f * current_ds * clamp(next_ds / current_ds, 0.1f, 4.f);

    next_ds = max(next_ds, MIN_STEP);
    next_ds = min(next_ds, 1.f);

    *next_ds_out = next_ds;

    //if(next_ds == MIN_STEP)
    //    return DS_RETURN;

    if(next_ds < current_ds/1.95f)
        return DS_SKIP;

    return DS_NONE;
}


__kernel
void trace_rays(__global float* cY0, __global float* cY1, __global float* cY2, __global float* cY3, __global float* cY4,
                __global float* cA0, __global float* cA1, __global float* cA2, __global float* cA3, __global float* cA4, __global float* cA5,
                __global float* cGi0, __global float* cGi1, __global float* cGi2, __global float* K, __global float* X, __global float* gA, __global float* gB0, __global float* gB1, __global float* gB2,
                #ifdef USE_gBB0
                __global float* gBB0, __global float* gBB1, __global float* gBB2,
                #endif // USE_gBB0
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

    float next_ds = 0.1f;

    bool deliberate_termination = false;
    bool last_skipped = false;

    for(int iteration=0; iteration < 16000; iteration++)
    {
        float3 cpos = {lp1, lp2, lp3};

        float3 voxel_pos = world_to_voxel(cpos, dim, scale);

        voxel_pos = clamp(voxel_pos, (float3)(BORDER_WIDTH,BORDER_WIDTH,BORDER_WIDTH), (float3)(dim.x, dim.y, dim.z) - BORDER_WIDTH - 1);

        float fx = voxel_pos.x;
        float fy = voxel_pos.y;
        float fz = voxel_pos.z;

        float TEMPORARIES6;

        float terminate_length = fast_length(cpos);

        if(terminate_length >= universe_size / 1.01f)
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

        float ds = next_ds;

        float dX0 = X0Diff;
        float dX1 = X1Diff;
        float dX2 = X2Diff;

        float dV0 = V0Diff;
        float dV1 = V1Diff;
        float dV2 = V2Diff;

        float3 next_acceleration = {dV0, dV1, dV2};

        int res = calculate_ds_error(ds, next_acceleration, &next_ds);

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

        lp1 += dX0 * ds;
        lp2 += dX1 * ds;
        lp3 += dX2 * ds;

        /*if(x == (int)width/2 && y == (int)height/2)
        {
            printf("%f %f %f  %f %f %f\n", V0, V1, V2, lp1, lp2, lp3);
        }*/

        if(fast_length((float3){dX0, dX1, dX2}) < 0.2f)
        {
            deliberate_termination = true;
            break;
        }
    }

    float4 col = {1,0,1,1};

    if(deliberate_termination || last_skipped)
    {
        col = (float4){0,0,0,1};
    }

    write_imagef(screen, (int2){x, y}, col);
}

struct lightray
{
    float4 pos;
    float4 vel;
    int x, y;
};

///todo: unify this with the above
///the memory overhead is extremely minimal for a huge performance boost
__kernel
void init_accurate_rays(__global float* cY0, __global float* cY1, __global float* cY2, __global float* cY3, __global float* cY4,
                        __global float* cA0, __global float* cA1, __global float* cA2, __global float* cA3, __global float* cA4, __global float* cA5,
                        __global float* cGi0, __global float* cGi1, __global float* cGi2, __global float* K, __global float* X, __global float* gA, __global float* gB0, __global float* gB1, __global float* gB2,
                        #ifdef USE_gBB0
                        __global float* gBB0, __global float* gBB1, __global float* gBB2,
                        #endif // USE_gBB0
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
void step_accurate_rays(__global float* cY0, __global float* cY1, __global float* cY2, __global float* cY3, __global float* cY4,
                        __global float* cA0, __global float* cA1, __global float* cA2, __global float* cA3, __global float* cA4, __global float* cA5,
                        __global float* cGi0, __global float* cGi1, __global float* cGi2, __global float* K, __global float* X, __global float* gA, __global float* gB0, __global float* gB1, __global float* gB2,
                        #ifdef USE_gBB0
                        __global float* gBB0, __global float* gBB1, __global float* gBB2,
                        #endif // USE_gBB0
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

float3 rot_quat(const float3 point, float4 quat)
{
    quat = fast_normalize(quat);

    float3 t = 2.f * cross(quat.xyz, point);

    return point + quat.w * t + cross(quat.xyz, t);
}

__kernel
void trace_metric(__global float* cY0, __global float* cY1, __global float* cY2, __global float* cY3, __global float* cY4,
                  __global float* cA0, __global float* cA1, __global float* cA2, __global float* cA3, __global float* cA4, __global float* cA5,
                  __global float* cGi0, __global float* cGi1, __global float* cGi2, __global float* K, __global float* X, __global float* gA, __global float* gB0, __global float* gB1, __global float* gB2,
                  #ifdef USE_gBB0
                  __global float* gBB0, __global float* gBB1, __global float* gBB2,
                  #endif // USE_gBB0
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
