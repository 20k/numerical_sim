///https://arxiv.org/pdf/1404.6523.pdf
///Gauge evolution equations

//#define SYMMETRY_BOUNDARY

//#define USE_GBB

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

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

///todo: This can be eliminated by use of local memory and using different approximations to the derivatives at the boundary
///need to work out why precaching the differentials of these didn't make much difference compared to bssnok_data
///it might be because they're arrays
struct intermediate_bssnok_data
{
    ///christoffel symbols are symmetric in the lower 2 indices
    //float christoffel[3 * 6];
    float dcYij[3 * 6];
    float digA[3];
    float digB[3*3];
    //float phi;
    float dX[3];
};

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

#define IDX(i, j, k) ((k) * dim.x * dim.y + (j) * dim.x + (i))

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
    float3 centre = {dim.x/2, dim.y/2, dim.z/2};
    float3 pos = {x, y, z};

    float3 diff = pos - centre;

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
    float3 centre = {dim.x/2, dim.y/2, dim.z/2};

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

    u_offset[IDX(ix, iy, iz)] = 0;
}

///https://learn.lboro.ac.uk/archive/olmp/olmp_resources/pages/workbooks_1_50_jan2008/Workbook33/33_2_elliptic_pde.pdf
///https://arxiv.org/pdf/1205.5111v1.pdf 78
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

    /*float B = (1.f/8.f) * pow(a, 7.f) * aij_aIJ;
    float RHS = -B * pow(1 + a * u, -7);*/

    float RHS = -(1/8.f) * aij_aIJ * pow(bl_s + u, -7);

    float h2f0 = scale * scale * RHS;

    float uxm1 = u_offset_in[IDX(ix-1, iy, iz)];
    float uxp1 = u_offset_in[IDX(ix+1, iy, iz)];
    float uym1 = u_offset_in[IDX(ix, iy-1, iz)];
    float uyp1 = u_offset_in[IDX(ix, iy+1, iz)];
    float uzm1 = u_offset_in[IDX(ix, iy, iz-1)];
    float uzp1 = u_offset_in[IDX(ix, iy, iz+1)];

    ///-6u0 + the rest of the terms = h^2 f0
    float u0n1 = (1/6.f) * (uxm1 + uxp1 + uym1 + uyp1 + uzm1 + uzp1 - h2f0);

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
void enforce_algebraic_constraints(__global float* cY0, __global float* cY1, __global float* cY2, __global float* cY3, __global float* cY4,
                                   __global float* cA0, __global float* cA1, __global float* cA2, __global float* cA3, __global float* cA4, __global float* cA5,
                                   __global float* cGi0, __global float* cGi1, __global float* cGi2, __global float* K, __global float* X, __global float* gA, __global float* gB0, __global float* gB1, __global float* gB2,
                                   #ifdef USE_gBB0
                                   __global float* gBB0, __global float* gBB1, __global float* gBB2,
                                   #endif // USE_gBB0
                                   float scale, int4 dim)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    //float TEMPORARIES3;

    int index = IDX(ix, iy, iz);

    //float fixed_cY0 = fix_cY0;
    //float fixed_cY1 = fix_cY1;
    //float fixed_cY2 = fix_cY2;
    //float fixed_cY3 = fix_cY3;
    //float fixed_cY4 = fix_cY4;

    float fixed_cA0 = fix_cA0;
    float fixed_cA1 = fix_cA1;
    float fixed_cA2 = fix_cA2;
    float fixed_cA3 = fix_cA3;
    float fixed_cA4 = fix_cA4;
    float fixed_cA5 = fix_cA5;

    //cY0[index] = fixed_cY0;
    //cY1[index] = fixed_cY1;
    //cY2[index] = fixed_cY2;
    //cY3[index] = fixed_cY3;
    //cY4[index] = fixed_cY4;

    cA0[index] = fixed_cA0;
    cA1[index] = fixed_cA1;
    cA2[index] = fixed_cA2;
    cA3[index] = fixed_cA3;
    cA4[index] = fixed_cA4;
    cA5[index] = fixed_cA5;
}

///https://en.wikipedia.org/wiki/Ricci_curvature#Definition_via_local_coordinates_on_a_smooth_manifold
__kernel
void calculate_intermediate_data(__global float* cY0, __global float* cY1, __global float* cY2, __global float* cY3, __global float* cY4,
                                 __global float* cA0, __global float* cA1, __global float* cA2, __global float* cA3, __global float* cA4, __global float* cA5,
                                 __global float* cGi0, __global float* cGi1, __global float* cGi2, __global float* K, __global float* X, __global float* gA, __global float* gB0, __global float* gB1, __global float* gB2,
                                 #ifdef USE_gBB0
                                 __global float* gBB0, __global float* gBB1, __global float* gBB2,
                                 #endif // USE_gBB0
                                 float scale, int4 dim, __global struct intermediate_bssnok_data* out)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    #ifndef SYMMETRY_BOUNDARY
    if(ix < BORDER_WIDTH || ix >= dim.x - BORDER_WIDTH - 1 || iy < BORDER_WIDTH || iy >= dim.y - BORDER_WIDTH - 1 || iz < BORDER_WIDTH || iz >= dim.z - BORDER_WIDTH - 1)
        return;
    #endif // SYMMETRY_BOUNDARY

    struct intermediate_bssnok_data* my_out = &out[IDX(ix, iy, iz)];

    float TEMPORARIES1;

    my_out->dcYij[0] = init_dcYij0;
    my_out->dcYij[1] = init_dcYij1;
    my_out->dcYij[2] = init_dcYij2;
    my_out->dcYij[3] = init_dcYij3;
    my_out->dcYij[4] = init_dcYij4;
    my_out->dcYij[5] = init_dcYij5;
    my_out->dcYij[6] = init_dcYij6;
    my_out->dcYij[7] = init_dcYij7;
    my_out->dcYij[8] = init_dcYij8;
    my_out->dcYij[9] = init_dcYij9;
    my_out->dcYij[10] = init_dcYij10;
    my_out->dcYij[11] = init_dcYij11;
    my_out->dcYij[12] = init_dcYij12;
    my_out->dcYij[13] = init_dcYij13;
    my_out->dcYij[14] = init_dcYij14;
    my_out->dcYij[15] = init_dcYij15;
    my_out->dcYij[16] = init_dcYij16;
    my_out->dcYij[17] = init_dcYij17;

    my_out->digA[0] = init_digA0;
    my_out->digA[1] = init_digA1;
    my_out->digA[2] = init_digA2;

    my_out->digB[0] = init_digB0;
    my_out->digB[1] = init_digB1;
    my_out->digB[2] = init_digB2;
    my_out->digB[3] = init_digB3;
    my_out->digB[4] = init_digB4;
    my_out->digB[5] = init_digB5;
    my_out->digB[6] = init_digB6;
    my_out->digB[7] = init_digB7;
    my_out->digB[8] = init_digB8;

    //my_out->phi = init_phi;

    my_out->dX[0] = init_dX0;
    my_out->dX[1] = init_dX1;
    my_out->dX[2] = init_dX2;
}

__kernel
void calculate_intermediate_data_thin(__global float* buffer, __global DERIV_PRECISION* buffer_out_1, __global DERIV_PRECISION* buffer_out_2, __global DERIV_PRECISION* buffer_out_3,
                                      float scale, int4 dim)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    #ifndef SYMMETRY_BOUNDARY
    if(ix < BORDER_WIDTH || ix >= dim.x - BORDER_WIDTH - 1 || iy < BORDER_WIDTH || iy >= dim.y - BORDER_WIDTH - 1 || iz < BORDER_WIDTH || iz >= dim.z - BORDER_WIDTH - 1)
        return;
    #endif // SYMMETRY_BOUNDARY

    float TEMPORARIES10;

    buffer_out_1[IDX(ix,iy,iz)] = init_buffer_intermediate0;
    buffer_out_2[IDX(ix,iy,iz)] = init_buffer_intermediate1;
    buffer_out_3[IDX(ix,iy,iz)] = init_buffer_intermediate2;
}

__kernel
void calculate_intermediate_data_thin_cY5(__global float* cY0, __global float* cY1, __global float* cY2, __global float* cY3, __global float* cY4,
                                          __global DERIV_PRECISION* buffer_out_1, __global DERIV_PRECISION* buffer_out_2, __global DERIV_PRECISION* buffer_out_3,
                                         float scale, int4 dim)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    #ifndef SYMMETRY_BOUNDARY
    if(ix < BORDER_WIDTH || ix >= dim.x - BORDER_WIDTH - 1 || iy < BORDER_WIDTH || iy >= dim.y - BORDER_WIDTH - 1 || iz < BORDER_WIDTH || iz >= dim.z - BORDER_WIDTH - 1)
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
void calculate_momentum_constraint(__global float* cY0, __global float* cY1, __global float* cY2, __global float* cY3, __global float* cY4,
            __global float* cA0, __global float* cA1, __global float* cA2, __global float* cA3, __global float* cA4, __global float* cA5,
            __global float* cGi0, __global float* cGi1, __global float* cGi2, __global float* K, __global float* X, __global float* gA, __global float* gB0, __global float* gB1, __global float* gB2,
            __global float* momentum0, __global float* momentum1, __global float* momentum2,
            float scale, int4 dim, float time)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    #ifndef SYMMETRY_BOUNDARY
    if(ix < BORDER_WIDTH*2 || ix >= dim.x - BORDER_WIDTH*2 - 1 || iy < BORDER_WIDTH*2 || iy >= dim.y - BORDER_WIDTH*2 - 1 || iz < BORDER_WIDTH*2 || iz >= dim.z - BORDER_WIDTH*2 - 1)
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
    float edge_half = scale * (dim.x/2);

    float sponge_r0 = scale * ((dim.x/2) - 64);
    //float sponge_r0 = edge_half/2;
    float sponge_r1 = scale * ((dim.x/2) - 8);

    /*if(time >= 4)
    {
        float time_frac = (time - 4) / 4;

        time_frac = clamp(time_frac, 0.f, 1.f);

        sponge_r0 = mix(sponge_r0, scale * ((dim.x/2) - 32), time_frac);
    }*/

    float3 fdim = (float3)(dim.x, dim.y, dim.z)/2.f;

    float r = fast_length((float3){x, y, z} - fdim) * scale;

    if(r <= sponge_r0)
        return 0.f;

    r = clamp(r, sponge_r0, sponge_r1);

    float r_frac = (r - sponge_r0) / (sponge_r1 - sponge_r0);

    //if(r_frac >= 0.2f)
    //    return 0.2f;

    return r_frac;

    //return r_frac * pow(r_frac, fabs(sin(time / (2 * M_PI))));
}

__kernel
void generate_sponge_points(__global ushort4* points, __global int* point_count, float scale, int4 dim)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    #ifdef SYMMETRY_BOUNDARY
    return;
    #endif // SYMMETRY_BOUNDARY

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    float sponge_factor = sponge_damp_coeff(ix, iy, iz, scale, dim, 0);

    if(sponge_factor <= 0)
        return;

    int idx = atomic_inc(point_count);

    points[idx].xyz = (ushort3)(ix, iy, iz);
}

///https://cds.cern.ch/record/517706/files/0106072.pdf
///boundary conditions
///todo: damp to schwarzschild, not initial conditions?
__kernel
void clean_data(__global ushort4* points, __global int* points_count,
                __global float* cY0, __global float* cY1, __global float* cY2, __global float* cY3, __global float* cY4,
                __global float* cA0, __global float* cA1, __global float* cA2, __global float* cA3, __global float* cA4, __global float* cA5,
                __global float* cGi0, __global float* cGi1, __global float* cGi2, __global float* K, __global float* X, __global float* gA, __global float* gB0, __global float* gB1, __global float* gB2,
                #ifdef USE_gBB0
                __global float* gBB0, __global float* gBB1, __global float* gBB2,
                #endif // USE_gBB0
                __global float* u_value,
                float scale, int4 dim, float time)
{
    int idx = get_global_id(0);

    if(idx >= *points_count)
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

    cY0[index] = mix(cY0[index],initial_cY0, sponge_factor);
    cY1[index] = mix(cY1[index],initial_cY1, sponge_factor);
    cY2[index] = mix(cY2[index],initial_cY2, sponge_factor);
    cY3[index] = mix(cY3[index],initial_cY3, sponge_factor);
    cY4[index] = mix(cY4[index],initial_cY4, sponge_factor);

    cA0[index] = mix(cA0[index],initial_cA0, sponge_factor);
    cA1[index] = mix(cA1[index],initial_cA1, sponge_factor);
    cA2[index] = mix(cA2[index],initial_cA2, sponge_factor);
    cA3[index] = mix(cA3[index],initial_cA3, sponge_factor);
    cA4[index] = mix(cA4[index],initial_cA4, sponge_factor);
    cA5[index] = mix(cA5[index],initial_cA5, sponge_factor);

    cGi0[index] = mix(cGi0[index],init_cGi0, sponge_factor);
    cGi1[index] = mix(cGi1[index],init_cGi1, sponge_factor);
    cGi2[index] = mix(cGi2[index],init_cGi2, sponge_factor);

    K[index] = mix(K[index],init_K, sponge_factor);
    X[index] = mix(X[index],initial_X, sponge_factor);

    gA[index] = mix(gA[index],fin_gA, sponge_factor);
    gB0[index] = mix(gB0[index],fin_gB0, sponge_factor);
    gB1[index] = mix(gB1[index],fin_gB1, sponge_factor);
    gB2[index] = mix(gB2[index],fin_gB2, sponge_factor);

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

/*__kernel
void dissipate(__global float* buffer_in, __global float* buffer_out, float scale, int4 dim, float timestep)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    #ifndef SYMMETRY_BOUNDARY
    if(ix < BORDER_WIDTH*2 || ix >= dim.x - BORDER_WIDTH*2 || iy < BORDER_WIDTH*2 || iy >= dim.y - BORDER_WIDTH*2 || iz < BORDER_WIDTH*2 || iz >= dim.z - BORDER_WIDTH*2)
        return;
    #endif // SYMMETRY_BOUNDARY

    float dissipation = get_dissipation(ix, iy, iz, dim, scale, buffer);

    int idx = IDX(ix, iy, iz);

    buffer_out[idx] = buffer_in[idx] +
}*/

#define NANCHECK(w) if(isnan(w[index])){printf("NAN " #w " %i %i %i\n", ix, iy, iz); debug = true;}

///todo: need to correctly evolve boundaries
///todo: need to factor out the differentials
__kernel
void evolve(__global float* cY0, __global float* cY1, __global float* cY2, __global float* cY3, __global float* cY4,
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
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    #ifndef SYMMETRY_BOUNDARY
    if(ix < BORDER_WIDTH*2 || ix >= dim.x - BORDER_WIDTH*2 - 1 || iy < BORDER_WIDTH*2 || iy >= dim.y - BORDER_WIDTH*2 - 1 || iz < BORDER_WIDTH*2 || iz >= dim.z - BORDER_WIDTH*2 - 1)
        return;
    #endif // SYMMETRY_BOUNDARY

    int index = IDX(ix, iy, iz);

    int3 icentre = dim.xyz/2;

    int3 ioffset = (int3)(ix, iy, iz) - icentre;

    //float sponge_factor = sponge_damp_coeff(ix, iy, iz, scale, dim, time);

    //if(sponge_factor == 1)
    //    return;

    float3 centre = {dim.x/2, dim.y/2, dim.z/2};
    float r = fast_length((float3){ix, iy, iz} - centre);


    float TEMPORARIES2;

    float f_dtcYij0 = dtcYij0;
    float f_dtcYij1 = dtcYij1;
    float f_dtcYij2 = dtcYij2;
    float f_dtcYij3 = dtcYij3;
    float f_dtcYij4 = dtcYij4;
    float f_dtcYij5 = dtcYij5;

    float f_dtcAij0 = dtcAij0;
    float f_dtcAij1 = dtcAij1;
    float f_dtcAij2 = dtcAij2;
    float f_dtcAij3 = dtcAij3;
    float f_dtcAij4 = dtcAij4;
    float f_dtcAij5 = dtcAij5;

    float f_dtcGi0 = dtcGi0;
    float f_dtcGi1 = dtcGi1;
    float f_dtcGi2 = dtcGi2;

    float f_dtK = dtK;
    float f_dtX = dtX;

    float f_dtgA = dtgA;
    float f_dtgB0 = dtgB0;
    float f_dtgB1 = dtgB1;
    float f_dtgB2 = dtgB2;

    #ifdef USE_GBB
    float f_dtgBB0 = dtgBB0;
    float f_dtgBB1 = dtgBB1;
    float f_dtgBB2 = dtgBB2;
    #endif // USE_GBB

    float debug1 = debug_p1;
    float debug2 = debug_p2;
    float debug3 = debug_p3;
    //float dbgdphi = ik.dphi[0];

    /*if(ix == 20 && iy == 20 && iz == 20)
    {
        printf("DISS %f %f %f %f %f %f\n", diss_cYij0, diss_cYij1, diss_cYij2, diss_cYij3, diss_cYij4, diss_cYij5);
    }*/

    //if(ix == 150 && iy == 150 && iz == 150)
    //printf("DISS %f\n", diss_cYij0);

    float I_cY0 = cY0[index];
    float I_cY1 = cY1[index];
    float I_cY2 = cY2[index];
    float I_cY3 = cY3[index];
    float I_cY4 = cY4[index];

    ocY0[index] = I_cY0 + (f_dtcYij0) * timestep;
    ocY1[index] = I_cY1 + (f_dtcYij1) * timestep;
    ocY2[index] = I_cY2 + (f_dtcYij2) * timestep;
    ocY3[index] = I_cY3 + (f_dtcYij3) * timestep;
    ocY4[index] = I_cY4 + (f_dtcYij4) * timestep;

    ocA0[index] = cA0[index] + (f_dtcAij0) * timestep;
    ocA1[index] = cA1[index] + (f_dtcAij1) * timestep;
    ocA2[index] = cA2[index] + (f_dtcAij2) * timestep;
    #ifndef NO_CAIJYY
    ocA3[index] = cA3[index] + (f_dtcAij3) * timestep;
    #endif // NO_CAIJYY
    ocA4[index] = cA4[index] + (f_dtcAij4) * timestep;
    ocA5[index] = cA5[index] + (f_dtcAij5) * timestep;

    ocGi0[index] = cGi0[index] + (f_dtcGi0) * timestep;
    ocGi1[index] = cGi1[index] + (f_dtcGi1) * timestep;
    ocGi2[index] = cGi2[index] + (f_dtcGi2) * timestep;

    oK[index] = K[index] + (f_dtK) * timestep;
    oX[index] = X[index] + (f_dtX) * timestep;

    ogA[index] = gA[index] + (f_dtgA) * timestep;
    ogB0[index] = gB0[index] + (f_dtgB0) * timestep;
    ogB1[index] = gB1[index] + (f_dtgB1) * timestep;
    ogB2[index] = gB2[index] + (f_dtgB2) * timestep;

    #ifdef USE_GBB
    ogBB0[index] = gBB0[index] + (dtgBB0 + diss_gBB0) * timestep;
    ogBB1[index] = gBB1[index] + (dtgBB1 + diss_gBB1) * timestep;
    ogBB2[index] = gBB2[index] + (dtgBB2 + diss_gBB2) * timestep;
    #endif // USE_GBB

    /*bool debug = false;

    NANCHECK(ocY0);
    NANCHECK(ocY1);
    NANCHECK(ocY2);
    NANCHECK(ocY3);
    NANCHECK(ocY4);
    //NANCHECK(ocY5);

    NANCHECK(ocA0);
    NANCHECK(ocA1);
    NANCHECK(ocA2);
    //NANCHECK(ocA3);
    NANCHECK(ocA4);
    NANCHECK(ocA5);

    NANCHECK(ocGi0);
    NANCHECK(ocGi1);
    NANCHECK(ocGi2);

    NANCHECK(oK);
    NANCHECK(oX);
    NANCHECK(ogA);
    NANCHECK(ogB0);
    NANCHECK(ogB1);
    NANCHECK(ogB2);*/

    #ifdef USE_GBB
    NANCHECK(gBB0);
    NANCHECK(gBB1);
    NANCHECK(gBB2);
    #endif // USE_GBB

    #if 0
    //if(debug)
    //if(x == 5 && y == 6 && z == 4)
    //if(ix == 125 && y == 100 && z == 125)
    //if(x == 125 && y == 125 && z == 125)
    ///NAN cA0 111 188 111
    ///the issue is the cGi term, again
    //if(ix == 171 && iy == 138 && iz == 141)
    //if(ix == 171 && iy == 137 && iz == 142)
    //if(ix == 140 && iy == 140 && iz == 140)
    //if(ix == 171 && iy == 140 && iz == 140)
    //if(ix == 162 && iy == 155 && iz == 211)
    ///161 148 211
    //if(ix == 161 && iy == 148 && iz == 211)
    //if(ix == 161 && iy == 151 && iz == 211)
    //if(ix == 161 && iy == 150 && iz == 150)

    //138.500000 150.500000 150.500000
    //if(ix == 139 && iy == 151 && iz == 150)
    //145.500000 150.500000 150.500000
    //if(ix == 146 && iy == 151 && iz == 151)
    if(ix == 134 && iy == 150 && iz == 152)
    {
        //float scalar = scalar_curvature;

        printf("DtY0 %f\n", dtcYij0);
        printf("DtA0 %f\n", dtcAij0);
        printf("Aij0 %f\n", cA0[index]);
        printf("Aij1 %f\n", cA1[index]);
        printf("Aij2 %f\n", cA2[index]);
        printf("Aij3 %f\n", cA3[index]);
        printf("Aij4 %f\n", cA4[index]);
        printf("Aij5 %f\n", cA5[index]);
        printf("Yij0 %f\n", cY0[index]);
        printf("Yij1 %f\n", cY1[index]);
        printf("Yij2 %f\n", cY2[index]);
        printf("Yij3 %f\n", cY3[index]);
        printf("Yij4 %f\n", cY4[index]);
        //printf("Yij5 %f\n", cY5[index]);
        printf("cGi0 %f\n", cGi0[index]);
        printf("cGi1 %f\n", cGi1[index]);
        printf("cGi2 %f\n", cGi2[index]);
        printf("X %f\n", X[index]);
        printf("K %f\n", K[index]);
        printf("gA %f\n", gA[index]);
        printf("gB0 %f\n", gB0[index]);
        printf("gB1 %f\n", gB1[index]);
        printf("gB2 %f\n", gB2[index]);
        //printf("Scalar %f\n", scalar);

        printf("Debugp1 %f", debug1);
        printf("Debugp2 %f", debug2);
        printf("Debugp3 %f", debug3);
        //printf("dphi %f", dbgdphi);

        /*#ifdef debug_val
        float dbg = debug_val;
        printf("Debug %f\n", debug_val);
        #endif // debug_val
        */

        /*float d0 = debug_val0;
        float d1 = debug_val1;
        float d2 = debug_val2;
        float d3 = debug_val3;
        float d4 = debug_val4;
        float d5 = debug_val5;
        float d6 = debug_val6;
        float d7 = debug_val7;
        float d8 = debug_val8;

        printf("Vals: %f %f %f %f %f %f %f %f %f\n", d0, d1, d2, d3, d4, d5, d6, d7, d8);*/
    }
    #endif // 0
}

#if 0
///kreiss
__kernel
void numerical_dissipate(__global float* cY0, __global float* cY1, __global float* cY2, __global float* cY3, __global float* cY4,
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
                        float scale, int4 dim, float timestep)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    #ifndef SYMMETRY_BOUNDARY
    if(ix < BORDER_WIDTH*2 || ix >= dim.x - BORDER_WIDTH*2 - 1 || iy < BORDER_WIDTH*2 || iy >= dim.y - BORDER_WIDTH*2 - 1 || iz < BORDER_WIDTH*2 || iz >= dim.z - BORDER_WIDTH*2 - 1)
        return;
    #endif // SYMMETRY_BOUNDARY

    int index = IDX(ix, iy, iz);

    float TEMPORARIES7;

    float diss_cYij0 = k_cYij0;
    float diss_cYij1 = k_cYij1;
    float diss_cYij2 = k_cYij2;
    float diss_cYij3 = k_cYij3;
    float diss_cYij4 = k_cYij4;

    float diss_cAij0 = k_cAij0;
    float diss_cAij1 = k_cAij1;
    float diss_cAij2 = k_cAij2;
    float diss_cAij3 = k_cAij3;
    float diss_cAij4 = k_cAij4;
    float diss_cAij5 = k_cAij5;

    float diss_cGi0 = k_cGi0;
    float diss_cGi1 = k_cGi1;
    float diss_cGi2 = k_cGi2;

    float diss_K = k_K;
    float diss_X = k_X;

    float diss_gA = k_gA;
    float diss_gB0 = k_gB0;
    float diss_gB1 = k_gB1;
    float diss_gB2 = k_gB2;

    ocY0[index] += (diss_cYij0) * timestep;
    ocY1[index] += (diss_cYij1) * timestep;
    ocY2[index] += (diss_cYij2) * timestep;
    ocY3[index] += (diss_cYij3) * timestep;
    ocY4[index] += (diss_cYij4) * timestep;

    ocA0[index] += (diss_cAij0) * timestep;
    ocA1[index] += (diss_cAij1) * timestep;
    ocA2[index] += (diss_cAij2) * timestep;
    ocA3[index] += (diss_cAij3) * timestep;
    ocA4[index] += (diss_cAij4) * timestep;
    ocA5[index] += (diss_cAij5) * timestep;

    ocGi0[index] += (diss_cGi0) * timestep;
    ocGi1[index] += (diss_cGi1) * timestep;
    ocGi2[index] += (diss_cGi2) * timestep;

    oK[index] += (diss_K) * timestep;
    oX[index] += (diss_X) * timestep;

    ogA[index] += (diss_gA) * timestep;
    ogB0[index] += (diss_gB0) * timestep;
    ogB1[index] += (diss_gB1) * timestep;
    ogB2[index] += (diss_gB2) * timestep;
}
#endif // 0

__kernel
void dissipate_single(__global float* buffer, __global float* obuffer,
                      float coefficient,
                      float scale, int4 dim, float timestep)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    #ifndef SYMMETRY_BOUNDARY
    if(ix < BORDER_WIDTH*2 || ix >= dim.x - BORDER_WIDTH*2 - 1 || iy < BORDER_WIDTH*2 || iy >= dim.y - BORDER_WIDTH*2 - 1 || iz < BORDER_WIDTH*2 || iz >= dim.z - BORDER_WIDTH*2 - 1)
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
            float scale, int4 dim, __write_only image2d_t screen, float time)
{
    int ix = get_global_id(0);
    //int iy = get_global_id(1);
    //int iz = dim.z/2;
    int iy = dim.y/2;
    int iz = get_global_id(1);


    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    if(ix <= 4 || ix >= dim.x - 5 || iy <= 4 || iy >= dim.y - 5 || iz <= 4 || iz >= dim.z - 5)
        return;


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

    /*if(ix == 125 && iy == 125)
    {
        printf("scalar %f\n", max_scalar);
    }*/

    max_scalar = max_scalar * 40;

    max_scalar = clamp(max_scalar, 0.f, 1.f);

    float3 col = {max_scalar, max_scalar, max_scalar};

    float3 lin_col = srgb_to_lin(col);

    write_imagef(screen, (int2){get_global_id(0), get_global_id(1)}, (float4)(lin_col.xyz, 1));
    //write_imagef(screen, (int2){ix, iy}, (float4){max_scalar, max_scalar, max_scalar, 1});
}

#if 0
__kernel
void extract_waveform(__global float* cY0, __global float* cY1, __global float* cY2, __global float* cY3, __global float* cY4, __global float* cY5,
                      __global float* cA0, __global float* cA1, __global float* cA2, __global float* cA3, __global float* cA4, __global float* cA5,
                      __global float* cGi0, __global float* cGi1, __global float* cGi2, __global float* K, __global float* X, __global float* gA, __global float* gB0, __global float* gB1, __global float* gB2,
                      #ifdef USE_gBB0
                      __global float* gBB0, __global float* gBB1, __global float* gBB2,
                      #endif // USE_gBB0
                      float scale, int4 dim, __global struct intermediate_bssnok_data* temp_in, int4 pos, __global float2* waveform_out)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    if(ix != pos.x || iy != pos.y || iz != pos.z)
        return;

    float3 offset = transform_position(ix, iy, iz, dim, scale);

    /*float s = length(offset);
    float theta = acos(z / s);
    float phi = atan2(y, x);*/

    struct intermediate_bssnok_data ik = temp_in[IDX(ix, iy, iz)];

    float TEMPORARIES4;

    /*for(int i=0; i < TEMP_COUNT4; i++)
    {
        if(!isfinite(pv[i]))
        {
            printf("%i idx is not finite %f\n", i, pv[i]);
        }
    }*/

    int index = IDX(ix, iy, iz);

    //printf("Scale %f\n", scale);

    //printf("X %f\n", X[index]);

    waveform_out[0].x = w4_real;
    waveform_out[0].y = w4_complex;

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
    float current_acceleration_err = fast_length(next_acceleration) * 0.01f;

    float experienced_acceleration_change = current_acceleration_err;

    #define MAX_ACCELERATION_CHANGE 0.0001

    float err = MAX_ACCELERATION_CHANGE;
    float i_hate_computers = 256*256;

    //#define MIN_STEP 0.00001f
    //#define MIN_STEP 0.000001f
    #define MIN_STEP 0.1f

    float max_timestep = 100000;

    float diff = experienced_acceleration_change * i_hate_computers;

    if(diff < err * i_hate_computers / pow(max_timestep, 2))
        diff = err * i_hate_computers / pow(max_timestep, 2);

    ///of course, as is tradition, whatever works for kerr does not work for alcubierre
    ///the sqrt error calculation is significantly better for alcubierre, largely in terms of having no visual artifacts at all
    ///whereas the pow version is nearly 2x faster for kerr
    float next_ds = native_sqrt(((err * i_hate_computers) / diff));

    ///produces strictly worse results for kerr
    next_ds = 0.99f * current_ds * clamp(next_ds / current_ds, 0.3f, 2.f);

    next_ds = max(next_ds, MIN_STEP);

    *next_ds_out = next_ds;

    if(next_ds == MIN_STEP && (diff/i_hate_computers) > err * 10000)
        return DS_RETURN;

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

    float next_ds = 0.00001f;

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

        if(isnan(dV0) || isnan(dV1) || isnan(dV2))
        {
            break;
        }

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
