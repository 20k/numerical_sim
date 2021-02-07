///https://arxiv.org/pdf/1404.6523.pdf
///Gauge evolution equations

//#define SYMMETRY_BOUNDARY

//#define USE_GBB

#define DIDX(x, y) data[x * 3 + y]

float determinant(float data[9])
{
    float a11 = DIDX(0, 0);
    float a12 = DIDX(0, 1);
    float a13 = DIDX(0, 2);

    float a21 = DIDX(1, 0);
    float a22 = DIDX(1, 1);
    float a23 = DIDX(1, 2);

    float a31 = DIDX(2, 0);
    float a32 = DIDX(2, 1);
    float a33 = DIDX(2, 2);

    return a11*a22*a33 + a21*a32*a13 + a31*a12*a23 - a11*a32*a23 - a31*a22*a13 - a21*a12*a33;
}

///for matrix with a unity determininant, which is ONLY cYij
void matrix_3x3_invert(float data[9], float out[9])
{
    float d = 1/determinant(data);

    float a11 = data[0 * 3 + 0];
    float a12 = data[0 * 3 + 1];
    float a13 = data[0 * 3 + 2];

    float a21 = data[1 * 3 + 0];
    float a22 = data[1 * 3 + 1];
    float a23 = data[1 * 3 + 2];

    float a31 = data[2 * 3 + 0];
    float a32 = data[2 * 3 + 1];
    float a33 = data[2 * 3 + 2];

    float x0 = (a22 * a33 - a23 * a32) * d;
    float y0 = (a13 * a32 - a12 * a33) * d;
    float z0 = (a12 * a23 - a13 * a22) * d;

    float x1 = (a23 * a31 - a21 * a33) * d;
    float y1 = (a11 * a33 - a13 * a31) * d;
    float z1 = (a13 * a21 - a11 * a23) * d;

    float x2 = (a21 * a32 - a22 * a31) * d;
    float y2 = (a12 * a31 - a11 * a32) * d;
    float z2 = (a11 * a22 - a12 * a21) * d;

    out[0 * 3 + 0] = x0;
    out[0 * 3 + 1] = y0;
    out[0 * 3 + 2] = z0;

    out[1 * 3 + 0] = x1;
    out[1 * 3 + 1] = y1;
    out[1 * 3 + 2] = z1;

    out[2 * 3 + 0] = x2;
    out[2 * 3 + 1] = y2;
    out[2 * 3 + 2] = z2;
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
    float dphi[3];
};

float finite_difference(float upper, float lower, float scale)
{
    return (upper - lower) / (2 * scale);
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

#define IDX(i, j, k) ((k) * dim.x * dim.y + (j) * dim.x + (i))

#define DIFFX(var) finite_difference(in[IDX(x+1, y, z)].var, in[IDX(x-1, y, z)].var, scale)
#define DIFFY(var) finite_difference(in[IDX(x, y+1, z)].var, in[IDX(x, y-1, z)].var, scale)
#define DIFFZ(var) finite_difference(in[IDX(x, y, z+1)].var, in[IDX(x, y, z-1)].var, scale)

#define INTERMEDIATE_DIFFX(var) finite_difference(temp_in[IDX(x+1, y, z)].var, temp_in[IDX(x-1, y, z)].var, scale)
#define INTERMEDIATE_DIFFY(var) finite_difference(temp_in[IDX(x, y+1, z)].var, temp_in[IDX(x, y-1, z)].var, scale)
#define INTERMEDIATE_DIFFZ(var) finite_difference(temp_in[IDX(x, y, z+1)].var, temp_in[IDX(x, y, z-1)].var, scale)

#define DIFFV(v) {DIFFX(v), DIFFY(v), DIFFZ(v)}

#define DERIV_IDX(derivative_matrix, coordinate_idx, vector_idx) derivative_matrix[(coordinate_idx) * 3 + (vector_idx)]

#define DIFFXI(v, i) DIFFX(v##i)
#define DIFFYI(v, i) DIFFY(v##i)
#define DIFFZI(v, i) DIFFZ(v##i)

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

float3 transform_position(int x, int y, int z, int4 dim, float scale)
{
    float3 centre = {dim.x/2, dim.y/2, dim.z/2};
    float3 pos = {x, y, z};

    float3 diff = pos - centre;

    //return diff;

    float len = length(diff);

    if(len == 0)
        return diff;

    if(len <= 0.0001)
        len = 0.0001;

    float real_len = len * scale;

    float edge = max(max(dim.x, dim.y), dim.z) * scale / 2;

    float r1 = 20.f;
    float r2 = edge - 32 * scale;
    float r3 = edge;

    float bulge_amount = 2;

    float r1b = r1 / bulge_amount;
    float r2b = r2;
    float r3b = r3;

    float rad = 0;

    if(real_len < r1)
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

    return diff * rad / len;
}

__kernel
void calculate_initial_conditions(__global float* cY0, __global float* cY1, __global float* cY2, __global float* cY3, __global float* cY4, __global float* cY5,
                                  __global float* cA0, __global float* cA1, __global float* cA2, __global float* cA3, __global float* cA4, __global float* cA5,
                                  __global float* cGi0, __global float* cGi1, __global float* cGi2, __global float* K, __global float* X, __global float* gA, __global float* gB0, __global float* gB1, __global float* gB2,
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

    float pv[TEMP_COUNT0] = {TEMPORARIES0};

    float conformal_factor = init_conformal_factor;

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

    float bl_conformal = init_bl_conformal;

    /*f->gA = 1/bl_conformal;
    f->gB0 = 1/bl_conformal;
    f->gB1 = 1/bl_conformal;
    f->gB2 = 1/bl_conformal;*/

    gA[index] = init_gA;
    gB0[index] = init_gB0;
    gB1[index] = init_gB1;
    gB2[index] = init_gB2;

    #ifdef USE_GBB
    f->gBB0 = init_gBB0;
    f->gBB1 = init_gBB1;
    f->gBB2 = init_gBB2;
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
void enforce_algebraic_constraints(__global float* cY0, __global float* cY1, __global float* cY2, __global float* cY3, __global float* cY4, __global float* cY5,
                                   __global float* cA0, __global float* cA1, __global float* cA2, __global float* cA3, __global float* cA4, __global float* cA5,
                                   __global float* cGi0, __global float* cGi1, __global float* cGi2, __global float* K, __global float* X, __global float* gA, __global float* gB0, __global float* gB1, __global float* gB2,
                                   float scale, int4 dim)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);

    if(x >= dim.x || y >= dim.y || z >= dim.z)
        return;

    float pv[TEMP_COUNT3] = {TEMPORARIES3};

    int index = IDX(x, y, z);

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
}

///https://en.wikipedia.org/wiki/Ricci_curvature#Definition_via_local_coordinates_on_a_smooth_manifold
__kernel
void calculate_intermediate_data(__global float* cY0, __global float* cY1, __global float* cY2, __global float* cY3, __global float* cY4, __global float* cY5,
                                 __global float* cA0, __global float* cA1, __global float* cA2, __global float* cA3, __global float* cA4, __global float* cA5,
                                 __global float* cGi0, __global float* cGi1, __global float* cGi2, __global float* K, __global float* X, __global float* gA, __global float* gB0, __global float* gB1, __global float* gB2,
                                 float scale, int4 dim, __global struct intermediate_bssnok_data* out)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);

    if(x >= dim.x || y >= dim.y || z >= dim.z)
        return;

    #ifndef SYMMETRY_BOUNDARY
    if(x < BORDER_WIDTH || x >= dim.x - BORDER_WIDTH || y < BORDER_WIDTH || y >= dim.y - BORDER_WIDTH || z < BORDER_WIDTH || z >= dim.z - BORDER_WIDTH)
        return;
    #endif // SYMMETRY_BOUNDARY

    struct intermediate_bssnok_data* my_out = &out[IDX(x, y, z)];

    float pv[TEMP_COUNT1] = {TEMPORARIES1};

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

    my_out->dphi[0] = init_dphi0;
    my_out->dphi[1] = init_dphi1;
    my_out->dphi[2] = init_dphi2;
}

float sponge_damp_coeff(float x, float y, float z, float scale, int4 dim, float time)
{
    float edge_half = scale * (dim.x/2);

    float sponge_r0 = scale * ((dim.x/2) - 16);
    //float sponge_r0 = edge_half/2;
    float sponge_r1 = scale * ((dim.x/2) - 4);

    float r = fast_length((float3){x, y, z}) * scale;

    r = clamp(r, sponge_r0, sponge_r1);

    float r_frac = (r - sponge_r0) / (sponge_r1 - sponge_r0);

    return r_frac;

    //return r_frac * pow(r_frac, fabs(sin(time / (2 * M_PI))));
}

///https://cds.cern.ch/record/517706/files/0106072.pdf
///boundary conditions
///todo: damp to schwarzschild, not initial conditions?
__kernel
void clean_data(__global float* cY0, __global float* cY1, __global float* cY2, __global float* cY3, __global float* cY4, __global float* cY5,
                __global float* cA0, __global float* cA1, __global float* cA2, __global float* cA3, __global float* cA4, __global float* cA5,
                __global float* cGi0, __global float* cGi1, __global float* cGi2, __global float* K, __global float* X, __global float* gA, __global float* gB0, __global float* gB1, __global float* gB2,
                __global struct intermediate_bssnok_data* iin, float scale, int4 dim, float time)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    #ifdef SYMMETRY_BOUNDARY
    return;
    #endif // SYMMETRY_BOUNDARY

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    //if(ix < BORDER_WIDTH*2 || ix >= dim.x - BORDER_WIDTH*2 || iy < BORDER_WIDTH*2 || iy >= dim.y - BORDER_WIDTH*2 || iz < BORDER_WIDTH*2 || iz >= dim.z - BORDER_WIDTH*2)
    {
        float3 offset = transform_position(ix, iy, iz, dim, scale);

        float ox = offset.x;
        float oy = offset.y;
        float oz = offset.z;

        float sponge_factor = sponge_damp_coeff(ox, oy, oz, scale, dim, time);

        if(sponge_factor <= 0)
            return;

        float pv[TEMP_COUNT0] = {TEMPORARIES0};

        float bl_conformal = init_bl_conformal;
        float conformal_factor = init_conformal_factor;

        /*float schwarzs_cY0 = schwarzs_init_cY0;
        float schwarzs_cY1 = schwarzs_init_cY1;
        float schwarzs_cY2 = schwarzs_init_cY2;
        float schwarzs_cY3 = schwarzs_init_cY3;
        float schwarzs_cY4 = schwarzs_init_cY4;
        float schwarzs_cY5 = schwarzs_init_cY5;

        float schwarzs_X = schwarzs_init_X;*/

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

        float fin_cY0 = initial_cY0;
        float fin_cY1 = initial_cY1;
        float fin_cY2 = initial_cY2;
        float fin_cY3 = initial_cY3;
        float fin_cY4 = initial_cY4;
        float fin_cY5 = initial_cY5;

        float fin_X = initial_X;

        float initial_error = 0;
        float schwarzs_error = 0;

        /*initial_error += fabs(initial_cY0 - v.cY0);
        initial_error += fabs(initial_cY1 - v.cY1);
        initial_error += fabs(initial_cY2 - v.cY2);
        initial_error += fabs(initial_cY3 - v.cY3);
        initial_error += fabs(initial_cY4 - v.cY4);
        initial_error += fabs(initial_cY5 - v.cY5);
        initial_error += fabs(initial_X - v.X);

        schwarzs_error += fabs(schwarzs_cY0 - v.cY0);
        schwarzs_error += fabs(schwarzs_cY1 - v.cY1);
        schwarzs_error += fabs(schwarzs_cY2 - v.cY2);
        schwarzs_error += fabs(schwarzs_cY3 - v.cY3);
        schwarzs_error += fabs(schwarzs_cY4 - v.cY4);
        schwarzs_error += fabs(schwarzs_cY5 - v.cY5);
        schwarzs_error += fabs(schwarzs_X - v.X);

        if(schwarzs_error < initial_error)
        {
            fin_cY0 = schwarzs_cY0;
            fin_cY1 = schwarzs_cY1;
            fin_cY2 = schwarzs_cY2;
            fin_cY3 = schwarzs_cY3;
            fin_cY4 = schwarzs_cY4;
            fin_cY5 = schwarzs_cY5;
            fin_X = schwarzs_X;
        }*/

        /*float time_frac = time / 6.f;

        time_frac = clamp(time_frac, 0.f, 1.f);

        fin_cY0 = mix(fin_cY0, schwarzs_cY0, time_frac);
        fin_cY1 = mix(fin_cY1, schwarzs_cY1, time_frac);
        fin_cY2 = mix(fin_cY2, schwarzs_cY2, time_frac);
        fin_cY3 = mix(fin_cY3, schwarzs_cY3, time_frac);
        fin_cY4 = mix(fin_cY4, schwarzs_cY4, time_frac);
        fin_cY5 = mix(fin_cY5, schwarzs_cY5, time_frac);
        fin_X = mix(fin_X, schwarzs_X, time_frac);*/

        int index = IDX(ix, iy, iz);

        cY0[index] = mix(cY0[index],fin_cY0, sponge_factor);
        cY1[index] = mix(cY1[index],fin_cY1, sponge_factor);
        cY2[index] = mix(cY2[index],fin_cY2, sponge_factor);
        cY3[index] = mix(cY3[index],fin_cY3, sponge_factor);
        cY4[index] = mix(cY4[index],fin_cY4, sponge_factor);
        cY5[index] = mix(cY5[index],fin_cY5, sponge_factor);

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
        X[index] = mix(X[index],fin_X, sponge_factor);

        gA[index] = mix(gA[index],init_gA, sponge_factor);
        gB0[index] = mix(gB0[index],init_gB0, sponge_factor);
        gB1[index] = mix(gB1[index],init_gB1, sponge_factor);
        gB2[index] = mix(gB2[index],init_gB2, sponge_factor);

        #ifdef USE_GBB
        out.gBB0 = init_gBB0;
        out.gBB1 = init_gBB1;
        out.gBB2 = init_gBB2;
        #endif // USE_GBB

        /*v.gA = 1;
        v.gB0 = 0;
        v.gB1 = 0;
        v.gB2 = 0;*/

        /*v.cY0 = 1;
        v.cY1 = 0;
        v.cY2 = 0;
        v.cY3 = 1;
        v.cY4 = 0;
        v.cY5 = 1;

        v.cA0 = 0;
        v.cA1 = 0;
        v.cA2 = 0;
        v.cA3 = 0;
        v.cA4 = 0;
        v.cA5 = 0;

        v.cGi0 = 0;
        v.cGi1 = 0;
        v.cGi2 = 0;

        v.X = 1;
        v.K = 0;
        v.gA = 0;
        v.gB0 = 0;
        v.gB1 = 0;
        v.gB2 = 0;*/
    }
}

#define NANCHECK(w) if(isnan(my_out->w)){printf("NAN " #w " %i %i %i\n", x, y, z); debug = true;}

///todo: need to correctly evolve boundaries
///todo: need to factor out the differentials
__kernel
void evolve(__global float* cY0, __global float* cY1, __global float* cY2, __global float* cY3, __global float* cY4, __global float* cY5,
            __global float* cA0, __global float* cA1, __global float* cA2, __global float* cA3, __global float* cA4, __global float* cA5,
            __global float* cGi0, __global float* cGi1, __global float* cGi2, __global float* K, __global float* X, __global float* gA, __global float* gB0, __global float* gB1, __global float* gB2,
            __global float* ocY0, __global float* ocY1, __global float* ocY2, __global float* ocY3, __global float* ocY4, __global float* ocY5,
            __global float* ocA0, __global float* ocA1, __global float* ocA2, __global float* ocA3, __global float* ocA4, __global float* ocA5,
            __global float* ocGi0, __global float* ocGi1, __global float* ocGi2, __global float* oK, __global float* oX, __global float* ogA, __global float* ogB0, __global float* ogB1, __global float* ogB2,
            float scale, int4 dim, __global const struct intermediate_bssnok_data* temp_in, float timestep, float time)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);

    if(x >= dim.x || y >= dim.y || z >= dim.z)
        return;

    #ifndef SYMMETRY_BOUNDARY
    if(x < BORDER_WIDTH*2 || x >= dim.x - BORDER_WIDTH*2 || y < BORDER_WIDTH*2 || y >= dim.y - BORDER_WIDTH*2 || z < BORDER_WIDTH*2 || z >= dim.z - BORDER_WIDTH*2)
        return;
    #endif // SYMMETRY_BOUNDARY

    float3 transform_pos = transform_position(x, y, z, dim, scale);
    float sponge_factor = sponge_damp_coeff(transform_pos.x, transform_pos.y, transform_pos.z, scale, dim, time);

    //if(sponge_factor == 1)
    //    return;

    float3 centre = {dim.x/2, dim.y/2, dim.z/2};
    float r = fast_length((float3){x, y, z} - centre);

    struct intermediate_bssnok_data ik = temp_in[IDX(x, y, z)];

    float pv[TEMP_COUNT2] = {TEMPORARIES2};

    int index_table[3][3] = {{0, 1, 2},
                             {1, 3, 4},
                             {2, 4, 5}};

    ///conformal christoffel derivatives
    /*float dcGijk[3 * 3 * 6];

    #pragma unroll
    for(int i=0; i < 3 * 6; i++)
    {
        dcGijk[0 * 3 * 6 + i] = INTERMEDIATE_DIFFX(christoffel[i]);
        dcGijk[1 * 3 * 6 + i] = INTERMEDIATE_DIFFY(christoffel[i]);
        dcGijk[2 * 3 * 6 + i] = INTERMEDIATE_DIFFZ(christoffel[i]);
    }*/

    int index = IDX(x, y, z);

    ocY0[index] = cY0[index] + dtcYij0 * timestep;
    ocY1[index] = cY1[index] + dtcYij1 * timestep;
    ocY2[index] = cY2[index] + dtcYij2 * timestep;
    ocY3[index] = cY3[index] + dtcYij3 * timestep;
    ocY4[index] = cY4[index] + dtcYij4 * timestep;
    ocY5[index] = cY5[index] + dtcYij5 * timestep;

    ocA0[index] = cA0[index] + dtcAij0 * timestep;
    ocA1[index] = cA1[index] + dtcAij1 * timestep;
    ocA2[index] = cA2[index] + dtcAij2 * timestep;
    ocA3[index] = cA3[index] + dtcAij3 * timestep;
    ocA4[index] = cA4[index] + dtcAij4 * timestep;
    ocA5[index] = cA5[index] + dtcAij5 * timestep;

    ocGi0[index] = cGi0[index] + dtcGi0 * timestep;
    ocGi1[index] = cGi1[index] + dtcGi1 * timestep;
    ocGi2[index] = cGi2[index] + dtcGi2 * timestep;

    oK[index] = K[index] + dtK * timestep;
    oX[index] = X[index] + dtX * timestep;

    ogA[index] = gA[index] + dtgA * timestep;
    ogB0[index] = gB0[index] + dtgB0 * timestep;
    ogB1[index] = gB1[index] + dtgB1 * timestep;
    ogB2[index] = gB2[index] + dtgB2 * timestep;

    #ifdef USE_GBB
    my_out->gBB0 = v.gBB0 + dtgBB0 * timestep;
    my_out->gBB1 = v.gBB1 + dtgBB1 * timestep;
    my_out->gBB2 = v.gBB2 + dtgBB2 * timestep;
    #endif // USE_GBB

    /*bool debug = false;

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
    NANCHECK(gBB0);
    NANCHECK(gBB1);
    NANCHECK(gBB2);
    #endif // USE_GBB*/

    #if 0
    //if(debug)
    //if(x == 5 && y == 6 && z == 4)
    if(x == 125 && y == 100 && z == 125)
    //if(x == 125 && y == 125 && z == 125)
    {
        float scalar = scalar_curvature;

        printf("DtY0 %f\n", dtcYij0);
        printf("DtA0 %f\n", dtcAij0);
        printf("Aij0 %f\n", v.cA0);
        printf("Aij1 %f\n", v.cA1);
        printf("Aij2 %f\n", v.cA2);
        printf("Aij3 %f\n", v.cA3);
        printf("Aij4 %f\n", v.cA4);
        printf("Aij5 %f\n", v.cA5);
        printf("Yij0 %f\n", v.cY0);
        printf("Yij1 %f\n", v.cY1);
        printf("Yij2 %f\n", v.cY2);
        printf("Yij3 %f\n", v.cY3);
        printf("Yij4 %f\n", v.cY4);
        printf("Yij5 %f\n", v.cY5);
        printf("cGi0 %f\n", v.cGi0);
        printf("cGi1 %f\n", v.cGi1);
        printf("cGi2 %f\n", v.cGi2);
        printf("X %f\n", v.X);
        printf("K %f\n", v.K);
        printf("gA %f\n", v.gA);
        printf("gB0 %f\n", v.gB0);
        printf("gB1 %f\n", v.gB1);
        printf("gB2 %f\n", v.gB2);
        printf("Scalar %f\n", scalar);

        #ifdef debug_val
        float dbg = debug_val;
        printf("Debug %f\n", debug_val);
        #endif // debug_val

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

__kernel
void render(__global float* cY0, __global float* cY1, __global float* cY2, __global float* cY3, __global float* cY4, __global float* cY5,
            __global float* cA0, __global float* cA1, __global float* cA2, __global float* cA3, __global float* cA4, __global float* cA5,
            __global float* cGi0, __global float* cGi1, __global float* cGi2, __global float* K, __global float* X, __global float* gA, __global float* gB0, __global float* gB1, __global float* gB2,
            float scale, int4 dim, __global struct intermediate_bssnok_data* temp_in, __write_only image2d_t screen, float time)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x >= dim.x || y >= dim.y)
        return;

    if(x <= 4 || x >= dim.x - 5 || y <= 4 || y >= dim.y - 5)
        return;



    int index_table[3][3] = {{0, 1, 2},
                             {1, 3, 4},
                             {2, 4, 5}};

    float max_scalar = 0;

    //for(int z = 20; z < dim.z-20; z++)

    int z = dim.z/2;
    {
        ///conformal christoffel derivatives
        /*float dcGijk[3 * 3 * 6];

        #pragma unroll
        for(int i=0; i < 3 * 6; i++)
        {
            dcGijk[0 * 3 * 6 + i] = INTERMEDIATE_DIFFX(christoffel[i]);
            dcGijk[1 * 3 * 6 + i] = INTERMEDIATE_DIFFY(christoffel[i]);
            dcGijk[2 * 3 * 6 + i] = INTERMEDIATE_DIFFZ(christoffel[i]);
        }*/

        float3 transform_pos = transform_position(x, y, z, dim, scale);
        float sponge_factor = sponge_damp_coeff(transform_pos.x, transform_pos.y, transform_pos.z, scale, dim, time);

        if(sponge_factor > 0)
        {
            write_imagef(screen, (int2){x, y}, (float4)(sponge_factor, 0, 0, 1));
            return;
        }

        struct intermediate_bssnok_data ik = temp_in[IDX(x, y, z)];

        ///reuses the evolve parameters
        float pv[TEMP_COUNT2] = {TEMPORARIES2};

        int index = IDX(x, y, z);

        //float curvature = scalar_curvature;

        /*if(x == 3 && y == 125)
        {
            printf("Ik %f\n", ik.Yij[0]);
        }*/

        float curvature = (fabs(cY0[index]/X[index]) + fabs(cY1[index]/X[index]) + fabs(cY2[index]/X[index]) + fabs(cY3[index]/X[index]) + fabs(cY4[index]/X[index]) + fabs(cY5[index]/X[index])) / 1000.f;

        //float curvature = (fabs(v.Yij[0]) + fabs(ik.Yij[1]) + fabs(ik.Yij[2]) + fabs(ik.Yij[3]) + fabs(ik.Yij[4]) + fabs(ik.Yij[5])) / 1000.;
        //float curvature = v.cY0 + v.cY1 + v.cY2 + v.cY3 + v.cY4 + v.cY5;

        float ascalar = fabs(curvature);

        max_scalar = max(ascalar, max_scalar);
    }

    if(x == 125 && y == 125)
    {
        //printf("scalar %f\n", max_scalar);
    }

    max_scalar = max_scalar * 10;

    max_scalar = clamp(max_scalar, 0.f, 1.f);

    write_imagef(screen, (int2){x, y}, (float4){max_scalar, max_scalar, max_scalar, 1});
}

__kernel
void extract_waveform(__global float* cY0, __global float* cY1, __global float* cY2, __global float* cY3, __global float* cY4, __global float* cY5,
                      __global float* cA0, __global float* cA1, __global float* cA2, __global float* cA3, __global float* cA4, __global float* cA5,
                      __global float* cGi0, __global float* cGi1, __global float* cGi2, __global float* K, __global float* X, __global float* gA, __global float* gB0, __global float* gB1, __global float* gB2,
                     float scale, int4 dim, __global struct intermediate_bssnok_data* temp_in, int4 pos, __global float2* waveform_out)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);

    if(x >= dim.x || y >= dim.y || z >= dim.z)
        return;

    if(x != pos.x || y != pos.y || z != pos.z)
        return;

    float3 offset = transform_position(x, y, z, dim, scale);

    /*float s = length(offset);
    float theta = acos(z / s);
    float phi = atan2(y, x);*/

    struct intermediate_bssnok_data ik = temp_in[IDX(x, y, z)];

    float pv[TEMP_COUNT4] = {TEMPORARIES4};

    /*for(int i=0; i < TEMP_COUNT4; i++)
    {
        if(!isfinite(pv[i]))
        {
            printf("%i idx is not finite %f\n", i, pv[i]);
        }
    }*/

    int index = IDX(x,y,z);

    printf("Scale %f\n", scale);

    printf("X %f\n", X[index]);

    waveform_out[0].x = w4_real;
    waveform_out[0].y = w4_complex;

    printf("WAV %f\n", waveform_out[0].x);

    #ifdef w4_debugr
    printf("Debugw4r %f\n", w4_debugr);
    #endif // w4_debug

    #ifdef w4_debugi
    printf("Debugw4i %f\n", w4_debugi);
    #endif // w4_debug
}
