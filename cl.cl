///https://arxiv.org/pdf/1404.6523.pdf
///Gauge evolution equations

struct bssnok_data
{
    /**
    conformal
    [0, 1, 2,
     X, 3, 4,
     X, X, 5]
    */
    float cY0, cY1, cY2, cY3, cY4, cY5;

    /**
    conformal
    [0, 1, 2,
     X, 3, 4,
     X, X, 5]
    */
    float cA0, cA1, cA2, cA3, cA4, cA5;

    float cGi0, cGi1, cGi2;

    float K;
    float X;

    float gA;
    float gB0;
    float gB1;
    float gB2;
    float gBB0;
    float gBB1;
    float gBB2;
};

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
struct intermediate_bssnok_data
{
    ///christoffel symbols are symmetric in the lower 2 indices
    float christoffel[3 * 6];
    float digA[3];
    float digB[3*3];
    float phi;
    float dphi[3];
    float Yij[6];
};

float finite_difference(float upper, float lower, float scale)
{
    return (upper - lower) / (2 * scale);
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

__kernel
void calculate_initial_conditions(__global struct bssnok_data* in, float scale, int4 dim)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    struct bssnok_data* f = &in[IDX(ix, iy, iz)];

    //float pv[TEMP_COUNT] = {TEMPORARIES};

    float x = ix;
    float y = iy;
    float z = iz;

    float conformal_factor = init_conformal_factor;

    f->cY0 = init_cY0;
    f->cY1 = init_cY1;
    f->cY2 = init_cY2;
    f->cY3 = init_cY3;
    f->cY4 = init_cY4;
    f->cY5 = init_cY5;

    f->cA0 = init_cA0;
    f->cA1 = init_cA1;
    f->cA2 = init_cA2;
    f->cA3 = init_cA3;
    f->cA4 = init_cA4;
    f->cA5 = init_cA5;

    f->cGi0 = init_cGi0;
    f->cGi1 = init_cGi1;
    f->cGi2 = init_cGi2;

    f->K = init_K;
    f->X = init_X;

    float bl_conformal = init_bl_conformal;

    /*f->gA = 1/bl_conformal;
    f->gB0 = 1/bl_conformal;
    f->gB1 = 1/bl_conformal;
    f->gB2 = 1/bl_conformal;*/

    f->gA = init_gA;
    f->gB0 = init_gB0;
    f->gB1 = init_gB1;
    f->gB2 = init_gB2;

    f->gBB0 = init_gBB0;
    f->gBB1 = init_gBB1;
    f->gBB2 = init_gBB2;

    //f->gBB0 = init_gBB0;
    //f->gBB1 = init_gBB1;
    //f->gBB2 = init_gBB2;

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

///https://en.wikipedia.org/wiki/Ricci_curvature#Definition_via_local_coordinates_on_a_smooth_manifold
__kernel
void calculate_intermediate_data(__global struct bssnok_data* in, float scale, int4 dim, __global struct intermediate_bssnok_data* out)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);

    if(x >= dim.x || y >= dim.y || z >= dim.z)
        return;

    if(x == 0 || x == dim.x-1 || y == 0 || y == dim.y - 1 || z == 0 || z == dim.z - 1)
        return;

    struct bssnok_data v = in[IDX(x, y, z)];

    struct intermediate_bssnok_data* my_out = &out[IDX(x, y, z)];

    float pv[TEMP_COUNT1] = {TEMPORARIES1};

    my_out->christoffel[0] = init_christoffel0;
    my_out->christoffel[1] = init_christoffel1;
    my_out->christoffel[2] = init_christoffel2;
    my_out->christoffel[3] = init_christoffel3;
    my_out->christoffel[4] = init_christoffel4;
    my_out->christoffel[5] = init_christoffel5;
    my_out->christoffel[6] = init_christoffel6;
    my_out->christoffel[7] = init_christoffel7;
    my_out->christoffel[8] = init_christoffel8;
    my_out->christoffel[9] = init_christoffel9;
    my_out->christoffel[10] = init_christoffel10;
    my_out->christoffel[11] = init_christoffel11;
    my_out->christoffel[12] = init_christoffel12;
    my_out->christoffel[13] = init_christoffel13;
    my_out->christoffel[14] = init_christoffel14;
    my_out->christoffel[15] = init_christoffel15;
    my_out->christoffel[16] = init_christoffel16;
    my_out->christoffel[17] = init_christoffel17;

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

    my_out->phi = init_phi;

    my_out->dphi[0] = init_dphi0;
    my_out->dphi[1] = init_dphi1;
    my_out->dphi[2] = init_dphi2;

    my_out->Yij[0] = init_Yij0;
    my_out->Yij[1] = init_Yij1;
    my_out->Yij[2] = init_Yij2;
    my_out->Yij[3] = init_Yij3;
    my_out->Yij[4] = init_Yij4;
    my_out->Yij[5] = init_Yij5;
}

///https://cds.cern.ch/record/517706/files/0106072.pdf
///boundary conditions
__kernel
void clean_data(__global struct bssnok_data* in, __global struct intermediate_bssnok_data* iin, int4 dim)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    if(ix <= 1 || ix >= dim.x - 2 || iy <= 1 || iy >= dim.y - 2 || iz <= 1 || iz >= dim.z - 2)
    {
        int xdir = 0;
        int ydir = 0;
        int zdir = 0;

        if(ix == 0)
            xdir = 2;
        if(ix == 1)
            xdir = 2;

        if(ix == dim.x - 1)
            xdir = -2;
        if(ix == dim.x - 2)
            xdir = -2;

        if(iy == 0)
            ydir = 2;
        if(iy == 1)
            ydir = 2;

        if(iy == dim.y - 1)
            ydir = -2;
        if(iy == dim.y - 2)
            ydir = -2;

        if(iz == 0)
            zdir = 2;
        if(iz == 1)
            zdir = 2;

        if(iz == dim.z - 1)
            zdir = -2;
        if(iz == dim.z - 2)
            zdir = -2;

        if(xdir == 0 && ydir == 0 && zdir == 0)
            return;

        /*in[IDX(x, y, z)] = in[IDX(x + xdir, y + ydir, z + zdir)];*/

        struct bssnok_data v = in[IDX(ix, iy, iz)];
        //struct bssnok_data o = v;
        struct bssnok_data o = in[IDX(ix + xdir, iy + ydir, iz + zdir)];

        float x = ix;
        float y = iy;
        float z = iz;

        float conformal_factor = init_conformal_factor;

        v.cY0 = init_cY0;
        v.cY1 = init_cY1;
        v.cY2 = init_cY2;
        v.cY3 = init_cY3;
        v.cY4 = init_cY4;
        v.cY5 = init_cY5;

        v.cA0 = init_cA0;
        v.cA1 = init_cA1;
        v.cA2 = init_cA2;
        v.cA3 = init_cA3;
        v.cA4 = init_cA4;
        v.cA5 = init_cA5;

        v.cGi0 = init_cGi0;
        v.cGi1 = init_cGi1;
        v.cGi2 = init_cGi2;

        v.K = init_K;
        v.X = init_X;

        float bl_conformal = init_bl_conformal;

        v.gA = init_gA;
        v.gB0 = init_gB0;
        v.gB1 = init_gB1;
        v.gB2 = init_gB2;

        /*v.gA = 1;
        v.gB0 = 0;
        v.gB1 = 0;
        v.gB2 = 0;*/

        float factor = 0;//0.25;

        v.cY0 = mix(v.cY0, o.cY0, factor);
        v.cY1 = mix(v.cY1, o.cY1, factor);
        v.cY2 = mix(v.cY2, o.cY2, factor);
        v.cY3 = mix(v.cY3, o.cY3, factor);
        v.cY4 = mix(v.cY4, o.cY4, factor);
        v.cY5 = mix(v.cY5, o.cY5, factor);

        v.cA0 = mix(v.cA0, o.cA0, factor);
        v.cA1 = mix(v.cA1, o.cA1, factor);
        v.cA2 = mix(v.cA2, o.cA2, factor);
        v.cA3 = mix(v.cA3, o.cA3, factor);
        v.cA4 = mix(v.cA4, o.cA4, factor);
        v.cA5 = mix(v.cA5, o.cA5, factor);

        v.cGi0 = mix(v.cGi0, o.cGi0, factor);
        v.cGi1 = mix(v.cGi1, o.cGi1, factor);
        v.cGi2 = mix(v.cGi2, o.cGi2, factor);

        v.K = mix(v.K, o.K, factor);
        v.X = mix(v.X, o.X, factor);

        //float bl_conformal = init_bl_conformal;

        v.gA = mix(v.gA, o.gA, factor);
        v.gB0 = mix(v.gB0, o.gB0, factor);
        v.gB1 = mix(v.gB1, o.gB1, factor);
        v.gB2 = mix(v.gB2, o.gB2, factor);
        v.gBB0 = mix(v.gBB0, o.gBB0, factor);
        v.gBB1 = mix(v.gBB1, o.gBB1, factor);
        v.gBB2 = mix(v.gBB2, o.gBB2, factor);

        in[IDX(ix, iy, iz)] = v;

        //iin[IDX(x, y, z)] = iin[IDX(x + xdir, y + ydir, z + zdir)];
    }
}

///todo: need to correctly evolve boundaries
///todo: need to factor out the differentials
__kernel
void evolve(__global const struct bssnok_data* restrict in, __global struct bssnok_data* restrict out, float scale, int4 dim, __global const struct intermediate_bssnok_data* temp_in, float timestep)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);

    if(x >= dim.x || y >= dim.y || z >= dim.z)
        return;

    if(x <= 1 || x >= dim.x - 2 || y <= 1 || y >= dim.y - 2 || z <= 1 || z >= dim.z - 2)
        return;

    float3 centre = {dim.x/2, dim.y/2, dim.z/2};
    float r = fast_length((float3){x, y, z} - centre);

    struct bssnok_data v = in[IDX(x, y, z)];
    struct intermediate_bssnok_data ik = temp_in[IDX(x, y, z)];

    float pv[TEMP_COUNT2] = {TEMPORARIES2};

    int index_table[3][3] = {{0, 1, 2},
                             {1, 3, 4},
                             {2, 4, 5}};

    ///conformal christoffel derivatives
    float dcGijk[3 * 3 * 6];

    #pragma unroll
    for(int i=0; i < 3 * 6; i++)
    {
        dcGijk[0 * 3 * 6 + i] = INTERMEDIATE_DIFFX(christoffel[i]);
        dcGijk[1 * 3 * 6 + i] = INTERMEDIATE_DIFFY(christoffel[i]);
        dcGijk[2 * 3 * 6 + i] = INTERMEDIATE_DIFFZ(christoffel[i]);
    }

    struct bssnok_data* my_out = &out[IDX(x, y, z)];

    my_out->cY0 = v.cY0 + dtcYij0 * timestep;
    my_out->cY1 = v.cY1 + dtcYij1 * timestep;
    my_out->cY2 = v.cY2 + dtcYij2 * timestep;
    my_out->cY3 = v.cY3 + dtcYij3 * timestep;
    my_out->cY4 = v.cY4 + dtcYij4 * timestep;
    my_out->cY5 = v.cY5 + dtcYij5 * timestep;

    my_out->cA0 = v.cA0 + dtcAij0 * timestep;
    my_out->cA1 = v.cA1 + dtcAij1 * timestep;
    my_out->cA2 = v.cA2 + dtcAij2 * timestep;
    my_out->cA3 = v.cA3 + dtcAij3 * timestep;
    my_out->cA4 = v.cA4 + dtcAij4 * timestep;
    my_out->cA5 = v.cA5 + dtcAij5 * timestep;

    my_out->cGi0 = v.cGi0 + dtcGi0 * timestep;
    my_out->cGi1 = v.cGi1 + dtcGi1 * timestep;
    my_out->cGi2 = v.cGi2 + dtcGi2 * timestep;

    my_out->K = v.K + dtK * timestep;
    my_out->X = v.X + dtX * timestep;

    my_out->gA = v.gA + dtgA * timestep;
    my_out->gB0 = v.gB0 + dtgB0 * timestep;
    my_out->gB1 = v.gB1 + dtgB1 * timestep;
    my_out->gB2 = v.gB2 + dtgB2 * timestep;

    my_out->gBB0 = v.gBB0 + dtgBB0 * timestep;
    my_out->gBB1 = v.gBB1 + dtgBB1 * timestep;
    my_out->gBB2 = v.gBB2 + dtgBB2 * timestep;


    #if 1
    if(z == 125 && x == 2 && y == 125)
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

        float dbg = debug_val;

        printf("Debug %f\n", debug_val);

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
void render(__global struct bssnok_data* in, float scale, int4 dim, __global struct intermediate_bssnok_data* temp_in, __write_only image2d_t screen)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x >= dim.x || y >= dim.y)
        return;

    if(x <= 2 || x >= dim.x - 3 || y <= 2 || y >= dim.y - 3)
        return;


    int index_table[3][3] = {{0, 1, 2},
                             {1, 3, 4},
                             {2, 4, 5}};

    float max_scalar = 0;

    //for(int z = 20; z < dim.z-20; z++)

    int z = dim.z/2;
    {
        ///conformal christoffel derivatives
        float dcGijk[3 * 3 * 6];

        #pragma unroll
        for(int i=0; i < 3 * 6; i++)
        {
            dcGijk[0 * 3 * 6 + i] = INTERMEDIATE_DIFFX(christoffel[i]);
            dcGijk[1 * 3 * 6 + i] = INTERMEDIATE_DIFFY(christoffel[i]);
            dcGijk[2 * 3 * 6 + i] = INTERMEDIATE_DIFFZ(christoffel[i]);
        }

        struct bssnok_data v = in[IDX(x, y, z)];
        struct intermediate_bssnok_data ik = temp_in[IDX(x, y, z)];

        ///reuses the evolve parameters
        float pv[TEMP_COUNT2] = {TEMPORARIES2};

        //float curvature = scalar_curvature;

        /*if(x == 3 && y == 125)
        {
            printf("Ik %f\n", ik.Yij[0]);
        }*/

        float curvature = (ik.Yij[0] + ik.Yij[1] + ik.Yij[2] + ik.Yij[3] + ik.Yij[4] + ik.Yij[5]) / 100000.;
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
