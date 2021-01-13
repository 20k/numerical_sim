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
};

///for matrix with a unity determininant, which is ONLY cYij
void matrix_3x3_invert(float data[9], float out[9])
{
    float d = 1;

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

    float Yij[9] = {v.cY0, v.cY1, v.cY2,
                    v.cY1, v.cY3, v.cY4,
                    v.cY2, v.cY4, v.cY5};

    float iYij[9];

    matrix_3x3_invert(Yij, iYij);

    float dkYij[3 * 6] = {0};

    dkYij[0 * 3 + 0] = DIFFXI(cY, 0);
    dkYij[0 * 3 + 1] = DIFFXI(cY, 1);
    dkYij[0 * 3 + 2] = DIFFXI(cY, 2);
    dkYij[0 * 3 + 3] = DIFFXI(cY, 3);
    dkYij[0 * 3 + 4] = DIFFXI(cY, 4);
    dkYij[0 * 3 + 5] = DIFFXI(cY, 5);

    dkYij[1 * 3 + 0] = DIFFYI(cY, 0);
    dkYij[1 * 3 + 1] = DIFFYI(cY, 1);
    dkYij[1 * 3 + 2] = DIFFYI(cY, 2);
    dkYij[1 * 3 + 3] = DIFFYI(cY, 3);
    dkYij[1 * 3 + 4] = DIFFYI(cY, 4);
    dkYij[1 * 3 + 5] = DIFFYI(cY, 5);

    dkYij[2 * 3 + 0] = DIFFZI(cY, 0);
    dkYij[2 * 3 + 1] = DIFFZI(cY, 1);
    dkYij[2 * 3 + 2] = DIFFZI(cY, 2);
    dkYij[2 * 3 + 3] = DIFFZI(cY, 3);
    dkYij[2 * 3 + 4] = DIFFZI(cY, 4);
    dkYij[2 * 3 + 5] = DIFFZI(cY, 5);

    int index_table[3][3] = {{0, 1, 2},
                             {1, 3, 4},
                             {2, 4, 5}};

    float christoff_big[3 * 3 * 3] = {0};

    #pragma unroll
    for(int k=0; k < 3; k++)
    {
        #pragma unroll
        for(int i=0; i < 3; i++)
        {
            #pragma unroll
            for(int j=0; j < 3; j++)
            {
                float sum = 0;

                #pragma unroll
                for(int l=0; l < 3; l++)
                {
                    float g_inv = iYij[k * 3 + l];

                    int symmetric_index1 = index_table[j][l];
                    int symmetric_index2 = index_table[i][l];
                    int symmetric_index3 = index_table[i][j];

                    sum += g_inv * dkYij[i * 3 + symmetric_index1];
                    sum += g_inv * dkYij[j * 3 + symmetric_index2];
                    sum -= g_inv * dkYij[l * 3 + symmetric_index3];
                }

                christoff_big[k * 3 * 3 + i * 3 + j] = 0.5 * sum;
            }
        }
    }

    struct intermediate_bssnok_data* my_out = &out[IDX(x, y, z)];

    #pragma unroll
    for(int k=0; k < 3; k++)
    {
        my_out->christoffel[k * 3 + 0] = christoff_big[k * 3 * 3 + 0 * 3 + 0];
        my_out->christoffel[k * 3 + 1] = christoff_big[k * 3 * 3 + 0 * 3 + 1];
        my_out->christoffel[k * 3 + 2] = christoff_big[k * 3 * 3 + 0 * 3 + 2];
        my_out->christoffel[k * 3 + 3] = christoff_big[k * 3 * 3 + 1 * 3 + 1];
        my_out->christoffel[k * 3 + 4] = christoff_big[k * 3 * 3 + 1 * 3 + 2];
        my_out->christoffel[k * 3 + 5] = christoff_big[k * 3 * 3 + 2 * 3 + 2];
    }

    my_out->digA[0] = DIFFX(gA);
    my_out->digA[1] = DIFFY(gA);
    my_out->digA[2] = DIFFZ(gA);

    my_out->digB[0 * 3 + 0] = DIFFX(gB0);
    my_out->digB[1 * 3 + 0] = DIFFY(gB0);
    my_out->digB[2 * 3 + 0] = DIFFZ(gB0);

    my_out->digB[0 * 3 + 1] = DIFFX(gB1);
    my_out->digB[1 * 3 + 1] = DIFFY(gB1);
    my_out->digB[2 * 3 + 1] = DIFFZ(gB1);

    my_out->digB[0 * 3 + 2] = DIFFX(gB2);
    my_out->digB[1 * 3 + 2] = DIFFY(gB2);
    my_out->digB[2 * 3 + 2] = DIFFZ(gB2);
}

__kernel
void evolve(__global struct bssnok_data* in, __global struct bssnok_data* out, float scale, int4 dim, __global struct intermediate_bssnok_data* temp_in)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);

    if(x >= dim.x || y >= dim.y || z >= dim.z)
        return;

    if(x == 0 || x == dim.x-1 || y == 0 || y == dim.y - 1 || z == 0 || z == dim.z - 1)
        return;

    struct bssnok_data v = in[IDX(x, y, z)];
    struct intermediate_bssnok_data ik = temp_in[IDX(x, y, z)];

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

    /*float Rjk[9];

    for(int j=0; j < 3; j++)
    {
        for(int k=0; k < 3; k++)
        {
            float sum = 0;

            for(int i=0; i < 3; i++)
            {
                for(int p=0; p < 3; p++)
                {
                    int symmetric_index1 = index_table[j][k];
                    int symmetric_index2 = index_table[i][k];
                    int symmetric_index3 = index_table[i][p]; ///freely
                    int symmetric_index4 = index_table[j][k];
                    int symmetric_index5 = index_table[j][p];
                    int symmetric_index6 = index_table[i][k];

                    sum += dcGijk[i * 3 * 6 + ]
                }
            }
        }
    }*/

    #if 0
    float3 dtB = {0,0,0};
    float dtA = 0;

    float dB_full[9] = {
        DIFFX(gB0), DIFFX(gB1), DIFFX(gB2),
        DIFFY(gB0), DIFFY(gB1), DIFFY(gB2),
        DIFFZ(gB0), DIFFZ(gB1), DIFFZ(gB2),
    };


    ///gauge B
    float gB[3] = {v.gB0, v.gB1, v.gB2};
    float dX[3] = DIFFV(X);

    ///https://arxiv.org/pdf/1404.6523.pdf (4)
    {
        float N = 1.375;

        dtB.x = (3.f/4.f) * v.cGi0 - N * v.gB0;
        dtB.y = (3.f/4.f) * v.cGi1 - N * v.gB1;
        dtB.z = (3.f/4.f) * v.cGi2 - N * v.gB2;

        float3 dA = DIFFV(gA);

                     ///b^i di a
        dtA = (gB[0] * dA.x + gB[1] * dA.y + gB[2] * dA.z) - 2 * v.gA * v.K;
    }

    int2 indices[6] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};

    /**
    [0, 1, 2,
     1, 3, 4,
     2, 4, 5]*/
    ///y, x
    int index_table[3][3] = {{0, 1, 2},
                             {1, 3, 4},
                             {2, 4, 5}};

    float lie_cYij[9] = {0};
    float lie_cAij[9] = {0};

    float Yij[6] = {v.cY0, v.cY1, v.cY2, v.cY3, v.cY4, v.cY5};
    float Aij[6] = {v.cA0, v.cA1, v.cA2, v.cA3, v.cA4, v.cA5};

    float dkYij[6 * 3] = {0};
    float dkAij[6 * 3] = {0};

    dkAij[0 * 3 + 0] = DIFFXI(cA, 0);
    dkAij[0 * 3 + 1] = DIFFXI(cA, 1);
    dkAij[0 * 3 + 2] = DIFFXI(cA, 2);
    dkAij[0 * 3 + 3] = DIFFXI(cA, 3);
    dkAij[0 * 3 + 4] = DIFFXI(cA, 4);
    dkAij[0 * 3 + 5] = DIFFXI(cA, 5);

    dkAij[1 * 3 + 0] = DIFFYI(cA, 0);
    dkAij[1 * 3 + 1] = DIFFYI(cA, 1);
    dkAij[1 * 3 + 2] = DIFFYI(cA, 2);
    dkAij[1 * 3 + 3] = DIFFYI(cA, 3);
    dkAij[1 * 3 + 4] = DIFFYI(cA, 4);
    dkAij[1 * 3 + 5] = DIFFYI(cA, 5);

    dkAij[2 * 3 + 0] = DIFFZI(cA, 0);
    dkAij[2 * 3 + 1] = DIFFZI(cA, 1);
    dkAij[2 * 3 + 2] = DIFFZI(cA, 2);
    dkAij[2 * 3 + 3] = DIFFZI(cA, 3);
    dkAij[2 * 3 + 4] = DIFFZI(cA, 4);
    dkAij[2 * 3 + 5] = DIFFZI(cA, 5);

    dkYij[0 * 3 + 0] = DIFFXI(cY, 0);
    dkYij[0 * 3 + 1] = DIFFXI(cY, 1);
    dkYij[0 * 3 + 2] = DIFFXI(cY, 2);
    dkYij[0 * 3 + 3] = DIFFXI(cY, 3);
    dkYij[0 * 3 + 4] = DIFFXI(cY, 4);
    dkYij[0 * 3 + 5] = DIFFXI(cY, 5);

    dkYij[1 * 3 + 0] = DIFFYI(cY, 0);
    dkYij[1 * 3 + 1] = DIFFYI(cY, 1);
    dkYij[1 * 3 + 2] = DIFFYI(cY, 2);
    dkYij[1 * 3 + 3] = DIFFYI(cY, 3);
    dkYij[1 * 3 + 4] = DIFFYI(cY, 4);
    dkYij[1 * 3 + 5] = DIFFYI(cY, 5);

    dkYij[2 * 3 + 0] = DIFFZI(cY, 0);
    dkYij[2 * 3 + 1] = DIFFZI(cY, 1);
    dkYij[2 * 3 + 2] = DIFFZI(cY, 2);
    dkYij[2 * 3 + 3] = DIFFZI(cY, 3);
    dkYij[2 * 3 + 4] = DIFFZI(cY, 4);
    dkYij[2 * 3 + 5] = DIFFZI(cY, 5);

    #pragma unroll
    for(int i=0; i < 3; i++)
    {
        #pragma unroll
        for(int j=0; j < 3; j++)
        {
            float sum = 0;

            #pragma unroll
            for(int k=0; k < 3; k++)
            {
                int symmetric_index1 = index_table[i][j];

                float v1 = gB[k] * dkAij[k * 3 + symmetric_index1];

                float v2 = Aij[index_table[i][k]] * DERIV_IDX(dB_full, j, k);

                float v3 = Aij[index_table[k][j]] * DERIV_IDX(dB_full, i, k);

                float v4 = -(2.f/3.f) * Aij[index_table[i][j]] * DERIV_IDX(dB_full, k, k);

                sum += v1 + v2 + v3 + v4;
            }

            lie_cAij[i * 3 + j] = sum;
        }
    }

    #pragma unroll
    for(int i=0; i < 3; i++)
    {
        #pragma unroll
        for(int j=0; j < 3; j++)
        {
            float sum = 0;

            #pragma unroll
            for(int k=0; k < 3; k++)
            {
                int symmetric_index1 = index_table[i][j];

                float v1 = gB[k] * dkYij[k * 3 + symmetric_index1];

                float v2 = Yij[index_table[i][k]] * DERIV_IDX(dB_full, j, k);

                float v3 = Yij[index_table[k][j]] * DERIV_IDX(dB_full, i, k);

                float v4 = -(2.f/3.f) * Yij[index_table[i][j]] * DERIV_IDX(dB_full, k, k);

                sum += v1 + v2 + v3 + v4;
            }

            lie_cYij[i * 3 + j] = sum;
        }
    }

    float dtYij[9] = {0};

    #pragma unroll
    for(int i=0; i < 3; i++)
    {
        #pragma unroll
        for(int j=0; j < 3; j++)
        {
            dtYij[i * 3 + j] = 2 * v.gA + lie_cYij[i * 3 + j];
        }
    }

    float dtX = 0;

    for(int i=0; i < 3; i++)
    {
        dtX += (2.f/3.f) * v.X * (v.gA * v.K - DERIV_IDX(dB_full, i, i)) + gB[i] * dX[i];
    }
    #endif // 0
}
