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

float finite_difference(float upper, float lower, float scale)
{
    return (upper - lower) / (2 * scale);
}

#define IDX(i, j, k) ((k) * dim.x * dim.y + (j) * dim.x + (i))

#define DIFFX(var) finite_difference(in[IDX(x+1, y, z)].var, in[IDX(x-1, y, z)].var, scale)
#define DIFFY(var) finite_difference(in[IDX(x, y+1, z)].var, in[IDX(x, y-1, z)].var, scale)
#define DIFFZ(var) finite_difference(in[IDX(x, y, z+1)].var, in[IDX(x, y, z-1)].var, scale)

#define DIFFV(v) {DIFFX(v), DIFFY(v), DIFFZ(v)}

#define DERIV_IDX(derivative_matrix, coordinate_idx, vector_idx) derivative_matrix[(coordinate_idx) * 3 + (vector_idx)]

#define DIFFXI(v, i) DIFFX(v##i)
#define DIFFYI(v, i) DIFFY(v##i)
#define DIFFZI(v, i) DIFFZ(v##i)

__kernel
void evolve(__global struct bssnok_data* in, __global struct bssnok_data* out, float scale, int4 dim)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);

    if(x >= dim.x || y >= dim.y || z >= dim.z)
        return;

    if(x == 0 || x == dim.x-1 || y == 0 || y == dim.y - 1 || z == 0 || z == dim.z - 1)
        return;

    float3 dtB = {0,0,0};
    float dtA = 0;

    float dB_full[9] = {
        DIFFX(gB0), DIFFX(gB1), DIFFX(gB2),
        DIFFY(gB0), DIFFY(gB1), DIFFY(gB2),
        DIFFZ(gB0), DIFFZ(gB1), DIFFZ(gB2),
    };

    struct bssnok_data v = in[IDX(x, y, z)];

    ///gauge B
    float gB[3] = {v.gB0, v.gB1, v.gB2};

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

    float Aij[6] = {v.cA0, v.cA1, v.cA2, v.cA3, v.cA4, v.cA5};

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
}
