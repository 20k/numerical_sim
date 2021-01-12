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

    struct bssnok_data v = in[IDX(x, y, z)];

    {
        ///https://arxiv.org/pdf/1404.6523.pdf (4)
        float N = 1.375;

        dtB.x = (3.f/4.f) * v.cGi0 - N * v.gB0;
        dtB.y = (3.f/4.f) * v.cGi1 - N * v.gB1;
        dtB.z = (3.f/4.f) * v.cGi2 - N * v.gB2;

        float3 dA = DIFFV(gA);

                     ///b^i di a
        dtA = (dB.x * dA.x + dB.y * dA.y + dB.z * dA.z) - 2 * v.gA * v.K;

        //float dAx = finite_difference(in[IDX(i-1, ])
    }
}
