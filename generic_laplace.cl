#include "common.cl"

void laplace_interior(__global float* buffer_in, __global float* buffer_out, float h2f0, int ix, int iy, int iz, float scale, int4 dim, __global int* still_going, float etol)
{
    bool x_degenerate = ix < 2 || ix >= dim.x - 2;
    bool y_degenerate = iy < 2 || iy >= dim.y - 2;
    bool z_degenerate = iz < 2 || iz >= dim.z - 2;

    float u0n1 = 0;

    if(x_degenerate || y_degenerate || z_degenerate)
    {
        float uxm1 = buffer_in[IDX(ix-1, iy, iz)];
        float uxp1 = buffer_in[IDX(ix+1, iy, iz)];
        float uym1 = buffer_in[IDX(ix, iy-1, iz)];
        float uyp1 = buffer_in[IDX(ix, iy+1, iz)];
        float uzm1 = buffer_in[IDX(ix, iy, iz-1)];
        float uzp1 = buffer_in[IDX(ix, iy, iz+1)];

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
        u0n1 = (1/6.f) * (Xs + Ys + Zs - h2f0);
    }
    else
    {
        float coeff1 = 4.f/3.f;
        float coeff2 = -1.f/12.f;
        float coeff_center = -5.f/2.f;

        float uxm1 = coeff1 * buffer_in[IDX(ix-1, iy, iz)];
        float uxp1 = coeff1 * buffer_in[IDX(ix+1, iy, iz)];
        float uym1 = coeff1 * buffer_in[IDX(ix, iy-1, iz)];
        float uyp1 = coeff1 * buffer_in[IDX(ix, iy+1, iz)];
        float uzm1 = coeff1 * buffer_in[IDX(ix, iy, iz-1)];
        float uzp1 = coeff1 * buffer_in[IDX(ix, iy, iz+1)];

        float uxm2 = coeff2 * buffer_in[IDX(ix-2, iy, iz)];
        float uxp2 = coeff2 * buffer_in[IDX(ix+2, iy, iz)];
        float uym2 = coeff2 * buffer_in[IDX(ix, iy-2, iz)];
        float uyp2 = coeff2 * buffer_in[IDX(ix, iy+2, iz)];
        float uzm2 = coeff2 * buffer_in[IDX(ix, iy, iz-2)];
        float uzp2 = coeff2 * buffer_in[IDX(ix, iy, iz+2)];

        ///so, floating point maths isn't associative
        ///which means that if we're on the other side of a symmetric boundary about the central plane
        ///the order of operations will be different
        ///the if statements correct this, which makes this method numerically symmetric, and implicitly
        ///converges to a symmetric solution if available
        float Xs1 = uxm1 + uxp1;
        float Xs2 = uxm2 + uxp2;
        float Ys1 = uyp1 + uym1;
        float Ys2 = uyp2 + uym2;
        float Zs1 = uzp1 + uzm1;
        float Zs2 = uzp2 + uzm2;

        if(ix > (dim.x - 1)/2)
        {
            Xs1 = uxp1 + uxm1;
            Xs2 = uxp2 + uxm2;
        }

        if(iy > (dim.y - 1)/2)
        {
            Ys1 = uym1 + uyp1;
            Ys2 = uym2 + uyp2;
        }

        if(iz > (dim.z - 1)/2)
        {
            Zs1 = uzm1 + uzp1;
            Zs2 = uzm2 + uzp2;
        }

        ///3 because 3 dimensions
        u0n1 = -(1/(3 * coeff_center)) * (Xs1 + Ys1 + Zs1 + Xs2 + Ys2 + Zs2 - h2f0);
    }

    float u = buffer_in[IDX(ix,iy,iz)];

    float err = u0n1 - u;

    if(fabs(err) > etol)
    {
        atomic_xchg(still_going, 1);
    }

    buffer_out[IDX(ix, iy, iz)] = mix(u, u0n1, 0.4f);
}
