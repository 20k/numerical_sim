#include "common.cl"

void laplace_interior(__global float* buffer_in, __global float* buffer_out, float h2f0, int ix, int iy, int iz, float scale, int4 dim, __global int* still_going, float etol)
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
    float u0n1 = (1/6.f) * (Xs + Ys + Zs - h2f0);

    float u = buffer_in[IDX(ix,iy,iz)];

    float err = u0n1 - u;

    if(fabs(err) > etol)
    {
        atomic_xchg(still_going, 1);
    }

    buffer_out[IDX(ix, iy, iz)] = mix(u, u0n1, 0.9f);
}

void laplace_interior_rb(__global float* buffer_in, float h2f0, int ix, int iy, int iz, float scale, int4 dim, __global int* still_going, float etol, int iteration)
{
    float uxm1 = buffer_in[IDX(ix-1, iy, iz)];
    float uxp1 = buffer_in[IDX(ix+1, iy, iz)];
    float uym1 = buffer_in[IDX(ix, iy-1, iz)];
    float uyp1 = buffer_in[IDX(ix, iy+1, iz)];
    float uzm1 = buffer_in[IDX(ix, iy, iz-1)];
    float uzp1 = buffer_in[IDX(ix, iy, iz+1)];

    int lix = ix;
    int liy = iy;

    if(iz % 2)
    {
        lix++;
        //liy++;
    }

    ///pretty crappy layout here
    if(((lix + liy) % 2) == iteration % 2)
        return;

    float Xs = uxm1 + uxp1;
    float Ys = uyp1 + uym1;
    float Zs = uzp1 + uzm1;

    ///-6u0 + the rest of the terms = h^2 f0
    float u0n1 = (1/6.f) * (Xs + Ys + Zs - h2f0);

    float u = buffer_in[IDX(ix,iy,iz)];

    float err = u0n1 - u;

    if(fabs(err) > etol)
    {
        atomic_xchg(still_going, 1);
    }

    buffer_in[IDX(ix, iy, iz)] = mix(u, u0n1, 0.9f);
}
