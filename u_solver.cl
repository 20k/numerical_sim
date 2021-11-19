#include "transform_position.cl"
#include "common.cl"

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

///out is > in
///this incorrectly does not produce a symmetric result
__kernel
void upscale_u(__global float* u_in, __global float* u_out, int4 in_dim, int4 out_dim)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= out_dim.x || iy >= out_dim.y || iz >= out_dim.z)
        return;

    float3 upper_pos = (float3)(ix, iy, iz);

    /*float3 normed = upper_pos / (float3)(out_dim.x - 1, out_dim.y - 1, out_dim.z - 1);

    float3 lower_pos = normed * (float3)(in_dim.x - 1, in_dim.y - 1, in_dim.z - 1);

    float3 lower_rounded = round(lower_pos);*/

    float3 upper_centre = convert_float3((out_dim.xyz - 1) / 2);

    float3 upper_offset = upper_pos - upper_centre;

    float scale = (out_dim.x - 1) / (in_dim.x - 1);

    float3 lower_offset = upper_offset / scale;

    ///symmetric, rounds away from 0
    float3 lower_pos = (lower_offset) + convert_float3((in_dim.xyz - 1) / 2);

    float val = buffer_read_linear(u_in, lower_pos, in_dim);

    //float val = buffer_read_nearest(u_in, convert_int3(lower_pos), in_dim);

    if(ix == 0 || iy == 0 || iz == 0 || ix == out_dim.x - 1 || iy == out_dim.y - 1 || iz == out_dim.z - 1)
        val = 1;

    u_out[IDXD(ix, iy, iz, out_dim)] = val;
}

///https://learn.lboro.ac.uk/archive/olmp/olmp_resources/pages/workbooks_1_50_jan2008/Workbook33/33_2_elliptic_pde.pdf
///https://arxiv.org/pdf/1205.5111v1.pdf 78
///https://arxiv.org/pdf/gr-qc/0007085.pdf 76?

///so, the laplacian is the sum of second derivatives in the same direction, ie
///didix + djdjx + dkdkx = 0
///so with first order stencil, we get [1, -2, 1] in each direction, which is why we get a central -6
///todo: this, but second order, because memory reads are heavily cached
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

    bool x_degenerate = ix < 2 || ix >= dim.x - 2;
    bool y_degenerate = iy < 2 || iy >= dim.y - 2;
    bool z_degenerate = iz < 2 || iz >= dim.z - 2;

    float3 offset = transform_position(ix, iy, iz, dim, scale);

    float ox = offset.x;
    float oy = offset.y;
    float oz = offset.z;

    float bl_s = init_BL_val;

    float aij_aIJ = init_aij_aIJ;

    float u = u_offset_in[IDX(ix, iy, iz)];

    ///https://arxiv.org/pdf/gr-qc/0007085.pdf (78) implies that the two formulations are not equivalent
    float X = 1/bl_s;
    float B = (1.f/8.f) * pow(X, 7.f) * aij_aIJ;
    float RHS = -B * pow(1 + X * u, -7);

    //float RHS = -(1/8.f) * aij_aIJ * pow(bl_s + u, -7);

    float h2f0 = scale * scale * RHS;

    float u0n1 = 0;

    //if(x_degenerate || y_degenerate || z_degenerate)
    {
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
        u0n1 = (1/6.f) * (Xs + Ys + Zs - h2f0);
    }
    #if 0
    else
    {
        float coeff1 = 4.f/3.f;
        float coeff2 = -1.f/12.f;
        float coeff_center = -5.f/2.f;

        float uxm1 = coeff1 * u_offset_in[IDX(ix-1, iy, iz)];
        float uxp1 = coeff1 * u_offset_in[IDX(ix+1, iy, iz)];
        float uym1 = coeff1 * u_offset_in[IDX(ix, iy-1, iz)];
        float uyp1 = coeff1 * u_offset_in[IDX(ix, iy+1, iz)];
        float uzm1 = coeff1 * u_offset_in[IDX(ix, iy, iz-1)];
        float uzp1 = coeff1 * u_offset_in[IDX(ix, iy, iz+1)];

        float uxm2 = coeff2 * u_offset_in[IDX(ix-2, iy, iz)];
        float uxp2 = coeff2 * u_offset_in[IDX(ix+2, iy, iz)];
        float uym2 = coeff2 * u_offset_in[IDX(ix, iy-2, iz)];
        float uyp2 = coeff2 * u_offset_in[IDX(ix, iy+2, iz)];
        float uzm2 = coeff2 * u_offset_in[IDX(ix, iy, iz-2)];
        float uzp2 = coeff2 * u_offset_in[IDX(ix, iy, iz+2)];

        ///so, floating point maths isn't associative
        ///which means that if we're on the other side of a symmetric boundary about the central plane
        ///the order of operations will be different
        ///the if statements correct this, which makes this method numerically symmetric, and implicitly
        ///converges to a symmetric solution if available
        /*float Xs1 = uxm1 + uxp1;
        float Xs2 = uxm2 + uxp2;
        float Ys1 = uyp1 + uym1;
        float Ys2 = uyp2 + uym2;
        float Zs1 = uzp1 + uzm1;
        float Zs2 = uzp2 + uzm2;*/

        float Xs = (uxm2 + uxp2) + (uxm1 + uxp1);
        float Ys = (uym2 + uyp2) + (uym1 + uyp1);
        float Zs = (uzm2 + uzp2) + (uzm1 + uzp1);

        if(ix > (dim.x - 1)/2)
        {
            /*Xs1 = uxp1 + uxm1;
            Xs2 = uxp2 + uxm2;*/

            Xs = (uxp2 + uxm2) + (uxp1 + uxm1);
        }

        if(iy > (dim.y - 1)/2)
        {
            /*Ys1 = uym1 + uyp1;
            Ys2 = uym2 + uyp2;*/

            Ys = (uyp2 + uym2) + (uyp1 + uym1);
        }

        if(iz > (dim.z - 1)/2)
        {
            /*Zs1 = uzm1 + uzp1;
            Zs2 = uzm2 + uzp2;*/

            Xs = (uzp2 + uzm2) + (uzp1 + uzm1);
        }

        ///3 because 3 dimensions
        u0n1 = -(1/(3 * coeff_center)) * (Xs + Ys + Zs - h2f0);
    }
    #endif // 0

    if(is_centre)
    {
        printf("Next! %f\n", u0n1);
        //u0n1 = 1;
    }

    //if(ix == 50 && iy == dim.y/2 && iz == dim.z/2)
    //    printf("hi %.23f\n", u0n1);

    /*if(ix == (dim.x - 1) / 2 && iy == (dim.y - 1) / 2)
    {
        int cz = (dim.z - 1) / 2;

        if(iz == cz - 1 || iz == cz || iz == cz + 1)
        {
            printf("Val %.24f %i\n", u0n1, iz);
        }
    }*/

    u_offset_out[IDX(ix, iy, iz)] = mix(u, u0n1, 0.6f);
    //u_offset_out[IDX(ix, iy, iz)] = u0n1;
}
