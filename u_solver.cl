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

float round_away_from(float in, float val)
{
    return in < val ? floor(in) : ceil(in);
}

float3 round_away_from_vec(float3 in, float3 val)
{
    return (float3){round_away_from(in.x, val.x), round_away_from(in.y, val.y), round_away_from(in.z, val.z)};
}

float get_scaled_coordinate(int in, int dimension_upper, int dimension_lower)
{
    int upper_centre = (dimension_upper - 1)/2;

    int upper_offset = in - upper_centre;

    float scale = (dimension_upper - 1) / (dimension_lower - 1);

    ///so lets say we have [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] with a dimension of 13
    ///this gives a middle value of 6, which is the 7th value
    ///Then we want to scale it to a dimension of 7
    ///to get [0:0, 1:0.5, 2:1, 3:1.5, 4:2, 5:2.5, 6:3, 7:3.5, 8:4, 9:4.5, 10:5, 11:5.5, 12:6]
    ///so... it should just be a straight division by the scale?

    return in / scale;
}

float3 get_scaled_coordinate_vec(int3 in, int3 dimension_upper, int3 dimension_lower)
{
    return (float3){get_scaled_coordinate(in.x, dimension_upper.x, dimension_lower.x),
                    get_scaled_coordinate(in.y, dimension_upper.y, dimension_lower.y),
                    get_scaled_coordinate(in.z, dimension_upper.z, dimension_lower.z)};
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

    float3 lower_pos = get_scaled_coordinate_vec((int3){ix, iy, iz}, out_dim.xyz, in_dim.xyz);

    float val = buffer_read_linear(u_in, lower_pos, in_dim);

    //int3 half_lower = (in_dim.xyz - 1) / 2;
    //float val = buffer_read_nearest(u_in, convert_int3(round_away_from_vec(lower_pos, convert_float3(half_lower))), in_dim);

    ///todo: remove this
    if(ix == 0 || iy == 0 || iz == 0 || ix == out_dim.x - 1 || iy == out_dim.y - 1 || iz == out_dim.z - 1)
        val = 1;

    u_out[IDXD(ix, iy, iz, out_dim)] = val;
}

///extracts an area from a larger volume, where both are the same grid size
__kernel
void extract_u_region(__global float* u_in, __global float* u_out, float c_at_max_in, float c_at_max_out, int4 dim)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    ///eg 0.25f
    float grid_fraction = c_at_max_out / c_at_max_in;

    float3 fpos = (float3)(ix, iy, iz);
    float3 half_dim = convert_float3(dim.xyz) / 2.f;

    float3 offset = fpos - half_dim;

    float3 scaled_offset = offset * grid_fraction;

    float3 new_pos = scaled_offset + half_dim;

    float val = buffer_read_linear(u_in, new_pos, dim);

    u_out[IDX(ix, iy, iz)] = val;
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
    float X = 1/init_BL_val;
    float B = (1.f/8.f) * pow(X, 7.f) * aij_aIJ;
    float RHS = -B * pow(1 + X * u, -7);

    //float RHS = -(1/8.f) * aij_aIJ * pow(bl_s + u, -7);

    float h2f0 = scale * scale * RHS;

    float u0n1 = 0;

    if(x_degenerate || y_degenerate || z_degenerate)
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

    u_offset_out[IDX(ix, iy, iz)] = mix(u, u0n1, 0.9f);
    //u_offset_out[IDX(ix, iy, iz)] = u0n1;
}

__kernel
void check_z_symmetry(__global float* u_in, int4 dim)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(iz >= (dim.z - 1)/2)
        return;

    if(ix >= dim.x)
        return;

    if(iy >= dim.y)
        return;

    float base_value = u_in[IDX(ix, iy, iz)];

    int mirrored_z = dim.z - iz - 1;

    float v_mirrored = u_in[IDX(ix, iy, mirrored_z)];

    if(base_value != v_mirrored)
    {
        printf("Failure in symmetry %.23f %i %i %i against %.23f %i %i %i with dim %i %i %i\n", base_value, ix, iy, iz, v_mirrored, ix, iy, mirrored_z, dim.x, dim.y, dim.z);
    }
}
