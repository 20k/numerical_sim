#include "transform_position.cl"
#include "common.cl"
#include "generic_laplace.cl"

__kernel
void setup_u_offset(__global float* u_offset,
                    int4 dim)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    u_offset[IDX(ix, iy, iz)] = U_BOUNDARY;
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
        val = U_BOUNDARY;

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
    //float grid_fraction = c_at_max_out / c_at_max_in;

    int inverse_grid_fraction = round(c_at_max_in / c_at_max_out);

    int3 fpos = (int3)(ix, iy, iz);
    int3 half_dim = (dim.xyz - 1) / 2;

    int3 offset = fpos - half_dim;

    float3 scaled_offset = convert_float3(offset) / inverse_grid_fraction;

    float3 new_pos = scaled_offset + convert_float3(half_dim);

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
                       float scale, int4 dim, __constant int* last_still_going, __global int* still_going, float etol)
{
    if(*last_still_going == 0)
        return;

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

    float RHS = U_RHS;

    float h2f0 = scale * scale * RHS;

    /*if(isnan(u_offset_in[IDX(ix,iy,iz)]))
    {
        printf("UOFF %f %i %i %i\n", u_offset_in[IDX(ix,iy,iz)], ix, iy, iz);
    }

    //if(ix == dim.x/2 && iy == dim.y/2 && iz == dim.z/2)
    {
        if(isnan(RHS))
            printf("RHS %f %i %i %i\n", RHS, ix, iy, iz);
    }

    ///107 125 125
    if(ix == 107 && iy == 125 && iz == 125)
    {
        printf("RHS %f %f\n", RHS, u_offset_in[IDX(ix,iy,iz)]);
    }*/

    /*if((ix == 107 || ix == 106 || ix == 108) && (iy == 125 || iy == 124 || iy == 126) && (iz == 125 || iz == 124 || iz == 126))
    {
        float u_in = u_offset_in[IDX(ix,iy,iz)];

        printf("U %f %f %i %i %i\n", RHS, u_in, ix, iy, iz);
    }*/

    laplace_interior(u_offset_in, u_offset_out, h2f0, ix, iy, iz, scale, dim, still_going, etol);
}
