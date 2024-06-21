#ifndef COMMON_CL_INCLUDED
#define COMMON_CL_INCLUDED

#define IDX(i, j, k) ((k) * dim.x * dim.y + (j) * dim.x + (i))
#define IDXD(i, j, k, d) ((k) * (d.x) * (d.y) + (j) * (d.x) + (i))

#include "bitflags.cl"

float buffer_read_nearest(__global const float* const buffer, int3 position, int3 dim)
{
    return buffer[position.z * dim.x * dim.y + position.y * dim.x + position.x];
}

float buffer_read_nearest_clamp(__global const float* const buffer, int3 position, int3 dim)
{
    position = clamp(position, (int3)(0,0,0), dim.xyz - 1);

    return buffer[position.z * dim.x * dim.y + position.y * dim.x + position.x];
}

float buffer_read_linear2(__global const float* const buffer, float px, float py, float pz, int dx, int dy, int dz)
{
    float3 floored = floor((float3)(px, py, pz));
    int3 dim = (int3)(dx, dy, dz);
    float3 position = (float3)(px, py, pz);

    int3 ipos = (int3)(floored.x, floored.y, floored.z);

    float c000 = buffer_read_nearest_clamp(buffer, ipos + (int3)(0,0,0), dim);
    float c100 = buffer_read_nearest_clamp(buffer, ipos + (int3)(1,0,0), dim);

    float c010 = buffer_read_nearest_clamp(buffer, ipos + (int3)(0,1,0), dim);
    float c110 = buffer_read_nearest_clamp(buffer, ipos + (int3)(1,1,0), dim);

    float c001 = buffer_read_nearest_clamp(buffer, ipos + (int3)(0,0,1), dim);
    float c101 = buffer_read_nearest_clamp(buffer, ipos + (int3)(1,0,1), dim);

    float c011 = buffer_read_nearest_clamp(buffer, ipos + (int3)(0,1,1), dim);
    float c111 = buffer_read_nearest_clamp(buffer, ipos + (int3)(1,1,1), dim);

    float3 frac = position - floored;

    /*float c00 = c000 * (1 - frac.x) + c100 * frac.x;
    float c01 = c001 * (1 - frac.x) + c101 * frac.x;

    float c10 = c010 * (1 - frac.x) + c110 * frac.x;
    float c11 = c011 * (1 - frac.x) + c111 * frac.x;

    float c0 = c00 * (1 - frac.y) + c10 * frac.y;
    float c1 = c01 * (1 - frac.y) + c11 * frac.y;

    return c0 * (1 - frac.z) + c1 * frac.z;*/

    ///numerically symmetric across the centre of dim
    float c00 = c000 - frac.x * (c000 - c100);
    float c01 = c001 - frac.x * (c001 - c101);

    float c10 = c010 - frac.x * (c010 - c110);
    float c11 = c011 - frac.x * (c011 - c111);

    float c0 = c00 - frac.y * (c00 - c10);
    float c1 = c01 - frac.y * (c01 - c11);

    return c0 - frac.z * (c0 - c1);
}

float buffer_read_linear(__global const float* const buffer, float3 position, int4 dim)
{
    return buffer_read_linear2(buffer, position.x, position.y, position.z, dim.x, dim.y, dim.z);
}

#ifdef DERIV_PRECISION
float buffer_read_nearest_clamph(__global const DERIV_PRECISION* const buffer, int3 position, int3 dim)
{
    position = clamp(position, (int3)(0,0,0), dim.xyz - 1);

    return buffer[position.z * dim.x * dim.y + position.y * dim.x + position.x];
}

float buffer_read_linear2h(__global const DERIV_PRECISION* const buffer, float px, float py, float pz, int dx, int dy, int dz)
{
    float3 floored = floor((float3)(px, py, pz));
    int3 dim = (int3)(dx, dy, dz);
    float3 position = (float3)(px, py, pz);

    int3 ipos = (int3)(floored.x, floored.y, floored.z);

    float c000 = buffer_read_nearest_clamph(buffer, ipos + (int3)(0,0,0), dim);
    float c100 = buffer_read_nearest_clamph(buffer, ipos + (int3)(1,0,0), dim);

    float c010 = buffer_read_nearest_clamph(buffer, ipos + (int3)(0,1,0), dim);
    float c110 = buffer_read_nearest_clamph(buffer, ipos + (int3)(1,1,0), dim);

    float c001 = buffer_read_nearest_clamph(buffer, ipos + (int3)(0,0,1), dim);
    float c101 = buffer_read_nearest_clamph(buffer, ipos + (int3)(1,0,1), dim);

    float c011 = buffer_read_nearest_clamph(buffer, ipos + (int3)(0,1,1), dim);
    float c111 = buffer_read_nearest_clamph(buffer, ipos + (int3)(1,1,1), dim);

    float3 frac = position - floored;

    ///numerically symmetric across the centre of dim
    float c00 = c000 - frac.x * (c000 - c100);
    float c01 = c001 - frac.x * (c001 - c101);

    float c10 = c010 - frac.x * (c010 - c110);
    float c11 = c011 - frac.x * (c011 - c111);

    float c0 = c00 - frac.y * (c00 - c10);
    float c1 = c01 - frac.y * (c01 - c11);

    return c0 - frac.z * (c0 - c1);
}
#endif // DERIV_PRECISION

float buffer_index(__global const float* const buffer, int x, int y, int z, int4 dim)
{
    return buffer[z * dim.x * dim.y + y * dim.x + x];
}

float sponge_damp_coeff(float x, float y, float z, float scale, int4 dim)
{
    float edge_half = scale * ((dim.x - 2)/2.f);

    //float sponge_r0 = scale * ((dim.x/2) - 48);
    float sponge_r0 = scale * (((dim.x - 1)/2.f) - 48);
    //float sponge_r0 = scale * ((dim.x/2) - 32);
    //float sponge_r0 = edge_half/2;
    float sponge_r1 = scale * (((dim.x - 1)/2.f) - 6);

    float3 fdim = ((float3)(dim.x, dim.y, dim.z) - 1)/2.f;

    float3 diff = ((float3){x, y, z} - fdim) * scale;

    #define MANHATTEN_SPONGE
    #ifdef MANHATTEN_SPONGE
    float r = max(fabs(diff.x), max(fabs(diff.y), fabs(diff.z)));
    #else
    float r = fast_length(diff);
    #endif // MANHATTEN_SPONGE

    if(r <= sponge_r0)
        return 0.f;

    if(r >= sponge_r1)
        return 1.f;

    r = clamp(r, sponge_r0, sponge_r1);

    //return (r - sponge_r0) / (sponge_r1 - sponge_r0);

    float sigma = (sponge_r1 - sponge_r0) / 3;
    return clamp(native_exp(-pow((r - sponge_r1) / sigma, 2)), 0.f, 1.f);
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

    if(base_value != v_mirrored && base_value != -v_mirrored)
    {
        printf("Failure in symmetry %.23f %i %i %i against %.23f %i %i %i with dim %i %i %i\n", base_value, ix, iy, iz, v_mirrored, ix, iy, mirrored_z, dim.x, dim.y, dim.z);
    }
}

float3 world_to_voxel(float3 world_pos, int4 dim, float scale)
{
    float3 centre = {(dim.x - 1)/2, (dim.y - 1)/2, (dim.z - 1)/2};

    return (world_pos / scale) + centre;
}

float world_to_voxel_x(float3 world_pos, int4 dim, float scale)
{
    return world_to_voxel(world_pos, dim, scale).x;
}

float world_to_voxel_y(float3 world_pos, int4 dim, float scale)
{
    return world_to_voxel(world_pos, dim, scale).y;
}

float world_to_voxel_z(float3 world_pos, int4 dim, float scale)
{
    return world_to_voxel(world_pos, dim, scale).z;
}

#endif // COMMON_CL_INCLUDED
