#define IDX(i, j, k) ((k) * dim.x * dim.y + (j) * dim.x + (i))
#define IDXD(i, j, k, d) ((k) * (d.x) * (d.y) + (j) * (d.x) + (i))

float buffer_read_nearest(__global const float* const buffer, int3 position, int4 dim)
{
    return buffer[position.z * dim.x * dim.y + position.y * dim.x + position.x];
}

float buffer_read_nearest_clamp(__global const float* const buffer, int3 position, int4 dim)
{
    position = clamp(position, (int3)(0,0,0), dim.xyz - 1);

    return buffer[position.z * dim.x * dim.y + position.y * dim.x + position.x];
}

float buffer_read_linear(__global const float* const buffer, float3 position, int4 dim)
{
    /*position = round(position);

    int3 ipos = (int3)(position.x, position.y, position.z);

    return buffer[ipos.z * dim.x * dim.y + ipos.y * dim.x + ipos.x];*/

    float3 floored = floor(position);

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

float buffer_index(__global const float* const buffer, int x, int y, int z, int4 dim)
{
    return buffer[z * dim.x * dim.y + y * dim.x + x];
}

#define MAKE_ARG(type, name) __global type* name
