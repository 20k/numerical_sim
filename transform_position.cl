struct mesh
{
    float4 position;
    int4 dim;
    float scale;
    int resolution;
};

float r_to_phys(float r)
{
    float a = 3;
    float r0 = 5.5f * 0.5f;
    float s = 1.2f * 0.5f;

    ///https://arxiv.org/pdf/gr-qc/0505055.pdf 5.5
    float R_r = (s / (2 * r * tanh(r0/s))) * log(cosh((r + r0)/s)/cosh((r - r0)/s));

    float r_phys = r * (a + (1 - a) * R_r);

    return r_phys;
}

/*float3 transform_position(int x, int y, int z, int4 dim, float scale)
{
    float3 centre = {dim.x/2, dim.y/2, dim.z/2};
    float3 pos = {x, y, z};

    float3 diff = pos - centre;

    float coordinate_r = fast_length(diff);
    coordinate_r = max(coordinate_r, 0.001f);

    float physical_r = r_to_phys(coordinate_r);

    float3 scaled_offset = diff * (physical_r / coordinate_r);
    //float3 scaled_offset = diff * (physical_r / coordinate_r);

    float3 unscaled = scaled_offset / 3;

    if(z == 125)
    {
        printf("%f %f\n", coordinate_r, physical_r);
    }

    return unscaled;
}*/

float polynomial(float x)
{
    return (1 + (-3 + 6 * (-1 + x)) * (-1 + x)) * x * x * x;
}

#define BULGE_AMOUNT 1

float3 transform_position(float x, float y, float z, struct mesh* m)
{
    int4 dim = m->dim;

    float3 centre = {(dim.x - 1)/2.f, (dim.y - 1)/2.f, (dim.z - 1)/2.f};
    float3 pos = {x, y, z};

    //float3 diff = pos - centre - mesh_position.xyz;

    float3 offset_from_grid = pos - centre;

    offset_from_grid = round(offset_from_grid * 2) / 2.f;

    float3 world_offset = offset_from_grid * m->scale * m->resolution;

    return world_offset + m->position.xyz;

    #if 0
    float len = length(diff);

    if(len == 0)
        return diff;

    //if(len <= 0.0001)
    //    len = 0.0001;

    float real_len = len * scale;

    float edge = max(max(dim.x, dim.y), dim.z) * scale / 2.0f;

    float real_distance_r1 = 10.f;

    float r1 = real_distance_r1;
    float r2 = edge - 16 * scale;
    float r3 = edge;

    float bulge_amount = BULGE_AMOUNT;

    float r1b = r1 * bulge_amount;
    float r2b = r2;
    float r3b = r3;

    float rad = 0;

    /*if(real_len < r1)
    {
        rad = (real_len * r1b / r1) / scale;
    }
    else if(real_len < r2)
    {
        float frac = (real_len - r1) / (r2 - r1);

        float polynomial_frac = polynomial(clamp(frac, 0.f, 1.f));

        rad = (r1b + (polynomial_frac * (r2b - r1b))) / scale;
    }
    else
    {
        rad = real_len / scale;
    }

    return diff * rad / len;*/

    float3 norm = normalize(diff);

    if(real_len < r1b)
    {
        return norm * (real_len * r1 / r1b);
    }

    else if(real_len < r2b)
    {
        float frac = (real_len - r1b) / (r2b - r1b);

        float polynomial_frac = polynomial(clamp(frac, 0.f, 1.f));

        float next_len = r1 + polynomial_frac * (r2 - r1);

        return norm * next_len;
    }
    else
    {
        return norm * real_len;
    }
    #endif // 0
}
