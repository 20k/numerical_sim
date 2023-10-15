float get_particle_radius(float scale)
{
    return 2 * scale;
}

///https://arxiv.org/pdf/1611.07906.pdf (20)
float dirac_distribution(float r, float radius)
{
    float rs = radius;

    float frac = r / rs;

    float mult = 1/(M_PI * pow(rs, 3.f));

    if(frac <= 1)
    {
        float val = 1.f - (3.f/2.f) * pow(frac, 2.f) + (3.f/4.f) * pow(frac, 3.f);

        return mult * val;
    }

    if(frac <= 2)
    {
        return mult * (1.f/4.f) * pow(2.f - frac, 3.f);
    }

    return 0.f;
}

float dirac_disc_xyz(float x, float y, float z, float radius)
{
    float r = native_sqrt(x*x + y*y + z*z);

    return dirac_distribution(r, radius);
}

#define INTEGRATE_N 4

float z_integral(float cx, float cy, float world_z, float world_radius, float world_cell_size)
{
    float min_world_z = world_z - world_cell_size/2;
    float max_world_z = world_z + world_cell_size/2;

    float prefix_z = (max_world_z - min_world_z) / INTEGRATE_N;

    float dz_integral_sum = 0;

    #pragma unroll
    for(int k=1; k < INTEGRATE_N; k++)
    {
        float cz = min_world_z + k * (max_world_z - min_world_z) / INTEGRATE_N;

        float f_val = dirac_disc_xyz(cx, cy, cz, world_radius);

        dz_integral_sum += f_val;
    }

    float dz_integrated = prefix_z * (dirac_disc_xyz(cx, cy, min_world_z, world_radius)/2.f + dz_integral_sum + dirac_disc_xyz(cx, cy, max_world_z, world_radius)/2.f);

    return dz_integrated;
}

float y_integral(float cx, float world_y, float world_z, float world_radius, float world_cell_size)
{
    float min_world_y = world_y - world_cell_size/2;
    float max_world_y = world_y + world_cell_size/2;

    float prefix_y = (max_world_y - min_world_y) / INTEGRATE_N;

    float dy_integral_sum = 0;

    #pragma unroll
    for(int k=1; k < INTEGRATE_N; k++)
    {
        float cy = min_world_y + k * (max_world_y - min_world_y) / INTEGRATE_N;

        float f_val = z_integral(cx, cy, world_z, world_radius, world_cell_size);

        dy_integral_sum += f_val;
    }

    float dy_integrated = prefix_y * (z_integral(cx, min_world_y, world_z, world_radius, world_cell_size)/2.f + dy_integral_sum + z_integral(cx, max_world_y, world_z, world_radius, world_cell_size)/2.f);

    return dy_integrated;
}

///i legitimately hate C
float x_integral(float world_x, float world_y, float world_z, float world_radius, float world_cell_size)
{
    float min_world_x = world_x - world_cell_size/2;
    float max_world_x = world_x + world_cell_size/2;

    float prefix_x = (max_world_x - min_world_x) / INTEGRATE_N;

    float dx_integral_sum = 0;

    #pragma unroll
    for(int k=1; k < INTEGRATE_N; k++)
    {
        float cx = min_world_x + k * (max_world_x - min_world_x) / INTEGRATE_N;

        float f_val = y_integral(cx, world_y, world_z, world_radius, world_cell_size);

        dx_integral_sum += f_val;
    }

    float dx_integrated = prefix_x * (y_integral(min_world_x, world_y, world_z, world_radius, world_cell_size)/2.f + dx_integral_sum + y_integral(max_world_x, world_y, world_z, world_radius, world_cell_size)/2.f);

    return dx_integrated;
}

float dirac_disc_volume(float world_x, float world_y, float world_z, float world_radius, float world_cell_size)
{
    ///integrate between -world_cell_size/2 and world_cell_size/2, relative to our coordinate

    return x_integral(world_x, world_y, world_z, world_radius, world_cell_size);
}
