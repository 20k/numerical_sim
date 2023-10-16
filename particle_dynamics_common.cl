float get_particle_radius(float scale)
{
    return 2 * scale;
}

float dirac_disc_volume(float world_x, float world_y, float world_z, float world_radius, float world_cell_size)
{
    ///integrate between -world_cell_size/2 and world_cell_size/2, relative to our coordinate

    return DIRAC_DISC_OUT;
}
