float get_particle_radius(float scale)
{
    return 3 * scale;
}

///https://arxiv.org/pdf/1611.07906.pdf (20)
float dirac_disc(float r, float radius)
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
