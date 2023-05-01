float get_particle_radius(float scale)
{
    return 2 * scale;
}

///https://arxiv.org/pdf/1611.07906.pdf (20)
float dirac_disc(float r_sq, float radius)
{
    float rs = radius;

    float frac_sq = r_sq / (rs * rs);

    float mult = 1/(M_PI * pow(rs, 3.f));

    if(frac_sq <= 1)
    {
        float val = 1.f - (3.f/2.f) * frac_sq + (3.f/4.f) * pow(frac_sq, 1.5f);

        return mult * val;
    }

    if(frac_sq <= 2*2)
    {
        return mult * (1.f/4.f) * (-pow(frac_sq, 1.5f) + 6 * frac_sq - 12 * sqrt(frac_sq) + 8);
        //return mult * (1.f/4.f) * pow(2.f - frac, 3.f);
    }

    return 0.f;
}
