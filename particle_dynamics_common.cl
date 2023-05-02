float get_particle_radius(float scale)
{
    return 4 * scale;
}

///https://arxiv.org/pdf/1611.07906.pdf (20)
float dirac_disc(float r_sq, float radius)
{
    float rs = radius;

    float frac_sq = r_sq / (rs * rs);

    float frac = sqrt(r_sq) / rs;

    float mult = 1/(M_PI * pow(rs, 3.f));

    if(frac <= 1)
    {
        //float val = 1.f - (3.f/2.f) * frac_sq + (3.f/4.f) * pow(frac_sq, 1.5f);

        float val = 1 - (3.f/2.f) * frac*frac + (3.f/4.f) * pow(frac, 3.f);

        return mult * val;
    }

    if(frac <= 2)
    {
        //return mult * (1.f/4.f) * (-pow(frac_sq, 1.5f) + 6 * frac_sq - 12 * sqrt(frac_sq) + 8);
        return mult * (1.f/4.f) * pow(2.f - frac, 3.f);
    }

    return 0.f;
}

float dirac_disc1(float diff, float dx)
{
    float adiff = fabs(diff);

    if(adiff < 0.5f * dx)
    {
        return (3.f/4.f) + pow(adiff/dx, 2.f);
    }

    if(adiff < 1.5f * dx)
    {
        return 0.5f * pow(3.f/2.f - (adiff/dx), 2.f);
    }

    return 0.f;
}

/*float dirac_disc2(float3 diff, float rad)
{
    float dx = rad/2;

    return dirac_disc1(diff.x, dx) * dirac_disc1(diff.y, dx) * dirac_disc1(diff.z, dx);
}*/

/*float dirac_disc2(float3 diff, float rad)
{
    float dx = rad/2;

    float rr = length(diff);

    float r = rr/dx;

    if(r < 0.5f)
        return 3.f/4.f - r*r;

    if(r < 1.5f)
        return 9.f/8.f - (3.f/2.f) * r + r*r/2;

    return 0;
}
*/

/*float bump_disc(float x)
{
    float bump = exp((4 * x)/(x*x - 1));

    return
}

float dirac_disc2(float3 diff, float rad)
{

}
*/

float dirac_disc2(float3 diff, float rad)
{
    return dirac_disc(dot(diff, diff), rad);
}
