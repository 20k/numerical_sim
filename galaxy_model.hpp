#ifndef GALAXY_MODEL_HPP_INCLUDED
#define GALAXY_MODEL_HPP_INCLUDED

#include <vector>
#include <vec/vec.hpp>

struct particle_data
{
    std::vector<vec3f> positions;
    std::vector<vec3f> velocities;
    std::vector<float> masses;

    std::vector<float> debug_velocities;
    std::vector<float> debug_analytic_mass;
    std::vector<float> debug_real_mass;
};

particle_data build_galaxy();

#endif // GALAXY_MODEL_HPP_INCLUDED
