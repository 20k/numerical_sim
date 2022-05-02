#ifndef SPHERICAL_DECOMPOSITION_HPP_INCLUDED
#define SPHERICAL_DECOMPOSITION_HPP_INCLUDED

#include "spherical_harmonics.hpp"

template<typename T, typename U>
inline
dual_types::complex<T> spherical_decompose_complex_cartesian_function(U&& cartesian_function, int s, int l, int m, vec<3, T> centre, T radius, int n)
{
    auto func_re = [&](T theta, T phi)
    {
        dual_types::complex<T> harmonic = sYlm_2(s, l, m, theta, phi);

        vec<3, T> pos = {radius * cos(phi) * sin(theta), radius * sin(phi) * sin(theta), radius * cos(theta)};
        pos += centre;

        dual_types::complex<T> result = cartesian_function(pos);

        return (result * conjugate(harmonic)).real;
    };

    auto func_im = [&](T theta, T phi)
    {
        dual_types::complex<T> harmonic = sYlm_2(s, l, m, theta, phi);

        vec<3, T> pos = {radius * cos(phi) * sin(theta), radius * sin(phi) * sin(theta), radius * cos(theta)};
        pos += centre;

        dual_types::complex<T> result = cartesian_function(pos);

        return (result * conjugate(harmonic)).imaginary;
    };

    return {spherical_integrate(func_re, n), spherical_integrate(func_im, n)};
}

inline
void test_spherical_decomp()
{
    ///http://scipp.ucsc.edu/~haber/ph116C/SphericalHarmonics_12.pdf 12
    {
        auto my_real_function = [](vec3f pos)
        {
            return pos.x() * sin(pos.y()) * cos(pos.z());
        };

        std::map<int, std::map<int, float>> lm;

        for(int l=0; l < 10; l++)
        {
            for(int m=-l; m <= l; m++)
            {
                lm[l][m] = spherical_decompose_complex_cartesian_function(my_real_function,  0, l, m, {0,0,0}, 1.f, 64).real;
            }
        }
    }
}

#endif // SPHERICAL_DECOMPOSITION_HPP_INCLUDED
