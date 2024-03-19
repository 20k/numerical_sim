#ifndef SPHERICAL_DECOMPOSITION_HPP_INCLUDED
#define SPHERICAL_DECOMPOSITION_HPP_INCLUDED

#include "spherical_harmonics.hpp"

template<typename T, typename U>
inline
dual_types::complex_v<T> spherical_decompose_complex_cartesian_function(U&& cartesian_function, int s, int l, int m, vec<3, T> centre, T radius, int n)
{
    auto func_re = [&](T theta, T phi)
    {
        dual_types::complex_v<T> harmonic = sYlm_2(s, l, m, theta, phi);

        vec<3, T> pos = {radius * cos(phi) * sin(theta), radius * sin(phi) * sin(theta), radius * cos(theta)};
        pos += centre;

        dual_types::complex_v<T> result = cartesian_function(pos);

        return (result * conjugate(harmonic)).real;
    };

    auto func_im = [&](T theta, T phi)
    {
        dual_types::complex_v<T> harmonic = sYlm_2(s, l, m, theta, phi);

        vec<3, T> pos = {radius * cos(phi) * sin(theta), radius * sin(phi) * sin(theta), radius * cos(theta)};
        pos += centre;

        dual_types::complex_v<T> result = cartesian_function(pos);

        return (result * conjugate(harmonic)).imaginary;
    };

    return {spherical_integrate(func_re, n), spherical_integrate(func_im, n)};
}

template<typename T, typename U>
inline
T spherical_decompose_cartesian_function(U&& cartesian_function, int s, int l, int m, vec<3, T> centre, T radius, int n)
{
    auto func_re = [&](T theta, T phi)
    {
        dual_types::complex_v<T> harmonic = sYlm_2(s, l, m, theta, phi);

        vec<3, T> pos = {radius * cos(phi) * sin(theta), radius * sin(phi) * sin(theta), radius * cos(theta)};
        pos += centre;

        dual_types::complex_v<T> result = cartesian_function(pos);

        return (result * conjugate(harmonic)).real;
    };

    return spherical_integrate(func_re, n);
}

#endif // SPHERICAL_DECOMPOSITION_HPP_INCLUDED
