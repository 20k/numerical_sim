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
    #if 1
    {
        float tol = 0.25f;

        auto my_real_function = [](vec3f pos)
        {
            float r = pos.length();
            float theta = acos(pos.z() / r);
            float phi = atan2(pos.y(), pos.x());

            //return 1;

            //return r * (4 + sin(theta)) + r * (4 + sin(phi));

            return dual_types::complex<float>(pos.x() * sin(pos.y()) * cos(pos.z()) + pos.x() + pos.y() + pos.z(), -1 * r * sin(theta) + 0.1f);
        };

        std::map<int, std::map<int, dual_types::complex<float>>> lm;

        float radius = 1;

        int max_l = 6;

        for(int l=0; l < max_l; l++)
        {
            for(int m=-l; m <= l; m++)
            {
                lm[l][m] = spherical_decompose_complex_cartesian_function<float>(my_real_function, 0, l, m, {0,0,0}, radius, 64);

                ///printf("Lm %i %i %f\n", l, m, lm[l][m]);
            }
        }

        for(float theta = 0; theta <= M_PI; theta += M_PI/8)
        {
            for(float phi = 0; phi <= 2 * M_PI; phi += M_PI/16)
            {
                vec3f cart = {radius * cos(phi) * sin(theta), radius * sin(phi) * sin(theta), radius * cos(theta)};

                auto my_function = my_real_function(cart);

                dual_types::complex<float> sum = {0,0};

                for(int l=0; l < max_l; l++)
                {
                    for(int m=-l; m <= l; m++)
                    {
                        sum += lm[l][m] * sYlm_2(0, l, m, theta, phi);
                    }
                }

                /*printf("Theta phi %f %f Cart %f %f %f\n", theta, phi, cart.x(), cart.y(), cart.z());

                printf("Sum R %.20f\n", sum.real);
                printf("Sum I %.20f\n", sum.imaginary);
                printf("My function R %.20f\n", my_function.real);
                printf("My function I %.20f\n", my_function.imaginary);*/

                assert(fabs(sum.real - my_function.real) < tol * fabs(my_function.real) + tol);
                assert(fabs(sum.imaginary - my_function.imaginary) < tol * fabs(my_function.imaginary) + tol);
            }
        }
    }
    #endif // 0
}

#endif // SPHERICAL_DECOMPOSITION_HPP_INCLUDED
