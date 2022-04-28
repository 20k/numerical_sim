#ifndef SPHERICAL_INTEGRATION_HPP_INCLUDED
#define SPHERICAL_INTEGRATION_HPP_INCLUDED

#include <vector>
#include "legendre_nodes.h"
#include "legendre_weights.h"
#include <geodesic/dual_value.hpp>

///https://pomax.github.io/bezierinfo/legendre-gauss.html
///https://cbeentjes.github.io/files/Ramblings/QuadratureSphere.pdf
///http://homepage.divms.uiowa.edu/~atkinson/papers/SphereQuad1982.pdf
template<typename T>
inline
auto integrate_1d_raw(const T& func, int n, float upper, float lower)
{
    std::vector<float> weights = get_legendre_weights(n);
    std::vector<float> nodes = get_legendre_nodes(n);

    using variable_type = decltype(func(0.f));

    variable_type sum = 0;

    for(int j=0; j < n; j++)
    {
        float w = weights[j];
        float xj = nodes[j];

        float value = ((upper - lower)/2.f) * xj + (upper + lower) / 2.f;

        auto func_eval = w * func(value);

        sum = sum + func_eval;
    }

    return ((upper - lower) / 2.f) * sum;
}

template<typename T>
inline
auto integrate_1d(const T& func, int n, float upper, float lower)
{
    using variable_type = decltype(func(0.f));
    variable_type sum =  0;

    int pieces = 16 * n;
    float step = (upper - lower) / pieces;

    /*for(int i=0; i < pieces; i++)
    {
        sum += integrate_1d_raw(func, n, i * step + lower, (i + 1) * step + lower);
    }*/

    for(int i=0; i < pieces; i++)
    {
        float base_x = i * step;
        float next_x = (i + 1) * step;

        float real_x = (base_x + next_x) / 2.f;

        auto i_1 = func(real_x);

        auto area = i_1 * (next_x - base_x);

        sum += area;
    }

    return sum;
}

template<typename T>
inline
auto spherical_integrate(const T& f_theta_phi, int n)
{
    using variable_type = decltype(f_theta_phi(0.f, 0.f));

    variable_type sum = 0;

    std::vector<float> weights = get_legendre_weights(n);
    std::vector<float> nodes = get_legendre_nodes(n);

    float iupper = 2 * M_PI;
    float ilower = 0;

    float jupper = M_PI;
    float jlower = 0;

    ///https://cbeentjes.github.io/files/Ramblings/QuadratureSphere.pdf7 7
    ///0 -> 2pi, phi
    auto outer_integral = [&](float phi)
    {
        auto inner_integral = [&](float theta){return sin(theta) * f_theta_phi(theta, phi);};

        return integrate_1d(inner_integral, n, jupper, jlower);
    };

    return integrate_1d(outer_integral, n, iupper, ilower);
}

///https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.728.5478&rep=rep1&type=pdf
inline
void test_integration()
{
    {
        auto f1_xyz = [](float x, float y, float z)
        {
            return (1 + tanh(-9 * x - 9 * y - 9 * z)) / 9;
        };

        auto f1_theta_phi = [&](float theta, float phi)
        {
            return f1_xyz(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));
        };

        float integral = spherical_integrate(f1_theta_phi, 64);

        assert(approx_equal(integral, 4 * M_PI/9, 0.0001f));
    }

    {
        auto f3_xyz = [](float x, float y, float z)
        {
            return (M_PI/2 + atan(300 * (z - 9999.f/10000.f))) / M_PI;
        };

        auto f3_theta_phi = [&](float theta, float phi)
        {
            return f3_xyz(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));
        };

        float integral = spherical_integrate(f3_theta_phi, 64);

        assert(approx_equal(integral, 0.049629692928687f, 0.0001f));
    }

}

#endif // SPHERICAL_INTEGRATION_HPP_INCLUDED
