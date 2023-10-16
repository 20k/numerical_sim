#ifndef SPHERICAL_INTEGRATION_HPP_INCLUDED
#define SPHERICAL_INTEGRATION_HPP_INCLUDED

#include <vector>
#include <assert.h>
#include "legendre_nodes.h"
#include "legendre_weights.h"
#include <geodesic/dual_value.hpp>
#include <toolkit/opencl.hpp>
#include "async_read_queue.hpp"

///https://pomax.github.io/bezierinfo/legendre-gauss.html
///https://cbeentjes.github.io/files/Ramblings/QuadratureSphere.pdf
///http://homepage.divms.uiowa.edu/~atkinson/papers/SphereQuad1982.pdf
template<typename T, typename U>
inline
auto integrate_1d_raw(const T& func, int n, const U& upper, const U& lower)
{
    std::vector<float> weights = get_legendre_weights(n);
    std::vector<float> nodes = get_legendre_nodes(n);

    using variable_type = decltype(func(0.f));

    variable_type sum = 0;

    for(int j=0; j < n; j++)
    {
        float w = weights[j];
        float xj = nodes[j];

        U value = ((upper - lower)/2.f) * xj + (upper + lower) / 2.f;

        auto func_eval = w * func(value);

        sum = sum + func_eval;
    }

    return ((upper - lower) / 2.f) * sum;
}

///why does this function exist?
template<typename T, typename U>
inline
auto integrate_1d(const T& func, int n, const U& upper, const U& lower)
{
    using variable_type = decltype(func(0.f));
    variable_type sum =  0;

    int pieces = 1;
    U step = (upper - lower) / pieces;

    for(int i=0; i < pieces; i++)
    {
        sum += integrate_1d_raw(func, n, (i + 1) * step + lower, i * step + lower);
    }

    return sum;
}

template<typename T, typename U>
inline
auto integrate_1d_trapezoidal(const T& func, int n, const U& upper, const U& lower)
{
    using variable_type = decltype(func(0.f));

    variable_type sum = 0;

    for(int k=1; k < n; k++)
    {
        auto coordinate = lower + k * (upper - lower) / n;

        auto val = func(coordinate);

        sum += val;
    }

    return ((upper - lower) / n) * (func(lower)/2.f + sum + func(upper)/2.f);
}

template<typename T, typename U>
inline
auto integrate_3d_trapezoidal(const T& func, int n, const U& upper, const U& lower)
{
    auto z_integral = [&](auto z)
    {
        auto y_integral = [&](auto y)
        {
            auto x_integral = [&](auto x)
            {
                return func(x,y,z);
            };

            return integrate_1d_trapezoidal(x_integral, n, upper[0], lower[0]);
        };

        return integrate_1d_trapezoidal(y_integral, n, upper[1], lower[1]);
    };

    return integrate_1d_trapezoidal(z_integral, n, upper[2], lower[2]);
}

template<typename T>
inline
auto spherical_integrate(const T& f_theta_phi, int n, float radius = 1)
{
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

    ///expand sphere area over unit sphere
    return integrate_1d(outer_integral, n, iupper, ilower) * radius * radius;
}

///cartesian domain is [-radius, +radius]
template<typename T>
inline
auto cartesian_integrate(const T& f_pos, int n, float cartesian_radius = 1.f, float sphere_radius = 1.f)
{
    auto cartesian_function = [&](float theta, float phi)
    {
        vec3f pos = {cartesian_radius * cos(phi) * sin(theta), cartesian_radius * sin(phi) * sin(theta), cartesian_radius * cos(theta)};

        return f_pos(pos);
    };

    return spherical_integrate(cartesian_function, n, sphere_radius);
}

std::vector<cl_ushort4> get_spherical_integration_points(vec3i dim, int extract_pixel);
float linear_interpolate(const std::map<int, std::map<int, std::map<int, float>>>& vals_map, vec3f pos, vec3i dim);

struct integrator
{
    int extract_pixel = 0;
    float scale = 0.f;

    std::vector<cl_ushort4> points;

    cl::command_queue& read_queue;
    async_read_queue<float> arq;
    cl::buffer gpu_points;
    vec3i dim;

    integrator(cl::context& ctx, vec3i _dim, float scale, cl::command_queue& _read_queue);

    std::vector<float> integrate();
};

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
