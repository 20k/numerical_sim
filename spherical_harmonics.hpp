#ifndef SPHERICAL_HARMONICS_HPP_INCLUDED
#define SPHERICAL_HARMONICS_HPP_INCLUDED

#include <map>
#include <tuple>
#include <geodesic/dual_value.hpp>

inline
int64_t factorial(int64_t i)
{
    if(i == 0)
        return 1;

    return i * factorial(i - 1);
}

inline
double factorial_d(int64_t i)
{
    return factorial(i);
}

template<typename T>
inline
dual_types::complex<T> expi_s(T val)
{
    return dual_types::complex<T>(cos(val), sin(val));
}

///https://arxiv.org/pdf/1906.03877.pdf 8
///aha!
///https://arxiv.org/pdf/0709.0093.pdf
///https://arxiv.org/pdf/gr-qc/0610128.pdf 40
///at last! A non horrible reference and non gpl reference for negative spin!
///this is where the cactus code comes from as well
template<typename T>
inline
dual_types::complex<T> sYlm_2(int s, int l, int m, T theta, T phi)
{
    thread_local std::map<std::tuple<int, int, int, T, T>, dual_types::complex<T>> cache;

    if(auto found_it = cache.find(std::tuple{s, l, m, theta, phi}); found_it != cache.end())
        return found_it->second;

    auto dlms = [](T theta, int l, int m, int s)
    {
        int k1 = std::max(0, m - s);
        int k2 = std::min(l + m, l - s);

        std::array<double, 4> top = {sqrt(factorial_d(l + m)), sqrt(factorial_d(l - m)), sqrt(factorial_d(l + s)), sqrt(factorial_d(l - s))};
        std::sort(top.begin(), top.end());

        T sum = 0;

        for(int k=k1; k <= k2; k++)
        {
            std::array<double, 4> bottom = {factorial_d(l + m - k), factorial_d(l - s - k), factorial_d(k), factorial_d(k + s - m)};
            std::sort(bottom.begin(), bottom.end());

            double run = 1;

            for(int sum = 3; sum >= 0; sum--)
            {
                run *= top[sum] / bottom[sum];
            }

            float cp1 = run;

            assert(isfinite(cp1));

            T cp2 = pow(cos(theta/2.f), 2 * l + m - s - 2 * k);
            T cp3 = pow(sin(theta/2.f), 2 * k + s - m);

            sum = sum + cp1 * cp2 * cp3;
        }

        return sum;
    };

    T coeff = pow(-1, s) * sqrt((2 * l + 1) / (4 * M_PI));

    dual_types::complex<T> ret = coeff * dlms(theta, l, m, -s) * expi_s(m * phi);

    cache[std::tuple{s, l, m, theta, phi}] = ret;

    return ret;
}

inline
void test_harmonics()
{
    float tol = 0.000001f;

    {
        for(float theta = 0; theta <= M_PI; theta += M_PI/8)
        {
            for(float phi = 0; phi <= 2 * M_PI; phi += M_PI/16)
            {
                {
                    dual_types::complex<float> ym222 = sqrt(5 / (64 * M_PI)) * (1 - cos(theta)) * (1 - cos(theta)) * expi_s(-2 * phi);

                    dual_types::complex<float> ym222_c = sYlm_2(-2, 2, -2, theta, phi);

                    assert(fabs(ym222.real - ym222_c.real) < tol);
                    assert(fabs(ym222.imaginary - ym222_c.imaginary) < tol);
                }

                {
                    dual_types::complex<float> y222 = sqrt(5 / (64 * M_PI)) * (1 + cos(theta)) * (1 + cos(theta)) * expi_s(2 * phi);

                    dual_types::complex<float> y222_c = sYlm_2(-2, 2, 2, theta, phi);

                    assert(fabs(y222.real - y222_c.real) < tol);
                    assert(fabs(y222.imaginary - y222_c.imaginary) < tol);
                }
            }
        }
    }
}


#endif // SPHERICAL_HARMONICS_HPP_INCLUDED
