#ifndef UNITS_HPP_INCLUDED
#define UNITS_HPP_INCLUDED

namespace units
{
    constexpr static double C = 299792458;

    constexpr
    double kg_to_m(double kg)
    {
        double G = 6.67430 * pow(10., -11.);

        return kg * G / (units::C*units::C);
    }
}

#endif // UNITS_HPP_INCLUDED
