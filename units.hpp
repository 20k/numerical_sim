#ifndef UNITS_HPP_INCLUDED
#define UNITS_HPP_INCLUDED

namespace units
{
    inline
    double kg_to_m(double kg)
    {
        double C = 299792458.;
        double G = 6.67430 * pow(10., -11.);

        return kg * G / (C*C);
    }

    static inline double C = 299792458;
}

#endif // UNITS_HPP_INCLUDED
