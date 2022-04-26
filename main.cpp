#include <iostream>
#include <toolkit/render_window.hpp>
#include <toolkit/texture.hpp>
#include <vec/vec.hpp>
#include <GLFW/glfw3.h>
#include <SFML/Graphics.hpp>
#include <CL/cl_ext.h>
#include <geodesic/dual.hpp>
#include <geodesic/dual_value.hpp>
#include <geodesic/numerical.hpp>
#include <fstream>
#include <imgui/misc/freetype/imgui_freetype.h>
#include <vec/tensor.hpp>
#include "gravitational_waves.hpp"
#include <execution>
#include <thread>
#include "mesh_manager.hpp"
#include "spherical_harmonics.hpp"
#include "spherical_integration.hpp"

/**
current paper set
https://arxiv.org/pdf/gr-qc/0505055.pdf - fourth order numerical paper, finite differencing
https://arxiv.org/pdf/gr-qc/0511048.pdf - good reference, th eone with lies and bssn
https://arxiv.org/pdf/gr-qc/0206072.pdf - binary initial conditions, caij
https://link.springer.com/article/10.12942/lrr-2000-5#Equ48 - initial conditions
https://arxiv.org/pdf/gr-qc/9703066.pdf - binary black hole initial conditions
https://arxiv.org/pdf/gr-qc/0605030.pdf - moving puncture gauge conditions
https://arxiv.org/pdf/1404.6523.pdf - gauge conditions, 2014 so check this one again
https://arxiv.org/pdf/gr-qc/0005043.pdf - initial conditions
https://cds.cern.ch/record/337814/files/9711015.pdf - initial conditions
https://cds.cern.ch/record/517706/files/0106072.pdf - general review
https://scholarworks.rit.edu/cgi/viewcontent.cgi?article=11286&context=theses - thesis
https://arxiv.org/pdf/0704.0782.pdf - boundary conditions
https://d-nb.info/999452606/34 - wave extraction
https://arxiv.org/pdf/gr-qc/0204002.pdf - bssn modification
https://arxiv.org/pdf/0811.1600.pdf - wave extraction
https://arxiv.org/pdf/1202.1038.pdf - bssn modification
https://arxiv.org/pdf/1606.02532.pdf - extracting waveforms
https://arxiv.org/pdf/gr-qc/0104063.pdf - this gives the tetrads in 5.6c
https://asd.gsfc.nasa.gov/archive/astrogravs/docs/lit/ringdown_date.html - stuff
https://www.black-holes.org/
https://aip.scitation.org/doi/am-pdf/10.1063/1.4962723
https://arxiv.org/pdf/1906.03877.pdf - spherical harmonics
https://arxiv.org/pdf/1412.4590.pdf - interesting glue of initial conditions to end up with a schwarzschild tail
http://gravity.psu.edu/numrel/jclub/jc/Cook___LivRev_2000-5.pdf - seems to be a good reference on initial conditions

https://arxiv.org/pdf/0812.3752.pdf - numerical methods, this paper seems very useful for finite differencing etc. Has 2nd order finite difference non uniform
https://arxiv.org/pdf/0706.0740.pdf - contains explicit upwind stencils, as well as material on numerical dissipation
https://physics.princeton.edu//~fpretori/AST523_NR_b.pdf - has an explicit expansion for kreiss-oliger
https://hal.archives-ouvertes.fr/hal-00569776/document - contains the D+- operators, and a higher order kreiss-oliger expansion

https://link.springer.com/article/10.12942/lrr-2007-3 - event horizon finding
https://arxiv.org/pdf/gr-qc/9412071.pdf - misc numerical relativity, old
https://arxiv.org/pdf/gr-qc/0703035.pdf - lots of hyper useful information on the adm formalism

https://arxiv.org/pdf/gr-qc/0007085.pdf - initial conditions, explanations and isotropic radial coordinates
https://indico.cern.ch/event/505595/contributions/1183661/attachments/1332828/2003830/sperhake.pdf - contains bssn in terms of X
https://arxiv.org/pdf/gr-qc/0612001.pdf - double kerr initial data, should be solvable via an elliptic solver
https://arxiv.org/pdf/gr-qc/0610128.pdf - this paper uses psi0 as the initial guess for lapse, not psibl
https://learn.lboro.ac.uk/archive/olmp/olmp_resources/pages/workbooks_1_50_jan2008/Workbook33/33_2_elliptic_pde.pdf five point stencil
https://arxiv.org/pdf/2008.12931.pdf - this contains a good set of equations to try for a more stable bssn
https://arxiv.org/pdf/gr-qc/0004050.pdf - ISCO explanation
https://core.ac.uk/download/pdf/144448463.pdf - 7.9 states you can split up trace free variables

https://github.com/GRChombo/GRChombo useful info
https://arxiv.org/pdf/gr-qc/0505055.pdf - explicit upwind stencils
https://arxiv.org/pdf/1205.5111v1.pdf - paper on numerical stability
https://arxiv.org/pdf/1208.3927.pdf - adm geodesics

https://arxiv.org/pdf/gr-qc/0404056.pdf - seems to suggest an analytic solution to bowen-york
0908.1063.pdf - analytic solution to initial data
https://arxiv.org/pdf/1410.8607.pdf - haven't read it yet but this promises everything i want

https://authors.library.caltech.edu/8284/1/RINcqg07.pdf - this gives a conflicting definition of kreiss oliger under B.4/B.5
https://arxiv.org/pdf/1503.03436.pdf - seems to have a usable radiative boundary condition
https://arxiv.org/pdf/gr-qc/0212039.pdf - apparent horizon?

https://iopscience.iop.org/article/10.1088/1361-6633/80/2/026901/ampdf - defines adm mass
https://arxiv.org/pdf/0707.0339.pdf - interesting paper analysing stability of bssn, has good links
https://arxiv.org/pdf/gr-qc/0209102.pdf - neutron stars, boundary conditions, hamiltonian constraints
https://arxiv.org/pdf/gr-qc/0501043.pdf - hamiltonian constraint
https://orca.cardiff.ac.uk/114952/1/PhysRevD.98.044014.pdf - this paper states that bowen-york suffers hamiltonian constraint violations, which... explains a lot
https://arxiv.org/pdf/0908.1063.pdf - gives an estimate of black hole mass from apparent horizon. Analysis of trumpet initial conditions shows they're unfortunately the same as non trumpet
https://arxiv.org/pdf/0912.2920.pdf - z4c
https://arxiv.org/pdf/astro-ph/0408492v1.pdf - kicks
https://arxiv.org/pdf/2108.05119.pdf - apparently horizons
https://arxiv.org/pdf/gr-qc/0209066.pdf - hamiltonian constraint damping
*/

///notes:
///off centre black hole results in distortion, probably due to boundary conditions contaminating things
///this is odd. Maybe don't boundary condition shift and lapse?

//#define USE_GBB

///https://scholarworks.rit.edu/cgi/viewcontent.cgi?article=11286&context=theses
///38.2

///https://arxiv.org/pdf/gr-qc/9810065.pdf
template<typename T, int N>
inline
T gpu_trace(const tensor<T, N, N>& mT, const metric<T, N, N>& met, const inverse_metric<T, N, N>& inverse)
{
    T ret = 0;

    for(int i=0; i < N; i++)
    {
        for(int j=0; j < N; j++)
        {
            ret = ret + inverse.idx(i, j) * mT.idx(i, j);
        }
    }

    return ret;
}

struct equation_context
{
    std::vector<std::pair<std::string, value>> values;
    std::vector<std::pair<std::string, value>> temporaries;
    std::vector<std::pair<value, value>> aliases;
    bool uses_linear = false;
    bool debug = false;

    int order = 2;

    void pin(value& v)
    {
        for(auto& i : temporaries)
        {
            if(dual_types::equivalent(v, i.second))
            {
                value facade;
                facade.make_value(i.first);

                v = facade;
                return;
            }
        }

        std::string name = "pv" + std::to_string(temporaries.size());
        //std::string name = "pv[" + std::to_string(temporaries.size()) + "]";

        value old = v;

        temporaries.push_back({name, old});

        value facade;
        facade.make_value(name);

        v = facade;
    }

    template<typename T, int N>
    void pin(tensor<T, N>& mT)
    {
        for(int i=0; i < N; i++)
        {
            pin(mT.idx(i));
        }
    }

    template<typename T, int N>
    void pin(vec<N, T>& mT)
    {
        for(int i=0; i < N; i++)
        {
            pin(mT[i]);
        }
    }

    template<typename T, int N>
    void pin(tensor<T, N, N>& mT)
    {
        for(int i=0; i < N; i++)
        {
            for(int j=0; j < N; j++)
            {
                pin(mT.idx(i, j));
            }
        }
    }

    template<typename T, int N>
    void pin(tensor<T, N, N, N>& mT)
    {
        for(int i=0; i < N; i++)
        {
            for(int j=0; j < N; j++)
            {
                for(int k=0; k < N; k++)
                {
                    pin(mT.idx(i, j, k));
                }
            }
        }
    }

    template<typename T, int N>
    void pin(inverse_metric<T, N, N>& mT)
    {
        for(int i=0; i < N; i++)
        {
            for(int j=0; j < N; j++)
            {
                pin(mT.idx(i, j));
            }
        }
    }

    template<typename T, int N>
    void pin(metric<T, N, N>& mT)
    {
        for(int i=0; i < N; i++)
        {
            for(int j=0; j < N; j++)
            {
                pin(mT.idx(i, j));
            }
        }
    }

    /*template<typename T>
    void pin(T& mT)
    {
        pin(mT.to_concrete());
    }*/

    void alias(const value& concrete, const value& alias)
    {
        for(auto& i : aliases)
        {
            if(dual_types::equivalent(concrete, i.first))
            {
                //std::cout << "CONC " << type_to_string(concrete)  << " ALIAS " << type_to_string(alias) << std::endl;
                //std::cout << "ALIAS " << type_to_string(alias) << " WITH " << type_to_string(i.second) << std::endl;

                assert(dual_types::equivalent(alias, i.second));
                return;
            }
        }

        aliases.push_back({concrete, alias});
    }

    void add(const std::string& name, const value& v)
    {
        values.push_back({name, v});
    }

    void strip_unused()
    {
        std::set<std::string> used_names;

        for(auto& i : values)
        {
            auto& name = i.first;
            value& v = i.second;

            used_names.insert(name);

            std::vector<std::string> all_used = v.get_all_variables();

            for(auto& k : all_used)
            {
                used_names.insert(k);
            }
        }

        std::vector<std::pair<std::string, value>> unprocessed_temporaries = temporaries;

        bool any_change = true;

        while(any_change && unprocessed_temporaries.size() > 0)
        {
            any_change = false;

            for(int i=0; i < (int)unprocessed_temporaries.size(); i++)
            {
                std::pair<std::string, value>& next = unprocessed_temporaries[i];

                //unprocessed_temporaries.erase(unprocessed_temporaries.begin());

                if(used_names.find(next.first) != used_names.end())
                {
                    std::vector<std::string> all_used = next.second.get_all_variables();

                    used_names.insert(next.first);

                    for(auto& kk : all_used)
                    {
                        used_names.insert(kk);
                    }

                    any_change = true;

                    unprocessed_temporaries.erase(unprocessed_temporaries.begin() + i);
                    i--;
                    continue;
                }
            }
        }

        std::set<std::string> to_erase;

        for(auto& i : unprocessed_temporaries)
        {
            to_erase.insert(i.first);
        }

        for(int i=0; i < (int)temporaries.size(); i++)
        {
            if(to_erase.find(temporaries[i].first) != to_erase.end())
            {
                temporaries.erase(temporaries.begin() + i);
                i--;
                continue;
            }
        }
    }

    void substutite_aliases()
    {
        for(auto& [name, v] : values)
        {
            v.substitute(aliases);
        }

        for(auto& [name, v] : temporaries)
        {
            v.substitute(aliases);
        }
    }

    void build_impl(std::string& argument_string, const std::string& str)
    {
        strip_unused();
        substutite_aliases();

        for(auto& i : values)
        {
            std::string str = "-D" + i.first + "=" + type_to_string(i.second) + " ";

            argument_string += str;
        }

        if(temporaries.size() == 0)
        {
            argument_string += "-DTEMPORARIES" + str + "=DUMMY ";
            return;
        }

        std::string temporary_string;

        for(auto& [current_name, value] : temporaries)
        {
            temporary_string += current_name + "=" + type_to_string(value) + ",";
        }

        ///remove trailing comma
        if(temporary_string.size() > 0)
            temporary_string.pop_back();

        argument_string += "-DTEMPORARIES" + str + "=" + temporary_string + " ";
    }

    void build(std::string& argument_string, const std::string& str)
    {
        int old_length = argument_string.size();

        build_impl(argument_string, str);

        int new_length = argument_string.size();

        std::cout << "EXTRA LENGTH " << (new_length - old_length) << " " << str << std::endl;
    }

    void build(std::string& argument_string, int idx)
    {
        build(argument_string, std::to_string(idx));
    }
};

//#define SYMMETRY_BOUNDARY
#define BORDER_WIDTH 4

std::string strip_variable(std::string in)
{
    std::string valid;

    for(auto i : in)
    {
        if(std::isalnum(i))
        {
            valid.push_back(i);
        }
        else
        {
            break;
        }
    }

    return valid;
}

struct variable
{
    std::string name;
    bool is_derivative = false;
};

inline
std::vector<variable> get_variables()
{
    std::array variables
    {
        "cY0",
        "cY1",
        "cY2",
        "cY3",
        "cY4",
        "cY5",
        "cA0",
        "cA1",
        "cA2",
        "cA3",
        "cA4",
        "cA5",
        "cGi0",
        "cGi1",
        "cGi2",
        "X",
        "K",
        "gA",
        "gB0",
        "gB1",
        "gB2",
        "gBB0",
        "gBB1",
        "gBB2",

        "momentum0",
        "momentum1",
        "momentum2",
    };

    std::array derivatives
    {
        "dcYij0",
        "dcYij1",
        "dcYij2",
        "dcYij3",
        "dcYij4",
        "dcYij5",
        "dcYij6",
        "dcYij7",
        "dcYij8",
        "dcYij9",
        "dcYij10",
        "dcYij11",
        "dcYij12",
        "dcYij13",
        "dcYij14",
        "dcYij15",
        "dcYij16",
        "dcYij17",

        "digA0",
        "digA1",
        "digA2",

        "digB0",
        "digB1",
        "digB2",
        "digB3",
        "digB4",
        "digB5",
        "digB6",
        "digB7",
        "digB8",

        "dX0",
        "dX1",
        "dX2",
    };

    std::vector<variable> ret;

    for(auto& i : variables)
    {
        variable v;
        v.name = i;
        v.is_derivative = false;

        ret.push_back(v);
    }

    for(auto& i : derivatives)
    {
        variable v;
        v.name = i;
        v.is_derivative = true;

        ret.push_back(v);
    }

    return ret;
}

inline
variable fetch_variable(const std::string& name)
{
    std::vector<variable> ret = get_variables();

    for(variable& v : ret)
    {
        if(v.name == name)
            return v;
    }

    throw std::runtime_error("Err in fetch variable for name " + name);
}

value as_float3(const value& x, const value& y, const value& z)
{
    return dual_types::apply("(float3)", x, y, z);
}

template<int elements = 5>
struct differentiation_context
{
    std::array<value, elements> vars;

    std::array<value, elements> xs;
    std::array<value, elements> ys;
    std::array<value, elements> zs;

    differentiation_context(const value& in, int idx, bool linear_interpolation = false)
    {
        for(int i=0; i < elements; i++)
        {
            if(linear_interpolation)
            {
                xs[i] = "fx";
                ys[i] = "fy";
                zs[i] = "fz";
            }
            else
            {
                xs[i] = "ix";
                ys[i] = "iy";
                zs[i] = "iz";
            }
        }

        for(int i=0; i < elements; i++)
        {
            int offset = i - (elements - 1)/2;

            if(idx == 0)
                xs[i] += offset;
            if(idx == 1)
                ys[i] += offset;
            if(idx == 2)
                zs[i] += offset;
        }

        std::vector<value> indexed_variables;

        in.recurse_arguments([&indexed_variables, linear_interpolation](const value& v)
        {
            if(v.type != dual_types::ops::UNKNOWN_FUNCTION)
                return;

            std::string function_name = type_to_string(v.args[0]);

            if(function_name != "buffer_index" && function_name != "buffer_indexh" &&
               function_name != "buffer_read_linear" && function_name != "buffer_read_linearh")
                return;

            if(linear_interpolation)
                assert(function_name == "buffer_read_linear" || function_name == "buffer_read_linearh");
            else
                assert(function_name == "buffer_index" || function_name == "buffer_indexh");

            #ifdef CHECK_HALF_PRECISION
            std::string vname = type_to_string(v.args[1]);

            std::vector<variable> test_vars = get_variables();

            for(const variable& them : test_vars)
            {
                if(vname == them.name)
                {
                    if(linear_interpolation && them.is_derivative)
                        assert(function_name == "buffer_read_linearh");
                    else if(linear_interpolation && !them.is_derivative)
                        assert(function_name == "buffer_read_linear");
                    else if(!linear_interpolation && them.is_derivative)
                        assert(function_name == "buffer_indexh");
                    else if(!linear_interpolation && !them.is_derivative)
                        assert(function_name == "buffer_index");
                    else
                    {
                        std::cout << "FNAME " << type_to_string(v) << std::endl;
                        assert(false);
                    }
                }
            }
            #endif // CHECK_HALF_PRECISION

            indexed_variables.push_back(v);
        });

        if(indexed_variables.size() == 0)
        {
            std::cout << "WHAT? " << type_to_string(in) << std::endl;
        }

        assert(indexed_variables.size() > 0);

        std::array<std::vector<value>, elements> substitutions;

        for(auto& variables : indexed_variables)
        {
            std::string function_name = type_to_string(variables.args.at(0));

            for(int kk=0; kk < elements; kk++)
            {
                value to_sub;

                if(function_name == "buffer_index" || function_name == "buffer_indexh")
                {
                    to_sub = apply(function_name, variables.args[1], xs[kk], ys[kk], zs[kk], "dim");
                }
                else if(function_name == "buffer_read_linear" || function_name == "buffer_read_linearh")
                {
                    to_sub = apply(function_name, variables.args[1], as_float3(xs[kk], ys[kk], zs[kk]), "dim");
                }
                else
                {
                    assert(false);
                }

                substitutions[kk].push_back(to_sub);
            }
        }

        for(int i=0; i < elements; i++)
        {
            vars[i] = in;

            ///look for a function which matches our indexed variables
            ///if we find it, substitute for the substitution
            vars[i].recurse_arguments([&substitutions, &indexed_variables, i](value& v)
            {
                assert(substitutions[i].size() == indexed_variables.size());

                ///search through the indexed variables
                for(int kk=0; kk < (int)indexed_variables.size(); kk++)
                {
                    ///its a me!
                    if(dual_types::equivalent(indexed_variables[kk], v))
                    {
                        ///substitute us for the directional derivative
                        v = substitutions[i][kk];
                        return;
                    }
                }
            });
        }
    }
};

value get_distance(const vec<3, value>& p1, const vec<3, value>& p2)
{
    return "get_distance(" + type_to_string(p1.x(), true) + "," + type_to_string(p1.y(), true) + "," + type_to_string(p1.z(), true) + "," + type_to_string(p2.x(), true) + "," + type_to_string(p2.y(), true) + "," + type_to_string(p2.z(), true) + ",dim,scale)";
}

value get_scale_distance(equation_context& ctx, const value& in, int idx, int which)
{
    differentiation_context<3> dctx(in, idx);

    value final_command;

    value ix = "ix";
    value iy = "iy";
    value iz = "iz";

    std::string ix0 = type_to_string(ix, true);
    std::string iy0 = type_to_string(iy, true);
    std::string iz0 = type_to_string(iz, true);

    value h = "get_distance(" + ix0 + "," + iy0 + "," + iz0 + "," + type_to_string(dctx.xs[2]) + "," + type_to_string(dctx.ys[2]) + "," + type_to_string(dctx.zs[2]) + ",dim,scale)";
    value k = "get_distance(" + ix0 + "," + iy0 + "," + iz0 + "," + type_to_string(dctx.xs[0]) + "," + type_to_string(dctx.ys[0]) + "," + type_to_string(dctx.zs[0]) + ",dim,scale)";

    if(which == 0)
        return h;

    if(which == 1)
        return k;

    return "error";

    //return "scale";
}

vec<3, value> get_idx_offset(int idx)
{
    if(idx == 0)
        return vec<3, value>{"1", "0", "0"};
    if(idx == 1)
        return vec<3, value>{"0", "1", "0"};
    if(idx == 2)
        return vec<3, value>{"0", "0", "1"};

    return {"error0", "error1", "error2"};
}

#define DIFFERENTIATION_WIDTH 3

value first_derivative(equation_context& ctx, const value& in, int idx)
{
    differentiation_context<3> dctx(in, idx);

    value h = get_scale_distance(ctx, in, idx, 0);
    value k = get_scale_distance(ctx, in, idx, 1);

    ///f(x + h) - f(x - k)
    value final_command = (dctx.vars[2] - dctx.vars[0]) / (h + k);

    return final_command;
}

value second_derivative(equation_context& ctx, const value& in, int idx)
{
    differentiation_context<5> dctx(in, idx);

    vec<3, value> pos = {"ix", "iy", "iz"};

    vec<3, value> local_offset = get_idx_offset(idx);

    value tdxm2 = get_distance(pos, pos - local_offset * 2);
    value tdxm1 = get_distance(pos, pos - local_offset);
    value tdxp1 = get_distance(pos, pos + local_offset);
    value tdxp2 = get_distance(pos, pos + local_offset * 2);

    value um2 = dctx.vars[0];
    value um1 = dctx.vars[1];
    value u0  = dctx.vars[2];
    value up1 = dctx.vars[3];
    value up2 = dctx.vars[4];

    return (um1 + up1 - 2 * u0) / (0.5 * (tdxp1 * tdxp1 + tdxm1 * tdxm1));
}

value fourth_derivative(equation_context& ctx, const value& in, int idx)
{
    differentiation_context<5> dctx(in, idx);

    vec<3, value> pos = {"ix", "iy", "iz"};

    vec<3, value> local_offset = get_idx_offset(idx);

    value tdxm2 = get_distance(pos, pos - local_offset * 2);
    value tdxm1 = get_distance(pos, pos - local_offset);
    value tdxp1 = get_distance(pos, pos + local_offset);
    value tdxp2 = get_distance(pos, pos + local_offset * 2);

    value um2 = dctx.vars[0];
    value um1 = dctx.vars[1];
    value u0  = dctx.vars[2];
    value up1 = dctx.vars[3];
    value up2 = dctx.vars[4];

    value lhs = um2 - 4 * um1 + 6 * u0 - 4 * up1 + up2;

    float coeff = 1.f/24.f;

    return lhs / (coeff * pow(tdxm2, 4) - 4 * coeff * pow(tdxm1, 4) - 4 * coeff * pow(tdxp1, 4) + coeff * pow(tdxp2, 4));
}

///https://hal.archives-ouvertes.fr/hal-00569776/document this paper implies you simply sum the directions
///dissipation is fixing some stuff, todo: investigate why so much dissipation is required
value kreiss_oliger_dissipate_dir(equation_context& ctx, const value& in, int idx)
{
    //std::cout << "TEST " << type_to_string(second_derivative(ctx, in, {"0", "0", "0"}, idx)) << std::endl;

    value h = get_scale_distance(ctx, in, idx, 0);
    value k = get_scale_distance(ctx, in, idx, 1);

    ///todo: fix this
    value effective_scale = (h + k) / 2.f;

    ///https://en.wikipedia.org/wiki/Finite_difference_coefficient according to wikipedia, this is the 6th derivative with 2nd order accuracy. I am confused, but at least I know where it came from
    value scale = "scale";

    //#define FOURTH
    #ifdef FOURTH
    differentiation_context<5> dctx(in, idx);
    //value stencil = -(1 / (16.f * effective_scale)) * (dctx.vars[0] - 4 * dctx.vars[1] + 6 * dctx.vars[2] - 4 * dctx.vars[3] + dctx.vars[4]);

    value stencil = (-1 / 16.f) * pow(effective_scale, 3.f) * fourth_derivative(ctx, in, idx);

    #endif // FOURTH

    #define SIXTH
    #ifdef SIXTH
    differentiation_context<7> dctx(in, idx);
    value stencil = (1 / (64.f * scale)) * (dctx.vars[0] - 6 * dctx.vars[1] + 15 * dctx.vars[2] - 20 * dctx.vars[3] + 15 * dctx.vars[4] - 6 * dctx.vars[5] + dctx.vars[6]);
    #endif // SIXTH


    return stencil;
}

value kreiss_oliger_dissipate(equation_context& ctx, const value& in)
{
    //#define NO_KREISS
    #ifndef NO_KREISS
    value fin = 0;

    for(int i=0; i < 3; i++)
    {
        fin = fin + kreiss_oliger_dissipate_dir(ctx, in, i);
    }

    return fin;
    #else
    return 0;
    #endif
}

void build_kreiss_oliger_dissipate_singular(equation_context& ctx)
{
    value buf = dual_types::apply("buffer_index", "buffer", "ix", "iy", "iz", "dim");

    value coeff = "coefficient";

    ctx.add("KREISS_DISSIPATE_SINGULAR", coeff * kreiss_oliger_dissipate(ctx, buf));
}

#if 0
template<int order = 2>
value hacky_differentiate(const value& in, int idx, bool pin = true, bool linear = false)
{
    value scale = "scale";

    value final_command;

    if(order == 1)
    {
        differentiation_context dctx(in, idx, linear);
        std::array<value, 5> vars = dctx.vars;

        final_command = (vars[3] - vars[1]) / (2 * scale);
    }
    else if(order == 2)
    {
        differentiation_context dctx(in, idx, linear);
        std::array<value, 5> vars = dctx.vars;

        final_command = (-vars[4] + 8 * vars[3] - 8 * vars[1] + vars[0]) / (12 * scale);
    }
    else if(order == 3)
    {
        differentiation_context<7> dctx(in, idx, linear);
        std::array<value, 7> vars = dctx.vars;

        final_command = (-(1/60.f) * vars[0] + (3/20.f) * vars[1] - (3/4.f) * vars[2] + 0 * vars[3] + (3/4.f) * vars[4] - (3/20.f) * vars[5] + (1/60.f) * vars[6]) / scale;
    }

    static_assert(order == 1 || order == 2 || order == 3);

    /*value final_command;

    {
        value h = "get_distance(ix,iy,iz," + dctx.xs[3] + "," + dctx.ys[3] + "," + dctx.zs[3] + ",dim,scale)";
        value k = "get_distance(ix,iy,iz," + dctx.xs[1] + "," + dctx.ys[1] + "," + dctx.zs[1] + ",dim,scale)";

        if(pin)
        {
            ctx.pin(h);
            ctx.pin(k);
        }

        ///f(x + h) - f(x - k)
        final_command = (vars[3] - vars[1]) / (h + k);
    }*/

    return final_command;
}
#endif // 0

value diff1(equation_context& ctx, const value& in, int idx)
{
    int order = ctx.order;

    value scale = "scale";

    if(order == 1)
    {
        differentiation_context<3> dctx(in, idx, ctx.uses_linear);
        std::array<value, 3> vars = dctx.vars;

        return (vars[2] - vars[0]) / (2 * scale);
    }
    else if(order == 2)
    {
        differentiation_context dctx(in, idx, ctx.uses_linear);
        std::array<value, 5> vars = dctx.vars;

        return (-vars[4] + 8 * vars[3] - 8 * vars[1] + vars[0]) / (12 * scale);
    }
    else if(order == 3)
    {
        differentiation_context<7> dctx(in, idx, ctx.uses_linear);
        std::array<value, 7> vars = dctx.vars;

        return (-(1/60.f) * vars[0] + (3/20.f) * vars[1] - (3/4.f) * vars[2] + 0 * vars[3] + (3/4.f) * vars[4] - (3/20.f) * vars[5] + (1/60.f) * vars[6]) / scale;
    }
    else if(order == 4)
    {
        differentiation_context<9> dctx(in, idx, ctx.uses_linear);
        std::array<value, 9> vars = dctx.vars;

        return ((1/280.f) * vars[0] - (4/105.f) * vars[1] + (1/5.f) * vars[2] - (4/5.f) * vars[3] + (4/5.f) * vars[5] - (1/5.f) * vars[6] + (4/105.f) * vars[7] - (1/280.f) * vars[8]) / scale;
    }

    assert(false);
    return 0;
}

value diff2(equation_context& ctx, const value& in, int idx, int idy, const value& first_x, const value& first_y)
{
    int order = ctx.order;

    value scale = "scale";

    /*if(idx == idy)
    {
        if(order == 1)
        {
            differentiation_context<3> dctx(in, idx, ctx.uses_linear);
            std::array<value, 3> vars = dctx.vars;

            return (vars[0] - 2 * vars[1] + vars[2]) / (scale * scale);
        }
        else if(order == 2)
        {
            differentiation_context<5> dctx(in, idx, ctx.uses_linear);
            std::array<value, 5> vars = dctx.vars;

            return (-vars[0] + 16 * vars[1] - 30 * vars[2] + 16 * vars[3] - vars[4]) / (12 * scale * scale);
        }
        else if(order == 3)
        {
            differentiation_context<7> dctx(in, idx, ctx.uses_linear);
            std::array<value, 7> vars = dctx.vars;

            return ((1/90.f) * vars[0] - (3/20.f) * vars[1] + (3/2.f) * vars[2] - (49/18.f) * vars[3] + (3/2.f) * vars[4] - (3/20.f) * vars[5] + (1/90.f) * vars[6]) / (scale * scale);
        }
        else if(order == 4)
        {
            differentiation_context<9> dctx(in, idx, ctx.uses_linear);
            std::array<value, 9> vars = dctx.vars;

            return ((-1/560.f) * vars[0] + (8/315.f) * vars[1] - (1/5.f) * vars[2] + (8/5.f) * vars[3] - (205/72.f) * vars[4] + (8/5.f) * vars[5] - (1/5.f) * vars[6] + (8/315.f) * vars[7] - (1/560.f) * vars[8]) / (scale * scale);
        }
    }*/

    #define SYMMETRIC_DERIVATIVES
    #ifdef SYMMETRIC_DERIVATIVES
    if(idx < idy)
    {
        ///we're using first_y, so alias unconditionally
        ctx.alias(diff1(ctx, in, idy), first_y);

        return diff1(ctx, first_y, idx);
    }
    else
    {
        ctx.alias(diff1(ctx, in, idx), first_x);

        return diff1(ctx, first_x, idy);
    }
    #endif // SYMMETRIC_DERIVATIVES

    return diff1(ctx, first_x, idy);
}

/*tensor<value, 3> tensor_derivative(equation_context& ctx, const value& in)
{
    tensor<value, 3> ret;

    for(int i=0; i < 3; i++)
    {
        ret.idx(i) = hacky_differentiate(in, i);
    }

    return ret;
}

tensor<value, 3, 3> tensor_derivative(equation_context& ctx, const tensor<value, 3>& in)
{
    tensor<value, 3, 3> ret;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            ret.idx(i, j) = hacky_differentiate(in.idx(j), i);
        }
    }

    return ret;
}*/

///B^i * Di whatever
value upwind_differentiate(equation_context& ctx, const value& prefix, const value& in, int idx, bool pin = true)
{
    /*differentiation_context dctx(in, idx);

    value scale = "scale";

    ///https://en.wikipedia.org/wiki/Upwind_scheme
    value a_p = max(prefix, 0);
    value a_n = min(prefix, 0);

    //value u_n = (dctx.vars[2] - dctx.vars[1]) / (2 * scale);
    //value u_p = (dctx.vars[3] - dctx.vars[2]) / (2 * scale);

    value u_n = (3 * dctx.vars[2] - 4 * dctx.vars[1] + dctx.vars[0]) / (6 * scale);
    value u_p = (-dctx.vars[4] + 4 * dctx.vars[3] - 3 * dctx.vars[2]) / (6 * scale);

    ///- here probably isn't right
    ///neither is correct, this is fundamentally wrong somewhere
    value final_command = (a_p * u_n + a_n * u_p);

    return final_command;*/

    return prefix * diff1(ctx, in, idx);

    /*differentiation_context<7> dctx(in, idx);

    value scale = "scale";

    auto vars = dctx.vars;

    ///https://arxiv.org/pdf/gr-qc/0505055.pdf 2.5
    value stencil_negative = (-vars[0] + 6 * vars[1] - 18 * vars[2] + 10 * vars[3] + 3 * vars[4]) / (12.f * scale);
    value stencil_positive = (vars[6] - 6 * vars[5] + 18 * vars[4] - 10 * vars[3] - 3 * vars[2]) / (12.f * scale);

    return dual_if(prefix >= 0, [&]()
    {
        return prefix * stencil_positive;
    },
    [&](){
        return prefix * stencil_negative;
    });*/
}

tensor<value, 3> tensor_upwind(equation_context& ctx, const tensor<value, 3>& prefix, const value& in)
{
    tensor<value, 3> ret;

    for(int i=0; i < 3; i++)
    {
        ret.idx(i) = upwind_differentiate(ctx, prefix.idx(i), in, i);
    }

    return ret;
}

template<typename T, int N>
inline
T lie_derivative(equation_context& ctx, const tensor<T, N>& gB, const T& variable)
{
    ///https://en.wikipedia.org/wiki/Lie_derivative#Coordinate_expressions
    return sum(tensor_upwind(ctx, gB, variable));
}

/*tensor<value, 3, 3> tensor_upwind(equation_context& ctx, const tensor<value, 3>& prefix, const tensor<value, 3>& in)
{
    tensor<value, 3, 3> ret;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            ret.idx(j, i) = upwind_differentiate(ctx, prefix.idx(i), in.idx(i), i);
        }
    }

    return ret;
}*/

template<typename T, int N, SizedTensor<T, N, N> S>
inline
tensor<T, N, N> gpu_lie_derivative_weight(equation_context& ctx, const tensor<T, N>& B, const S& mT)
{
    tensor<T, N, N> lie;

    for(int i=0; i < N; i++)
    {
        for(int j=0; j < N; j++)
        {
            T sum = 0;

            for(int k=0; k < N; k++)
            {
                sum = sum + upwind_differentiate(ctx, B.idx(k), mT.idx(i, j), k);
                sum = sum + mT.idx(i, k) * diff1(ctx, B.idx(k), j);
                sum = sum + mT.idx(j, k) * diff1(ctx, B.idx(k), i);
                sum = sum - (2.f/3.f) * mT.idx(i, j) * diff1(ctx, B.idx(k), k);
            }

            lie.idx(i, j) = sum;
        }
    }

    return lie;
}

tensor<value, 3, 3, 3> get_eijk()
{
    auto eijk_func = [](int i, int j, int k)
    {
        if((i == 0 && j == 1 && k == 2) || (i == 1 && j == 2 && k == 0) || (i == 2 && j == 0 && k == 1))
            return 1;

        if(i == j || j == k || k == i)
            return 0;

        return -1;
    };

    tensor<value, 3, 3, 3> eijk;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                eijk.idx(i, j, k) = eijk_func(i, j, k);
            }
        }
    }

    return eijk;
}

///mT symmetric?
template<typename T, int N>
tensor<T, N, N> raise_index(const tensor<T, N, N>& mT, const metric<T, N, N>& met, const inverse_metric<T, N, N>& inverse)
{
    tensor<T, N, N> ret;

    for(int i=0; i < N; i++)
    {
        for(int j=0; j < N; j++)
        {
            T sum = 0;

            for(int k=0; k < N; k++)
            {
                sum = sum + inverse.idx(i, k) * mT.idx(k, j);
                //sum = sum + mT.idx(i, k) * inverse.idx(k, j);
            }

            ret.idx(i, j) = sum;
        }
    }

    return ret;
}

template<typename T, int N>
tensor<T, N, N> raise_second_index(const tensor<T, N, N>& mT, const metric<T, N, N>& met, const inverse_metric<T, N, N>& inverse)
{
    tensor<T, N, N> ret;

    for(int i=0; i < N; i++)
    {
        for(int j=0; j < N; j++)
        {
            T sum = 0;

            for(int k=0; k < N; k++)
            {
                sum = sum + inverse.idx(k, j) * mT.idx(i, k);
                //sum = sum + mT.idx(i, k) * inverse.idx(k, j);
            }

            ret.idx(i, j) = sum;
        }
    }

    return ret;
}

template<typename T, int N>
tensor<T, N> raise_index(const tensor<T, N>& mT, const metric<T, N, N>& met, const inverse_metric<T, N, N>& inverse)
{
    tensor<T, N> ret;

    for(int i=0; i < N; i++)
    {
        T sum = 0;

        for(int j=0; j < N; j++)
        {
            sum = sum + inverse.idx(i, j) * mT.idx(j);
        }

        ret.idx(i) = sum;
    }

    return ret;
}

template<int N>
tensor<value, N> lower_index(const tensor<value, N>& mT, const metric<value, N, N>& met)
{
    tensor<value, N> ret;

    for(int i=0; i < N; i++)
    {
        value sum = 0;

        for(int j=0; j < N; j++)
        {
            sum = sum + met.idx(i, j) * mT.idx(j);
        }

        ret.idx(i) = sum;
    }

    return ret;
}

///https://en.wikipedia.org/wiki/Covariant_derivative#Covariant_derivative_by_field_type
///for the tensor DcDa, this returns idx(a, c)
template<typename T, int N>
inline
tensor<T, N, N> gpu_covariant_derivative_low_vec(equation_context& ctx, const tensor<T, N>& v_in, const metric<T, N, N>& met, const inverse_metric<T, N, N>& inverse)
{
    auto christoff = gpu_christoffel_symbols_2(ctx, met, inverse);

    tensor<T, N, N> lac;

    for(int a=0; a < N; a++)
    {
        for(int c=0; c < N; c++)
        {
            T sum = 0;

            for(int b=0; b < N; b++)
            {
                sum = sum + christoff.idx(b, c, a) * v_in.idx(b);
            }

            lac.idx(a, c) = diff1(ctx, v_in.idx(a), c) - sum;
        }
    }

    return lac;
}

template<typename T, int N>
inline
tensor<T, N, N> gpu_double_covariant_derivative(equation_context& ctx, const T& in, const tensor<T, N>& first_derivatives,
                                                const metric<T, N, N>& met, const inverse_metric<T, N, N>& inverse,
                                                const tensor<T, N, N, N>& christoff2)
{
    tensor<T, N, N> lac;

    for(int a=0; a < N; a++)
    {
        for(int c=0; c < N; c++)
        {
            T sum = 0;

            for(int b=0; b < N; b++)
            {
                sum += christoff2.idx(b, c, a) * diff1(ctx, in, b);
            }

            lac.idx(a, c) = diff2(ctx, in, a, c, first_derivatives.idx(a), first_derivatives.idx(c)) - sum;
        }
    }

    return lac;
}

/*template<typename T, int N>
inline
tensor<T, N, N> gpu_high_covariant_derivative_vec(equation_context& ctx, const tensor<T, N>& in, const metric<T, N, N>& met, const inverse_metric<T, N, N>& inverse)
{
    tensor<T, N, N> deriv_low = gpu_covariant_derivative_low_vec(ctx, in, met, inverse);

    tensor<T, N, N> ret;

    for(int s=0; s < N; s++)
    {
        for(int j=0; j < N; j++)
        {
            T sum = 0;

            for(int p=0; p < N; p++)
            {
                sum = sum + inverse.idx(s, p) * deriv_low.idx(j, p);
            }

            ret.idx(j, s) = sum;
        }
    }

    return ret;
}*/

template<typename T, int N>
inline
tensor<T, N, N> gpu_trace_free(const tensor<T, N, N>& mT, const metric<T, N, N>& met, const inverse_metric<T, N, N>& inverse)
{
    tensor<T, N, N> TF;
    T t = gpu_trace(mT, met, inverse);

    for(int i=0; i < N; i++)
    {
        for(int j=0; j < N; j++)
        {
            TF.idx(i, j) = mT.idx(i, j) - (1/3.f) * met.idx(i, j) * t;
        }
    }

    return TF;
}

template<typename T, int N>
inline
tensor<T, N, N, N> gpu_christoffel_symbols_2(equation_context& ctx, const metric<T, N, N>& met, const inverse_metric<T, N, N>& inverse)
{
    tensor<T, N, N, N> christoff;

    for(int i=0; i < N; i++)
    {
        for(int k=0; k < N; k++)
        {
            for(int l=0; l < N; l++)
            {
                T sum = 0;

                for(int m=0; m < N; m++)
                {
                    value local = 0;

                    local = local + diff1(ctx, met.idx(m, k), l);
                    local = local + diff1(ctx, met.idx(m, l), k);
                    local = local - diff1(ctx, met.idx(k, l), m);

                    sum = sum + local * inverse.idx(i, m);
                }

                christoff.idx(i, k, l) = 0.5 * sum;
            }
        }
    }

    return christoff;
}

template<typename T, int N>
inline
tensor<T, N, N, N> gpu_christoffel_symbols_1(equation_context& ctx, const metric<T, N, N>& met)
{
    tensor<T, N, N, N> christoff;

    for(int c=0; c < N; c++)
    {
        for(int a=0; a < N; a++)
        {
            for(int b=0; b < N; b++)
            {
                christoff.idx(c, a, b) = 0.5f * (diff1(ctx, met.idx(c, a), b) + diff1(ctx, met.idx(c, b), a) - diff1(ctx, met.idx(a, b), c));
            }
        }
    }

    return christoff;
}

template<typename T, int N>
inline
tensor<T, N, N> raise_both(const tensor<T, N, N>& mT, const metric<T, N, N>& met, const inverse_metric<T, N, N>& inverse)
{
    tensor<T, N, N> ret;

    for(int a=0; a < N; a++)
    {
        for(int b=0; b < N; b++)
        {
            T sum = 0;

            for(int g = 0; g < N; g++)
            {
                for(int d = 0; d < N; d++)
                {
                    sum = sum + inverse.idx(a, g) * inverse.idx(b, d) * mT.idx(g, d);
                }
            }

            ret.idx(a, b) = sum;
        }
    }

    return ret;
}

template<typename T, int N>
inline
tensor<T, N, N> lower_both(const tensor<T, N, N>& mT, const metric<T, N, N>& met)
{
    tensor<T, N, N> ret;

    for(int a=0; a < N; a++)
    {
        for(int b=0; b < N; b++)
        {
            T sum = 0;

            for(int g = 0; g < N; g++)
            {
                for(int d = 0; d < N; d++)
                {
                    sum = sum + met.idx(a, g) * met.idx(b, d) * mT.idx(g, d);
                }
            }

            ret.idx(a, b) = sum;
        }
    }

    return ret;
}

value bidx(const std::string& buf, bool interpolate)
{
    if(interpolate)
    {
        variable v = fetch_variable(buf);

        if(v.is_derivative)
        {
            return dual_types::apply("buffer_read_linearh", buf, as_float3("fx", "fy", "fz"), "dim");
        }
        else
        {
            return dual_types::apply("buffer_read_linear", buf, as_float3("fx", "fy", "fz"), "dim");
        }
    }
    else
    {
        variable v = fetch_variable(buf);

        if(v.is_derivative)
        {
            return dual_types::apply("buffer_indexh", buf, "ix", "iy", "iz", "dim");
        }
        else
        {
            return dual_types::apply("buffer_index", buf, "ix", "iy", "iz", "dim");
        }
    }
}

struct standard_arguments
{
    value gA;
    tensor<value, 3> gB;
    tensor<value, 3> gBB;

    unit_metric<value, 3, 3> cY;
    tensor<value, 3, 3> cA;

    value X;
    value K;

    tensor<value, 3> cGi;

    value gA_X;

    metric<value, 3, 3> Yij;
    inverse_metric<value, 3, 3> iYij;
    tensor<value, 3, 3> Kij;

    tensor<value, 3> momentum_constraint;

    vec2i linear_indices[6] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};

    int index_table[3][3] = {{0, 1, 2},
                             {1, 3, 4},
                             {2, 4, 5}};


    tensor<value, 3, 3, 3> dcYij;
    tensor<value, 3, 3> digB;
    tensor<value, 3> digA;
    tensor<value, 3> dX;

    tensor<value, 3> bigGi;
    tensor<value, 3> derived_cGi;

    tensor<value, 3, 3, 3> christoff2;

    standard_arguments(equation_context& ctx)
    {
        bool interpolate = ctx.uses_linear;

        gA = (bidx("gA", interpolate));

        gA = max(gA, 0.f);
        //gA = max(gA, 0.00001f);

        gB.idx(0) = (bidx("gB0", interpolate));
        gB.idx(1) = (bidx("gB1", interpolate));
        gB.idx(2) = (bidx("gB2", interpolate));

        gBB.idx(0) = (bidx("gBB0", interpolate));
        gBB.idx(1) = (bidx("gBB1", interpolate));
        gBB.idx(2) = (bidx("gBB2", interpolate));

        std::array<int, 9> arg_table
        {
            0, 1, 2,
            1, 3, 4,
            2, 4, 5,
        };

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                int index = arg_table[i * 3 + j];

                cY.idx(i, j) = bidx("cY" + std::to_string(index), interpolate);
            }
        }

        //cY.idx(2, 2) = (1 + cY.idx(1, 1) * cY.idx(0, 2) * cY.idx(0, 2) - 2 * cY.idx(0, 1) * cY.idx(1, 2) * cY.idx(0, 2) + cY.idx(0, 0) * cY.idx(1, 2) * cY.idx(1, 2)) / (cY.idx(0, 0) * cY.idx(1, 1) - cY.idx(0, 1) * cY.idx(0, 1));

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                int index = arg_table[i * 3 + j];

                cA.idx(i, j) = bidx("cA" + std::to_string(index), interpolate);
            }
        }

        inverse_metric<value, 3, 3> icY = cY.invert();

        //tensor<value, 3, 3> raised_cAij = raise_index(cA, cY, icY);

        //cA.idx(1, 1) = -(raised_cAij.idx(0, 0) + raised_cAij.idx(2, 2) + cA.idx(0, 1) * icY.idx(0, 1) + cA.idx(1, 2) * icY.idx(1, 2)) / (icY.idx(1, 1));

        X = (bidx("X", interpolate));
        K = (bidx("K", interpolate));

        //X = max(X, 0.0001f);

        gA_X = gA / max(X, 0.001f);

        cGi.idx(0) = (bidx("cGi0", interpolate));
        cGi.idx(1) = (bidx("cGi1", interpolate));
        cGi.idx(2) = (bidx("cGi2", interpolate));

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                Yij.idx(i, j) = cY.idx(i, j) / max(X, 0.001f);
                iYij.idx(i, j) = X * icY.idx(i, j);
            }
        }

        tensor<value, 3, 3> Aij = cA / max(X, 0.001f);

        Kij = Aij + Yij.to_tensor() * (K / 3.f);

        momentum_constraint.idx(0) = (bidx("momentum0", interpolate));
        momentum_constraint.idx(1) = (bidx("momentum1", interpolate));
        momentum_constraint.idx(2) = (bidx("momentum2", interpolate));

        for(int k=0; k < 3; k++)
        {
            for(int i=0; i < 3; i++)
            {
                for(int j=0; j < 3; j++)
                {
                    int symmetric_index = index_table[i][j];

                    int final_index = k + symmetric_index * 3;

                    dcYij.idx(k, i, j) = bidx("dcYij" + std::to_string(final_index), interpolate);
                }
            }
        }

        digA.idx(0) = (bidx("digA0", interpolate));
        digA.idx(1) = (bidx("digA1", interpolate));
        digA.idx(2) = (bidx("digA2", interpolate));

        dX.idx(0) = (bidx("dX0", interpolate));
        dX.idx(1) = (bidx("dX1", interpolate));
        dX.idx(2) = (bidx("dX2", interpolate));

        ///derivative
        for(int i=0; i < 3; i++)
        {
            ///value
            for(int j=0; j < 3; j++)
            {
                int idx = i + j * 3;

                digB.idx(i, j)  = bidx("digB" + std::to_string(idx), interpolate);
            }
        }

        /*tensor<value, 3, 3, 3> christoff2 = gpu_christoffel_symbols_2(ctx, cY, icY);

        auto pinned_christoff2 = christoff2;
        ctx.pin(pinned_christoff2);

        auto pinned_icY = icY;
        //ctx.pin(pinned_icY);

        tensor<value, 3> cGi_G;*/

        /*for(int i=0; i < 3; i++)
        {
            value sum = 0;

            for(int j=0; j < 3; j++)
            {
                for(int k=0; k < 3; k++)
                {
                    sum += pinned_icY.idx(j, k) * pinned_christoff2.idx(i, j, k);
                }
            }

            cGi_G.idx(i) = sum;
        }*/

        tensor<value, 3> cGi_G;

        for(int i=0; i < 3; i++)
        {
            value sum = 0;

            for(int j=0; j < 3; j++)
            {
                sum += -diff1(ctx, icY.idx(i, j), j);
            }

            cGi_G.idx(i) = sum;
        }

        //ctx.pin(cGi_G);

        ///https://arxiv.org/pdf/1205.5111v1.pdf 34
        for(int i=0; i < 3; i++)
        {
            bigGi.idx(i) = cGi.idx(i) - cGi_G.idx(i);
        }

        //#define USE_DERIVED_CGI
        #ifdef USE_DERIVED_CGI
        derived_cGi = cGi_G;
        #else
        derived_cGi = cGi;
        #endif

        /// https://arxiv.org/pdf/1507.00570.pdf (1)
        //#define MODIFIED_CHRISTOFFEL
        #ifdef MODIFIED_CHRISTOFFEL
        tensor<value, 3> bigGi_lower = lower_index(bigGi, cY);

        tensor<value, 3, 3, 3> raw_christoff2 = gpu_christoffel_symbols_2(ctx, cY, icY);

        tensor<value, 3> Tk;

        for(int i=0; i < 3; i++)
        {
            value sum = 0;

            for(int k=0; k < 3; k++)
            {
                sum += raw_christoff2.idx(k, k, i);
            }

            Tk.idx(i) = sum;
        }

        tensor<value, 3, 3, 3> djtk;
        tensor<value, 3, 3, 3> djgk;
        tensor<value, 3, 3, 3> yjkGi;

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                for(int k=0; k < 3; k++)
                {
                    float kroneck = (i == j) ? 1 : 0;

                    djtk.idx(i, j, k) = (-3.f/5.f) * kroneck * Tk.idx(k);
                    djgk.idx(i, j, k) = (-1.f/5.f) * kroneck * bigGi_lower.idx(k);
                    yjkGi.idx(i, j, k) = (1.f/3.f) * cY.idx(j, k) * bigGi.idx(i);
                }
            }
        }


        ///Qab_TF = Qab - (1/3) Q * met
        ///where Q = iMet * Qab

        tensor<value, 3, 3, 3> djtk_TF;
        tensor<value, 3, 3, 3> djgk_TF;

        for(int k=0; k < 3; k++)
        {
            value T1 = 0;
            value T2 = 0;

            for(int i=0; i < 3; i++)
            {
                for(int j=0; j < 3; j++)
                {
                    T1 += icY.idx(i, j) * djtk.idx(k, i, j);
                    T2 += icY.idx(i, j) * djgk.idx(k, i, j);
                }
            }

            for(int i=0; i < 3; i++)
            {
                for(int j=0; j < 3; j++)
                {
                    djtk_TF.idx(k, i, j) = djtk.idx(k, i, j) - (1.f/3.f) * T1 * cY.idx(i, j);
                    djgk_TF.idx(k, i, j) = djgk.idx(k, i, j) - (1.f/3.f) * T2 * cY.idx(i, j);
                }
            }
        }

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                for(int k=0; k < 3; k++)
                {
                    christoff2.idx(i, j, k) = raw_christoff2.idx(i, j, k) + djtk_TF.idx(i, j, k) + djgk_TF.idx(i, j, k) + yjkGi.idx(i, j, k);
                }
            }
        }

        #else
        christoff2 = gpu_christoffel_symbols_2(ctx, cY, icY);
        #endif

        /*
        ///dcgA alias
        for(int i=0; i < 3; i++)
        {
            value v = diff1(ctx, gA, i);

            ctx.alias(v, digA.idx(i));
        }

        ///dcgB alias
        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                value v = diff1(ctx, gB.idx(j), i);

                ctx.alias(v, digB.idx(i, j));
            }
        }

        ///dcYij alias
        for(int k=0; k < 3; k++)
        {
            for(int i=0; i < 3; i++)
            {
                for(int j=0; j < 3; j++)
                {
                    value v = diff1(ctx, cY.idx(i, j), k);

                    ctx.alias(v, dcYij.idx(k, i, j));
                }
            }
        }

        for(int i=0; i < 3; i++)
        {
            ctx.alias(diff1(ctx, X, i), dX.idx(i));
        }*/
    }
};

template<typename T>
struct black_hole
{
    ///in world coordinates
    tensor<T, 3> position;
    ///this is not a parameter that maps to any straightforward concept of mass
    T bare_mass = 0;
    tensor<T, 3> momentum;
    tensor<T, 3> angular_momentum;
};

struct adm_black_hole
{
    float bare_mass_guess = 0.5f;

    tensor<float, 3> position;
    float adm_mass = 0;
    tensor<float, 3> velocity;
    tensor<float, 3> angular_velocity;
};

///https://arxiv.org/pdf/gr-qc/0610128.pdf initial conditions, see (7)
template<typename T>
inline
tensor<value, 3, 3> calculate_single_bcAij(const tensor<value, 3>& pos, const black_hole<T>& hole)
{
    tensor<value, 3, 3, 3> eijk = get_eijk();

    tensor<value, 3, 3> bcAij;

    metric<value, 3, 3> flat;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            flat.idx(i, j) = (i == j) ? 1 : 0;
            bcAij.idx(i, j) = 0;
        }
    }

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            tensor<value, 3> momentum_tensor = {hole.momentum.x(), hole.momentum.y(), hole.momentum.z()};

            tensor<value, 3> vri = {hole.position.x(), hole.position.y(), hole.position.z()};

            value ra = (pos - vri).length();

            ra = max(ra, 1e-6);

            tensor<value, 3> nia = (pos - vri) / ra;

            tensor<value, 3> momentum_lower = lower_index(momentum_tensor, flat);
            tensor<value, 3> nia_lower = lower_index(tensor<value, 3>{nia.x(), nia.y(), nia.z()}, flat);

            bcAij.idx(i, j) += (3 / (2.f * ra * ra)) * (momentum_lower.idx(i) * nia_lower.idx(j) + momentum_lower.idx(j) * nia_lower.idx(i) - (flat.idx(i, j) - nia_lower.idx(i) * nia_lower.idx(j)) * sum_multiply(momentum_tensor, nia_lower));

            ///spin
            value s1 = 0;
            value s2 = 0;

            for(int k=0; k < 3; k++)
            {
                for(int l=0; l < 3; l++)
                {
                    s1 += eijk.idx(k, i, l) * hole.angular_momentum[l] * nia[k] * nia_lower.idx(j);
                    s2 += eijk.idx(k, j, l) * hole.angular_momentum[l] * nia[k] * nia_lower.idx(i);
                }
            }

            bcAij.idx(i, j) += (3 / (ra*ra*ra)) * (s1 + s2);
        }
    }

    return bcAij;
}

///this does not return the same kind of conformal cAij as bssn uses, need to reconstruct Kij!
template<typename T>
inline
tensor<value, 3, 3> calculate_bcAij(const tensor<value, 3>& pos, const std::vector<black_hole<T>>& holes)
{
    tensor<value, 3, 3> bcAij;

    for(const black_hole<T>& hole : holes)
    {
        bcAij += calculate_single_bcAij(pos, hole);

    }

    return bcAij;
}

tensor<float, 3> world_to_voxel(const tensor<float, 3>& world_pos, vec3i dim, float scale)
{
    tensor<float, 3> centre = {(dim.x() - 1)/2, (dim.y() - 1)/2, (dim.z() - 1)/2};

    return (world_pos / scale) + centre;
}

//https://arxiv.org/pdf/gr-qc/0610128.pdf (6)
float get_nonspinning_adm_mass(cl::command_queue& cqueue, int idx, const std::vector<black_hole<float>>& holes, vec3i dim, float scale, cl::buffer& u_buffer)
{
    assert(idx >= 0 && idx < holes.size());

    const black_hole<float>& my_hole = holes[idx];

    tensor<float, 3> voxel_pos = world_to_voxel(my_hole.position, dim, scale);

    int read_idx = (int)voxel_pos.z() * dim.x() * dim.y() + (int)voxel_pos.y() * dim.x() + (int)voxel_pos.x();

    cl_float u_read = 0;

    u_buffer.read(cqueue, (char*)&u_read, sizeof(cl_float), read_idx * sizeof(cl_float));

    ///should be integer
    printf("VXP %f %f %f\n", voxel_pos.x(), voxel_pos.y(), voxel_pos.z());

    printf("Got u %f\n", u_read);

    float sum = 0;

    for(int i=0; i < (int)holes.size(); i++)
    {
        if(i == idx)
            continue;

        sum += holes[i].bare_mass / (2 * (holes[idx].position - holes[i].position).length());
    }

    ///the reason this isn't 1+ is the differences in definition of u
    return holes[idx].bare_mass * (u_read + sum);
}

struct initial_conditions
{
    std::vector<black_hole<float>> holes;
    cl::buffer gpu_holes;

    initial_conditions(cl::context& ctx) : gpu_holes(ctx){}
};

template<typename T>
inline
value calculate_aij_aIJ(const metric<value, 3, 3>& flat_metric, const tensor<value, 3, 3>& bcAij, const std::vector<black_hole<T>>& holes)
{
    value aij_aIJ = 0;

    tensor<value, 3, 3> ibcAij = raise_both(bcAij, flat_metric, flat_metric.invert());

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            aij_aIJ += ibcAij.idx(i, j) * bcAij.idx(i, j);
        }
    }

    return aij_aIJ;
}

template<typename T>
inline
value calculate_conformal_guess(const tensor<value, 3>& pos, const std::vector<black_hole<T>>& holes)
{
    //https://arxiv.org/pdf/gr-qc/9703066.pdf (8)
    value BL_s = 0;

    for(const black_hole<T>& hole : holes)
    {
        value Mi = hole.bare_mass;
        tensor<value, 3> ri = {hole.position.x(), hole.position.y(), hole.position.z()};

        value dist = (pos - ri).length();

        dist = max(dist, 1e-6);

        BL_s += Mi / (2 * dist);
    }

    return BL_s;
}


void check_symmetry(const std::string& debug_name, cl::command_queue& cqueue, cl::buffer& arg, vec<4, cl_int> size)
{
    //#define CHECK_SYMMETRY
    #ifdef CHECK_SYMMETRY
    std::cout << debug_name << std::endl;

    cl::args check;
    check.push_back(arg);
    check.push_back(size);

    cqueue.exec("check_z_symmetry", check, {size.x(), size.y(), size.z()}, {8, 8, 1});

    cqueue.block();
    #endif // CHECK_SYMMETRY
}

cl::buffer solve_for_u(cl::context& ctx, cl::command_queue& cqueue, vec<4, cl_int> base_size, float c_at_max, int scale_factor, cl::buffer& gpu_holes, std::optional<cl::buffer> base, cl_float etol)
{
    vec<4, cl_int> reduced_clsize = ((base_size - 1) / scale_factor) + 1;

    std::array<cl::buffer, 2> reduced_u_args{ctx, ctx};

    if(base.has_value())
        reduced_u_args[0] = base.value();
    else
        reduced_u_args[0].alloc(reduced_clsize.x() * reduced_clsize.y() * reduced_clsize.z() * sizeof(cl_float));

    reduced_u_args[1].alloc(reduced_clsize.x() * reduced_clsize.y() * reduced_clsize.z() * sizeof(cl_float));

    int which_reduced = 0;

    if(!base.has_value())
    {
        cl::args initial_u_args;
        initial_u_args.push_back(reduced_u_args[0]);
        initial_u_args.push_back(reduced_clsize);

        cqueue.exec("setup_u_offset", initial_u_args, {reduced_clsize.x(), reduced_clsize.y(), reduced_clsize.z()}, {8, 8, 1});
    }

    cl::copy(cqueue, reduced_u_args[0], reduced_u_args[1]);

    int N = 8000;

    #ifdef GPU_PROFILE
    N = 1000;
    #endif // GPU_PROFILE

    #ifdef QUICKSTART
    N = 200;
    #endif // QUICKSTART

    //cl_int still_going = 0;

    std::array<cl::buffer, 2> still_going{ctx, ctx};

    cl_int one = 1;

    for(int i=0; i < 2; i++)
    {
        still_going[i].alloc(sizeof(cl_int));
        still_going[i].fill(cqueue, one);
    }

    int which_still_going = 0;

    for(int i=0; i < N; i++)
    {
        float local_scale = calculate_scale(c_at_max, reduced_clsize);

        cl::args iterate_u_args;
        iterate_u_args.push_back(reduced_u_args[which_reduced]);
        iterate_u_args.push_back(reduced_u_args[(which_reduced + 1) % 2]);
        iterate_u_args.push_back(local_scale);
        iterate_u_args.push_back(reduced_clsize);
        iterate_u_args.push_back(still_going[which_still_going]);
        iterate_u_args.push_back(still_going[(which_still_going + 1) % 2]);
        iterate_u_args.push_back(etol);
        iterate_u_args.push_back(gpu_holes);

        cqueue.exec("iterative_u_solve", iterate_u_args, {reduced_clsize.x(), reduced_clsize.y(), reduced_clsize.z()}, {8, 8, 1});

        if(((i % 50) == 0) && still_going[(which_still_going + 1) % 2].read<cl_int>(cqueue)[0] == 0)
            break;

        still_going[which_still_going].set_to_zero(cqueue);

        which_reduced = (which_reduced + 1) % 2;
        which_still_going = (which_still_going + 1) % 2;
    }

    check_symmetry("post_iterate", cqueue, reduced_u_args[which_reduced], reduced_clsize);

    return reduced_u_args[which_reduced];
}

#ifdef OLD_FAST_U
cl::buffer upscale_u(cl::context& ctx, cl::command_queue& cqueue, cl::buffer& source_buffer, vec<4, cl_int> base_size, int upscale_scale, int source_scale)
{
    vec<4, cl_int> reduced_clsize = ((base_size - 1) / source_scale) + 1;
    vec<4, cl_int> upper_clsize = ((base_size - 1) / upscale_scale) + 1;

    check_symmetry("pre_iterate", cqueue, source_buffer, reduced_clsize);

    cl::buffer u_arg(ctx);
    u_arg.alloc(upper_clsize.x() * upper_clsize.y() * upper_clsize.z() * sizeof(cl_float));

    cl::args upscale_args;
    upscale_args.push_back(source_buffer);
    upscale_args.push_back(u_arg);
    upscale_args.push_back(reduced_clsize);
    upscale_args.push_back(upper_clsize);

    cqueue.exec("upscale_u", upscale_args, {upper_clsize.x(), upper_clsize.y(), upper_clsize.z()}, {8, 8, 1});

    check_symmetry("post_upscale", cqueue, u_arg, upper_clsize);

    return u_arg;
}

cl::buffer iterate_u(cl::context& ctx, cl::command_queue& cqueue, vec3i size, float c_at_max)
{
    vec<4, cl_int> clsize = {size.x(), size.y(), size.z(), 0};

    std::optional<cl::buffer> last;

    for(int i=2; i >= 0; i--)
    {
        int up_size = pow(2, i+1);
        int current_size = pow(2, i);

        cl::buffer reduced = solve_for_u(ctx, cqueue, clsize, c_at_max, up_size, last);

        cl::buffer upscaled = upscale_u(ctx, cqueue, reduced, clsize, current_size, up_size);

        last = upscaled;
    }

    return solve_for_u(ctx, cqueue, clsize, c_at_max, 1, last);
}
#endif // OLD_FAST_U

cl::buffer extract_u_region(cl::context& ctx, cl::command_queue& cqueue, cl::buffer& in, float c_at_max_in, float c_at_max_out, vec<4, cl_int> clsize)
{
    cl::buffer out(ctx);
    out.alloc(in.alloc_size);

    cl::args upscale_args;
    upscale_args.push_back(in);
    upscale_args.push_back(out);
    upscale_args.push_back(c_at_max_in);
    upscale_args.push_back(c_at_max_out);
    upscale_args.push_back(clsize);

    cqueue.exec("extract_u_region", upscale_args, {clsize.x(), clsize.y(), clsize.z()}, {8, 8, 1});

    check_symmetry("extract_u_region", cqueue, out, clsize);

    return out;
}

cl::buffer iterate_u(cl::context& ctx, cl::command_queue& cqueue, vec3i size, float c_at_max, cl::buffer& gpu_holes, cl_float etol)
{
    float boundaries[4] = {c_at_max, c_at_max * 4, c_at_max * 8, c_at_max * 16};

    vec<4, cl_int> clsize = {size.x(), size.y(), size.z(), 0};

    std::optional<cl::buffer> last;

    for(int i=2; i >= 0; i--)
    {
        float current_boundary = boundaries[i + 1];
        float next_boundary = boundaries[i];

        cl::buffer reduced = solve_for_u(ctx, cqueue, clsize, current_boundary, 1, gpu_holes, last, etol);

        cl::buffer extracted = extract_u_region(ctx, cqueue, reduced, current_boundary, next_boundary, clsize);

        last = extracted;
    }

    return solve_for_u(ctx, cqueue, clsize, c_at_max, 1, gpu_holes, last, etol);
}

cl::buffer construct_black_holes(cl::context& ctx, cl::command_queue& cqueue, const std::vector<black_hole<float>>& holes)
{
    cl::buffer gpu_holes(ctx);

    gpu_holes.alloc(sizeof(cl_float) * 10 * holes.size());

    std::vector<cl_float> to_write;

    ///matches the gpu hole struct
    for(const black_hole<float>& v : holes)
    {
        to_write.push_back(v.bare_mass);

        to_write.push_back(v.position.x());
        to_write.push_back(v.position.y());
        to_write.push_back(v.position.z());

        to_write.push_back(v.momentum.x());
        to_write.push_back(v.momentum.y());
        to_write.push_back(v.momentum.z());

        to_write.push_back(v.angular_momentum.x());
        to_write.push_back(v.angular_momentum.y());
        to_write.push_back(v.angular_momentum.z());
    }

    gpu_holes.write(cqueue, to_write);

    return gpu_holes;
}

std::vector<float> calculate_adm_mass(const std::vector<black_hole<float>>& holes, cl::context& ctx, cl::command_queue& cqueue, float err = 0.0001f)
{
    std::vector<float> ret;

    cl::buffer buf = construct_black_holes(ctx, cqueue, holes);

    vec3i dim = {281, 281, 281};

    cl::buffer u_arg = iterate_u(ctx, cqueue, dim, get_c_at_max(), buf, err);

    for(int i=0; i < (int)holes.size(); i++)
    {
        ret.push_back(get_nonspinning_adm_mass(cqueue, i, holes, dim, calculate_scale(get_c_at_max(), dim), u_arg));
    }

    return ret;
}

void create_u_program(cl::context& clctx, int holes, const std::string& u_argument_string)
{
    tensor<value, 3> pos = {"ox", "oy", "oz"};

    std::vector<black_hole<value>> gpu_holes;

    for(int i=0; i < holes; i++)
    {
        std::string sidx = std::to_string(i);

        std::string p = "holes[" + sidx + "].";

        auto suff = [&](const std::string& in)
        {
            return p + in;
        };

        black_hole<value> bhv;
        bhv.position = {suff("px"), suff("py"), suff("pz")};
        bhv.momentum = {suff("mx"), suff("my"), suff("mz")};
        bhv.angular_momentum = {suff("amx"), suff("amy"), suff("amz")};
        bhv.bare_mass = suff("mass");

        gpu_holes.push_back(bhv);
    }

    metric<value, 3, 3> flat_metric;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            flat_metric.idx(i, j) = (i == j) ? 1 : 0;
        }
    }

    equation_context eqs;

    //https://arxiv.org/pdf/gr-qc/9703066.pdf (8)
    value BL_s_dyn = calculate_conformal_guess(pos, gpu_holes);
    tensor<value, 3, 3> bcAij_dyn = calculate_bcAij(pos, gpu_holes);
    value aij_aIJ_dyn = calculate_aij_aIJ(flat_metric, bcAij_dyn, gpu_holes);

    eqs.add("init_aij_aIJ_dyn", aij_aIJ_dyn);
    eqs.add("init_BL_val_dyn", BL_s_dyn);

    std::string local_build_str = u_argument_string;

    eqs.build(local_build_str, 8);

    {
        cl::program u_program(clctx, "u_solver.cl");
        u_program.build(clctx, local_build_str);

        clctx.register_program(u_program);
    }
}

inline
initial_conditions get_bare_initial_conditions(cl::context& clctx, cl::command_queue& cqueue, float scale, std::vector<black_hole<float>> holes)
{
    initial_conditions ret(clctx);

    float bulge = 1;

    auto san_black_hole_pos = [&](const tensor<float, 3>& in)
    {
        tensor<float, 3> scaled = round((in / scale) * bulge);

        return scaled * scale / bulge;
    };

    for(black_hole<float>& hole : holes)
    {
        hole.position = san_black_hole_pos(hole.position);
    }

    ret.holes = holes;
    ret.gpu_holes = construct_black_holes(clctx, cqueue, holes);

    return ret;
}

inline
initial_conditions get_adm_initial_conditions(cl::context& clctx, cl::command_queue& cqueue, float scale, std::vector<adm_black_hole> adm_holes)
{
    float bulge = 1;

    auto san_black_hole_pos = [&](const tensor<float, 3>& in)
    {
        tensor<float, 3> scaled = round((in / scale) * bulge);

        return scaled * scale / bulge;
    };

    for(adm_black_hole& hole : adm_holes)
    {
        hole.position = san_black_hole_pos(hole.position);
    }

    std::vector<black_hole<float>> raw_holes;

    for(adm_black_hole& adm : adm_holes)
    {
        black_hole<float> black;

        black.position = adm.position;
        black.bare_mass = adm.bare_mass_guess;
        black.momentum = adm.adm_mass * adm.velocity;
        black.angular_momentum = adm.adm_mass * adm.angular_velocity;

        raw_holes.push_back(black);
    }

    float start_err = 0.001f;
    float fin_err = 0.0001f;

    int iterations = 8;

    for(int i=0; i < iterations; i++)
    {
        float err = ((float)i / (iterations - 1)) * (fin_err - start_err) + start_err;

        std::vector<float> masses = calculate_adm_mass(raw_holes, clctx, cqueue, err);

        printf("Masses ");

        for(int kk=0; kk < (int)masses.size(); kk++)
        {
            printf("%f ", masses[kk]);
        }

        printf("\n");

        for(int kk=0; kk < (int)masses.size(); kk++)
        {
            float error_delta = adm_holes[kk].adm_mass - masses[kk];

            float relaxation = 0.5;

            float bare_mass_adjust = raw_holes[kk].bare_mass + raw_holes[kk].bare_mass * (error_delta / masses[kk]) * relaxation;

            raw_holes[kk].bare_mass = bare_mass_adjust;

            printf("new bare mass %f\n", raw_holes[kk].bare_mass);
        }
    }

    return get_bare_initial_conditions(clctx, cqueue, scale, raw_holes);
}

inline
initial_conditions setup_dynamic_initial_conditions(const std::string& u_argument_string, cl::context& clctx, cl::command_queue& cqueue, vec3f centre, float scale)
{
    initial_conditions ret(clctx);

    #if 0
    ///https://arxiv.org/pdf/gr-qc/0505055.pdf
    ///https://arxiv.org/pdf/1205.5111v1.pdf under binary black hole with punctures
    std::vector<float> black_hole_m{0.5, 0.5};
    std::vector<vec3f> black_hole_pos{san_black_hole_pos({-4, 0, 0}), san_black_hole_pos({4, 0, 0})};
    //std::vector<vec3f> black_hole_velocity{{0, 0, 0}, {0, 0, 0}};
    std::vector<vec3f> black_hole_velocity{{0, 0, 0}, {0, 0, 0}};
    std::vector<vec3f> black_hole_spin{{0, 0, 0}, {0, 0, 0}};

    //#define KEPLER
    #ifdef KEPLER
    {
        float R = (black_hole_pos[1] - black_hole_pos[0]).length();

        float m0 = black_hole_m[0];
        float m1 = black_hole_m[1];

        float M = m0 + m1;

        float v0 = sqrt(m1 * m1 / (R * M));
        float v1 = sqrt(m0 * m0 / (R * M));

        vec3f v0_v = {0.0, -1, 0};

        ///made it to 532 after 2 + 3/4s orbits
        ///1125
        black_hole_velocity[0] = v0 * v0_v.norm() * 0.6575f;
        black_hole_velocity[1] = v1 * -v0_v.norm() * 0.6575f;

        float r0 = m1 * R / M;
        float r1 = m0 * R / M;

        black_hole_pos[0] = {-r0, 0, 0};
        black_hole_pos[1] = {r1, 0, 0};
    }
    #endif // KEPLER

    //#define TRIPLEHOLES
    #ifdef TRIPLEHOLES
    black_hole_m = {0.5, 0.4, 0.4};
    black_hole_pos = {{0, 3.f, 0.f}, {6, 0, 0}, {-5, 0, 0}};
    black_hole_velocity = {{0, 0, 0}, {0, -0.2, 0}, {0, 0.2, 0}};
    black_hole_spin = {{0,0,0}, {0,0,0}, {0,0,0}};
    #endif // TRIPLEHOLES

    //#define SINGLEHOLE
    #ifdef SINGLEHOLE
    black_hole_m = {0.5};
    black_hole_pos = {{-2, 0.f, 0.f}};
    black_hole_velocity = {{0.5f, 0, 0}};
    black_hole_spin = {{0,0,0}};
    #endif // SINGLEHOLE
    #endif // 0

    #define BARE_BLACK_HOLES
    #ifdef BARE_BLACK_HOLES
    std::vector<black_hole<float>> holes;

    ///https://arxiv.org/pdf/gr-qc/0610128.pdf
    #define PAPER_0610128
    #ifdef PAPER_0610128
    black_hole<float> h1;
    h1.bare_mass = 0.483;
    h1.momentum = {0, 0.133 * 0.8, 0};
    h1.position = {-3.257, 0.f, 0.f};

    black_hole<float> h2;
    h2.bare_mass = 0.483;
    h2.momentum = {0, -0.133 * 0.8, 0};
    h2.position = {3.257, 0.f, 0.f};

    holes.push_back(h1);
    holes.push_back(h2);
    #endif // PAPER_0610128

    create_u_program(clctx, holes.size(), u_argument_string);

    return get_bare_initial_conditions(clctx, cqueue, scale, holes);
    #endif

    //#define USE_ADM_HOLE
    #ifdef USE_ADM_HOLE
    std::vector<adm_black_hole> adm_holes;

    #ifdef TEST_CSE
    adm_black_hole adm1;
    adm1.position = {-4, 0, 0};
    adm1.adm_mass = 0.5f;
    adm1.velocity = {0.f, 0.f, 0.f};
    adm1.angular_velocity = {0.f, 0.f, 0.f};

    adm_black_hole adm2;
    adm2.position = {4, 0, 0};
    adm2.adm_mass = 0.6f;
    adm2.velocity = {0.f, 0.f, 0.f};
    adm2.angular_velocity = {0.f, 0.f, 0.f};

    adm_holes.push_back(adm1);
    adm_holes.push_back(adm2);
    #endif // TEST_CSE

    //#define NAKED_CASE
    #ifdef NAKED_CASE
    adm_black_hole adm1;
    adm1.position = {-3, 0, 0};
    adm1.adm_mass = 0.5f;
    adm1.velocity = {0.f, 0.f, 0.f};
    adm1.angular_velocity = {0.6f, 0.f, 0.f};

    adm_black_hole adm2;
    adm2.position = {3, 0, 0};
    adm2.adm_mass = 0.7f;
    adm2.velocity = {0.f, 0.f, 0.f};
    adm2.angular_velocity = {-0.7f, 0.f, 0.f};

    adm_holes.push_back(adm1);
    adm_holes.push_back(adm2);
    #endif // NAKED_CASE

    //#define KICK
    #ifdef KICK
    adm_black_hole adm1;
    adm1.position = {-3, 0, 0};
    adm1.adm_mass = 0.85f;
    adm1.velocity = {0.f, 0.15f / adm1.adm_mass, 0.f};
    adm1.angular_velocity = {0.f, 0.f, 0.f};

    adm_black_hole adm2;
    adm2.position = {3, 0, 0};
    adm2.adm_mass = 0.4f;
    adm2.velocity = {0.f, -0.15f / adm2.adm_mass, 0.f};
    adm2.angular_velocity = {0.f, 0.f, 0.f};

    adm_holes.push_back(adm1);
    adm_holes.push_back(adm2);
    #endif // KICK

    create_u_program(clctx, adm_holes.size(), u_argument_string);

    return get_adm_initial_conditions(clctx, cqueue, scale, adm_holes);
    #endif // USE_ADM_HOLE

    assert(false);
}

void setup_static_conditions(cl::context& clctx, cl::command_queue& cqueue, equation_context& ctx, const std::vector<black_hole<float>>& holes)
{
    tensor<value, 3> pos = {"ox", "oy", "oz"};

    metric<value, 3, 3> flat_metric;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            flat_metric.idx(i, j) = (i == j) ? 1 : 0;
        }
    }

    value BL_s = calculate_conformal_guess(pos, holes);
    tensor<value, 3, 3> bcAij_static = calculate_bcAij(pos, holes);
    value aij_aIJ_static = calculate_aij_aIJ(flat_metric, bcAij_static, holes);

    ctx.add("init_BL_val", BL_s);
    ctx.add("init_aij_aIJ", aij_aIJ_static);

    vec2i linear_indices[6] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};

    for(int i=0; i < 6; i++)
    {
        ctx.add("init_bcA" + std::to_string(i), bcAij_static.idx(linear_indices[i].x(), linear_indices[i].y()));
    }

    ///https://arxiv.org/pdf/gr-qc/0206072.pdf see 69
    ///https://arxiv.org/pdf/gr-qc/9810065.pdf, 11
    ///phi
}

///https://arxiv.org/pdf/gr-qc/0206072.pdf alternative initial conditions
///https://cds.cern.ch/record/337814/files/9711015.pdf
///https://cds.cern.ch/record/517706/files/0106072.pdf this paper has a lot of good info on soaking up boundary conditions
///https://arxiv.org/pdf/1309.2960.pdf double fisheye
///https://arxiv.org/pdf/gr-qc/0505055.pdf better differentiation. Enforces the algebraic constraints det(cY) = 1, and subtracts the trace of Aij each frame
///manually enforce the conditions when X=0
///todo: even schwarzschild explodes after t=7
///todo: this paper suggests having a very short timestep initially while the gauge conditions settle down: https://arxiv.org/pdf/1404.6523.pdf, then increasing it
inline
void get_initial_conditions_eqs(equation_context& ctx, vec3f centre, float scale)
{
    value bl_conformal = "bl_conformal";
    value u = dual_types::apply("buffer_index", "u_value", "ix", "iy", "iz", "dim");

    tensor<value, 3, 3> bcAij;

    bcAij.idx(0, 0) = "init_bcA0"; bcAij.idx(0, 1) = "init_bcA1"; bcAij.idx(0, 2) = "init_bcA2";
    bcAij.idx(1, 0) = "init_bcA1"; bcAij.idx(1, 1) = "init_bcA3"; bcAij.idx(1, 2) = "init_bcA4";
    bcAij.idx(2, 0) = "init_bcA2"; bcAij.idx(2, 1) = "init_bcA4"; bcAij.idx(2, 2) = "init_bcA5";

    metric<value, 3, 3> Yij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            float kronecker = (i == j) ? 1 : 0;

            Yij.idx(i, j) = pow(bl_conformal + u, 4) * kronecker;
        }
    }

    value Y = Yij.det();

    ///https://arxiv.org/pdf/gr-qc/0206072.pdf see 10
    ///https://arxiv.org/pdf/gr-qc/9810065.pdf, 11
    ///phi
    value conformal_factor = (1/12.f) * log(Y);

    ctx.pin(conformal_factor);

    ///https://indico.cern.ch/event/505595/contributions/1183661/attachments/1332828/2003830/sperhake.pdf the york-lichnerowicz split
    tensor<value, 3, 3> Aij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            Aij.idx(i, j) = pow(bl_conformal + u, -2) * bcAij.idx(i, j);
        }
    }

    //value gA = 1;
    ///https://arxiv.org/pdf/gr-qc/0206072.pdf (95)
    //value gA = 1/(pow(bl_conformal + 1, 2));

    value gA = 1/(pow(bl_conformal + u, 2));
    value gB0 = 0;
    value gB1 = 0;
    value gB2 = 0;

    tensor<value, 3> cGi;
    value K = 0;

    ///https://arxiv.org/pdf/gr-qc/0206072.pdf (58)

    value X = exp(-4 * conformal_factor);

    tensor<value, 3, 3> cAij = X * Aij;

    vec2i linear_indices[6] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};

    #define OLDFLAT
    #ifdef OLDFLAT
    for(int i=0; i < 3; i++)
    {
        cGi.idx(i) = 0;
    }

    K = 0;
    #endif // OLDFLAT

    tensor<value, 3, 3> cYij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            cYij.idx(i, j) = (i == j) ? 1 : 0;
        }
    }

    for(int i=0; i < 6; i++)
    {
        vec2i index = linear_indices[i];

        std::string y_name = "init_cY" + std::to_string(i);

        ctx.add(y_name, cYij.idx(index.x(), index.y()));
    }

    for(int i=0; i < 6; i++)
    {
        ctx.add("init_cA" + std::to_string(i), cAij.idx(linear_indices[i].x(), linear_indices[i].y()));
    }

    ctx.add("init_cGi0", cGi.idx(0));
    ctx.add("init_cGi1", cGi.idx(1));
    ctx.add("init_cGi2", cGi.idx(2));

    ctx.add("init_K", K);
    ctx.add("init_X", X);

    ctx.add("init_gA", gA);
    ctx.add("init_gB0", gB0);
    ctx.add("init_gB1", gB1);
    ctx.add("init_gB2", gB2);

    //#define USE_GBB
    #ifdef USE_GBB
    value gBB0 = 0;
    value gBB1 = 0;
    value gBB2 = 0;

    ctx.add("init_gBB0", gBB0);
    ctx.add("init_gBB1", gBB1);
    ctx.add("init_gBB2", gBB2);
    #endif // USE_GBB
}

///algebraic_constraints
///https://arxiv.org/pdf/1507.00570.pdf says that the cY modification is bad
inline
void build_constraints(equation_context& ctx)
{
    standard_arguments args(ctx);

    unit_metric<value, 3, 3> cY = args.cY;
    tensor<value, 3, 3> cA = args.cA;

    value det_cY_pow = pow(cY.det(), 1.f/3.f);

    /// / det_cY_pow
    metric<value, 3, 3> fixed_cY = cY / det_cY_pow;

    /*tensor<value, 3, 3> fixed_cA = cA;

    ///https://arxiv.org/pdf/0709.3559.pdf b.49
    fixed_cA = fixed_cA / det_cY_pow;*/

    vec2i linear_indices[6] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};

    //#define NO_CAIJYY

    #ifdef NO_CAIJYY
    inverse_metric<value, 3, 3> icY = cY.invert();
    tensor<value, 3, 3> raised_cA = raise_second_index(cA, cY, icY);
    tensor<value, 3, 3> fixed_cA = cA;

    fixed_cA.idx(1, 1) = -(raised_cA.idx(0, 0) + raised_cA.idx(2, 2) + cA.idx(0, 1) * icY.idx(0, 1) + cA.idx(1, 2) * icY.idx(1, 2)) / icY.idx(1, 1);

    ctx.add("NO_CAIJYY", 1);
    #else
    ///https://arxiv.org/pdf/0709.3559.pdf b.48?? note: this defines a seemingly alternate method to do the below, but it doesn't work amazingly well
    tensor<value, 3, 3> fixed_cA = gpu_trace_free(cA, cY, cY.invert());
    #endif

    for(int i=0; i < 6; i++)
    {
        vec2i idx = linear_indices[i];

        ctx.add("fix_cY" + std::to_string(i), fixed_cY.idx(idx.x(), idx.y()));
        ctx.add("fix_cA" + std::to_string(i), fixed_cA.idx(idx.x(), idx.y()));
    }
}

void build_intermediate_thin(equation_context& ctx)
{
    standard_arguments args(ctx);

    value buffer = dual_types::apply("buffer_index", "buffer", "ix", "iy", "iz", "dim");

    value v1 = diff1(ctx, buffer, 0);
    value v2 = diff1(ctx, buffer, 1);
    value v3 = diff1(ctx, buffer, 2);

    ctx.add("init_buffer_intermediate0", v1);
    ctx.add("init_buffer_intermediate1", v2);
    ctx.add("init_buffer_intermediate2", v3);
}

void build_intermediate_thin_cY5(equation_context& ctx)
{
    standard_arguments args(ctx);

    for(int k=0; k < 3; k++)
    {
        ctx.add("init_cY5_intermediate" + std::to_string(k), diff1(ctx, args.cY.idx(2, 2), k));
    }
}

tensor<value, 3, 3, 3> gpu_covariant_derivative_low_tensor(equation_context& ctx, const tensor<value, 3, 3>& mT, const metric<value, 3, 3>& met, const inverse_metric<value, 3, 3>& inverse)
{
    tensor<value, 3, 3, 3> christoff2 = gpu_christoffel_symbols_2(ctx, met, inverse);

    tensor<value, 3, 3, 3> ret;

    for(int a=0; a < 3; a++)
    {
        for(int b=0; b < 3; b++)
        {
            for(int c=0; c < 3; c++)
            {
                value sum = 0;

                for(int d=0; d < 3; d++)
                {
                    sum += -christoff2.idx(d, c, a) * mT.idx(d, b) - christoff2.idx(d, c, b) * mT.idx(a, d);
                }

                ret.idx(c, a, b) = diff1(ctx, mT.idx(a, b), c) + sum;
            }
        }
    }

    return ret;
}

void build_momentum_constraint(equation_context& ctx)
{
    standard_arguments args(ctx);

    inverse_metric<value, 3, 3> icY = args.cY.invert();
    auto unpinned_icY = icY;
    ctx.pin(icY);

    /*value X_recip = 0.f;

    {
        float min_X = 0.001;

        X_recip = dual_if(args.X <= min_X,
        [&]()
        {
            return 1.f / min_X;
        },
        [&]()
        {
            return 1.f / args.X;
        });
    }*/

    value X_clamped = max(args.X, 0.001f);

    tensor<value, 3> Mi;

    for(int i=0; i < 3; i++)
    {
        Mi.idx(i) = 0;
    }

    //#define BETTERDAMP_DTCAIJ
    //#define DAMP_DTCAIJ
    #if defined(DAMP_DTCAIJ) || defined(BETTERDAMP_DTCAIJ)
    #define CALCULATE_MOMENTUM_CONSTRAINT
    #endif // defined

    #ifdef CALCULATE_MOMENTUM_CONSTRAINT
    #if 0
    tensor<value, 3, 3, 3> dmni = gpu_covariant_derivative_low_tensor(ctx, args.cA, args.cY, icY);

    tensor<value, 3, 3> mixed_cAij = raise_index(args.cA, args.cY, icY);

    for(int i=0; i < 3; i++)
    {
        value s1 = 0;

        for(int m=0; m < 3; m++)
        {
            for(int n=0; n < 3; n++)
            {
                s1 += icY.idx(m, n) * dmni.idx(m, n, i);
            }
        }

        value s2 = -(2.f/3.f) * diff1(ctx, args.K, i);

        value s3 = 0;

        for(int m=0; m < 3; m++)
        {
            s3 += -(3.f/2.f) * mixed_cAij.idx(m, i) * diff1(ctx, args.X, m) / X_clamped;
        }

        /*Mi.idx(i) = dual_if(args.X <= 0.001f,
        []()
        {
            return 0.f;
        },
        [&]()
        {
            return s1 + s2 + s3;
        });*/

        Mi.idx(i) = s1 + s2 + s3;
    }
    #endif // 0

    tensor<value, 3, 3> second_cAij = raise_second_index(args.cA, args.cY, unpinned_icY);

    for(int i=0; i < 3; i++)
    {
        value s1 = 0;

        for(int j=0; j < 3; j++)
        {
            s1 += diff1(ctx, second_cAij.idx(i, j), j);
        }

        value s2 = 0;

        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                s2 += -0.5f * icY.idx(j, k) * diff1(ctx, args.cA.idx(j, k), i);
            }
        }

        value s3 = 0;

        for(int j=0; j < 3; j++)
        {
            s3 += -0.25f * 6 * (1/X_clamped) * diff1(ctx, args.X, j) * second_cAij.idx(i, j);
        }

        value s4 = -(2.f/3.f) * diff1(ctx, args.K, i);

        Mi.idx(i) = s1 + s2 + s3 + s4;
    }
    #endif // 0

    /*tensor<value, 3> Mi;

    tensor<value, 3, 3> second_cAij = raise_second_index(args.cA, args.cY, unpinned_icY);

    for(int i=0; i < 3; i++)
    {
        value s1 = 0;

        for(int j=0; j < 3; j++)
        {
            s1 += hacky_differentiate(second_cAij.idx(i, j), j);
        }

        value s2 = 0;

        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                s2 += -0.5f * icY.idx(j, k) * hacky_differentiate(args.cA.idx(j, k), i);
            }
        }

        value s3 = 0;

        for(int j=0; j < 3; j++)
        {
            s3 += -0.25f * 6 * X_recip * hacky_differentiate(args.X, j) * second_cAij.idx(i, j);
        }

        value s4 = -(2.f/3.f) * hacky_differentiate(args.K, i);

        Mi.idx(i) = s1 + s2 + s3 + s4;
    }*/

    for(int i=0; i < 3; i++)
    {
        ctx.add("init_momentum" + std::to_string(i), Mi.idx(i));
    }
}

inline
void build_cY(equation_context& ctx)
{
    standard_arguments args(ctx);

    metric<value, 3, 3> unpinned_cY = args.cY;

    ctx.pin(args.cY);

    tensor<value, 3> bigGi_lower = lower_index(args.bigGi, args.cY);
    tensor<value, 3> gB_lower = lower_index(args.gB, args.cY);

    ctx.pin(bigGi_lower);
    ctx.pin(gB_lower);

    tensor<value, 3, 3> lie_cYij = gpu_lie_derivative_weight(ctx, args.gB, unpinned_cY);

    ///https://arxiv.org/pdf/gr-qc/0511048.pdf (1)
    ///https://scholarworks.rit.edu/cgi/viewcontent.cgi?article=11286&context=theses 3.66
    tensor<value, 3, 3> dtcYij = -2 * args.gA * args.cA + lie_cYij;

    ///makes it to 50 with this enabled
    #define USE_DTCYIJ_MODIFICATION
    #ifdef USE_DTCYIJ_MODIFICATION
    ///https://arxiv.org/pdf/1205.5111v1.pdf 46
    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            float sigma = 4/5.f;

            dtcYij.idx(i, j) += sigma * 0.5f * (gB_lower.idx(i) * bigGi_lower.idx(j) + gB_lower.idx(j) * bigGi_lower.idx(i));

            dtcYij.idx(i, j) += -(1.f/5.f) * args.cY.idx(i, j) * sum_multiply(args.gB, bigGi_lower);
        }
    }
    #endif // USE_DTCYIJ_MODIFICATION

    for(int i=0; i < 6; i++)
    {
        std::string name = "dtcYij" + std::to_string(i);

        vec2i idx = args.linear_indices[i];

        ctx.add(name, dtcYij.idx(idx.x(), idx.y()));
    }
}

inline
tensor<value, 3, 3> calculate_xgARij(equation_context& ctx, standard_arguments& args, const inverse_metric<value, 3, 3>& icY, const tensor<value, 3, 3, 3>& christoff1, const tensor<value, 3, 3, 3>& christoff2)
{
    value gA_X = args.gA_X;

    tensor<value, 3, 3> cRij;

    tensor<value, 3> derived_cGi = args.derived_cGi;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value s1 = 0;

            for(int l=0; l < 3; l++)
            {
                for(int m=0; m < 3; m++)
                {
                    s1 = s1 + -0.5f * icY.idx(l, m) * diff2(ctx, args.cY.idx(i, j), m, l, args.dcYij.idx(m, i, j), args.dcYij.idx(l, i, j));
                }
            }

            value s2 = 0;

            for(int k=0; k < 3; k++)
            {
                s2 = s2 + 0.5f * (args.cY.idx(k, i) * diff1(ctx, args.cGi.idx(k), j) + args.cY.idx(k, j) * diff1(ctx, args.cGi.idx(k), i));
            }

            value s3 = 0;

            for(int k=0; k < 3; k++)
            {
                s3 = s3 + 0.5f * derived_cGi.idx(k) * (christoff1.idx(i, j, k) + christoff1.idx(j, i, k));
            }

            value s4 = 0;

            for(int m=0; m < 3; m++)
            {
                for(int l=0; l < 3; l++)
                {
                    value inner1 = 0;
                    value inner2 = 0;

                    for(int k=0; k < 3; k++)
                    {
                        inner1 = inner1 + 0.5f * (2 * christoff2.idx(k, l, i) * christoff1.idx(j, k, m) + 2 * christoff2.idx(k, l, j) * christoff1.idx(i, k, m));
                    }

                    for(int k=0; k < 3; k++)
                    {
                        inner2 = inner2 + christoff2.idx(k, i, m) * christoff1.idx(k, l, j);
                    }

                    s4 = s4 + icY.idx(l, m) * (inner1 + inner2);
                }
            }

            cRij.idx(i, j) = s1 + s2 + s3 + s4;
        }
    }

    tensor<value, 3, 3> cov_div_X = gpu_double_covariant_derivative(ctx, args.X, args.dX, args.cY, icY, christoff2);
    ctx.pin(cov_div_X);

    ///https://indico.cern.ch/event/505595/contributions/1183661/attachments/1332828/2003830/sperhake.pdf
    tensor<value, 3, 3> xgARphiij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value s1 = 0;
            value s2 = 0;

            for(int m=0; m < 3; m++)
            {
                for(int n=0; n < 3; n++)
                {
                    s1 += icY.idx(m, n) * cov_div_X.idx(n, m);
                    s2 += icY.idx(m, n) * args.dX.idx(m) * args.dX.idx(n);
                }
            }

            value s3 = (1/2.f) * (args.gA * cov_div_X.idx(j, i) - gA_X * (1/2.f) * args.dX.idx(i) * args.dX.idx(j));

            s1 = args.gA * (args.cY.idx(i, j) / 2.f) * s1;
            s2 = gA_X * (args.cY.idx(i, j) / 2.f) * -(3.f/2.f) * s2;

            xgARphiij.idx(i, j) = s1 + s2 + s3;
        }
    }

    tensor<value, 3, 3> xgARij = xgARphiij + args.X * args.gA * cRij;

    ctx.pin(xgARij);

    return xgARij;
}

value calculate_hamiltonian(const metric<value, 3, 3>& Yij, const inverse_metric<value, 3, 3>& iYij, const tensor<value, 3, 3>& Rij, const value& K, const tensor<value, 3, 3>& Kij)
{
    value R = gpu_trace(Rij, Yij, iYij);

    tensor<value, 3, 3> KIJ = raise_both(Kij, Yij, iYij);

    value Kij_KIJ = 0;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            Kij_KIJ += Kij.idx(i, j) * KIJ.idx(i, j);
        }
    }

    return R + K*K - Kij_KIJ;
}

value calculate_hamiltonian(equation_context& ctx, standard_arguments& args)
{
    auto icY = args.cY.invert();

    tensor<value, 3, 3, 3> christoff1 = gpu_christoffel_symbols_1(ctx, args.cY);

    tensor<value, 3, 3> xgARij = calculate_xgARij(ctx, args, icY, christoff1, args.christoff2);

    return calculate_hamiltonian(args.Yij, args.iYij, (xgARij / (args.X * args.gA)), args.K, args.Kij);
}

inline
void build_cA(equation_context& ctx)
{
    standard_arguments args(ctx);

    value scale = "scale";

    ctx.pin(args.derived_cGi);

    inverse_metric<value, 3, 3> icY = args.cY.invert();

    //tensor<value, 3, 3, 3> christoff1 = gpu_christoffel_symbols_1(ctx, args.cY);
    tensor<value, 3, 3, 3> christoff2 = args.christoff2;

    tensor<value, 3, 3, 3> christoff1;

    ///Gak Ckbc
    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                value sum = 0;

                for(int d = 0; d < 3; d++)
                {
                    sum += args.cY.idx(i, d) * christoff2.idx(d, j, k);
                }

                christoff1.idx(i, j, k) = sum;
            }
        }
    }

    ctx.pin(christoff1);
    ctx.pin(christoff2);

    unit_metric<value, 3, 3> cY = args.cY;

    inverse_metric<value, 3, 3> unpinned_icY = cY.invert();

    ctx.pin(icY);

    tensor<value, 3, 3> cA = args.cA;

    auto unpinned_cA = cA;

    ///the christoffel symbol
    tensor<value, 3> cGi = args.cGi;

    value gA = args.gA;

    tensor<value, 3> gB = args.gB;

    value X = args.X;
    value K = args.K;

    tensor<value, 3> derived_cGi = args.derived_cGi;

    ///a / X
    value gA_X = args.gA_X;

    tensor<value, 3, 3> xgARij = calculate_xgARij(ctx, args, icY, christoff1, christoff2);

    ctx.pin(xgARij);

    tensor<value, 3, 3> Xdidja;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value Xderiv = X * gpu_double_covariant_derivative(ctx, args.gA, args.digA, cY, icY, args.christoff2).idx(j, i);
            //value Xderiv = X * gpu_covariant_derivative_low_vec(ctx, args.digA, cY, icY).idx(j, i);

            value s2 = 0.5f * (diff1(ctx, X, i) * diff1(ctx, gA, j) + diff1(ctx, X, j) * diff1(ctx, gA, i));

            value s3 = 0;

            for(int m=0; m < 3; m++)
            {
                for(int n=0; n < 3; n++)
                {
                    value v = icY.idx(m, n) * diff1(ctx, X, m) * diff1(ctx, gA, n);

                    s3 += v;
                }
            }

            Xdidja.idx(i, j) = Xderiv + s2 + -0.5f * cY.idx(i, j) * s3;
        }
    }

    ctx.pin(Xdidja);

    ///recover Yij from X and cYij
    ///https://arxiv.org/pdf/gr-qc/0511048.pdf
    ///https://arxiv.org/pdf/gr-qc/9810065.pdf
    ///X = exp(-4 phi)
    ///consider trying to eliminate via https://arxiv.org/pdf/gr-qc/0206072.pdf (27). I think this is what you're meant to do
    ///to eliminate the dependency on the non conformal metric entirely. This would improve stability quite significantly
    ///near the puncture

    ///Aki G^kj
    tensor<value, 3, 3> mixed_cAij = raise_index(cA, cY, icY);

    ctx.pin(mixed_cAij);

    ///not sure dtcaij is correct, need to investigate
    tensor<value, 3, 3> dtcAij;

    ///https://indico.cern.ch/event/505595/contributions/1183661/attachments/1332828/2003830/sperhake.pdf replaced with definition under bssn aux
    tensor<value, 3, 3> with_trace = -Xdidja + xgARij;

    tensor<value, 3, 3> without_trace = gpu_trace_free(with_trace, cY, icY);

    #ifdef BETTERDAMP_DTCAIJ
    tensor<value, 3, 3> momentum_deriv;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            momentum_deriv.idx(i, j) = diff1(ctx, args.momentum_constraint.idx(i), j);
        }
    }

    tensor<value, 3, 3> symmetric_momentum_deriv;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            symmetric_momentum_deriv.idx(i, j) = 0.5f * (momentum_deriv.idx(i, j) + momentum_deriv.idx(j, i));
        }
    }

    ctx.pin(symmetric_momentum_deriv);

    #endif // BETTERDAMP_DTCAIJ

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value sum = 0;

            for(int k=0; k < 3; k++)
            {
                sum = sum + cA.idx(i, k) * mixed_cAij.idx(k, j);
            }

            ///so
            ///the trace is calculated as iYij Vij, where Vij is whatever
            ///if Yij = cYij / X
            ///https://en.wikipedia.org/wiki/Invertible_matrix#Other_properties
            ///then iYij = = X * icYij
            ///the trace is the sum X * icYij * Vij
            ///making something trace free is denoted as Vij - (1/3) metij * V, where V = trace
            ///= Vij - (1/3) Yij * V
            ///= Vij - (1/3) (cYij / X) * V
            ///but the trace is the sum of something multiplied by X
            ///= Vij - (1/3) cYij (icYkl Vkl)
            ///therefore I think constant factor multiplications to the metric make no difference to the trace calculation, so we can use
            ///cY here instead of Yij

            ///not convinced its correct to push x inside of trace free?
            ///what if the riemann quantity is made trace free by cY instead of Yij like I assumed?
            value p1 = without_trace.idx(i, j);

            value p2 = gA * (K * cA.idx(i, j) - 2 * sum);

            value p3 = gpu_lie_derivative_weight(ctx, gB, unpinned_cA).idx(i, j);

            if(i == 0 && j == 0)
            {
                ctx.add("debug_p1", p1);
                ctx.add("debug_p2", p2);
                ctx.add("debug_p3", p3);
            }

            dtcAij.idx(i, j) = p1 + p2 + p3;

            #ifdef DAMP_DTCAIJ
            float Ka = 0.0001f;

            dtcAij.idx(i, j) += Ka * gA * 0.5f *
                                                (gpu_covariant_derivative_low_vec(ctx, args.momentum_constraint, cY, icY).idx(i, j)
                                                 + gpu_covariant_derivative_low_vec(ctx, args.momentum_constraint, cY, icY).idx(j, i));
            #endif // DAMP_DTCAIJ

            #ifdef BETTERDAMP_DTCAIJ
            value F_a = scale * gA;

            ///https://arxiv.org/pdf/1205.5111v1.pdf (56)
            dtcAij.idx(i, j) += scale * F_a * gpu_trace_free(symmetric_momentum_deriv, cY, icY).idx(i, j);
            #endif // BETTERDAMP_DTCAIJ
        }
    }

    for(int i=0; i < 6; i++)
    {
        std::string name = "dtcAij" + std::to_string(i);

        vec2i idx = args.linear_indices[i];

        ctx.add(name, dtcAij.idx(idx.x(), idx.y()));
    }
}

inline
void build_cGi(equation_context& ctx)
{
    standard_arguments args(ctx);

    inverse_metric<value, 3, 3> icY = args.cY.invert();

    tensor<value, 3, 3, 3> christoff2 = args.christoff2;

    ctx.pin(christoff2);

    unit_metric<value, 3, 3> cY = args.cY;

    inverse_metric<value, 3, 3> unpinned_icY = cY.invert();

    tensor<value, 3, 3> cA = args.cA;

    ///the christoffel symbol
    tensor<value, 3> cGi = args.cGi;

    value gA = args.gA;

    tensor<value, 3> gB = args.gB;

    value X = args.X;
    value K = args.K;

    tensor<value, 3, 3> icAij = raise_both(cA, cY, icY);

    value gA_X = args.gA_X;

    ///these seem to suffer from oscillations
    tensor<value, 3> dtcGi;

    tensor<value, 3> derived_cGi = args.derived_cGi;

    ///https://arxiv.org/pdf/gr-qc/0511048.pdf

    ///https://arxiv.org/pdf/1205.5111v1.pdf 49
    ///made it to 58 with this
    #define CHRISTOFFEL_49
    #ifdef CHRISTOFFEL_49
    tensor<value, 3, 3> littlekij = unpinned_icY.to_tensor() * K;

    ///PAPER_12055111_SUBST

    tensor<value, 3> Yij_Kj;

    #define PAPER_1205_5111
    #ifdef PAPER_1205_5111
    for(int i=0; i < 3; i++)
    {
        value sum = 0;

        for(int j=0; j < 3; j++)
        {
            sum += diff1(ctx, littlekij.idx(i, j), j);
        }

        Yij_Kj.idx(i) = sum + args.K * derived_cGi.idx(i);
    }
    #else
    for(int i=0; i < 3; i++)
    {
        value sum = 0;

        for(int j=0; j < 3; j++)
        {
            sum += icY.idx(i, j) * diff1(ctx, args.K, j);
        }

        Yij_Kj.idx(i) = sum;
    }
    #endif // PAPER_1205_5111

    for(int i=0; i < 3; i++)
    {
        value s1 = 0;

        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                s1 += 2 * gA * christoff2.idx(i, j, k) * icAij.idx(j, k);
            }
        }

        value s2 = 2 * gA * -(2.f/3.f) * Yij_Kj.idx(i);

        value s3 = 0;

        for(int j=0; j < 3; j++)
        {
            s3 += 2 * (-1.f/4.f) * gA_X * 6 * icAij.idx(i, j) * diff1(ctx, X, j);
        }

        value s4 = 0;

        for(int j=0; j < 3; j++)
        {
            s4 += -2 * icAij.idx(i, j) * diff1(ctx, gA, j);
        }

        value s5 = 0;

        for(int j=0; j < 3; j++)
        {
            s5 += upwind_differentiate(ctx, gB.idx(j), cGi.idx(i), j);
        }

        value s6 = 0;

        for(int j=0; j < 3; j++)
        {
            s6 += -derived_cGi.idx(j) * args.digB.idx(j, i);
        }

        value s7 = 0;

        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                //s7 += icY.idx(j, k) * hacky_differentiate(args.digB.idx(k, i), j);
                s7 += icY.idx(j, k) * diff2(ctx, args.gB.idx(i), k, j, args.digB.idx(k, i), args.digB.idx(j, i));
            }
        }

        value s8 = 0;

        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                //s8 += (1.f/3.f) * icY.idx(i, j) * hacky_differentiate(args.digB.idx(k, k), j);
                s8 += (1.f/3.f) * icY.idx(i, j) * diff2(ctx, args.gB.idx(k), k, j, args.digB.idx(k, k), args.digB.idx(j, k));
            }
        }

        value s9 = 0;

        for(int k=0; k < 3; k++)
        {
            s9 += (2.f/3.f) * args.digB.idx(k, k) * derived_cGi.idx(i);
        }

        ///this is the only instanced of derived_cGi that might want to be regular cGi
        //value s10 = (2.f/3.f) * -2 * gA * K * derived_cGi.idx(i);

        dtcGi.idx(i) = s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9;

        ///https://arxiv.org/pdf/1205.5111v1.pdf 50
        ///made it to 70+ and then i got bored, but the simulation was meaningfully different
        #define EQ_50
        #ifdef EQ_50
        auto step = [](const value& in)
        {
            return dual_if(in >= 0,
            [](){return 1;},
            [](){return 0;});
        };

        value bkk = 0;

        for(int k=0; k < 3; k++)
        {
            bkk += args.digB.idx(k, k);
        }

        float E = 1;

        value lambdai = (2.f/3.f) * (bkk - 2 * gA * K)
                        - args.digB.idx(i, i)
                        - (2.f/5.f) * gA * raise_second_index(cA, cY, icY).idx(i, i);

        dtcGi.idx(i) += -(1 + E) * step(lambdai) * lambdai * args.bigGi.idx(i);
        #endif // EQ_50

        //#define YBS
        #ifdef YBS
        value E = 1;

        {
            value sum = 0;

            for(int k=0; k < 3; k++)
            {
                sum += diff1(ctx, args.gB.idx(k), k);
            }

            dtcGi.idx(i) += (-2.f/3.f) * (E + 1) * args.bigGi.idx(i) * sum;
        }
        #endif // YBS
    }
    #endif // CHRISTOFFEL_49

    for(int i=0; i < 3; i++)
    {
        std::string name = "dtcGi" + std::to_string(i);

        ctx.add(name, dtcGi.idx(i));
    }
}

inline
void build_K(equation_context& ctx)
{
    standard_arguments args(ctx);

    inverse_metric<value, 3, 3> icY = args.cY.invert();

    unit_metric<value, 3, 3> cY = args.cY;

    ctx.pin(icY);

    tensor<value, 3, 3> cA = args.cA;

    ///the christoffel symbol
    tensor<value, 3> cGi = args.cGi;

    value gA = args.gA;

    tensor<value, 3> gB = args.gB;

    value X = args.X;
    value K = args.K;

    tensor<value, 3, 3> Xdidja;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value Xderiv = X * gpu_double_covariant_derivative(ctx, args.gA, args.digA, cY, icY, args.christoff2).idx(j, i);
            //value Xderiv = X * gpu_covariant_derivative_low_vec(ctx, args.digA, cY, icY).idx(j, i);

            value s2 = 0.5f * (diff1(ctx, X, i) * diff1(ctx, gA, j) + diff1(ctx, X, j) * diff1(ctx, gA, i));

            value s3 = 0;

            for(int m=0; m < 3; m++)
            {
                for(int n=0; n < 3; n++)
                {
                    value v = -0.5f * cY.idx(i, j) * icY.idx(m, n) * diff1(ctx, X, m) * diff1(ctx, gA, n);

                    s3 += v;
                }
            }

            Xdidja.idx(i, j) = Xderiv + s2 + s3;
        }
    }

    tensor<value, 3, 3> icAij = raise_both(cA, cY, icY);

    value dtK = sum(tensor_upwind(ctx, gB, K)) - sum_multiply(icY.to_tensor(), Xdidja) + gA * (sum_multiply(icAij, cA) + (1/3.f) * K * K);

    ctx.add("dtK", dtK);
}

inline
void build_X(equation_context& ctx)
{
    standard_arguments args(ctx);

    tensor<value, 3> linear_dB;

    for(int i=0; i < 3; i++)
    {
        linear_dB.idx(i) = diff1(ctx, args.gB.idx(i), i);
    }

    value dtX = (2.f/3.f) * args.X * (args.gA * args.K - sum(linear_dB)) + sum(tensor_upwind(ctx, args.gB, args.X));

    ctx.add("dtX", dtX);
}

inline
void build_gA(equation_context& ctx)
{
    standard_arguments args(ctx);

    //value bl_s = "(init_BL_val)";
    //value bl = bl_s + 1;

    ///https://arxiv.org/pdf/gr-qc/0206072.pdf (94)
    ///this breaks immediately
    //int m = 4;
    //value dtgA = lie_derivative(ctx, args.gB, args.gA) - 2 * args.gA * args.K * pow(bl, m);

    value dtgA = lie_derivative(ctx, args.gB, args.gA) - 2 * args.gA * args.K;

    ctx.add("dtgA", dtgA);
}

inline
void build_gB(equation_context& ctx)
{
    standard_arguments args(ctx);

    tensor<value, 3> bjdjbi;

    for(int i=0; i < 3; i++)
    {
        value v = 0;

        for(int j=0; j < 3; j++)
        {
           v += upwind_differentiate(ctx, args.gB.idx(j), args.gB.idx(i), j);
        }

        bjdjbi.idx(i) = v;
    }

    #ifndef USE_GBB
    ///https://arxiv.org/pdf/gr-qc/0605030.pdf 26
    ///todo: remove this

    float N = 2;

    tensor<value, 3> dtgB = (3.f/4.f) * args.derived_cGi + bjdjbi - N * args.gB;

    tensor<value, 3> dtgBB;
    dtgBB.idx(0) = 0;
    dtgBB.idx(1) = 0;
    dtgBB.idx(2) = 0;

    #else

    tensor<value, 3> bjdjBi;

    for(int i=0; i < 3; i++)
    {
        value v = 0;

        for(int j=0; j < 3; j++)
        {
           v += upwind_differentiate(ctx, args.gB.idx(j), args.gBB.idx(i), j);
        }

        bjdjBi.idx(i) = v;
    }

    tensor<value, 3> christoffd;

    for(int i=0; i < 3; i++)
    {
        value sum = 0;

        for(int j=0; j < 3; j++)
        {
            sum += args.gB.idx(j) * diff1(ctx, args.cGi.idx(i), j);
        }

        christoffd.idx(i) = sum;
    }

    tensor<value, 3> dtcGi;
    dtcGi.idx(0).make_value("f_dtcGi0");
    dtcGi.idx(1).make_value("f_dtcGi1");
    dtcGi.idx(2).make_value("f_dtcGi2");

    tensor<value, 3> dtgB;
    tensor<value, 3> dtgBB;

    ///https://arxiv.org/pdf/gr-qc/0511048.pdf (11)
    /*for(int i=0; i < 3; i++)
    {
        dtgB.idx(i) = (3.f/4.f) * args.gBB.idx(i) + bjdjbi.idx(i);
    }*/

    /*#ifdef PAPER_0610128
    float N = 1;

    dtgB = (3.f/4.f) * args.gBB;

    dtgBB = dtcGi - N * args.gBB;
    #else*/
    dtgB = (3.f/4.f) * args.gBB + bjdjbi;

    float N = 1;

    dtgBB = dtcGi - N * args.gBB + bjdjBi - christoffd;
    //#endif // PAPER_0610128
    #endif // USE_GBB

    for(int i=0; i < 3; i++)
    {
        std::string name = "dtgB" + std::to_string(i);

        ctx.add(name, dtgB.idx(i));
    }

    for(int i=0; i < 3; i++)
    {
        std::string name = "dtgBB" + std::to_string(i);

        ctx.add(name, dtgBB.idx(i));
    }
}

template<typename T, int N>
inline
tensor<T, N, N> raise_index_impl(const tensor<T, N, N>& mT, const inverse_metric<T, N, N>& met, int index)
{
    tensor<T, N, N> ret;

    for(int i=0; i < N; i++)
    {
        for(int j=0; j < N; j++)
        {
            T sum = 0;

            for(int s=0; s < N; s++)
            {
                if(index == 0)
                {
                    sum = sum + met.idx(i, s) * mT.idx(s, j);
                }

                if(index == 1)
                {
                    sum = sum + met.idx(j, s) * mT.idx(i, s);
                }
            }

            ret.idx(i, j) = sum;
        }
    }

    return ret;
}

template<typename T, int N>
inline
tensor<T, N, N, N> raise_index_impl(const tensor<T, N, N, N>& mT, const inverse_metric<T, N, N>& met, int index)
{
    tensor<T, N, N, N> ret;

    for(int i=0; i < N; i++)
    {
        for(int j=0; j < N; j++)
        {
            for(int k=0; k < N; k++)
            {
                T sum = 0;

                for(int s=0; s < N; s++)
                {
                    if(index == 0)
                    {
                        sum = sum + met.idx(i, s) * mT.idx(s, j, k);
                    }

                    if(index == 1)
                    {
                        sum = sum + met.idx(j, s) * mT.idx(i, s, k);
                    }

                    if(index == 2)
                    {
                        sum = sum + met.idx(k, s) * mT.idx(i, j, s);
                    }
                }

                ret.idx(i, j, k) = sum;
            }
        }
    }

    return ret;
}

template<typename T, int U, int... N>
inline
tensor<T, N...> raise_index_generic(const tensor<T, N...>& mT, const inverse_metric<T, U, U>& met, int index)
{
    return raise_index_impl(mT, met, index);
}

template<int N>
value dot_product(const vec<N, value>& u, const vec<N, value>& v, const metric<value, N, N>& met)
{
    tensor<value, N> as_tensor;

    for(int i=0; i < N; i++)
    {
        as_tensor.idx(i) = u[i];
    }

    auto lowered_as_tensor = lower_index(as_tensor, met);

    vec<N, value> lowered;

    for(int i=0; i < N; i++)
    {
        lowered[i] = lowered_as_tensor.idx(i);
    }

    return dot(lowered, v);
}

template<int N>
vec<N, value> gram_proj(const vec<N, value>& u, const vec<N, value>& v, const metric<value, N, N>& met)
{
    value top = dot_product(u, v, met);

    value bottom = dot_product(u, u, met);

    return (top / bottom) * u;
}

template<int N>
vec<N, value> normalize_big_metric(const vec<N, value>& in, const metric<value, N, N>& met)
{
    value dot = dot_product(in, in, met);

    return in / sqrt(fabs(dot));
}

///raised indices for v/1/2/3
std::array<vec<3, value>, 3> orthonormalise(equation_context& ctx, const vec<3, value>& v1, const vec<3, value>& v2, const vec<3, value>& v3, const metric<value, 3, 3>& met)
{
    ctx.add("dbgv0", v1[0]);
    ctx.add("dbgv1", v1[1]);
    ctx.add("dbgv2", v1[2]);

    ctx.add("dbgv3", v2[0]);
    ctx.add("dbgv4", v2[1]);
    ctx.add("dbgv5", v2[2]);

    vec<3, value> u1 = v1;

    ctx.pin(u1);

    vec<3, value> u2 = v2;

    ctx.pin(u2);

    u2 = u2 - gram_proj(u1, u2, met);

    ctx.pin(u2);

    vec<3, value> u3 = v3;

    ctx.pin(u3);

    u3 = u3 - gram_proj(u1, u3, met);
    u3 = u3 - gram_proj(u2, u3, met);

    ctx.pin(u3);

    /*u1 = u1.norm();
    u2 = u2.norm();
    u3 = u3.norm();*/

    ctx.pin(u1);
    ctx.pin(u2);
    ctx.pin(u3);

    //ctx.add("dbgw2", dot_product(u2, u2, met));
    //ctx.add("dbgw2", lower_index(as_tensor, met).idx(0));

    u1 = normalize_big_metric(u1, met);
    u2 = normalize_big_metric(u2, met);
    u3 = normalize_big_metric(u3, met);

    ctx.pin(u1);
    ctx.pin(u2);
    ctx.pin(u3);

    return {u1, u2, u3};
}


///https://arxiv.org/pdf/1503.08455.pdf (8)
metric<value, 4, 4> calculate_real_metric(const metric<value, 3, 3>& adm, const value& gA, const tensor<value, 3>& gB)
{
    tensor<value, 3> lower_gB = lower_index(gB, adm);

    metric<value, 4, 4> ret;

    value gB_sum = 0;

    for(int i=0; i < 3; i++)
    {
        gB_sum = gB_sum + lower_gB.idx(i) * gB.idx(i);
    }

    ///https://arxiv.org/pdf/gr-qc/0703035.pdf 4.43
    ret.idx(0, 0) = -gA * gA + gB_sum;

    ///latin indices really run from 1-4
    for(int i=1; i < 4; i++)
    {
        ///https://arxiv.org/pdf/gr-qc/0703035.pdf 4.45
        ret.idx(i, 0) = lower_gB.idx(i - 1);
        ret.idx(0, i) = ret.idx(i, 0); ///symmetry
    }

    for(int i=1; i < 4; i++)
    {
        for(int j=1; j < 4; j++)
        {
            ret.idx(i, j) = adm.idx(i - 1, j - 1);
        }
    }

    return ret;
}

///https://scholarworks.rit.edu/cgi/viewcontent.cgi?article=11286&context=theses (3.33)
tensor<value, 4> get_adm_hypersurface_normal_raised(const value& gA, const tensor<value, 3>& gB)
{
    return {1/gA, -gB.idx(0)/gA, -gB.idx(1)/gA, -gB.idx(2)/gA};
}

tensor<value, 4> get_adm_hypersurface_normal_lowered(const value& gA)
{
    return {-gA, 0, 0, 0};
}

///https://arxiv.org/pdf/1503.08455.pdf (10)
///also the projection tensor
metric<value, 4, 4> calculate_induced_metric(const metric<value, 3, 3>& adm, const value& gA, const tensor<value, 3>& gB)
{
    metric<value, 4, 4> spacetime = calculate_real_metric(adm, gA, gB);

    tensor<value, 4> nu = get_adm_hypersurface_normal_lowered(gA);

    metric<value, 4, 4> induced;

    for(int i=0; i < 4; i++)
    {
        for(int j=0; j < 4; j++)
        {
            induced.idx(i, j) = spacetime.idx(i, j) + nu.idx(i) * nu.idx(j);
        }
    }

    return induced;
}

///https://indico.cern.ch/event/505595/contributions/1183661/attachments/1332828/2003830/sperhake.pdf 28
tensor<value, 4, 4> calculate_projector(const value& gA, const tensor<value, 3>& gB)
{
    tensor<value, 4> nu_u = get_adm_hypersurface_normal_raised(gA, gB);
    tensor<value, 4> nu_l = get_adm_hypersurface_normal_lowered(gA);

    tensor<value, 4, 4> proj;

    for(int i=0; i < 4; i++)
    {
        for(int j=0; j < 4; j++)
        {
            value kronecker = (i == j) ? 1 : 0;

            proj.idx(i, j) = kronecker + nu_u.idx(i) * nu_l.idx(j);
        }
    }

    return proj;
}

template<typename T>
tensor<T, 4> tensor_project_lower(const tensor<T, 4>& in, const value& gA, const tensor<value, 3>& gB)
{
    tensor<value, 4, 4> projector = calculate_projector(gA, gB);

    tensor<T, 4> full_ret;

    for(int i=0; i < 4; i++)
    {
        T sum = 0;

        for(int j=0; j < 4; j++)
        {
            sum += projector.idx(j, i) * in.idx(j);
        }

        full_ret.idx(i) = sum;
    }

    return full_ret;
}

template<typename T>
tensor<T, 4> tensor_project_upper(const tensor<T, 4>& in, const value& gA, const tensor<value, 3>& gB)
{
    tensor<value, 4, 4> projector = calculate_projector(gA, gB);

    tensor<T, 4> full_ret;

    for(int i=0; i < 4; i++)
    {
        T sum = 0;

        for(int j=0; j < 4; j++)
        {
            sum += projector.idx(i, j) * in.idx(j);
        }

        full_ret.idx(i) = sum;
    }

    return full_ret;
}

///https://scc.ustc.edu.cn/zlsc/sugon/intel/ipp/ipp_manual/IPPM/ippm_ch9/ch9_SHT.htm this states you can approximate
///a spherical harmonic transform integral with simple summation
///assumes unigrid
///https://arxiv.org/pdf/1606.02532.pdf 104 etc
#if 1
inline
void extract_waveforms(equation_context& ctx)
{
    printf("Extracting waveforms\n");

    tensor<value, 3, 3> kronecker;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            if(i == j)
                kronecker.idx(i, j) = 1;
            else
                kronecker.idx(i, j) = 0;
        }
    }

    standard_arguments args(ctx);

    inverse_metric<value, 3, 3> icY = args.cY.invert();
    ctx.pin(icY);

    tensor<value, 3, 3, 3> christoff1 = gpu_christoffel_symbols_1(ctx, args.cY);
    tensor<value, 3, 3, 3> christoff2 = gpu_christoffel_symbols_2(ctx, args.cY, icY);

    ctx.pin(christoff1);
    ctx.pin(christoff2);

    ///NEEDS SCALING???
    vec<3, value> pos;
    pos.x() = "offset.x";
    pos.y() = "offset.y";
    pos.z() = "offset.z";

    tensor<value, 3, 3> xgARij = calculate_xgARij(ctx, args, icY, christoff1, christoff2);

    tensor<value, 3, 3> Rij = xgARij / (args.gA * args.X);

    ctx.pin(Rij);

    tensor<value, 3, 3> Kij = args.Kij;
    tensor<value, 3, 3> unpinned_Kij = Kij;

    ctx.pin(Kij);

    metric<value, 3, 3> Yij = args.Yij;

    ctx.pin(Yij);

    inverse_metric<value, 3, 3> iYij = args.X * icY;
    ctx.pin(iYij);

    //inverse_metric<value, 3, 3> iYij = Yij.invert();

    //ctx.pin(iYij);

    //auto christoff_Y = gpu_christoffel_symbols_2(Yij, iYij);

    ///lecture slides
    tensor<value, 3, 3, 3> christoff_Y;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                value sum = 0;

                for(int m=0; m < 3; m++)
                {
                    sum += -args.cY.idx(j, k) * icY.idx(i, m) * diff1(ctx, args.X, m);
                }

                christoff_Y.idx(i, j, k) = christoff2.idx(i, j, k) - (1 / (2 * args.X)) *
                (kronecker.idx(i, k) * diff1(ctx, args.X, j) +
                 kronecker.idx(i, j) * diff1(ctx, args.X, k) +
                 sum);
            }
        }
    }

    ctx.pin(christoff_Y);

    ///l, j, k
    ///aka: i, j, covariant derivative
    ///or a, b; c in wikipedia notation
    tensor<value, 3, 3, 3> cdKij;

    for(int c=0; c < 3; c++)
    {
        for(int b=0; b < 3; b++)
        {
            for(int a=0; a < 3; a++)
            {
                value deriv = diff1(ctx, unpinned_Kij.idx(a, b), c);

                value sum = 0;

                for(int d=0; d < 3; d++)
                {
                    sum = sum - christoff_Y.idx(d, c, a) * Kij.idx(d, b) - christoff_Y.idx(d, c, b) * Kij.idx(a, d);
                }

                cdKij.idx(a, b, c) = deriv + sum;
            }
        }
    }

    ctx.pin(cdKij);

    tensor<value, 3, 3, 3> eijk = get_eijk();

    ///can raise and lower the indices... its a regular tensor bizarrely
    tensor<value, 3, 3, 3> eijk_tensor = sqrt(Yij.det()) * eijk;

    ctx.pin(eijk_tensor);

    /*value s = pos.length();
    value theta = acos(pos.z() / s);
    value phi = atan2(pos.y(), pos.x());

    ///covariant derivative is just partial differentiation

    dual_types::complex<value, value> i = dual_types::unit_i();

    vec<3, value> pd_s;
    pd_s.x() = s.differentiate("offset.x");
    pd_s.y() = s.differentiate("offset.y");
    pd_s.z() = s.differentiate("offset.z");

    vec<3, value> pd_theta;
    pd_theta.x() = theta.differentiate("offset.x");
    pd_theta.y() = theta.differentiate("offset.y");
    pd_theta.z() = theta.differentiate("offset.z");

    vec<3, value> pd_phi;
    pd_phi.x() = phi.differentiate("offset.x");
    pd_phi.y() = phi.differentiate("offset.y");
    pd_phi.z() = phi.differentiate("offset.z");

    vec<4, dual_types::complex<value, value>> cval;

    cval.x() = 0;
    cval.y() = pd_theta.x() + i * pd_phi.x();
    cval.z() = pd_theta.y() + i * pd_phi.y();
    cval.w() = pd_theta.z() + i * pd_phi.z();*/

    ///https://arxiv.org/pdf/gr-qc/0104063.pdf
    ///so, according to the lazarus project:
    ///https://d-nb.info/999452606/34 and 6.16c.. though there's a typo there

    ///gab is their spatial metric in lazarus

    pos.x() += 0.005f;
    pos.y() += 0.005f;
    pos.z() += 0.005f;

    vec<3, value> v1ai = {-pos.y(), pos.x(), 0};
    vec<3, value> v2ai = {pos.x(), pos.y(), pos.z()};
    //vec<3, value> v1ai = {pos.x(), pos.y(), pos.z()};
    //vec<3, value> v2ai = {pos.x() * pos.z(), pos.y() * pos.z(), -pos.x() * pos.x() - pos.y() * pos.y()};
    vec<3, value> v3ai;

    for(int a=0; a < 3; a++)
    {
        value sum = 0;

        for(int d=0; d < 3; d++)
        {
            for(int c=0; c < 3; c++)
            {
                for(int b=0; b < 3; b++)
                {
                    sum += iYij.idx(a, d) * eijk_tensor.idx(d, b, c) * v1ai[b] * v2ai[c];
                }
            }
        }

        v3ai[a] = sum;
    }

    ///https://arxiv.org/pdf/gr-qc/0610128.pdf
    ///https://arxiv.org/pdf/gr-qc/0104063.pdf 5.7. I already have code for doing this but lets stay exact

    auto [v1a, v2a, v3a] = orthonormalise(ctx, v1ai, v2ai, v3ai, Yij);

    ctx.pin(v1a);
    ctx.pin(v2a);
    ctx.pin(v3a);

    /*{
        value r = pos.length();

        value theta = atan2(sqrt(pos.x() * pos.x() + pos.y() * pos.y()), pos.z());

        value phi = dual_types::dual_if(pos.x() >= 0,
                                        [&](){return atan2(pos.y(), pos.x());},
                                        [&](){return atan2(pos.y(), pos.x()) + M_PI;}
                                        );
    }*/

    //ctx.add("dbgw", v1a[0]);

    //vec<4, value> thetau = {0, v3a[0], v3a[1], v3a[2]};
    //vec<4, value> phiu = {0, v1a[0], v1a[1], v1a[2]};

    dual_types::complex<value> unit_i = dual_types::unit_i();

    tensor<dual_types::complex<value>, 4> mu;

    for(int i=1; i < 4; i++)
    {
        mu.idx(i) = dual_types::complex<value>(1.f/sqrt(2.f)) * (v1a[i - 1] + unit_i * v3a[i - 1]);

        //mu.idx(i) = dual_types::complex<value>(1.f/sqrt(2.f)) * (thetau[i] + unit_i * phiu[i]);
        ctx.pin(mu.idx(i).real);
        ctx.pin(mu.idx(i).imaginary);
    }

    ///https://en.wikipedia.org/wiki/Newman%E2%80%93Penrose_formalism
    tensor<dual_types::complex<value>, 4> mu_dash;

    for(int i=0; i < 4; i++)
    {
        mu_dash.idx(i) = dual_types::conjugate(mu.idx(i));
    }

    tensor<dual_types::complex<value>, 4> mu_dash_p = tensor_project_upper(mu_dash, args.gA, args.gB);

    tensor<value, 3, 3, 3> raised_eijk = raise_index_generic(raise_index_generic(eijk_tensor, iYij, 1), iYij, 2);

    ctx.pin(raised_eijk);

    dual_types::complex<value> w4;

    {
        dual_types::complex<value> sum(0.f);

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                value k_sum_1 = 0;

                dual_types::complex<value> k_sum_2(0.f);

                for(int k=0; k < 3; k++)
                {
                    for(int l=0; l < 3; l++)
                    {
                        k_sum_2 += unit_i * raised_eijk.idx(i, k, l) * cdKij.idx(l, j, k);
                    }

                    k_sum_1 += Kij.idx(i, k) * raise_index_generic(Kij, iYij, 0).idx(k, j);
                }

                dual_types::complex<value> inner_sum = -Rij.idx(i, j) - args.K * Kij.idx(i, j) + k_sum_1 + k_sum_2;

                ///mu is a 4 vector, but we use it spatially
                ///this exposes the fact that i really runs from 1-4 instead of 0-3
                sum += inner_sum * mu_dash_p.idx(i + 1) * mu_dash_p.idx(j + 1);
            }
        }

        w4 = sum;
    }

    //std::cout << "MUr " << type_to_string(mu.idx(1).real) << std::endl;
    //std::cout << "MUi " << type_to_string(mu.idx(1).imaginary) << std::endl;

    ctx.add("w4_real", w4.real);
    ctx.add("w4_complex", w4.imaginary);
    //ctx.add("w4_debugr", mu.idx(1).real);
    //ctx.add("w4_debugi", mu.idx(1).imaginary);

    //vec<4, dual_types::complex<value>> mu = (1.f/sqrt(2)) * (thetau + i * phiu);
}
#endif // 0


struct frame_basis
{
    vec<4, value> v1;
    vec<4, value> v2;
    vec<4, value> v3;
    vec<4, value> v4;
};

frame_basis calculate_frame_basis(equation_context& ctx, const metric<value, 4, 4>& met)
{
    tensor<value, 4> ti1 = {met.idx(0, 0), met.idx(0, 1), met.idx(0, 2), met.idx(0, 3)};
    tensor<value, 4> ti2 = {met.idx(0, 1), met.idx(1, 1), met.idx(1, 2), met.idx(1, 3)};
    tensor<value, 4> ti3 = {met.idx(0, 2), met.idx(1, 2), met.idx(2, 2), met.idx(2, 3)};
    tensor<value, 4> ti4 = {met.idx(0, 3), met.idx(1, 3), met.idx(2, 3), met.idx(3, 3)};

    auto metric_inverse = met.invert();

    ctx.pin(metric_inverse);

    ti1 = raise_index(ti1, met, metric_inverse);
    ti2 = raise_index(ti2, met, metric_inverse);
    ti3 = raise_index(ti3, met, metric_inverse);
    ti4 = raise_index(ti4, met, metric_inverse);

    ctx.pin(ti1);
    ctx.pin(ti2);
    ctx.pin(ti3);
    ctx.pin(ti4);

    vec<4, value> i1 = {ti1.idx(0), ti1.idx(1), ti1.idx(2), ti1.idx(3)};
    vec<4, value> i2 = {ti2.idx(0), ti2.idx(1), ti2.idx(2), ti2.idx(3)};
    vec<4, value> i3 = {ti3.idx(0), ti3.idx(1), ti3.idx(2), ti3.idx(3)};
    vec<4, value> i4 = {ti4.idx(0), ti4.idx(1), ti4.idx(2), ti4.idx(3)};

    vec<4, value> u1 = i1;

    vec<4, value> u2 = i2;
    u2 = u2 - gram_proj(u1, u2, met);

    ctx.pin(u2);

    vec<4, value> u3 = i3;
    u3 = u3 - gram_proj(u1, u3, met);
    u3 = u3 - gram_proj(u2, u3, met);

    ctx.pin(u3);

    vec<4, value> u4 = i4;
    u4 = u4 - gram_proj(u1, u4, met);
    u4 = u4 - gram_proj(u2, u4, met);
    u4 = u4 - gram_proj(u3, u4, met);

    ctx.pin(u4);

    u1 = u1.norm();
    u2 = u2.norm();
    u3 = u3.norm();
    u4 = u4.norm();

    ctx.pin(u1);
    ctx.pin(u2);
    ctx.pin(u3);
    ctx.pin(u4);

    u1 = normalize_big_metric(u1, met);
    u2 = normalize_big_metric(u2, met);
    u3 = normalize_big_metric(u3, met);
    u4 = normalize_big_metric(u4, met);

    frame_basis ret;
    ret.v1 = u1;
    ret.v2 = u2;
    ret.v3 = u3;
    ret.v4 = u4;

    return ret;
}

vec<3, value> rotate_vector(const vec<3, value>& bx, const vec<3, value>& by, const vec<3, value>& bz, const vec<3, value>& v)
{
    /*
    [nxx, nyx, nzx,   [vx]
     nxy, nyy, nzy,   [vy]
     nxz, nyz, nzz]   [vz] =

     nxx * vx + nxy * vy + nzx * vz
     nxy * vx + nyy * vy + nzy * vz
     nxz * vx + nzy * vy + nzz * vz*/

     return {
        bx.x() * v.x() + by.x() * v.y() + bz.x() * v.z(),
        bx.y() * v.x() + by.y() * v.y() + bz.y() * v.z(),
        bx.z() * v.x() + by.z() * v.y() + bz.z() * v.z()
    };
}

vec<3, value> unrotate_vector(const vec<3, value>& bx, const vec<3, value>& by, const vec<3, value>& bz, const vec<3, value>& v)
{
    /*
    nxx, nxy, nxz,   vx,
    nyx, nyy, nyz,   vy,
    nzx, nzy, nzz    vz*/

    return rotate_vector({bx.x(), by.x(), bz.x()}, {bx.y(), by.y(), bz.y()}, {bx.z(), by.z(), bz.z()}, v);
}

void process_geodesics(equation_context& ctx)
{
    ctx.order = 1;

    standard_arguments args(ctx);

    /*vec<3, value> camera;
    camera.x().make_value("camera_pos.x");
    camera.y().make_value("camera_pos.y");
    camera.z().make_value("camera_pos.z");*/

    vec<3, value> world_position;
    world_position.x().make_value("world_pos.x");
    world_position.y().make_value("world_pos.y");
    world_position.z().make_value("world_pos.z");

    quaternion_base<value> camera_quat;
    camera_quat.q.x().make_value("camera_quat.x");
    camera_quat.q.y().make_value("camera_quat.y");
    camera_quat.q.z().make_value("camera_quat.z");
    camera_quat.q.w().make_value("camera_quat.w");

    value width, height;
    width.make_value("width");
    height.make_value("height");

    value cx, cy;
    cx.make_value("x");
    cy.make_value("y");

    float FOV = 90;

    float fov_rad = (FOV / 360.f) * 2 * M_PI;

    value nonphysical_plane_half_width = width/2;
    value nonphysical_f_stop = nonphysical_plane_half_width / tan(fov_rad/2);

    vec<3, value> pixel_direction = {cx - width/2, cy - height/2, nonphysical_f_stop};

    pixel_direction = rot_quat(pixel_direction, camera_quat);

    ctx.pin(pixel_direction);

    metric<value, 4, 4> real_metric = calculate_real_metric(args.Yij, args.gA, args.gB);

    ctx.pin(real_metric);

    frame_basis basis = calculate_frame_basis(ctx, real_metric);

    vec<4, value> basis_x = basis.v3;
    vec<4, value> basis_y = basis.v4;
    vec<4, value> basis_z = basis.v2;

    ctx.pin(basis_x);
    ctx.pin(basis_y);
    ctx.pin(basis_z);

    pixel_direction = pixel_direction.norm();

    vec<3, value> basis3_x = {basis_x.y(), basis_x.z(), basis_x.w()};
    vec<3, value> basis3_y = {basis_y.y(), basis_y.z(), basis_y.w()};
    vec<3, value> basis3_z = {basis_z.y(), basis_z.z(), basis_z.w()};

    ctx.pin(basis3_x);
    ctx.pin(basis3_y);
    ctx.pin(basis3_z);

    pixel_direction = unrotate_vector(basis3_x.norm(), basis3_y.norm(), basis3_z.norm(),  pixel_direction);

    ctx.pin(pixel_direction);

    vec<4, value> pixel_x = pixel_direction.x() * basis_x;
    vec<4, value> pixel_y = pixel_direction.y() * basis_y;
    vec<4, value> pixel_z = pixel_direction.z() * basis_z;
    vec<4, value> pixel_t = -basis.v1;

    #define INVERT_TIME
    #ifdef INVERT_TIME
    pixel_t = -pixel_t;
    #endif // INVERT_TIME

    vec<4, value> lightray_velocity = pixel_x + pixel_y + pixel_z + pixel_t;
    vec<4, value> lightray_position = {0, world_position.x(), world_position.y(), world_position.z()};

    ctx.add("lv0_d", lightray_velocity.x());
    ctx.add("lv1_d", lightray_velocity.y());
    ctx.add("lv2_d", lightray_velocity.z());
    ctx.add("lv3_d", lightray_velocity.w());

    ctx.add("lp0_d", lightray_position.x());
    ctx.add("lp1_d", lightray_position.y());
    ctx.add("lp2_d", lightray_position.z());
    ctx.add("lp3_d", lightray_position.w());

    tensor<value, 4> lightray_velocity_t = {lightray_velocity.x(), lightray_velocity.y(), lightray_velocity.z(), lightray_velocity.w()};

    tensor<value, 4> velocity_lower = lower_index(lightray_velocity_t, real_metric);

    tensor<value, 3> adm_V_lower = {velocity_lower.idx(1), velocity_lower.idx(2), velocity_lower.idx(3)};

    tensor<value, 3> adm_V_higher = raise_index(adm_V_lower, args.Yij, args.Yij.invert());

    ctx.add("V0_d", adm_V_higher.idx(0));
    ctx.add("V1_d", adm_V_higher.idx(1));
    ctx.add("V2_d", adm_V_higher.idx(2));

    /*vec<4, value> loop_lightray_velocity = {"lv0", "lv1", "lv2", "lv3"};
    vec<4, value> loop_lightray_position = {"lp0", "lp1", "lp2", "lp3"};

    float step = 0.01;

    vec<4, value> ipos = {"(int)round(lpv0)", "(int)round(lpv1)", "(int)round(lpv2)", "(int)round(lpv3)"};

    float universe_length = (dim/2.f).max_elem();

    ctx.pin("universe_size", universe_length);*/
}

template<typename T, int N>
T dot(const tensor<T, N>& v1, const tensor<T, N>& v2)
{
    T ret = 0;

    for(int i=0; i < N; i++)
    {
        ret += v1.idx(i) * v2.idx(i);
    }

    return ret;
}

template<int N>
value dot_metric(const tensor<value, N>& v1_upper, const tensor<value, N>& v2_upper, const metric<value, N, N>& met)
{
    return dot(v1_upper, lower_index(v2_upper, met));
}

///https://arxiv.org/pdf/1208.3927.pdf (28a)
void loop_geodesics(equation_context& ctx, vec3f dim)
{
    ctx.order = 1;

    standard_arguments args(ctx);

    ctx.pin(args.Kij);
    ctx.pin(args.Yij);

    /*ctx.pin(args.gA);
    ctx.pin(args.gB);
    ctx.pin(args.cY);
    ctx.pin(args.X);
    ctx.pin(args.Yij);*/

    //ctx.pin(args.Yij);

    ///upper index, aka contravariant
    vec<4, value> loop_lightray_velocity = {"lv0", "lv1", "lv2", "lv3"};
    vec<4, value> loop_lightray_position = {"lp0", "lp1", "lp2", "lp3"};

    /*for(int i=0; i < 3; i++)
    {
        value v = diff1(ctx, args.gA, i);

        ctx.alias(v, args.digA.idx(i));
    }

    ///dcgB alias
    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value v = diff1(ctx, args.gB.idx(j), i);

            ctx.alias(v, args.digB.idx(i, j));
        }
    }

    ///dcYij alias
    for(int k=0; k < 3; k++)
    {
        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                value v = diff1(ctx, args.cY.idx(i, j), k);

                ctx.alias(v, args.dcYij.idx(k, i, j));
            }
        }
    }

    for(int i=0; i < 3; i++)
    {
        value v = diff1(ctx, args.X, i);

        ctx.alias(v, args.dX.idx(i));
    }*/

    tensor<value, 3, 3> digB;

    ///derivative
    for(int i=0; i < 3; i++)
    {
        ///index
        for(int j=0; j < 3; j++)
        {
            digB.idx(i, j) = diff1(ctx, args.gB.idx(j), i);
        }
    }

    tensor<value, 3> digA;

    for(int i=0; i < 3; i++)
    {
        digA.idx(i) = diff1(ctx, args.gA, i);
    }

    float universe_length = (dim/2.f).max_elem();

    value scale = "scale";

    ctx.add("universe_size", universe_length * scale);

    //tensor<value, 3> X_upper = {"lp1", "lp2", "lp3"};
    tensor<value, 3> V_upper = {"V0", "V1", "V2"};

    /*inverse_metric<value, 3, 3> iYij = args.Yij.invert();

    ctx.pin(iYij);*/

    inverse_metric<value, 3, 3> iYij = args.X * args.cY.invert();

    inverse_metric<value, 3, 3> icY = args.cY.invert();

    tensor<value, 3, 3, 3> conformal_christoff2 = gpu_christoffel_symbols_2(ctx, args.cY, icY);

    tensor<value, 3, 3, 3> full_christoffel2;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                float kronecker_ik = (i == k) ? 1 : 0;
                float kronecker_ij = (i == j) ? 1 : 0;

                value sm = 0;

                for(int m=0; m < 3; m++)
                {
                    sm += icY.idx(i, m) * diff1(ctx, args.X, m);
                }

                full_christoffel2.idx(i, j, k) = conformal_christoff2.idx(i, j, k) -
                                                 (1.f/(2.f * args.X)) * (kronecker_ik * diff1(ctx, args.X, j) + kronecker_ij * diff1(ctx, args.X, k) - args.cY.idx(j, k) * sm);
            }
        }
    }

    //tensor<value, 3, 3, 3> full_christoffel2 = gpu_christoffel_symbols_2(ctx, args.Yij, iYij);

    value length_sq = dot_metric(V_upper, V_upper, args.Yij);

    value length = sqrt(fabs(length_sq));

    V_upper = (V_upper * 1 / length);

    ///https://arxiv.org/pdf/1208.3927.pdf (28a)
    #define PAPER_1
    #ifdef PAPER_1
    tensor<value, 3> dx = args.gA * V_upper - args.gB;

    tensor<value, 3> V_upper_diff;

    for(int i=0; i < 3; i++)
    {
        V_upper_diff.idx(i) = 0;

        for(int j=0; j < 3; j++)
        {
            value kjvk = 0;

            for(int k=0; k < 3; k++)
            {
                kjvk += args.Kij.idx(j, k) * V_upper.idx(k);
            }

            value christoffel_sum = 0;

            for(int k=0; k < 3; k++)
            {
                christoffel_sum += full_christoffel2.idx(i, j, k) * V_upper.idx(k);
            }

            value dlog_gA = diff1(ctx, args.gA, j) / args.gA;

            V_upper_diff.idx(i) += args.gA * V_upper.idx(j) * (V_upper.idx(i) * (dlog_gA - kjvk) + 2 * raise_index(args.Kij, args.Yij, iYij).idx(i, j) - christoffel_sum)
                                   - iYij.idx(i, j) * diff1(ctx, args.gA, j) - V_upper.idx(j) * diff1(ctx, args.gB.idx(i), j);

        }
    }
    #endif // PAPER_1

    /*value length_sq = dot_metric(V_upper, V_upper, args.Yij);

    value length = sqrt(fabs(length_sq));

    V_upper_diff += (V_upper * 1 / length) - V_upper;*/

    ///https://authors.library.caltech.edu/88020/1/PhysRevD.49.4004.pdf
    //#define PAPER_2
    #ifdef PAPER_2
    tensor<value, 3> p_lower = lower_index(V_upper, args.Yij);

    value p0 = 0;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            p0 += iYij.idx(i, j) * p_lower.idx(i) * p_lower.idx(j);
        }
    }

    p0 = (1/args.gA) * sqrt(p0);

    tensor<value, 3> dx = {0, 0, 0};

    for(int j=0; j < 3; j++)
    {
        for(int i=0; i < 3; i++)
        {
            dx.idx(j) += iYij.idx(i, j) * p_lower.idx(i);
        }

        dx.idx(j) += -args.gB.idx(j) * p0;
    }

    tensor<value, 3> V_lower_diff = {0, 0, 0};

    for(int i=0; i < 3; i++)
    {
        value s1 = -args.gA * diff1(ctx, args.gA, i) * p0 * p0;

        value s2 = 0;

        for(int k=0; k < 3; k++)
        {
            s2 += diff1(ctx, args.gB.idx(k), i) * p_lower.idx(k) * p0;
        }

        value s3 = 0;

        for(int l=0; l < 3; l++)
        {
            for(int m=0; m < 3; m++)
            {
                s3 += -0.5f * diff1(ctx, iYij.idx(l, m), i) * p_lower.idx(l) * p_lower.idx(m);
            }
        }

        V_lower_diff.idx(i) = s1 + s2 + s3;
    }

    tensor<value, 3> V_upper_diff = raise_index(V_lower_diff, args.Yij, iYij);

    #endif // PAPER_2

    ctx.add("V0Diff", V_upper_diff.idx(0));
    ctx.add("V1Diff", V_upper_diff.idx(1));
    ctx.add("V2Diff", V_upper_diff.idx(2));

    ctx.add("X0Diff", dx.idx(0));
    ctx.add("X1Diff", dx.idx(1));
    ctx.add("X2Diff", dx.idx(2));

    /**
    [tt, tx, ty, tz,
    xt, xx, xy, xz,
    yt, yx, yy, yz,
    zt, zx, zy, zz,] //???
    */

    //metric<value, 4, 4> real_metric = calculate_real_metric(Yij, gA, gB);
    ///calculate metric partial derivatives, with a time pd of 0

    //tensor<value, 4> adm_normal = get_adm_hypersurface_normal(args.gA, args.gB);

    ///https://scholarworks.rit.edu/cgi/viewcontent.cgi?article=11286&context=theses 3.81
}

void build_hamiltonian_constraint(equation_context& ctx)
{
    standard_arguments args(ctx);

    ctx.add("HAMILTONIAN", calculate_hamiltonian(ctx, args));
}

/*float fisheye(float r)
{
    float a = 3;
    float r0 = 5.5f * 0.5f;
    float s = 1.2f * 0.5f;

    float other = 0;

    ///https://arxiv.org/pdf/gr-qc/0505055.pdf 5.5
    float R_r = (s / (2 * r * tanh(r0/s))) * log(cosh((r + r0)/s)/cosh((r - r0)/s));

    float r_phys = r * (a + (1 - a) * R_r);

    return r_phys;
}*/

cl::image_with_mipmaps load_mipped_image(const std::string& fname, opencl_context& clctx)
{
    sf::Image img;
    img.loadFromFile(fname);

    std::vector<uint8_t> as_uint8;

    for(int y=0; y < (int)img.getSize().y; y++)
    {
        for(int x=0; x < (int)img.getSize().x; x++)
        {
            auto col = img.getPixel(x, y);

            as_uint8.push_back(col.r);
            as_uint8.push_back(col.g);
            as_uint8.push_back(col.b);
            as_uint8.push_back(col.a);
        }
    }

    texture_settings bsett;
    bsett.width = img.getSize().x;
    bsett.height = img.getSize().y;
    bsett.is_srgb = false;

    texture opengl_tex;
    opengl_tex.load_from_memory(bsett, &as_uint8[0]);

    #define MIP_LEVELS 20

    int max_mips = floor(log2(std::min(img.getSize().x, img.getSize().y))) + 1;

    max_mips = std::min(max_mips, MIP_LEVELS);

    cl::image_with_mipmaps image_mipped(clctx.ctx);
    image_mipped.alloc((vec2i){img.getSize().x, img.getSize().y}, max_mips, {CL_RGBA, CL_FLOAT});

    int swidth = img.getSize().x;
    int sheight = img.getSize().y;

    for(int i=0; i < max_mips; i++)
    {
        printf("I is %i\n", i);

        int cwidth = swidth;
        int cheight = sheight;

        swidth /= 2;
        sheight /= 2;

        std::vector<vec4f> converted = opengl_tex.read(i);

        assert((int)converted.size() == (cwidth * cheight));

        image_mipped.write(clctx.cqueue, (char*)&converted[0], vec<2, size_t>{0, 0}, vec<2, size_t>{cwidth, cheight}, i);
    }

    return image_mipped;
}

struct lightray
{
    cl_float4 pos;
    cl_float4 vel;
    cl_int x, y;
};

///it seems like basically i need numerical dissipation of some form
///if i didn't evolve where sponge = 1, would be massively faster
int main()
{
    steady_timer time_to_main;

    test_harmonics();
    test_integration();

    int width = 800;
    int height = 600;

    render_settings sett;
    sett.width = width;
    sett.height = height;
    sett.opencl = true;
    sett.no_double_buffer = true;
    sett.is_srgb = true;

    render_window win(sett, "Geodesics");

    assert(win.clctx);

    ImFontAtlas* atlas = ImGui::GetIO().Fonts;
    atlas->FontBuilderFlags = ImGuiFreeTypeBuilderFlags_LCD | ImGuiFreeTypeBuilderFlags_FILTER_DEFAULT | ImGuiFreeTypeBuilderFlags_LoadColor;

    ImFontConfig font_cfg;
    font_cfg.GlyphExtraSpacing = ImVec2(0, 0);
    font_cfg.FontBuilderFlags = ImGuiFreeTypeBuilderFlags_LCD | ImGuiFreeTypeBuilderFlags_FILTER_DEFAULT | ImGuiFreeTypeBuilderFlags_LoadColor;

    ImGuiIO& io = ImGui::GetIO();

    io.Fonts->Clear();
    io.Fonts->AddFontFromFileTTF("VeraMono.ttf", 14, &font_cfg);

    opencl_context& clctx = *win.clctx;

    cl::managed_command_queue mqueue(clctx.ctx, 0, 16);

    std::cout << "EXT " << cl::get_extensions(clctx.ctx) << std::endl;

    std::string argument_string = "-I ./ -O3 -cl-std=CL2.0 -cl-uniform-work-group-size -cl-mad-enable -cl-finite-math-only -cl-denorms-are-zero ";

    std::string u_argument_string = argument_string;

    ///the simulation domain is this * 2
    int current_simulation_boundary = 1024;
    ///must be a multiple of DIFFERENTIATION_WIDTH
    vec3i size = {251, 251, 251};
    //vec3i size = {250, 250, 250};
    //float c_at_max = 160;
    float c_at_max = get_c_at_max();
    float scale = calculate_scale(c_at_max, size);
    vec3f centre = {size.x()/2.f, size.y()/2.f, size.z()/2.f};

    initial_conditions holes = setup_dynamic_initial_conditions(u_argument_string, clctx.ctx, clctx.cqueue, centre, scale);

    equation_context setup_static;

    setup_static_conditions(clctx.ctx, clctx.cqueue, setup_static, holes.holes);

    cl::buffer gpu_holes = holes.gpu_holes;

    cl::buffer u_arg(clctx.ctx);

    auto u_thread = [c_at_max, size, &clctx, &u_arg, &gpu_holes]()
    {
        cl::command_queue cqueue(clctx.ctx);

        u_arg = iterate_u(clctx.ctx, cqueue, size, c_at_max, gpu_holes, 0.000001f).as_read_only();

        cqueue.block();
    };

    std::thread async_u(u_thread);

    cl_sampler_properties sampler_props[] = {
    CL_SAMPLER_NORMALIZED_COORDS, CL_TRUE,
    CL_SAMPLER_ADDRESSING_MODE, CL_ADDRESS_REPEAT,
    CL_SAMPLER_FILTER_MODE, CL_FILTER_LINEAR,
    CL_SAMPLER_MIP_FILTER_MODE_KHR, CL_FILTER_LINEAR,
    0
    };

    cl_sampler sam = clCreateSamplerWithProperties(clctx.ctx.native_context.data, sampler_props, nullptr);

    cl::image_with_mipmaps background_mipped = load_mipped_image("background.png", clctx);

    vec<4, cl_int> clsize = {size.x(), size.y(), size.z(), 0};

    equation_context ctx1;
    get_initial_conditions_eqs(ctx1, centre, scale);

    equation_context dtcY;
    build_cY(dtcY);

    equation_context dtcA;
    build_cA(dtcA);

    equation_context dtcGi;
    build_cGi(dtcGi);

    equation_context dtK;
    build_K(dtK);

    equation_context dtX;
    build_X(dtX);

    equation_context dtgA;
    build_gA(dtgA);

    equation_context dtgB;
    build_gB(dtgB);

    equation_context ctx4;
    build_constraints(ctx4);

    equation_context ctx5;
    extract_waveforms(ctx5);

    equation_context ctx6;
    ctx6.uses_linear = true;
    process_geodesics(ctx6);

    equation_context ctx7;
    ctx7.uses_linear = true;
    loop_geodesics(ctx7, {size.x(), size.y(), size.z()});

    equation_context ctx10;
    build_kreiss_oliger_dissipate_singular(ctx10);

    equation_context ctx11;
    build_intermediate_thin(ctx11);

    equation_context ctx12;
    build_intermediate_thin_cY5(ctx12);

    equation_context ctx13;
    build_momentum_constraint(ctx13);

    equation_context ctx14;
    build_hamiltonian_constraint(ctx14);

    ctx1.build(argument_string, 0);
    ctx4.build(argument_string, 3);
    ctx5.build(argument_string, 4);
    ctx6.build(argument_string, 5);
    ctx7.build(argument_string, 6);
    setup_static.build(argument_string, 8);
    ctx10.build(argument_string, 9);
    ctx11.build(argument_string, 10);
    ctx12.build(argument_string, 11);
    ctx13.build(argument_string, 12);
    ctx14.build(argument_string, "unused1");

    dtcY.build(argument_string, "tcy");
    dtcA.build(argument_string, "tca");
    dtcGi.build(argument_string, "tcgi");
    dtK.build(argument_string, "tk");
    dtX.build(argument_string, "tx");
    dtgA.build(argument_string, "tga");
    dtgB.build(argument_string, "tgb");

    argument_string += "-DBORDER_WIDTH=" + std::to_string(BORDER_WIDTH) + " ";

    #ifdef USE_GBB
    argument_string += "-DUSE_GBB ";
    #endif // USE_GBB

    ///seems to make 0 difference to instability time
    #define USE_HALF_INTERMEDIATE
    #ifdef USE_HALF_INTERMEDIATE
    int intermediate_data_size = sizeof(cl_half);
    argument_string += "-DDERIV_PRECISION=half ";
    #else
    int intermediate_data_size = sizeof(cl_float);
    argument_string += "-DDERIV_PRECISION=float ";
    #endif

    {
        std::ofstream out("args.txt");
        out << argument_string;
    }

    std::cout << "Size " << argument_string.size() << std::endl;

    cl::program prog(clctx.ctx, "cl.cl");
    prog.build(clctx.ctx, argument_string);

    async_u.join();

    for(int i=0; i < (int)holes.holes.size(); i++)
    {
        printf("Black hole test mass %f %i\n", get_nonspinning_adm_mass(clctx.cqueue, i, holes.holes, size, scale, u_arg), i);
    }

    ///this is not thread safe
    clctx.ctx.register_program(prog);

    texture_settings tsett;
    tsett.width = width;
    tsett.height = height;
    tsett.is_srgb = false;
    tsett.generate_mipmaps = false;

    std::array<texture, 2> tex;
    tex[0].load_from_memory(tsett, nullptr);
    tex[1].load_from_memory(tsett, nullptr);

    std::array<cl::gl_rendertexture, 2> rtex{clctx.ctx, clctx.ctx};
    rtex[0].create_from_texture(tex[0].handle);
    rtex[1].create_from_texture(tex[1].handle);

    std::array<cl::buffer, 2> ray_buffer{clctx.ctx, clctx.ctx};
    ray_buffer[0].alloc(sizeof(cl_float) * 10 * width * height);
    ray_buffer[1].alloc(sizeof(cl_float) * 10 * width * height);

    cl::buffer rays_terminated(clctx.ctx);
    rays_terminated.alloc(sizeof(cl_float) * 10 * width * height);

    std::array<cl::buffer, 2> ray_count{clctx.ctx, clctx.ctx};
    ray_count[0].alloc(sizeof(cl_int));
    ray_count[1].alloc(sizeof(cl_int));

    cl::buffer ray_count_terminated(clctx.ctx);
    ray_count_terminated.alloc(sizeof(cl_int));

    cl::buffer texture_coordinates(clctx.ctx);
    texture_coordinates.alloc(sizeof(cl_float2) * width * height);

    cpu_mesh_settings base_settings;

    #ifdef USE_HALF_INTERMEDIATE
    base_settings.use_half_intermediates = true;
    #else
    base_settings.use_half_intermediates = false;
    #endif // USE_HALF_INTERMEDIATE

    #ifdef CALCULATE_MOMENTUM_CONSTRAINT
    base_settings.calculate_momentum_constraint = true;
    #endif // CALCULATE_MOMENTUM_CONSTRAINT

    cpu_mesh base_mesh(clctx.ctx, clctx.cqueue, {0,0,0}, size, base_settings);

    cl_float time_elapsed_s = 0;

    thin_intermediates_pool thin_pool;

    gravitational_wave_manager wave_manager(clctx.ctx, size, c_at_max, scale);

    base_mesh.init(clctx.cqueue, u_arg);

    std::vector<float> real_graph;
    std::vector<float> real_decomp;

    int which_texture = 0;
    int steps = 0;

    bool run = false;
    bool should_render = true;

    vec3f camera_pos = {0, 0, -c_at_max/2.f + 1};
    quat camera_quat;
    camera_quat.load_from_axis_angle({1, 0, 0, 0});

    std::optional<cl::event> last_event;

    int rendering_method = 1;

    bool trapezoidal_init = false;

    bool pao = false;

    bool render_skipping = false;
    int skip_frames = 4;
    int current_skip_frame = 0;

    clctx.cqueue.block();

    bool ever_stepped = false;

    std::cout << "Init time " << time_to_main.get_elapsed_time_s() << std::endl;

    std::vector<cl::buffer> last_valid_thin;
    std::vector<cl::buffer> last_valid_buffer;

    float rendering_err = 0.01f;

    while(!win.should_close())
    {
        steady_timer frametime;

        win.poll();

        if(!ImGui::GetIO().WantCaptureKeyboard)
        {
            float speed = 0.001;

            if(ImGui::IsKeyDown(GLFW_KEY_LEFT_SHIFT))
                speed = 0.1;

            if(ImGui::IsKeyDown(GLFW_KEY_LEFT_CONTROL))
                speed = 0.00001;

            if(ImGui::IsKeyDown(GLFW_KEY_LEFT_ALT))
                speed /= 1000;

            if(ImGui::IsKeyDown(GLFW_KEY_Z))
                speed *= 100;

            if(ImGui::IsKeyDown(GLFW_KEY_X))
                speed *= 100;

            if(ImGui::IsKeyPressed(GLFW_KEY_B))
            {
                camera_pos = {0, 0, -100};
            }

            if(ImGui::IsKeyPressed(GLFW_KEY_C))
            {
                camera_pos = {0, 0, 0};
            }

            if(ImGui::IsKeyDown(GLFW_KEY_RIGHT))
            {
                mat3f m = mat3f().ZRot(-M_PI/128);

                quat q;
                q.load_from_matrix(m);

                camera_quat = q * camera_quat;
            }

            if(ImGui::IsKeyDown(GLFW_KEY_LEFT))
            {
                mat3f m = mat3f().ZRot(M_PI/128);

                quat q;
                q.load_from_matrix(m);

                camera_quat = q * camera_quat;
            }

            vec3f up = {0, 0, -1};
            vec3f right = rot_quat({1, 0, 0}, camera_quat);
            vec3f forward_axis = rot_quat({0, 0, 1}, camera_quat);

            if(ImGui::IsKeyDown(GLFW_KEY_DOWN))
            {
                quat q;
                q.load_from_axis_angle({right.x(), right.y(), right.z(), M_PI/128});

                camera_quat = q * camera_quat;
            }

            if(ImGui::IsKeyDown(GLFW_KEY_UP))
            {
                quat q;
                q.load_from_axis_angle({right.x(), right.y(), right.z(), -M_PI/128});

                camera_quat = q * camera_quat;
            }

            vec3f offset = {0,0,0};

            offset += forward_axis * ((ImGui::IsKeyDown(GLFW_KEY_W) - ImGui::IsKeyDown(GLFW_KEY_S)) * speed);
            offset += right * (ImGui::IsKeyDown(GLFW_KEY_D) - ImGui::IsKeyDown(GLFW_KEY_A)) * speed;
            offset += up * (ImGui::IsKeyDown(GLFW_KEY_E) - ImGui::IsKeyDown(GLFW_KEY_Q)) * speed;

            /*camera.y() += offset.x();
            camera.z() += offset.y();
            camera.w() += offset.z();*/

            camera_pos += offset;
        }

        //std::cout << camera_quat.q << std::endl;
        //std::cout << "POS " << camera_pos << std::endl;

        auto buffer_size = rtex[which_texture].size<2>();

        if((vec2i){buffer_size.x(), buffer_size.y()} != win.get_window_size())
        {
            width = win.get_window_size().x();
            height = win.get_window_size().y();

            if(width < 4)
                width = 4;

            if(height < 4)
                height = 4;

            texture_settings new_sett;
            new_sett.width = width;
            new_sett.height = height;
            new_sett.is_srgb = false;
            new_sett.generate_mipmaps = false;

            std::cout << "DIMS " << width << " " << height << std::endl;

            tex[0].load_from_memory(new_sett, nullptr);
            tex[1].load_from_memory(new_sett, nullptr);

            rtex[0].create_from_texture(tex[0].handle);
            rtex[1].create_from_texture(tex[1].handle);

            ray_buffer[0].alloc(sizeof(cl_float) * 10 * width * height);
            ray_buffer[1].alloc(sizeof(cl_float) * 10 * width * height);
            rays_terminated.alloc(sizeof(cl_float) * 10 * width * height);

            texture_coordinates.alloc(sizeof(cl_float2) * width * height);
        }

        rtex[which_texture].acquire(clctx.cqueue);

        bool step = false;

            ImGui::Begin("Test Window", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

            if(ImGui::Button("Step"))
                step = true;

            ImGui::Checkbox("Run", &run);
            ImGui::Checkbox("Render", &should_render);

            ImGui::Text("Time: %f\n", time_elapsed_s);

            bool snap = ImGui::Button("Snapshot");

            ImGui::InputInt("Render Method", &rendering_method, 1);

            if(ImGui::IsKeyPressed(GLFW_KEY_1))
                rendering_method = 1;

            ImGui::InputFloat("Rendering Error", &rendering_err, 0.0001f, 0.01f, "%.5f");

            ImGui::Checkbox("Render Skipping", &render_skipping);

            if(real_graph.size() > 0)
            {
                ImGui::PushItemWidth(400);
                ImGui::PlotLines("w4      ", &real_graph[0], real_graph.size());
                ImGui::PopItemWidth();
            }

            ImGui::Checkbox("pao", &pao);

            if(real_decomp.size() > 0)
            {
                ImGui::PushItemWidth(400);
                ImGui::PlotLines("w4_l2_m0", &real_decomp[0], real_decomp.size());
                ImGui::PopItemWidth();
            }

            ImGui::End();

        if(run)
            step = true;

        ///rk4
        ///though no signs of any notable instability for backwards euler
        /*float timestep = 0.08;

        if(steps < 20)
           timestep = 0.016;

        if(steps < 10)
            timestep = 0.0016;*/

        ///todo: backwards euler test
        float timestep = 0.035;

        //timestep = 0.04;

        if(steps < 20)
           timestep = 0.001;

        if(steps < 10)
            timestep = 0.0001;

        if(pao && time_elapsed_s > 250)
            step = false;

        if(step || !ever_stepped)
        {
            ever_stepped = true;

            steps++;

            std::tie(last_valid_buffer, last_valid_thin) = base_mesh.full_step(clctx.ctx, clctx.cqueue, mqueue, timestep, thin_pool, u_arg);

            if(!should_render)
            {
                cl::args render;

                for(auto& i : last_valid_buffer)
                {
                    render.push_back(i.as_device_read_only());
                }

                for(auto& i : last_valid_thin)
                {
                    render.push_back(i.as_device_read_only());
                }

                //render.push_back(bssnok_datas[which_data]);
                render.push_back(scale);
                render.push_back(clsize);
                render.push_back(rtex[which_texture]);

                clctx.cqueue.exec("render", render, {size.x(), size.y()}, {16, 16});
            }

            {
                wave_manager.issue_extraction(clctx.cqueue, last_valid_buffer, last_valid_thin, scale, clsize, rtex[which_texture]);

                std::vector<float> values = wave_manager.process();

                for(float v : values)
                {
                    if(!isnanf(v))
                        real_decomp.push_back(v);
                }
            }

            time_elapsed_s += timestep;
            current_simulation_boundary += DIFFERENTIATION_WIDTH;

            current_simulation_boundary = clamp(current_simulation_boundary, 0, size.x()/2);
        }

        if(should_render || snap)
        {
            if(rendering_method == 0)
            {
                cl::args render_args;

                auto buffers = last_valid_buffer;

                for(auto& i : buffers)
                {
                    render_args.push_back(i.as_device_read_only());
                }

                cl_float3 ccamera_pos = {camera_pos.x(), camera_pos.y(), camera_pos.z()};
                cl_float4 ccamera_quat = {camera_quat.q.x(), camera_quat.q.y(), camera_quat.q.z(), camera_quat.q.w()};

                render_args.push_back(scale);
                render_args.push_back(ccamera_pos);
                render_args.push_back(ccamera_quat);
                render_args.push_back(clsize);
                render_args.push_back(rtex[which_texture]);

                clctx.cqueue.exec("trace_metric", render_args, {width, height}, {16, 16});
            }

            bool not_skipped = render_skipping ? ((current_skip_frame % skip_frames) == 0) : true;

            not_skipped = not_skipped || snap;

            current_skip_frame = (current_skip_frame + 1) % skip_frames;

            if(rendering_method == 1 && not_skipped)
            {
                cl_float3 ccamera_pos = {camera_pos.x(), camera_pos.y(), camera_pos.z()};
                cl_float4 ccamera_quat = {camera_quat.q.x(), camera_quat.q.y(), camera_quat.q.z(), camera_quat.q.w()};

                ray_count_terminated.set_to_zero(clctx.cqueue);

                {
                    cl::args init_args;

                    init_args.push_back(ray_buffer[0]);
                    init_args.push_back(ray_count[0]);

                    for(auto& i : last_valid_buffer)
                    {
                        init_args.push_back(i.as_device_read_only());
                    }

                    for(auto& i : last_valid_thin)
                    {
                        init_args.push_back(i.as_device_read_only());
                    }

                    init_args.push_back(scale);
                    init_args.push_back(ccamera_pos);
                    init_args.push_back(ccamera_quat);
                    init_args.push_back(clsize);
                    init_args.push_back(width);
                    init_args.push_back(height);

                    clctx.cqueue.exec("init_rays", init_args, {width, height}, {8, 8});
                }

                {
                    cl::args render_args;

                    render_args.push_back(ray_buffer[0]);
                    render_args.push_back(rays_terminated);

                    for(auto& i : last_valid_buffer)
                    {
                        render_args.push_back(i.as_device_read_only());
                    }

                    for(auto& i : last_valid_thin)
                    {
                        render_args.push_back(i.as_device_read_only());
                    }

                    render_args.push_back(scale);
                    render_args.push_back(clsize);
                    render_args.push_back(width);
                    render_args.push_back(height);
                    render_args.push_back(rendering_err);

                    clctx.cqueue.exec("trace_rays", render_args, {width, height}, {8, 8});
                }

                {
                    cl::args texture_args;
                    texture_args.push_back(rays_terminated.as_device_read_only());
                    texture_args.push_back(texture_coordinates);
                    texture_args.push_back(width);
                    texture_args.push_back(height);
                    texture_args.push_back(ccamera_pos);
                    texture_args.push_back(ccamera_quat);

                    for(auto& i : last_valid_buffer)
                    {
                        texture_args.push_back(i.as_device_read_only());
                    }

                    for(auto& i : last_valid_thin)
                    {
                        texture_args.push_back(i.as_device_read_only());
                    }

                    texture_args.push_back(scale);
                    texture_args.push_back(clsize);

                    clctx.cqueue.exec("calculate_adm_texture_coordinates", texture_args, {width, height}, {8, 8});
                }

                {
                    cl::args render_args;
                    render_args.push_back(rays_terminated.as_device_read_only());
                    render_args.push_back(ray_count_terminated.as_device_read_only());
                    render_args.push_back(rtex[which_texture]);

                    for(auto& i : last_valid_buffer)
                    {
                        render_args.push_back(i.as_device_read_only());
                    }

                    for(auto& i : last_valid_thin)
                    {
                        render_args.push_back(i.as_device_read_only());
                    }

                    render_args.push_back(scale);
                    render_args.push_back(clsize);
                    render_args.push_back(width);
                    render_args.push_back(height);
                    render_args.push_back(background_mipped);
                    render_args.push_back(texture_coordinates);
                    render_args.push_back(sam);

                    clctx.cqueue.exec("render_rays", render_args, {width * height}, {128});
                }
            }
        }

        if(rendering_method == 2 && snap)
        {
            cl_float3 ccamera_pos = {camera_pos.x(), camera_pos.y(), camera_pos.z()};
            cl_float4 ccamera_quat = {camera_quat.q.x(), camera_quat.q.y(), camera_quat.q.z(), camera_quat.q.w()};

            cl::args init_args;

            auto buffers = base_mesh.get_input().buffers;

            for(auto& i : buffers)
            {
                init_args.push_back(i);
            }

            init_args.push_back(scale);
            init_args.push_back(ccamera_pos);
            init_args.push_back(ccamera_quat);
            init_args.push_back(clsize);
            init_args.push_back(rtex[which_texture]);
            init_args.push_back(ray_buffer);

            clctx.cqueue.exec("init_accurate_rays", init_args, {width, height}, {8, 8});

            printf("Init\n");
        }

        if(rendering_method == 2 && step)
        {
            cl_float3 ccamera_pos = {camera_pos.x(), camera_pos.y(), camera_pos.z()};
            cl_float4 ccamera_quat = {camera_quat.q.x(), camera_quat.q.y(), camera_quat.q.z(), camera_quat.q.w()};

            std::cout << camera_pos << std::endl;

            cl::args step_args;

            auto buffers = base_mesh.get_input().buffers;

            for(auto& i : buffers)
            {
                step_args.push_back(i);
            }

            step_args.push_back(scale);
            step_args.push_back(ccamera_pos);
            step_args.push_back(ccamera_quat);
            step_args.push_back(clsize);
            step_args.push_back(rtex[which_texture]);
            step_args.push_back(ray_buffer);
            step_args.push_back(timestep);

            clctx.cqueue.exec("step_accurate_rays", step_args, {width * height}, {128});
        }

        cl::event next_event = rtex[which_texture].unacquire(clctx.cqueue);

        if(last_event.has_value())
            last_event.value().block();

        last_event = next_event;

        {
            ImDrawList* lst = ImGui::GetBackgroundDrawList();

            ImVec2 screen_pos = ImGui::GetMainViewport()->Pos;

            ImVec2 tl = {0,0};
            ImVec2 br = {rtex[which_texture].size<2>().x(), rtex[which_texture].size<2>().y()};

            if(win.get_render_settings().viewports)
            {
                tl.x += screen_pos.x;
                tl.y += screen_pos.y;

                br.x += screen_pos.x;
                br.y += screen_pos.y;
            }

            lst->AddImage((void*)rtex[which_texture].texture_id, tl, br, ImVec2(0, 0), ImVec2(1.f, 1.f));
        }

        win.display();

        printf("Time: %f\n", frametime.restart() * 1000.);
    }
}
