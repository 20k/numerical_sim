#include <iostream>
#include <toolkit/render_window.hpp>
#include <toolkit/texture.hpp>
#include <vec/vec.hpp>
#include <GLFW/glfw3.h>
#include <SFML/Graphics.hpp>
#include <CL/cl_ext.h>
#include <geodesic/dual.hpp>
#include <geodesic/dual_value.hpp>
#include <fstream>
#include <imgui/misc/freetype/imgui_freetype.h>
#include <vec/tensor.hpp>
#include "gravitational_waves.hpp"
#include <execution>
#include <thread>
#include "mesh_manager.hpp"
#include "spherical_harmonics.hpp"
#include "spherical_integration.hpp"
#include "equation_context.hpp"
#include "laplace_solver.hpp"
#include "tensor_algebra.hpp"
#include "bssn.hpp"
#include <toolkit/fs_helpers.hpp>
#include "hydrodynamics.hpp"
#include "particle_dynamics.hpp"

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
https://air.unipr.it/retrieve/handle/11381/2783927/20540/0912.2920.pdf - relativistic hydrodynamics
https://arxiv.org/pdf/gr-qc/9910044.pdf - the volume smeary integral thing for w4
https://orca.cardiff.ac.uk/id/eprint/114952/1/PhysRevD.98.044014.pdf - trumpet data, claims reduced junk
https://arxiv.org/pdf/0912.1285.pdf - cauchy gravitational waves

///neutron stars
https://arxiv.org/pdf/gr-qc/0209102.pdf - basic paper
https://www.db-thueringen.de/servlets/MCRFileNodeServlet/dbt_derivate_00024818/thierfelder/diss_v3.1_pdfa.pdf - inversion
https://www.researchgate.net/publication/50818590_Relativistic_simulations_of_rotational_core_collapse_I_Methods_initial_models_and_code_tests
https://arxiv.org/pdf/gr-qc/9811015.pdf - hydrodynamics, 1998
https://arxiv.org/pdf/gr-qc/0403029.pdf - collapse to black hole
https://air.unipr.it/retrieve/handle/11381/2783927/20540/0912.2920.pdf - contains a definition for E
https://arxiv.org/pdf/gr-qc/0003101.pdf - fat hydrodynamic paper
https://arxiv.org/pdf/1606.04881.pdf - this provides mixed neutron star black hole initial conditions
https://www.hindawi.com/journals/aa/2017/6127031/ - neutron star initial conditions
https://www.aanda.org/articles/aa/pdf/2010/06/aa12738-09.pdf - this gives exact formulas for pressure and density
https://gwic.ligo.org/assets/docs/theses/Read_Thesis.pdf - 3.4 definition of e. This also contains some interesting post newtonian expansions
https://arxiv.org/pdf/gr-qc/0403029.pdf - 2.13 definition of e. I don't think they're the same as the other e, but I think this is fixable
https://arxiv.org/pdf/2101.10252.pdf - another source which uses this bowen-york data with different notation (yay!)
https://arxiv.org/pdf/gr-qc/9908027.pdf - hydrodynamic paper off which the one I'm implementing is based
*/

///notes:
///off centre black hole results in distortion, probably due to boundary conditions contaminating things
///this is odd. Maybe don't boundary condition shift and lapse?

//#define USE_GBB

//#define SYMMETRY_BOUNDARY
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

        #define DETECT_INCORRECT_DIFFERENTIATION
        #ifdef DETECT_INCORRECT_DIFFERENTIATION
        std::vector<std::string> variables = in.get_all_variables();

        for(auto& v : variables)
        {
            if(v == "dim" || v == "scale" || v == "ix" || v == "iy" || v == "iz" || v == "fx" || v == "fy" || v == "fz")
                continue;

            bool found = false;

            for(auto& o : indexed_variables)
            {
                if(v == type_to_string(o.args[1]))
                {
                    found = true;
                    break;
                }
            }

            if(!found)
            {
                std::cout << "Could not find " << v << std::endl;

                assert(false);
            }
        }
        #endif // DETECT_INCORRECT_DIFFERENTIATION

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

///https://hal.archives-ouvertes.fr/hal-00569776/document this paper implies you simply sum the directions
///dissipation is fixing some stuff, todo: investigate why so much dissipation is required
value kreiss_oliger_dissipate_dir(equation_context& ctx, const value& in, int idx)
{
    ///https://en.wikipedia.org/wiki/Finite_difference_coefficient according to wikipedia, this is the 6th derivative with 2nd order accuracy. I am confused, but at least I know where it came from
    value scale = "scale";

    //#define FOURTH
    #ifdef FOURTH
    differentiation_context<5> dctx(in, idx);
    value stencil = -(1 / (16.f * scale)) * (dctx.vars[0] - 4 * dctx.vars[1] + 6 * dctx.vars[2] - 4 * dctx.vars[3] + dctx.vars[4]);
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

value diff1_interior(equation_context& ctx, const value& in, int idx, int order, int direction)
{
    value scale = "scale";

    if(direction != 0)
        assert(order == 1);

    if(order == 1)
    {
        #if 0
        differentiation_context<5> dctx(in, idx, ctx.uses_linear);
        std::array<value, 5> vars = dctx.vars;

        if(direction == 0)
            return (vars[3] - vars[1]) / (2 * scale);

        /*if(direction == 1)
            return (vars[2] - vars[1]) / scale;

        if(direction == -1)
            return (vars[1] - vars[0]) / scale;*/

        if(direction == 1)
            return (-vars[4] + 4 * vars[3] - 3 * vars[2]) / (2 * scale);

        if(direction == -1)
            return (vars[0] - 4 * vars[1] + 3 * vars[2]) / (2 * scale);
        #endif // 0

        differentiation_context<5> dctx(in, idx, ctx.uses_linear);
        std::array<value, 5> vars = dctx.vars;

        if(direction == 0)
            return (vars[3] - vars[1]) / (2 * scale);

        if(direction == 1)
            return (vars[3] - vars[2]) / scale;

        if(direction == -1)
            return -(vars[1] - vars[2]) / scale;
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

value diff1(equation_context& ctx, const value& in, int idx)
{
    //ctx.use_precise_differentiation = false;

    assert(!ctx.is_derivative_free);

    if(!ctx.use_precise_differentiation)
    {
        assert(!ctx.always_directional_derivatives);

        return diff1_interior(ctx, in, idx, ctx.order, 0);
    }
    else
    {
        value d_low = "D_LOW";
        value d_full = "D_FULL";
        value d_only_px = "D_ONLY_PX";
        value d_only_py = "D_ONLY_PY";
        value d_only_pz = "D_ONLY_PZ";
        value d_both_px = "D_BOTH_PX";
        value d_both_py = "D_BOTH_PY";
        value d_both_pz = "D_BOTH_PZ";

        value directional_single = 0;
        value directional_both = 0;

        if(idx == 0)
        {
            directional_single = d_only_px;
            directional_both = d_both_px;
        }
        else if(idx == 1)
        {
            directional_single = d_only_py;
            directional_both = d_both_py;
        }
        else if(idx == 2)
        {
            directional_single = d_only_pz;
            directional_both = d_both_pz;
        }

        value order = "order";

        value is_high_order = (order & d_full) > 0;
        //value is_low_order = (order & d_low) > 0;

        value is_forward = (order & directional_single) > 0;
        value is_bidi = (order & directional_both) > 0;

        value regular_d = diff1_interior(ctx, in, idx, ctx.order, 0);
        value low_d = diff1_interior(ctx, in, idx, 1, 0);

        value forward_d = diff1_interior(ctx, in, idx, 1, 1);
        value back_d = diff1_interior(ctx, in, idx, 1, -1);

        value selected_directional = dual_types::if_v(is_forward, forward_d, back_d);

        value selected_full = dual_types::if_v(is_bidi, low_d, selected_directional);

        if(ctx.always_directional_derivatives)
            return selected_full;
        else
            return dual_types::if_v(is_high_order, regular_d, low_d);
    }
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

template<typename T>
inline
T chi_to_e_6phi(const T& chi)
{
    return pow(1/(max(chi, 0.001f)), (3.f/2.f));
}

template<typename T>
inline
T chi_to_e_m6phi(const T& chi)
{
    return pow(max(chi, 0.001f), (3.f/2.f));
}

template<typename T>
inline
T chi_to_e_6phi_unclamped(const T& chi)
{
    return pow(1/(max(chi, T{0.f})), (3.f/2.f));
}

template<typename T>
inline
T chi_to_e_m6phi_unclamped(const T& chi)
{
    return pow(max(chi, T{0.f}), (3.f/2.f));
}

#define DIVISION_TOL 0.00001f

///https://arxiv.org/pdf/gr-qc/0209102.pdf (29)
///constant_1 is chi * icYij * cSi cSj
template<typename T>
inline
T w_next_interior(const T& w_in, const T& p_star, const T& chi, const T& constant_1, float gamma, const T& e_star)
{
    ///p*^(2-G) * (w e6phi)^G-1 is equivalent to the divisor

    T em6_phi_G = pow(chi_to_e_m6phi_unclamped(chi), gamma - 1);

    T geg = gamma * pow(e_star, gamma);

    T divisor = pow(w_in, gamma - 1);

    ///so: Limits
    ///when w -> 0, w = 0
    ///when p_star -> 0, w = 0
    ///when e_star -> 0.... I think pstar and w have to tend to 0?

    ///So! I think this equation has a regular fomulation. The non finite quantities are em6_phi_G which tends to 0, geg which can be zero
    ///pstar doesn't actually matter here, though theoretically it might
    //T non_regular_interior = divisor / (divisor + em6_phi_G * geg * pow(p_star, gamma - 2));

    ///I'm not sure this equation tends to 0, but constant_1 tends to 0 because Si = p* h uk
    T non_regular_interior = dual_types::divide_with_limit(divisor, divisor + em6_phi_G * geg * pow(p_star, gamma - 2), T{0.f}, DIVISION_TOL);

    return sqrt(p_star * p_star + constant_1 * pow(non_regular_interior, 2));
    //return sqrt(p_star * p_star + constant_1 * pow(1 + em6_phi_G * geg * pow(p_star, gamma - 2) / divisor, -2));
}

float w_next_interior_nonregular(float w_in, float p_star, float chi, float constant_1, float gamma, float e_star)
{
    float geg = gamma * pow(e_star, gamma);

    float pstarwe6phipstar = p_star * pow(w_in * chi_to_e_6phi_unclamped(chi)/p_star, gamma - 1);

    return sqrt(p_star * p_star + constant_1 * pow(1 + geg / pstarwe6phipstar, -2));
}

template<typename T>
inline
T w_next(const T& w_in, const T& p_star, const T& chi, const inverse_metric<T, 3, 3>& icY, const tensor<T, 3>& cS, float gamma, const T& e_star)
{
    T constant_1 = 0;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            constant_1 += chi * icY.idx(i, j) * cS.idx(i) * cS.idx(j);
        }
    }

    return w_next_interior(w_in, p_star, chi, constant_1, gamma, e_star);
}

tensor<value, 3> calculate_dPhi(const value& chi, const tensor<value, 3>& dchi)
{
    return -dchi/(4 * max(chi, 0.001f));
}

#if 0
///https://arxiv.org/pdf/1606.04881.pdf (72)
///this relates rest mass density to pressure
template<typename T>
inline
T eos_polytropic(const T& rest_mass_density) ///aka p0
{
    float Gamma = 2;
    float K = 123.641f;

    return K * pow(rest_mass_density, Gamma);
}
#endif // 0

namespace neutron_star
{
    template<typename T>
    struct params
    {
        T mass = 0; ///... bare mass? adm mass? rest mass?
        T compactness = 0;
        tensor<T, 3> position;
        tensor<T, 3> linear_momentum;
        tensor<T, 3> angular_momentum;

        T get_radius() const
        {
            if constexpr(std::is_same_v<T, float>)
                assert(compactness != 0);

            return mass / compactness;
        }
    };

    template<typename T>
    struct data
    {
        T Gamma = 2;
        T pressure = 0;
        T rest_mass_density = 0; ///p0
        T mass_energy_density = 0; /// = rho, looks like p
        T k = 0;

        T compactness = 0;
        T radius =  0; ///neutron star parameter
        T mass = 0;

        ///this is not littlee. This is the convention that h = (e + p)/p0
        T energy_density = 0;
        ///this is littlee. This is the convention that h = 1 + e + p/p0
        T specific_energy_density = 0;
    };

    struct conformal_data
    {
        value pressure;
        value rest_mass_density;
        value mass_energy_density;
    };

    /*inline
    float compactness()
    {
        return 0.06f;
    }*/

    template<typename T>
    inline
    T mass_to_radius(const T& mass, const T& compactness)
    {
        T radius = mass / compactness;

        return radius;
    }

    ///https://www.aanda.org/articles/aa/pdf/2010/06/aa12738-09.pdf
    ///notes on equations of state https://arxiv.org/pdf/gr-qc/9911047.pdf
    ///this is only valid for coordinate radius < radius
    ///todo: make a general sampler
    template<typename T>
    inline
    data<T> sample_interior(const T& coordinate_radius, const T& mass, const T& compactness)
    {
        if constexpr(std::is_same_v<T, float>)
            assert(compactness != 0);

        T radius = mass_to_radius(mass, compactness);

        T xi = M_PI * coordinate_radius / radius;

        ///oh thank god
        T pc = mass / ((4 / M_PI) * pow(radius, 3.f));

        float xi_boundary = 0.0001f;
        float value_at_boundary = sin(xi_boundary) / xi_boundary;

        T xi_fraction = xi / xi_boundary;

        auto lmix = [](const T& v1, const T& v2, const T& a)
        {
            return (1-a) * v1 + a * v2;
        };

        ///piecewise sin(x) / x, enforce sin(x)/x = 1 at x = 0
        T sin_fraction = dual_types::if_v(xi <= xi_boundary,
                                          lmix(value_at_boundary, 1.f, 1.f - xi_fraction),
                                          sin(xi) / xi);

        T p_xi = pc * sin_fraction;

        T k = (2 / M_PI) * radius * radius;

        T pressure = k * p_xi * p_xi;

        data<T> ret;
        ret.Gamma = 2;
        ret.pressure = pressure;
        ret.rest_mass_density = p_xi;
        ret.k = k;

        ret.compactness = compactness;
        ret.radius = radius;
        ret.mass = mass;

        ///https://arxiv.org/pdf/gr-qc/0403029.pdf (2.13)

        ret.energy_density = ret.rest_mass_density + (ret.pressure / (ret.Gamma - 1));


        ///ok so. https://gwic.ligo.org/assets/docs/theses/Read_Thesis.pdf 1.5 defines h as (little1 + pressure) / rest_density
        ///https://arxiv.org/pdf/1606.04881.pdf defines h as 1 + little2 + pressure / rest_density

        ///so 1 + little2 + pressure / rest_density = (little1 + pressure) / rest_density

        ///little2 = (little1 / rest_density) - 1

        ret.specific_energy_density = (ret.energy_density / ret.rest_mass_density) - 1;

        ///https://arxiv.org/pdf/1606.04881.pdf (before 6)
        ret.mass_energy_density = ret.rest_mass_density * (1 + ret.specific_energy_density);


        ret.pressure = dual_types::if_v(coordinate_radius >= radius, 0.f, ret.pressure);
        ret.rest_mass_density = dual_types::if_v(coordinate_radius >= radius, 0.f, ret.rest_mass_density);
        ret.energy_density = dual_types::if_v(coordinate_radius >= radius, 0.f, ret.energy_density);
        ret.specific_energy_density = dual_types::if_v(coordinate_radius >= radius, 0.f, ret.specific_energy_density);
        ret.mass_energy_density = dual_types::if_v(coordinate_radius >= radius, 0.f, ret.mass_energy_density);

        return ret;
    }

    ///remember tov_phi_at_coordinate samples in WORLD coordinates
    template<typename T, typename U>
    inline
    conformal_data sample_conformal(const value& coordinate_radius, const params<T>& p, U&& tov_phi_at_coordinate)
    {
        ///samples phi at pos + z * radius. Assumes that phi is symmetric
        tensor<value, 3> direction = {0, 0, 1};
        tensor<value, 3> vposition = {p.position.x(), p.position.y(), p.position.z()};
        tensor<value, 3> phi_position = direction * coordinate_radius + vposition;

        data<value> non_conformal = sample_interior(coordinate_radius, value{p.mass}, value{p.compactness});

        value phi = tov_phi_at_coordinate(phi_position);

        conformal_data cret;
        cret.pressure = pow(phi, 8) * non_conformal.pressure;
        cret.mass_energy_density = pow(phi, 8) * non_conformal.mass_energy_density;
        cret.rest_mass_density = pow(phi, 8) * non_conformal.rest_mass_density;

        return cret;
    }

    ///https://arxiv.org/pdf/1606.04881.pdf (59)
    template<typename T, typename U>
    inline
    value calculate_M_factor(const params<T>& p, U&& tov_phi_at_coordinate)
    {
        T radius = p.get_radius();

        auto integration_func = [&](const T& coordinate_radius)
        {
            conformal_data ndata = sample_conformal(coordinate_radius, p, tov_phi_at_coordinate);

            return (ndata.mass_energy_density + ndata.pressure) * pow(coordinate_radius, 2);
        };

        return 4 * M_PI * integrate_1d(integration_func, 32, radius, T{0.f});
    }

    ///squiggly N
    ///https://arxiv.org/pdf/1606.04881.pdf (64)
    template<typename T, typename U>
    inline
    value calculate_squiggly_N_factor(const params<T>& p, U&& tov_phi_at_coordinate)
    {
        T radius = p.get_radius();

        auto integration_func = [&](const T& coordinate_radius)
        {
            conformal_data ndata = sample_conformal(coordinate_radius, p, tov_phi_at_coordinate);

            return (ndata.mass_energy_density + ndata.pressure) * pow(coordinate_radius, 4.f);
        };

        return (8 * M_PI/3) * integrate_1d(integration_func, 32, radius, T{0.f});
    }

    ///https://arxiv.org/pdf/1606.04881.pdf (57)
    template<typename T, typename U>
    inline
    value calculate_sigma(const value& coordinate_radius, const params<T>& p, const value& M_factor, U&& tov_phi_at_coordinate)
    {
        conformal_data cdata = sample_conformal(coordinate_radius, p, tov_phi_at_coordinate);

        return (cdata.mass_energy_density + cdata.pressure) / M_factor;
    }

    template<typename T, typename U>
    inline
    value calculate_kappa(const value& coordinate_radius, const params<T>& p, const value& squiggly_N_factor, U&& tov_phi_at_coordinate)
    {
        conformal_data cdata = sample_conformal(coordinate_radius, p, tov_phi_at_coordinate);

        return (cdata.mass_energy_density + cdata.pressure) / squiggly_N_factor;
    }

    ///https://arxiv.org/pdf/1606.04881.pdf (43)
    template<typename T, typename U>
    inline
    value calculate_integral_Q(equation_context& ctx, const value& coordinate_radius, const params<T>& p, const value& M_factor, U&& tov_phi_at_coordinate)
    {
        ///currently impossible, but might as well
        //if(p.get_radius() == 0)
        //    return 1;

        auto integral_func = [&ctx, p, &M_factor, tov_phi_at_coordinate](const value& rp)
        {
            return 4 * M_PI * calculate_sigma(rp, p, M_factor, tov_phi_at_coordinate) * pow(rp, 2.f);
        };

        value integrated = integrate_1d(integral_func, 16, coordinate_radius, value{0.f});

        return dual_types::if_v(coordinate_radius > value{p.get_radius()},
                                1.f,
                                integrated);
    }

    ///https://arxiv.org/pdf/1606.04881.pdf (45)
    template<typename T, typename U>
    inline
    value calculate_integral_C(equation_context& ctx, const value& coordinate_radius, const params<T>& p, const value& M_factor, U&& tov_phi_at_coordinate)
    {
        //if(p.get_radius() == 0)
        //    return 0;

        auto integral_func = [p, &M_factor, tov_phi_at_coordinate](const value& rp)
        {
            return (2.f/3.f) * M_PI * calculate_sigma(rp, p, M_factor, tov_phi_at_coordinate) * pow(rp, 4.f);
        };

        value integrated = integrate_1d(integral_func, 16, coordinate_radius, value{0.f});

        value full_integrated = integrate_1d(integral_func, 16, p.get_radius(), T{0.f});

        ///for precision reasons. I don't think integrates to a known constant, but sigma  is 0 > radius
        return dual_types::if_v(coordinate_radius > value{p.get_radius()},
                                full_integrated,
                                integrated);
    }

    template<typename T, typename U>
    inline
    value calculate_integral_unsquiggly_N(equation_context& ctx, const value& coordinate_radius, const params<T>& p, const value& squiggly_N_factor, U&& tov_phi_at_coordinate)
    {
        ///if radius == 0, return 1

        auto integral_func = [p, &squiggly_N_factor, tov_phi_at_coordinate](const value& rp)
        {
            return (8.f/3.f) * M_PI * calculate_kappa(rp, p, squiggly_N_factor, tov_phi_at_coordinate) * pow(rp, 4.f);
        };

        value integrated = integrate_1d(integral_func, 16, coordinate_radius, value{0.f});

        value full_integrated = integrate_1d(integral_func, 16, p.get_radius(), T{0.f});

        ///N tends to 1 exterior to the source
        return dual_types::if_v(coordinate_radius > value{p.get_radius()},
                                full_integrated,
                                integrated);
    }

    ///https://arxiv.org/pdf/1606.04881.pdf (60)
    template<typename T>
    value calculate_W2_linear_momentum(const metric<T, 3, 3>& flat, const tensor<T, 3>& linear_momentum, const value& M_factor)
    {
        T p2 = 0;

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                p2 += flat.idx(i, j) * linear_momentum.idx(i) * linear_momentum.idx(j);
            }
        }

        value M2 = M_factor * M_factor;

        return 0.5f * (1 + sqrt(1 + (4 * p2 / M2)));
    }

    template<typename T>
    value calculate_W2_angular_momentum(const tensor<value, 3>& coordinate, const tensor<T, 3>& ns_pos, const metric<T, 3, 3>& flat, const tensor<T, 3>& angular_momentum, const value& squiggly_N_factor)
    {
        tensor<value, 3> relative_pos = coordinate - ns_pos.template as<value>();

        value r = relative_pos.length();

        r = max(r, value{1e-3});

        ///angular momentum is J, with upper index. Also called S in other papers
        ///https://arxiv.org/pdf/1606.04881.pdf (65)
        tensor<value, 3> li = relative_pos / r;

        value J2 = 0;

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                J2 += flat.idx(i, j) * angular_momentum.idx(i) * angular_momentum.idx(j);
            }
        }

        tensor<value, 3> lowered = lower_index(li, flat, 0);

        value cos_angle = dot(angular_momentum / max(angular_momentum.length(), value{1e-6}), lowered / max(lowered.length(), value{1e-5}));

        value sin2 = 1 - cos_angle * cos_angle;

        return 0.5f * (1 + sqrt(1 + 4 * J2 * r*r * sin2 / (squiggly_N_factor * squiggly_N_factor)));
    }

    ///only handles linear momentum currently
    ///https://arxiv.org/pdf/2101.10252.pdf (20)
    template<typename T, typename U>
    inline
    tensor<value, 3, 3> calculate_aij_single(equation_context& ctx, const tensor<value, 3>& coordinate, const metric<T, 3, 3>& flat, const params<T>& param, U&& tov_phi_at_coordinate)
    {
        tensor<value, 3> vposition = {param.position.x(), param.position.y(), param.position.z()};

        tensor<value, 3> relative_pos = coordinate - vposition;

        value r = relative_pos.length();

        r = max(r, 1e-3f);

        ctx.pin(r);

        tensor<value, 3> li = relative_pos / r;

        tensor<T, 3> linear_momentum_lower = lower_index(param.linear_momentum, flat, 0);

        value M_factor = calculate_M_factor(param, tov_phi_at_coordinate);

        ctx.pin(M_factor);

        value iQ = calculate_integral_Q(ctx, r, param, M_factor, tov_phi_at_coordinate);
        value iC = calculate_integral_C(ctx, r, param, M_factor, tov_phi_at_coordinate);

        ctx.pin(iQ);
        ctx.pin(iC);

        value coeff1 = 3 * iQ / (2 * r * r);
        value coeff2 = 3 * iC / pow(r, 4);

        tensor<value, 3, 3> aIJ;

        tensor<T, 3> P = param.linear_momentum;

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                value p1 = 2 * (0.5f * P.idx(i) * li.idx(j) + 0.5f * P.idx(j) * li.idx(i));

                value pklk = 0;

                for(int k=0; k < 3; k++)
                {
                    pklk += linear_momentum_lower.idx(k) * li.idx(k);
                }

                value p2 = (flat.idx(i, j) - li.idx(i) * li.idx(j)) * pklk;

                value p3 = p1;

                value p4 = (flat.idx(i, j) - 5 * li.idx(i) * li.idx(j)) * pklk;

                aIJ.idx(i, j) = coeff1 * (p1 - p2) + coeff2 * (p3 + p4);
            }
        }

        ctx.pin(aIJ);

        tensor<value, 3, 3, 3> eijk = get_eijk();

        value squiggly_N_factor = calculate_squiggly_N_factor(param, tov_phi_at_coordinate);

        ctx.pin(squiggly_N_factor);

        value unsquiggly_N_factor = calculate_integral_unsquiggly_N(ctx, r, param, squiggly_N_factor, tov_phi_at_coordinate);

        ctx.pin(unsquiggly_N_factor);

        tensor<T, 3> angular_momentum_lower = lower_index(param.angular_momentum, flat, 0);
        tensor<value, 3> li_lower = lower_index(li, get_flat_metric<value, 3>(), 0);

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                value sum = 0;

                for(int k=0; k < 3; k++)
                {
                    for(int l=0; l < 3; l++)
                    {
                        sum += (3 / (r*r*r)) * (li.idx(i) * eijk.idx(j, k, l) + li.idx(j) * eijk.idx(i, k, l)) * angular_momentum_lower.idx(k) * li_lower.idx(l) * unsquiggly_N_factor;
                    }
                }

                aIJ.idx(i, j) += sum;
            }
        }

        return aIJ;
    }

    template<typename T, typename U>
    inline
    value calculate_ppw2_p(const tensor<value, 3>& coordinate, const metric<T, 3, 3>& flat, const params<T>& param, U&& tov_phi_at_coordinate)
    {
        tensor<value, 3> vposition = {param.position.x(), param.position.y(), param.position.z()};

        tensor<value, 3> relative_pos = coordinate - vposition;

        value r = relative_pos.length();

        value M_factor = calculate_M_factor(param, tov_phi_at_coordinate);
        value squiggly_N_factor = calculate_squiggly_N_factor(param, tov_phi_at_coordinate);

        value W2_linear = calculate_W2_linear_momentum(flat, param.linear_momentum, M_factor);
        value W2_angular = calculate_W2_angular_momentum(coordinate, param.position, flat, param.angular_momentum, squiggly_N_factor);

        value linear_rapidity = acosh(sqrt(W2_linear));
        value angular_rapidity = acosh(sqrt(W2_angular));

        value final_W = cosh(linear_rapidity + angular_rapidity);

        conformal_data cdata = sample_conformal(r, param, tov_phi_at_coordinate);

        ///so. The paper specifically says superimpose ppw2p terms
        ///which presumably means add. Which would translate to adding the W2 terms
        ///this superposing is incorrect. I do not know how to combine linear and angular boost
        value ppw2p = (cdata.mass_energy_density + cdata.pressure) * (final_W*final_W) - cdata.pressure;

        return if_v(r > param.get_radius(),
                    value{0},
                    ppw2p);
    }

    #if 0
    ///https://arxiv.org/pdf/1606.04881.pdf (64)
    float calculate_N_factor(float mass)
    {
        float radius = mass_to_radius(mass);

        auto integration_func = [&](float coordinate_radius)
        {
            data<float> ndata = sample_interior(coordinate_radius, mass);

            return (ndata.mass_energy_density + ndata.pressure) * pow(coordinate_radius, 4.f);
        };

        return (8 * M_PI/3) * integrate_1d(integration_func, 64, radius, 0.f);
    }

    ///https://arxiv.org/pdf/1606.04881.pdf (57)
    template<typename T>
    T calculate_sigma(const T& coordinate_radius, float mass, float M_factor)
    {
        data<T> ndata = sample_interior<T>(coordinate_radius, T{mass});

        return (ndata.mass_energy_density + ndata.pressure) / M_factor;
    }

    ///https://arxiv.org/pdf/1606.04881.pdf (62)
    template<typename T>
    T calculate_kappa(const T& coordinate_radius, float mass, float N_factor)
    {
        data<T> ndata = sample_interior<T>(coordinate_radius, T{mass});

        return (ndata.mass_energy_density + ndata.pressure) / N_factor;
    }

    ///https://arxiv.org/pdf/1606.04881.pdf (55)
    template<typename T>
    T calculate_integral_N(const T& coordinate_radius, float mass, float N_factor)
    {
        auto integral_func = [mass, N_factor](const T& rp)
        {
            return (8 * M_PI / 3.f) * calculate_kappa(rp, mass, N_factor) * pow(rp, 4.f);
        };

        T integrated = integrate_1d(integral_func, 16, coordinate_radius, T{0.f});

        return if_v(coordinate_radius > mass_to_radius(mass),
                    1.f,
                    integrated);
    }

    ///only handles linear momentum currently
    ///https://arxiv.org/pdf/2101.10252.pdf (20)
    tensor<value, 3, 3> calculate_aij_single(const tensor<value, 3>& coordinate, const metric<float, 3, 3>& flat, const params& param)
    {
        tensor<value, 3> vposition = {param.position.x(), param.position.y(), param.position.z()};

        tensor<value, 3> relative_pos = coordinate - vposition;

        value r = relative_pos.length();

        r = max(r, 1e-3f);

        tensor<value, 3> li = relative_pos / r;

        tensor<float, 3> linear_momentum_lower = lower_index(param.linear_momentum, flat, 0);

        float M_factor = calculate_M_factor(param.mass);

        printf("M fac %f\n", M_factor);

        value iQ = calculate_integral_Q(r, param.mass, M_factor);
        value iC = calculate_integral_C(r, param.mass, M_factor);

        value coeff1 = 3 * iQ / (2 * r * r);
        value coeff2 = 3 * iC / pow(r, 4);

        tensor<value, 3, 3> aIJ;

        tensor<float, 3> P = param.linear_momentum;

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                value p1 = 2 * (0.5f * P.idx(i) * li.idx(j) + 0.5f * P.idx(j) * li.idx(i));

                value pklk = 0;

                for(int k=0; k < 3; k++)
                {
                    pklk += linear_momentum_lower.idx(k) * li.idx(k);
                }

                value p2 = (flat.idx(i, j) - li.idx(i) * li.idx(j)) * pklk;

                value p3 = p1;

                value p4 = (flat.idx(i, j) - 5 * li.idx(i) * li.idx(j)) * pklk;

                aIJ.idx(i, j) = coeff1 * (p1 - p2) + coeff2 * (p3 + p4);
            }
        }

        return aIJ;
    }

    value calculate_ppw2_p(const tensor<value, 3>& coordinate, const metric<float, 3, 3>& flat, const params& param)
    {
        tensor<value, 3> vposition = {param.position.x(), param.position.y(), param.position.z()};

        tensor<value, 3> relative_pos = coordinate - vposition;

        value r = relative_pos.length();

        data<value> dat = sample_interior<value>(r, value{param.mass});

        float M_factor = calculate_M_factor(param.mass);

        #if 0
        {
            auto ifunc = [&](float cr)
            {
                return calculate_sigma(cr, param.mass, M_factor) * cr * cr;
            };

            float test_integral = 4 * M_PI * integrate_1d(ifunc, 64, mass_to_radius(param.mass) * 10, 0.f);

            printf("Test integral %f\n", test_integral);
        }
        #endif // 0

        #if 0
        {
            float Q_test = calculate_integral_Q(mass_to_radius(param.mass) * 10, param.mass, M_factor);

            printf("QT %f\n", Q_test);
        }
        #endif // 0

        /*{
            float Q_test = calculate_integral_Q(mass_to_radius(param.mass) * 0.5f, param.mass, M_factor);

            printf("QT %f\n", Q_test);
        }*/

        float W2 = calculate_W2_linear_momentum(flat, param.linear_momentum, M_factor);

        //dat.mass_energy_density = 1;
        //dat.pressure = 0;

        value ppw2p = (dat.mass_energy_density + dat.pressure) * W2 - dat.pressure;

        return if_v(r > mass_to_radius(param.mass),
                    value{0},
                    ppw2p);
    }
    #endif // 0
}

inline
value calculate_h_with_gamma_eos(const value& eps)
{
    float Gamma = 2;

    return 1 + Gamma * eps;
}

//#define USE_MATTER

///https://arxiv.org/pdf/0812.0641.pdf just before 23
///X = e-4phi
struct matter
{
    value p_star;
    value e_star;
    tensor<value, 3> cS;

    float Gamma = 2;

    matter(equation_context& ctx)
    {
        p_star = bidx("Dp_star", ctx.uses_linear, false);
        e_star = bidx("De_star", ctx.uses_linear, false);

        for(int i=0; i < 3; i++)
        {
            cS.idx(i) = bidx("DcS" + std::to_string(i), ctx.uses_linear, false);
        }

        p_star = max(p_star, 0.f);
        e_star = max(e_star, 0.f);
    }

    /*value calculate_W(const inverse_metric<value, 3, 3>& icY, const value& chi)
    {
        value W = 0.5f;

        int iterations = 5;

        for(int i=0; i < iterations; i++)
        {
            W = w_next(W, p_star, chi, icY, cS, Gamma, e_star);
        }

        return W;
    }*/

    ///??? comes from initial conditions
    value p_star_max()
    {
        return 1;
    }

    value p_star_is_degenerate()
    {
        return p_star < 1e-5f * p_star_max();
    }

    value p_star_below_e_star_threshold()
    {
        float e_factor = 1e-4f;

        return p_star < e_factor * p_star_max();
    }

    value e_star_clamped()
    {
        return min(e_star, 10 * p_star);
    }

    value calculate_p0(const value& chi, const value& W)
    {
        return divide_with_limit(chi_to_e_m6phi(chi) * p_star * p_star, W, 0.f, DIVISION_TOL);
    }

    value calculate_eps(const value& chi, const value& W)
    {
        value e_m6phi = chi_to_e_m6phi(chi);

        /*value p0 = calculate_p0(chi, W);

        value au0 = divide_with_limit(W, p_star, 0.f);

        value lhs = divide_with_limit(pow(divide_with_limit(e_star * e_m6phi, au0, 0.f), Gamma), p0, 0.f);

        return lhs;*/

        return pow(divide_with_limit(e_m6phi, W, 0.f, DIVISION_TOL), Gamma - 1) * pow(e_star, Gamma) * pow(p_star, Gamma - 2);
    }

    /*value gamma_eos(const value& p0, const value& eps)
    {
        return (Gamma - 1) * p0 * eps;
    }*/

    value calculate_p0e(const value& chi, const value& W)
    {
        value iv_au0 = divide_with_limit(p_star, W, 0.f, DIVISION_TOL);

        value e_m6phi = chi_to_e_m6phi_unclamped(chi);

        return pow(max(e_star * e_m6phi * iv_au0, 0.f), Gamma);
    }

    value gamma_eos_from_e_star(const value& chi, const value& W)
    {
        value p0e = calculate_p0e(chi, W);

        return p0e * (Gamma - 1);
    }

    value calculate_h_with_gamma_eos(const value& chi, const value& W)
    {
        return ::calculate_h_with_gamma_eos(calculate_eps(chi, W));
    }

    tensor<value, 3> get_u_lower(const value& chi, const value& W)
    {
        tensor<value, 3> ret;

        for(int i=0; i < 3; i++)
        {
            ret.idx(i) = divide_with_limit(cS.idx(i), p_star * calculate_h_with_gamma_eos(chi, W), 0.f, DIVISION_TOL);
        }

        for(int i=0; i < 3; i++)
        {
            //ret.idx(i) = dual_types::clamp(ret.idx(i), value{-0.1f}, value{0.1f});
        }

        return ret;
    }

    tensor<value, 3> get_u_upper(const inverse_metric<value, 3, 3>& icY, const value& chi, const value& W)
    {
        tensor<value, 3> ui_lower = get_u_lower(chi, W);

        return raise_index(ui_lower, chi * icY, 0);
    }

    #if 0
    tensor<value, 3> get_v_upper(const inverse_metric<value, 3, 3>& icY, const value& gA, const value& chi, const value& W)
    {
        value u0 = divide_with_limit(W, (p_star * gA), 0);

        tensor<value, 3> u_up = get_u_upper(icY, chi, W);

        tensor<value, 3> clamped;

        for(int i=0; i < 3; i++)
        {
            clamped.idx(i) = divide_with_limit(u_up.idx(i), u0, 0.f);

            ///todo: tensor if_v
            //clamped.idx(i) = dual_types::if_v(p_star_is_degenerate(), 0.f, u_up.idx(i) / u0);
        }

        return clamped;
    }
    #endif // 0

    ///https://arxiv.org/pdf/gr-qc/9908027.pdf 2.12
    ///except sk = p* h uj, and uhatj = h uj
    ///and w = p* a u0 not a u0
    tensor<value, 3> get_v_upper(const inverse_metric<value, 3, 3>& icY, const value& gA, const tensor<value, 3>& gB, const value& chi, const value& W)
    {
        //#define V_UPPER_PRIMARY
        #ifdef V_UPPER_PRIMARY
        value u0 = divide_with_limit(W, (p_star * gA), 0, DIVISION_TOL);

        tensor<value, 3> u_up = get_u_upper(icY, chi, W);

        tensor<value, 3> clamped;

        for(int i=0; i < 3; i++)
        {
            clamped.idx(i) = divide_with_limit(u_up.idx(i), u0, 0.f, DIVISION_TOL);

            ///todo: tensor if_v
            //clamped.idx(i) = dual_types::if_v(p_star_is_degenerate(), 0.f, u_up.idx(i) / u0);
        }

        return clamped;
        #endif // V_UPPER_PRIMARY

        #define V_UPPER_ALT
        #ifdef V_UPPER_ALT
        value h = calculate_h_with_gamma_eos(chi, W);

        tensor<value, 3> ret = -gB;

        for(int i=0; i < 3; i++)
        {
            value sum = 0;

            for(int j=0; j < 3; j++)
            {
                sum += divide_with_limit(gA * icY.idx(i, j) * cS.idx(j) * chi, W * h, 0.f, DIVISION_TOL);
            }

            ret.idx(i) += sum;
        }

        return ret;
        #endif // V_UPPER_ALT
    }

    tensor<value, 3> p_star_vi(const inverse_metric<value, 3, 3>& icY, const value& gA, const tensor<value, 3>& gB, const value& chi, const value& W)
    {
        tensor<value, 3> v_upper = get_v_upper(icY, gA, gB, chi, W);

        return p_star * v_upper;
    }

    tensor<value, 3> e_star_vi(const inverse_metric<value, 3, 3>& icY, const value& gA, const tensor<value, 3>& gB, const value& chi, const value& W)
    {
        tensor<value, 3> v_upper = get_v_upper(icY, gA, gB, chi, W);

        return e_star * v_upper;
    }

    tensor<value, 3, 3> cSk_vi(const inverse_metric<value, 3, 3>& icY, const value& gA, const tensor<value, 3>& gB, const value& chi, const value& W)
    {
        tensor<value, 3> v_upper = get_v_upper(icY, gA, gB, chi, W);

        tensor<value, 3, 3> cSk_vi;

        for(int k=0; k < 3; k++)
        {
            for(int i=0; i < 3; i++)
            {
                cSk_vi.idx(k, i) = cS.idx(k) * v_upper.idx(i);
            }
        }

        return cSk_vi;
    }

    value calculate_adm_p(const value& chi, const value& W)
    {
        //return {};

        value h = calculate_h_with_gamma_eos(chi, W);
        value em6phi = chi_to_e_m6phi_unclamped(chi);

        value p0 = calculate_p0(chi, W);
        value eps = calculate_eps(chi, W);

        return h * W * em6phi - gamma_eos_from_e_star(chi, W);
    }

    tensor<value, 3> calculate_adm_Si(const value& chi)
    {
        value em6phi = chi_to_e_m6phi_unclamped(chi);

        return cS * em6phi;
    }

    ///the reason to calculate X_Sij is that its regular in terms of chi
    tensor<value, 3, 3> calculate_adm_X_Sij(const value& chi, const value& W, const metric<value, 3, 3>& cY)
    {
        value em6phi = chi_to_e_m6phi_unclamped(chi);
        value h = calculate_h_with_gamma_eos(chi, W);

        tensor<value, 3, 3> Sij;

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                Sij.idx(i, j) = divide_with_limit(em6phi, W * h, 0.f, DIVISION_TOL) * cS.idx(i) * cS.idx(j);
            }
        }

        ///this gamma eos is the specific problem
        tensor<value, 3, 3> X_P_Yij = gamma_eos_from_e_star(chi, W) * cY.to_tensor();
        //tensor<value, 3, 3> X_P_Yij = gamma_eos(p0, eps) * cY.to_tensor();

        return Sij * chi + X_P_Yij;
    }

    value calculate_adm_S(const metric<value, 3, 3>& cY, const inverse_metric<value, 3, 3>& icY, const value& chi, const value& W)
    {
        ///so. Raise Sij with iYij, which is X * icY
        ///now I'm actually raising X * Sij which means....... i can use icY?
        ///because iYij * Sjk = X * icYij * Sjk, and icYij * X * Sjk = X * icYij * Sjk
        tensor<value, 3, 3> XSij = calculate_adm_X_Sij(chi, W, cY);

        tensor<value, 3, 3> raised = raise_index(XSij, icY, 0);

        value sum = 0;

        for(int i=0; i < 3; i++)
        {
            sum += raised.idx(i, i);
        }

        return sum;
    }

    value calculate_PQvis(equation_context& ctx, const value& gA, const tensor<value, 3>& gB, const inverse_metric<value, 3, 3>& icY, const value& chi, const value& W)
    {
        #define QUADRATIC_VISCOSITY
        #ifndef QUADRATIC_VISCOSITY
        return 0;
        #endif // QUADRATIC_VISCOSITY

        value e_m6phi = chi_to_e_m6phi_unclamped(chi);
        value e_6phi = chi_to_e_6phi(chi);

        tensor<value, 3> vk = get_v_upper(icY, gA, gB, chi, W);

        value scale = "scale";

        value dkvk = 0;

        for(int k=0; k < 3; k++)
        {
            dkvk += 2 * diff1(ctx, vk.idx(k), k);
        }

        value littledv = dkvk * scale;

        value A = divide_with_limit(pow(e_star, Gamma) * pow(p_star, Gamma - 1) * pow(e_m6phi, Gamma - 1), pow(W, Gamma - 1), 0.f);
        //value A = divide_with_limit(pow(e_star, Gamma) * pow(p_star, Gamma - 1), pow(W * e_6phi, Gamma - 1), 0.f);

        //ctx.add("DBG_A", A);

        ///[0.1, 1.0}
        value CQvis = 1.f;

        value PQvis = if_v(littledv < 0, CQvis * A * pow(littledv, 2), 0.f);

        return PQvis;
    }

    ///I suspect we shouldn't quadratic viscosity near the event horizon, there's an infinite term to_diff
    value estar_vi_rhs(equation_context& ctx, const value& gA, const tensor<value, 3>& gB, const inverse_metric<value, 3, 3>& icY, const value& chi, const value& W)
    {
        #ifndef QUADRATIC_VISCOSITY
        return 0;
        #endif // QUADRATIC_VISCOSITY

        value e_m6phi = chi_to_e_m6phi_unclamped(chi);

        value PQvis = calculate_PQvis(ctx, gA, gB, icY, chi, W);

        tensor<value, 3> vk = get_v_upper(icY, gA, gB, chi, W);

        value sum_interior_rhs = 0;

        for(int k=0; k < 3; k++)
        {
            value to_diff = divide_with_limit(W * vk.idx(k), p_star * e_m6phi, 0.f);

            sum_interior_rhs += diff1(ctx, to_diff, k);
        }

        value p0e = calculate_p0e(chi, W);

        value degenerate = divide_with_limit(value{1}, pow(p0e, 1 - 1/Gamma), 0.f);

        /*ctx.add("DBG_IRHS", sum_interior_rhs);

        ctx.add("DBG_p0eps", p0 * eps);

        ctx.add("DINTERIOR", p0 * eps);

        ctx.add("DP2", PQvis / Gamma);

        ctx.add("DP1", degenerate);*/

        return -degenerate * (PQvis / Gamma) * sum_interior_rhs;
    }

    tensor<value, 3> cSkvi_rhs(equation_context& ctx, const inverse_metric<value, 3, 3>& icY, const value& gA, const tensor<value, 3>& gB, const value& chi, const tensor<value, 3>& dchi, const value& P, const value& W)
    {
        tensor<value, 3> dX = dchi;

        //value PQvis = calculate_PQvis(ctx, gA, gB, icY, chi, W);

        //ctx.pin(PQvis);

        value h = calculate_h_with_gamma_eos(chi, W);

        tensor<value, 3> ret;

        for(int k=0; k < 3; k++)
        {
            ret.idx(k) += -gA * divide_with_limit(value{1}, chi_to_e_m6phi(chi), 0.f, DIVISION_TOL) * diff1(ctx, P, k);

            ret.idx(k) += -W * h * diff1(ctx, gA, k);

            {
                value sum = 0;

                for(int j=0; j < 3; j++)
                {
                    sum += -cS.idx(j) * diff1(ctx, gB.idx(j), k);
                }

                ret.idx(k) += sum;
            }

            {
                value sum = 0;

                for(int i=0; i < 3; i++)
                {
                    for(int j=0; j < 3; j++)
                    {
                        sum += divide_with_limit(gA * chi * cS.idx(i) * cS.idx(j), (2 * W * h), 0.f, DIVISION_TOL) * diff1(ctx, icY.idx(i, j), k);
                    }
                }

                ret.idx(k) += sum;
            }

            ret.idx(k) += -divide_with_limit((2 * gA * h * (W * W - p_star * p_star)), W, 0.f, DIVISION_TOL) * calculate_dPhi(chi, dX).idx(k);
        }

        return ret;
    }
};

namespace compact_object
{
    enum type
    {
        BLACK_HOLE,
        NEUTRON_STAR,
    };

    template<typename T>
    struct base_matter_data
    {
        T compactness = 0.06;
        tensor<T, 3> colour = {1,1,1};
    };

    using matter_data = base_matter_data<float>;

    template<typename T>
    struct base_data
    {
        ///in coordinate space
        tensor<T, 3> position;
        ///for black holes this is bare mass. For neutron stars its a solid 'something'
        T bare_mass = 0;
        tensor<T, 3> momentum;
        tensor<T, 3> angular_momentum;

        base_matter_data<T> matter;

        type t = BLACK_HOLE;

        base_data(){}

        /*template<typename U>
        base_data(const base_data<U>& other)
        {
            position = other.position.template as<T>();
            bare_mass = T{other.bare_mass};
            momentum = other.momentum.template as<T>();
            angular_momentum = other.angular_momentum.template as<T>();

            matter.compactness = T{other.matter.compactness};
            matter.colour = other.matter.colour.template as<T>();

            t = other.t;
        }*/
    };

    using data = base_data<float>;
}

#if 0
///come back to adm
struct adm_black_hole
{
    float bare_mass_guess = 0.5f;

    tensor<float, 3> position;
    float adm_mass = 0;
    tensor<float, 3> velocity;
    tensor<float, 3> angular_velocity;
};
#endif // 0

namespace black_hole
{
    ///https://arxiv.org/pdf/gr-qc/0610128.pdf initial conditions, see (7)
    template<typename T>
    inline
    tensor<value, 3, 3> calculate_single_bcAij(const tensor<value, 3>& pos, const tensor<T, 3>& bh_position, const tensor<T, 3>& bh_momentum, const tensor<T, 3>& bh_angular_momentum)
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
                tensor<value, 3> momentum_tensor = {bh_momentum.x(), bh_momentum.y(), bh_momentum.z()};

                tensor<value, 3> vri = {bh_position.x(), bh_position.y(), bh_position.z()};

                value ra = (pos - vri).length();

                ra = max(ra, 1e-6);

                tensor<value, 3> nia = (pos - vri) / ra;

                tensor<value, 3> momentum_lower = lower_index(momentum_tensor, flat, 0);
                tensor<value, 3> nia_lower = lower_index(tensor<value, 3>{nia.x(), nia.y(), nia.z()}, flat, 0);

                bcAij.idx(i, j) += (3 / (2.f * ra * ra)) * (momentum_lower.idx(i) * nia_lower.idx(j) + momentum_lower.idx(j) * nia_lower.idx(i) - (flat.idx(i, j) - nia_lower.idx(i) * nia_lower.idx(j)) * sum_multiply(momentum_tensor, nia_lower));

                ///spin
                value s1 = 0;
                value s2 = 0;

                for(int k=0; k < 3; k++)
                {
                    for(int l=0; l < 3; l++)
                    {
                        s1 += eijk.idx(k, i, l) * bh_angular_momentum.idx(l) * nia[k] * nia_lower.idx(j);
                        s2 += eijk.idx(k, j, l) * bh_angular_momentum.idx(l) * nia[k] * nia_lower.idx(i);
                    }
                }

                bcAij.idx(i, j) += (3 / (ra*ra*ra)) * (s1 + s2);
            }
        }

        return bcAij;
    }
}

template<typename T, typename U>
inline
tensor<value, 3, 3> calculate_bcAij_generic(equation_context& ctx, const tensor<value, 3>& pos, const std::vector<compact_object::base_data<T>>& objs, U&& tov_phi_at_coordinate)
{
    tensor<value, 3, 3> bcAij;

    for(const compact_object::base_data<T>& obj : objs)
    {
        if(obj.t == compact_object::BLACK_HOLE)
        {
            tensor<value, 3, 3> bcAij_single = black_hole::calculate_single_bcAij(pos, obj.position, obj.momentum, obj.angular_momentum);

            bcAij += bcAij_single;
        }

        if(obj.t == compact_object::NEUTRON_STAR)
        {
            neutron_star::params<T> p;
            p.position = obj.position;
            p.mass = obj.bare_mass;
            p.compactness = obj.matter.compactness;
            p.linear_momentum = obj.momentum;
            p.angular_momentum = obj.angular_momentum;

            auto flat = get_flat_metric<T, 3>();
            auto flatv = get_flat_metric<value, 3>();

            tensor<value, 3, 3> bcAIJ_single = neutron_star::calculate_aij_single(ctx, pos, flat, p, tov_phi_at_coordinate);

            tensor<value, 3, 3> bcAij_single = lower_both(bcAIJ_single, flatv);

            bcAij += bcAij_single;
        }
    }

    return bcAij;
}

tensor<float, 3> world_to_voxel(const tensor<float, 3>& world_pos, vec3i dim, float scale)
{
    tensor<float, 3> centre = {(dim.x() - 1)/2, (dim.y() - 1)/2, (dim.z() - 1)/2};

    return (world_pos / scale) + centre;
}

#if 0
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
#endif // 0

struct initial_conditions
{
    bool use_matter = false;
    bool use_particles = false;

    std::vector<compact_object::data> objs;
    particle_data particles;
};

inline
value calculate_aij_aIJ(const metric<value, 3, 3>& flat_metric, const tensor<value, 3, 3>& bcAij)
{
    value aij_aIJ = 0;

    tensor<value, 3, 3> ibcAij = raise_both(bcAij, flat_metric.invert());

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            aij_aIJ += ibcAij.idx(i, j) * bcAij.idx(i, j);
        }
    }

    return aij_aIJ;
}

inline
value calculate_conformal_guess(const tensor<value, 3>& pos, const std::vector<compact_object::data>& holes)
{
    //https://arxiv.org/pdf/gr-qc/9703066.pdf (8)
    value BL_s = 0;

    for(const compact_object::data& hole : holes)
    {
        if(hole.t == compact_object::NEUTRON_STAR)
            continue;

        value Mi = hole.bare_mass;
        tensor<value, 3> ri = {hole.position.x(), hole.position.y(), hole.position.z()};

        value dist = (pos - ri).length();

        ///I'm not sure this is correct to do for neutron stars
        //if(hole.t == compact_object::BLACK_HOLE)
            dist = max(dist, 1e-3);
        //else
        //    dist = max(dist, 1e-1);

        BL_s += Mi / (2 * dist);
    }

    return BL_s;
}

value tov_phi_at_coordinate_general(const tensor<value, 3>& world_position)
{
    value fl3 = as_float3(world_position.x(), world_position.y(), world_position.z());

    /*value vx = dual_types::apply("world_to_voxel_x", fl3, "dim", "scale");
    value vy = dual_types::apply("world_to_voxel_y", fl3, "dim", "scale");
    value vz = dual_types::apply("world_to_voxel_z", fl3, "dim", "scale");*/

    value v = dual_types::apply("world_to_voxel", fl3, "dim", "scale");

    return dual_types::apply("buffer_read_linear", "tov_phi", v, "dim");

    //return dual_types::apply("tov_phi", as_float3(vx, vy, vz), "dim");
}

///so. Need to awkward mash particle ph in here
laplace_data setup_u_laplace(cl::context& clctx, const std::vector<compact_object::data>& cpu_holes, cl::buffer& aij_aIJ_buf, cl::buffer& ppw2p_buf, cl::buffer& nonconformal_pH)
{
    tensor<value, 3> pos = {"ox", "oy", "oz"};

    value u_value = dual_types::apply("buffer_index", "u_offset_in", "ix", "iy", "iz", "dim");

    equation_context eqs;

    //https://arxiv.org/pdf/gr-qc/9703066.pdf (8)
    ///todo when I forget: I'm using the conformal guess here for neutron stars which probably isn't right
    value BL_s_dyn = calculate_conformal_guess(pos, cpu_holes);

    ///https://arxiv.org/pdf/1606.04881.pdf 74
    value phi = BL_s_dyn + u_value;

    value cached_aij_aIJ = bidx("cached_aij_aIJ", false, false);
    value cached_ppw2p = bidx("cached_ppw2p", false, false);

    ///https://arxiv.org/pdf/1606.04881.pdf I think I need to do (85)
    ///ok no: I think what it is is that they're solving for ph in ToV, which uses tov's conformally flat variable
    ///whereas I'm getting values directly out of an analytic solution
    value U_RHS = (-1.f/8.f) * cached_aij_aIJ * pow(phi, -7) - 2 * M_PI * pow(phi, -3) * cached_ppw2p;

    laplace_data solve(aij_aIJ_buf, ppw2p_buf);

    solve.rhs = U_RHS;
    solve.boundary = 1;
    solve.ectx = eqs;

    return solve;
}

std::pair<cl::program, std::vector<cl::kernel>> build_and_fetch_kernel(cl::context& clctx, equation_context& ctx, const std::string& filename, const std::vector<std::string>& kernel_name, const std::string& temporaries_name)
{
    std::string local_build_str = "-I ./ -cl-std=CL1.2 -cl-finite-math-only ";

    ctx.build(local_build_str, temporaries_name);

    cl::program t_program(clctx, filename);
    t_program.build(clctx, local_build_str);

    std::vector<cl::kernel> kerns;

    for(auto& i : kernel_name)
    {
        kerns.emplace_back(t_program, i);
    }

    return {t_program, kerns};
}

std::pair<cl::program, cl::kernel> build_and_fetch_kernel(cl::context& clctx, equation_context& ctx, const std::string& filename, const std::string& kernel_name, const std::string& temporaries_name)
{
    auto result = build_and_fetch_kernel(clctx, ctx, filename, std::vector<std::string>{kernel_name}, temporaries_name);

    return {result.first, result.second[0]};
}

struct black_hole_gpu_data
{
    std::array<cl::buffer, 6> bcAij;

    black_hole_gpu_data(cl::context& ctx) :bcAij{ctx, ctx, ctx, ctx, ctx, ctx}
    {

    }

    void create(cl::context& ctx, cl::command_queue& cqueue, const compact_object::data& obj, float scale, vec3i dim)
    {
        int64_t cells = dim.x() * dim.y() * dim.z();

        for(int i=0; i < 6; i++)
        {
            bcAij[i].alloc(cells * sizeof(cl_float));
            bcAij[i].fill(cqueue, cl_float{0.f});
        }

        calculate_bcAij(ctx, cqueue, obj, scale, dim);
    }

private:
    void calculate_bcAij(cl::context& clctx, cl::command_queue& cqueue, const compact_object::data& obj, float scale, vec3i dim)
    {
        vec<4, cl_int> clsize = {dim.x(), dim.y(), dim.z(), 0};

        equation_context ctx;

        tensor<value, 3> pos = {"ox", "oy", "oz"};

        auto pinning_tov_phi = [&](const tensor<value, 3>& world_position)
        {
            assert(false);

            return value{0.f};
        };

        tensor<value, 3, 3> bcAij_dyn = calculate_bcAij_generic(ctx, pos, std::vector{obj}, pinning_tov_phi);

        vec2i linear_indices[6] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};

        for(int i=0; i < 6; i++)
        {
            vec2i index = linear_indices[i];

            ctx.add("B_BCAIJ_" + std::to_string(i), bcAij_dyn.idx(index.x(), index.y()));
        }

        ctx.add("INITIAL_BCAIJ", 1);

        auto [prog, calculate_bcAij_k] = build_and_fetch_kernel(clctx, ctx, "initial_conditions.cl", "calculate_bcAij", "bcaij");

        cl::args args;
        args.push_back(nullptr);

        for(int i=0; i < 6; i++)
        {
            args.push_back(bcAij[i]);
        }

        args.push_back(scale);
        args.push_back(clsize);

        calculate_bcAij_k.set_args(args);

        cqueue.exec(calculate_bcAij_k, {dim.x(), dim.y(), dim.z()}, {8,8,1}, {});
    }
};

struct gpu_matter_data
{
    cl_float4 position = {};
    cl_float4 linear_momentum = {};
    cl_float4 angular_momentum = {};
    cl_float4 colour = {};
    cl_float mass = 0;
    cl_float compactness = 0;
};

///no longer convinced its correct to sum aij_aIJ, instead of aIJ and then calculate
struct neutron_star_gpu_data
{
    cl::buffer tov_phi;
    std::array<cl::buffer, 6> bcAij;
    cl::buffer ppw2p;

    neutron_star_gpu_data(cl::context& clctx) : tov_phi(clctx), bcAij{clctx, clctx, clctx, clctx, clctx, clctx}, ppw2p(clctx)
    {

    }

    void create(cl::context& ctx, cl::command_queue& cqueue, cl::kernel calculate_ppw2p_kernel, cl::kernel calculate_bcAij_matter_kernel, const compact_object::data& obj, float scale, vec3i dim)
    {
        int64_t cells = dim.x() * dim.y() * dim.z();

        tov_phi.alloc(cells * sizeof(cl_float));
        tov_phi.fill(cqueue, cl_float{1.f});

        for(int i=0; i < 6; i++)
        {
            bcAij[i].alloc(cells * sizeof(cl_float));
            bcAij[i].fill(cqueue, cl_float{0.f});
        }

        ppw2p.alloc(cells * sizeof(cl_float));
        ppw2p.fill(cqueue, cl_float{0.f});

        calculate_tov_phi(ctx, cqueue, obj, scale, dim);
        calculate_bcAij(ctx, cqueue, calculate_bcAij_matter_kernel, obj, scale, dim);
        calculate_ppw2p(ctx, cqueue, calculate_ppw2p_kernel, obj, scale, dim);
    }

private:
    void calculate_tov_phi(cl::context& clctx, cl::command_queue& cqueue, const compact_object::data& obj, float scale, vec3i dim)
    {
        cl::buffer scratch(clctx);
        scratch.alloc(tov_phi.alloc_size);

        tensor<value, 3> pos = {"ox", "oy", "oz"};
        tensor<value, 3> from_object = pos - obj.position.as<value>();

        value coordinate_radius = from_object.length();

        neutron_star::data<value> dat = neutron_star::sample_interior(coordinate_radius, value{obj.bare_mass}, value{obj.matter.compactness});

        value rho = dat.mass_energy_density;

        value u_value = dual_types::apply("buffer_index", "u_offset_in", "ix", "iy", "iz", "dim");
        value phi = u_value;
        value phi_rhs = -2 * M_PI * pow(phi, 5) * rho;

        equation_context ctx;
        ctx.add("B_PHI_RHS", phi_rhs);

        float radius = neutron_star::mass_to_radius(obj.bare_mass, obj.matter.compactness);

        value cst = 1 + obj.bare_mass / (2 * max(coordinate_radius, 1e-3f));

        value integration_constant = if_v(coordinate_radius > radius, cst, 0);
        value within_star = if_v(coordinate_radius <= radius, value{1.f}, value{0.f});

        ctx.add("SHOULD_NOT_USE_INTEGRATION_CONSTANT", within_star);
        ctx.add("INTEGRATION_CONSTANT", integration_constant);

        cl_float err = 0.000001f;

        {
            vec<4, cl_int> clsize = {dim.x(), dim.y(), dim.z(), 0};

            auto [prog, iterate_phi] = build_and_fetch_kernel(clctx, ctx, "tov_solver.cl", "simple_tov_solver_phi", "UNUSEDTOVSOLVE");

            std::array<cl::buffer, 2> still_going{clctx, clctx};

            for(int i=0; i < 2; i++)
            {
                still_going[i].alloc(sizeof(cl_int));
                still_going[i].fill(cqueue, cl_int{1});
            }

            for(int i=0; i < 10000; i++)
            {
                cl::args iterate_u_args;
                iterate_u_args.push_back(tov_phi);
                iterate_u_args.push_back(scratch);
                iterate_u_args.push_back(scale);
                iterate_u_args.push_back(clsize);
                iterate_u_args.push_back(still_going[0]);
                iterate_u_args.push_back(still_going[1]);
                iterate_u_args.push_back(err);

                iterate_phi.set_args(iterate_u_args);

                cqueue.exec(iterate_phi, {clsize.x(), clsize.y(), clsize.z()}, {8, 8, 1}, {});

                if(((i % 50) == 0) && still_going[1].read<cl_int>(cqueue)[0] == 0)
                    break;

                still_going[0].set_to_zero(cqueue);

                std::swap(still_going[0], still_going[1]);
                std::swap(tov_phi, scratch);
            }
        }
    }

    void calculate_bcAij(cl::context& clctx, cl::command_queue& cqueue, cl::kernel& calculate_bcAij_matter_kernel, const compact_object::data& obj, float scale, vec3i dim)
    {
        vec<4, cl_int> clsize = {dim.x(), dim.y(), dim.z(), 0};

        gpu_matter_data gmd;
        gmd.position = {obj.position.x(), obj.position.y(), obj.position.z()};
        gmd.mass = obj.bare_mass;
        gmd.compactness = obj.matter.compactness;
        gmd.linear_momentum = {obj.momentum.x(), obj.momentum.y(), obj.momentum.z()};
        gmd.angular_momentum = {obj.angular_momentum.x(), obj.angular_momentum.y(), obj.angular_momentum.z()};
        gmd.colour = {obj.matter.colour.x(), obj.matter.colour.y(), obj.matter.colour.z()};

        cl::buffer buf(clctx);
        buf.alloc(sizeof(gpu_matter_data));
        buf.write(cqueue, std::vector<gpu_matter_data>{gmd});

        cl::args args;
        args.push_back(buf);
        args.push_back(tov_phi);

        for(int i=0; i < 6; i++)
        {
            args.push_back(bcAij[i]);
        }

        args.push_back(scale);
        args.push_back(clsize);

        calculate_bcAij_matter_kernel.set_args(args);

        cqueue.exec(calculate_bcAij_matter_kernel, {dim.x(), dim.y(), dim.z()}, {8,8,1}, {});
    }

    void calculate_ppw2p(cl::context& clctx, cl::command_queue& cqueue, cl::kernel calculate_ppw2p_kernel, const compact_object::data& obj, float scale, vec3i dim)
    {
        vec<4, cl_int> clsize = {dim.x(), dim.y(), dim.z(), 0};

        gpu_matter_data gmd;
        gmd.position = {obj.position.x(), obj.position.y(), obj.position.z()};
        gmd.mass = obj.bare_mass;
        gmd.compactness = obj.matter.compactness;
        gmd.linear_momentum = {obj.momentum.x(), obj.momentum.y(), obj.momentum.z()};
        gmd.angular_momentum = {obj.angular_momentum.x(), obj.angular_momentum.y(), obj.angular_momentum.z()};
        gmd.colour = {obj.matter.colour.x(), obj.matter.colour.y(), obj.matter.colour.z()};

        cl::buffer buf(clctx);
        buf.alloc(sizeof(gpu_matter_data));
        buf.write(cqueue, std::vector<gpu_matter_data>{gmd});

        cl::args args;
        args.push_back(buf);
        args.push_back(tov_phi);
        args.push_back(ppw2p);
        args.push_back(scale);
        args.push_back(clsize);

        calculate_ppw2p_kernel.set_args(args);

        cqueue.exec(calculate_ppw2p_kernel, {dim.x(), dim.y(), dim.z()}, {8,8,1}, {});
    }
};

struct superimposed_gpu_data
{
    ///this isn't added to, its forcibly imposed
    cl::buffer tov_phi;

    std::array<cl::buffer, 6> bcAij;
    ///derived from bcAij
    cl::buffer aij_aIJ;
    cl::buffer ppw2p;

    ///not conformal variables
    cl::buffer pressure_buf;
    cl::buffer rho_buf;
    cl::buffer rhoH_buf;
    cl::buffer p0_buf;
    std::array<cl::buffer, 3> Si_buf;
    std::array<cl::buffer, 3> colour_buf;

    cl::buffer particle_position;
    cl::buffer particle_mass;
    cl::buffer particle_lorentz;

    cl::buffer particle_counts;
    cl::buffer particle_indices;
    cl::buffer particle_memory_count;

    cl::buffer particle_grid_E_without_conformal;

    cl_int max_particle_memory = 1024 * 1024 * 120;

    cl::buffer u_arg;

    cl::program ppw2p_program;
    cl::kernel calculate_ppw2p_kernel;

    cl::program bcAij_matter_program;
    cl::kernel calculate_bcAij_matter_kernel;

    cl::program multi_matter_program;
    cl::kernel multi_matter_kernel;

    superimposed_gpu_data(cl::context& ctx, cl::command_queue& cqueue, vec3i dim) : tov_phi{ctx}, bcAij{ctx, ctx, ctx, ctx, ctx, ctx}, aij_aIJ{ctx}, ppw2p{ctx},
                                                                                                      pressure_buf{ctx}, rho_buf{ctx}, rhoH_buf{ctx}, p0_buf{ctx}, Si_buf{ctx, ctx, ctx}, u_arg{ctx},
                                                                                                      colour_buf{ctx, ctx, ctx},
                                                                                                      ppw2p_program(ctx), bcAij_matter_program(ctx), multi_matter_program(ctx),
                                                                                                      particle_position(ctx), particle_mass(ctx), particle_lorentz(ctx),
                                                                                                      particle_counts(ctx), particle_indices(ctx), particle_memory_count(ctx),
                                                                                                      particle_grid_E_without_conformal(ctx)
    {
        int cells = dim.x() * dim.y() * dim.z();

        ///unsure why I previously filled this with a 1
        tov_phi.alloc(cells * sizeof(cl_float));
        tov_phi.fill(cqueue, cl_float{1});

        for(int i=0; i < 6; i++)
        {
            bcAij[i].alloc(cells * sizeof(cl_float));
            bcAij[i].fill(cqueue, cl_float{0});
        }

        aij_aIJ.alloc(cells * sizeof(cl_float));
        aij_aIJ.fill(cqueue, cl_float{0});

        ppw2p.alloc(cells * sizeof(cl_float));
        ppw2p.fill(cqueue, cl_float{0});

        pressure_buf.alloc(cells * sizeof(cl_float));
        pressure_buf.fill(cqueue,cl_float{0});

        rho_buf.alloc(cells * sizeof(cl_float));
        rho_buf.fill(cqueue,cl_float{0});

        rhoH_buf.alloc(cells * sizeof(cl_float));
        rhoH_buf.fill(cqueue,cl_float{0});

        p0_buf.alloc(cells * sizeof(cl_float));
        p0_buf.fill(cqueue, cl_float{0});

        for(cl::buffer& i : Si_buf)
        {
            i.alloc(cells * sizeof(cl_float));
            i.fill(cqueue, cl_float{0});
        }

        for(cl::buffer& i : colour_buf)
        {
            i.alloc(cells * sizeof(cl_float));
            i.fill(cqueue, cl_float{0});
        }

        particle_counts.alloc(cells * sizeof(cl_ulong));
        particle_counts.fill(cqueue, cl_ulong{0});

        particle_indices.alloc(max_particle_memory * sizeof(cl_ulong));

        particle_memory_count.alloc(sizeof(cl_ulong));
        particle_memory_count.fill(cqueue, cl_ulong{0});

        particle_grid_E_without_conformal.alloc(cells * sizeof(cl_float));
        particle_grid_E_without_conformal.fill(cqueue, cl_float{0});
    }

    void build_accumulation_matter_programs(cl::context& ctx)
    {
        ///ppw2p generic kernel
        {
            equation_context ectx;

            ectx.add("INITIAL_PPW2P_2", 1);

            tensor<value, 3> pos = {"ox", "oy", "oz"};

            auto pinning_tov_phi = [&](const tensor<value, 3>& world_position)
            {
                value v = tov_phi_at_coordinate_general(world_position);
                ectx.pin(v);
                return v;
            };

            auto flat = get_flat_metric<value, 3>();

            neutron_star::params<value> p;
            p.position = {"data->position.x", "data->position.y", "data->position.z"};
            p.mass = "data->mass";
            p.compactness = "data->compactness";
            p.linear_momentum = {"data->linear_momentum.x", "data->linear_momentum.y", "data->linear_momentum.z"};
            p.angular_momentum = {"data->angular_momentum.x", "data->angular_momentum.y", "data->angular_momentum.z"};

            value ppw2p_equation = neutron_star::calculate_ppw2_p(pos, flat, p, pinning_tov_phi);

            ectx.add("B_PPW2P", ppw2p_equation);

            std::tie(ppw2p_program, calculate_ppw2p_kernel) = build_and_fetch_kernel(ctx, ectx, "initial_conditions.cl", "calculate_ppw2p", "ppw2p");
        }

        {

            compact_object::base_data<value> base_data;
            base_data.position = {"data->position.x", "data->position.y", "data->position.z"};
            base_data.bare_mass = "data->mass";
            base_data.momentum = {"data->linear_momentum.x", "data->linear_momentum.y", "data->linear_momentum.z"};
            base_data.angular_momentum = {"data->angular_momentum.x", "data->angular_momentum.y", "data->angular_momentum.z"};
            base_data.matter.compactness = "data->compactness";
            base_data.t = compact_object::NEUTRON_STAR;

            equation_context ectx;

            tensor<value, 3> pos = {"ox", "oy", "oz"};

            auto pinning_tov_phi = [&](const tensor<value, 3>& world_position)
            {
                value v = tov_phi_at_coordinate_general(world_position);
                ectx.pin(v);
                return v;
            };

            tensor<value, 3, 3> bcAij_dyn = calculate_bcAij_generic(ectx, pos, std::vector{base_data}, pinning_tov_phi);

            vec2i linear_indices[6] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};

            for(int i=0; i < 6; i++)
            {
                vec2i index = linear_indices[i];

                ectx.add("B_BCAIJ_" + std::to_string(i), bcAij_dyn.idx(index.x(), index.y()));
            }

            ectx.add("INITIAL_BCAIJ_2", 1);

            std::tie(bcAij_matter_program, calculate_bcAij_matter_kernel) = build_and_fetch_kernel(ctx, ectx, "initial_conditions.cl", "calculate_bcAij", "bcaij");
        }
    }

    void build_matter_program(cl::context& ctx, const value& conformal_guess)
    {
        tensor<value, 3> pos = {"ox", "oy", "oz"};

        equation_context ectx;

        auto pinning_tov_phi = [&](const tensor<value, 3>& world_position)
        {
            value v = tov_phi_at_coordinate_general(world_position);

            ectx.pin(v);

            return v;
        };

        value u_value = dual_types::apply("buffer_index", "u_value", "ix", "iy", "iz", "dim");

        ///https://arxiv.org/pdf/1606.04881.pdf 74
        value phi = conformal_guess + u_value;

        ///we need respectively
        ///(rhoH, Si, Sij), all lower indices

        ///pH is the adm variable, NOT P
        value p0_conformal = 0;

        value pressure_conformal = 0;
        value rho_conformal = 0;
        value rhoH_conformal = 0;
        tensor<value, 3> Si_conformal;
        tensor<value, 3> colour;

        {
            ///todo: remove the duplication?
            neutron_star::params<value> p;
            p.position = {"data->position.x", "data->position.y", "data->position.z"};
            p.mass = "data->mass";
            p.compactness = "data->compactness";
            p.linear_momentum = {"data->linear_momentum.x", "data->linear_momentum.y", "data->linear_momentum.z"};
            p.angular_momentum = {"data->angular_momentum.x", "data->angular_momentum.y", "data->angular_momentum.z"};

            tensor<value, 3> in_colour = {"data->colour.x", "data->colour.y", "data->colour.z"};

            value rad = (pos - p.position).length();

            ectx.pin(rad);

            value M_factor = neutron_star::calculate_M_factor(p, pinning_tov_phi);

            ectx.pin(M_factor);

            auto flat = get_flat_metric<value, 3>();

            //value W2_factor = neutron_star::calculate_W2_linear_momentum(flat, p.linear_momentum, M_factor);

            //neutron_star::data<value> sampled = neutron_star::sample_interior<value>(rad, value{p.mass});

            neutron_star::conformal_data cdata = neutron_star::sample_conformal(rad, p, pinning_tov_phi);

            ectx.pin(cdata.mass_energy_density);
            ectx.pin(cdata.pressure);
            ectx.pin(cdata.rest_mass_density);

            pressure_conformal += cdata.pressure;
            //unused_conformal_rest_mass += sampled.mass_energy_density;

            rho_conformal += cdata.mass_energy_density;
            //rhoH_conformal += (cdata.mass_energy_density + cdata.pressure) * W2_factor - cdata.pressure;
            rhoH_conformal += neutron_star::calculate_ppw2_p(pos, flat, p, pinning_tov_phi);
            //eps += sampled.specific_energy_density;

            //enthalpy += 1 + sampled.specific_energy_density + pressure_conformal / sampled.mass_energy_density;

            p0_conformal += cdata.mass_energy_density;

            ///https://arxiv.org/pdf/1606.04881.pdf (61)
            {
                value squiggly_N_factor = neutron_star::calculate_squiggly_N_factor(p, pinning_tov_phi);

                value kappa = neutron_star::calculate_kappa(rad, p, squiggly_N_factor, pinning_tov_phi);

                ectx.pin(kappa);

                /// p.angular_momentum *
                tensor<value, 3> Si_conformal_angular_lower;

                auto eijk = get_eijk();

                ///X
                tensor<value, 3> relative_pos = pos - p.position;

                for(int i=0; i < 3; i++)
                {
                    for(int j=0; j < 3; j++)
                    {
                        for(int k=0; k < 3; k++)
                        {
                            Si_conformal_angular_lower.idx(i) += eijk.idx(i, j, k) * p.angular_momentum.idx(j) * relative_pos.idx(k) * kappa;
                        }
                    }
                }

                tensor<value, 3> Si_cai = raise_index(Si_conformal_angular_lower, flat.invert(), 0);

                Si_conformal += Si_cai;
            }

            ///https://arxiv.org/pdf/1606.04881.pdf (56)
            ///we could avoid the triple calculation of sigma here
            Si_conformal += p.linear_momentum * neutron_star::calculate_sigma(rad, p, M_factor, pinning_tov_phi);

            colour.x() += if_v(rad <= p.get_radius(), value{in_colour.x()}, value{0.f});
            colour.y() += if_v(rad <= p.get_radius(), value{in_colour.y()}, value{0.f});
            colour.z() += if_v(rad <= p.get_radius(), value{in_colour.z()}, value{0.f});
        }

        value pressure = pow(phi, -8.f) * pressure_conformal;
        value rho = pow(phi, -8.f) * rho_conformal;
        value rhoH = pow(phi, -8.f) * rhoH_conformal;
        value p0 = pow(phi, -8.f) * p0_conformal;
        tensor<value, 3> Si = pow(phi, -10.f) * Si_conformal; // upper

        ectx.add("ALL_MATTER_VARIABLES", 1);
        ectx.add("ACCUM_PRESSURE", pressure);
        ectx.add("ACCUM_RHO", rho);
        ectx.add("ACCUM_RHOH", rhoH);
        ectx.add("ACCUM_P0", p0);

        for(int i=0; i < 3; i++)
        {
            ectx.add("ACCUM_SI" + std::to_string(i), Si.idx(i));
            ectx.add("ACCUM_COLOUR" + std::to_string(i), colour.idx(i));
        }

        std::tie(multi_matter_program, multi_matter_kernel) = build_and_fetch_kernel(ctx, ectx, "initial_conditions.cl", "multi_accumulate", "multiaccumulate");
    }

    void pull_all(cl::context& clctx, cl::command_queue& cqueue, const std::vector<compact_object::data>& objs, const particle_data& particles, float scale, vec3i dim)
    {
        bool built_accum_matter_kernel = false;

        for(const compact_object::data& obj : objs)
        {
            if(obj.t == compact_object::NEUTRON_STAR)
            {
                if(!built_accum_matter_kernel)
                {
                    build_accumulation_matter_programs(clctx);
                    built_accum_matter_kernel = true;
                }

                neutron_star_gpu_data dat(clctx);
                dat.create(clctx, cqueue, calculate_ppw2p_kernel, calculate_bcAij_matter_kernel, obj, scale, dim);

                pull(clctx, cqueue, dat, obj, scale, dim);
            }
            else
            {
                black_hole_gpu_data dat(clctx);
                dat.create(clctx, cqueue, obj, scale, dim);

                pull(clctx, cqueue, dat, scale, dim);
            }
        }

        ///need to handle particle data here. Currently only doing stationary particles, which do not have a velocity component
        ///So write the particle positions, set up everything, do the fast method (sigh), and then go

        pull(clctx, cqueue, particles, scale, dim);

        laplace_data solve = setup_u_laplace(clctx, objs, aij_aIJ, ppw2p, particle_grid_E_without_conformal);
        u_arg = laplace_solver(clctx, cqueue, solve, scale, dim, 0.000001f);

        tensor<value, 3> pos = {"ox", "oy", "oz"};

        //https://arxiv.org/pdf/gr-qc/9703066.pdf (8)
        value conformal_guess = calculate_conformal_guess(pos, objs);;

        bool built_matter_program = false;;

        for(const compact_object::data& obj : objs)
        {
            if(obj.t == compact_object::NEUTRON_STAR)
            {
                if(!built_matter_program)
                {
                    build_matter_program(clctx, conformal_guess);
                    built_matter_program = true;
                }

                accumulate_matter_variables(clctx, cqueue, scale, dim, obj, conformal_guess);
            }
        }
    }

    void accumulate_matter_variables(cl::context& clctx, cl::command_queue& cqueue, float scale, vec3i dim, const compact_object::data& obj, const value& conformal_guess)
    {
        vec<4, cl_int> clsize = {dim.x(), dim.y(), dim.z(), 0};

        gpu_matter_data gmd;
        gmd.position = {obj.position.x(), obj.position.y(), obj.position.z()};
        gmd.mass = obj.bare_mass;
        gmd.compactness = obj.matter.compactness;
        gmd.linear_momentum = {obj.momentum.x(), obj.momentum.y(), obj.momentum.z()};
        gmd.angular_momentum = {obj.angular_momentum.x(), obj.angular_momentum.y(), obj.angular_momentum.z()};
        gmd.colour = {obj.matter.colour.x(), obj.matter.colour.y(), obj.matter.colour.z()};

        cl::buffer buf(clctx);
        buf.alloc(sizeof(gpu_matter_data));
        buf.write(cqueue, std::vector<gpu_matter_data>{gmd});

        cl::args args;
        args.push_back(buf);
        args.push_back(pressure_buf);
        args.push_back(rho_buf);
        args.push_back(rhoH_buf);
        args.push_back(p0_buf);
        args.push_back(Si_buf[0]);
        args.push_back(Si_buf[1]);
        args.push_back(Si_buf[2]);
        args.push_back(colour_buf[0]);
        args.push_back(colour_buf[1]);
        args.push_back(colour_buf[2]);
        args.push_back(u_arg);
        args.push_back(tov_phi);
        args.push_back(scale);
        args.push_back(clsize);

        multi_matter_kernel.set_args(args);

        cqueue.exec(multi_matter_kernel, {dim.x(), dim.y(), dim.z()}, {8,8,1}, {});
    }

    void pull(cl::context& clctx, cl::command_queue& cqueue, const particle_data& particles, float scale, vec3i dim)
    {
        if(particles.positions.size() == 0)
            return;


    }

    void pull(cl::context& clctx, cl::command_queue& cqueue, neutron_star_gpu_data& dat, const compact_object::data& obj, float scale, vec3i dim)
    {
        auto flat = get_flat_metric<float, 3>();

        equation_context ctx;

        auto pinning_tov_phi = [&](const tensor<value, 3>& world_position)
        {
            value v = tov_phi_at_coordinate_general(world_position);
            ctx.pin(v);
            return v;
        };

        tensor<value, 3> pos = {"ox", "oy", "oz"};
        tensor<value, 3> from_object = pos - obj.position.as<value>();

        value coordinate_radius = from_object.length();

        float radius = neutron_star::mass_to_radius(obj.bare_mass, obj.matter.compactness);

        ///todo: remove the duplication?
        neutron_star::params<float> p;
        p.position = obj.position;
        p.mass = obj.bare_mass;
        p.compactness = obj.matter.compactness;
        p.linear_momentum = obj.momentum;
        p.angular_momentum = obj.angular_momentum;

        value superimposed_tov_phi_eq = dual_types::if_v(coordinate_radius <= radius, pinning_tov_phi(pos), 0.f);

        vec<4, cl_int> clsize = {dim.x(), dim.y(), dim.z(), 0};

        ctx.add("ACCUM_MATTER_VARIABLES", 1);

        ctx.add("B_TOV_PHI", superimposed_tov_phi_eq);

        auto [prog, accum_matter_variables_k] = build_and_fetch_kernel(clctx, ctx, "initial_conditions.cl", "accum_matter_variables", "accum");

        cl::args args;
        args.push_back(dat.tov_phi);

        for(int i=0; i < 6; i++)
        {
            args.push_back(dat.bcAij[i]);
        }

        args.push_back(dat.ppw2p);

        args.push_back(tov_phi);

        for(int i=0; i < 6; i++)
        {
            args.push_back(bcAij[i]);
        }

        args.push_back(ppw2p);
        args.push_back(scale);
        args.push_back(clsize);

        accum_matter_variables_k.set_args(args);

        cqueue.exec(accum_matter_variables_k, {dim.x(), dim.y(), dim.z()}, {8,8,1}, {});

        recalculate_aij_aIJ(clctx, cqueue, scale, dim);
    }

    void pull(cl::context& clctx, cl::command_queue& cqueue, black_hole_gpu_data& dat, float scale, vec3i dim)
    {
        equation_context ctx;

        vec<4, cl_int> clsize = {dim.x(), dim.y(), dim.z(), 0};

        ctx.add("ACCUM_BLACK_HOLE_VARIABLES", 1);

        auto [prog, accum_black_hole_variables_k] = build_and_fetch_kernel(clctx, ctx, "initial_conditions.cl", "accum_black_hole_variables", "accummatter");

        cl::args args;

        for(int i=0; i < 6; i++)
        {
            args.push_back(dat.bcAij[i]);
        }

        for(int i=0; i < 6; i++)
        {
            args.push_back(bcAij[i]);
        }

        args.push_back(scale);
        args.push_back(clsize);

        accum_black_hole_variables_k.set_args(args);

        cqueue.exec(accum_black_hole_variables_k, {dim.x(), dim.y(), dim.z()}, {8,8,1}, {});

        recalculate_aij_aIJ(clctx, cqueue, scale, dim);
    }

    void recalculate_aij_aIJ(cl::context& clctx, cl::command_queue& cqueue, float scale, vec3i dim)
    {
        equation_context ctx;
        ctx.add("CALCULATE_AIJ_AIJ", 1);

        vec<4, cl_int> clsize = {dim.x(), dim.y(), dim.z(), 0};

        auto flat = get_flat_metric<value, 3>();

        int index_table[3][3] = {{0, 1, 2},
                                 {1, 3, 4},
                                 {2, 4, 5}};

        tensor<value, 3, 3> bcAija;

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                bcAija.idx(i, j) = bidx("bcAij" + std::to_string(index_table[i][j]), false, false);
            }
        }

        value aij_aIJ_eq = calculate_aij_aIJ(flat, bcAija);

        ctx.add("B_AIJ_AIJ", aij_aIJ_eq);

        auto [prog, calculate_aij_aIJ_k] = build_and_fetch_kernel(clctx, ctx, "initial_conditions.cl", "calculate_aij_aIJ", "calcaijaij");

        cl::args args;

        for(int i=0; i < 6; i++)
        {
            args.push_back(bcAij[i]);
        }

        args.push_back(aij_aIJ);
        args.push_back(scale);
        args.push_back(clsize);

        calculate_aij_aIJ_k.set_args(args);

        cqueue.exec(calculate_aij_aIJ_k, {dim.x(), dim.y(), dim.z()}, {8,8,1}, {});
    }
};

///this I suspect is very slow
void construct_hydrodynamic_quantities(equation_context& ctx, const std::vector<compact_object::data>& cpu_holes)
{
    tensor<value, 3> pos = {"ox", "oy", "oz"};

    value gA = bidx("gA", ctx.uses_linear, false);

    //https://arxiv.org/pdf/gr-qc/9703066.pdf (8)
    value BL_s_dyn = calculate_conformal_guess(pos, cpu_holes);

    value u_value = dual_types::apply("buffer_index", "u_value", "ix", "iy", "iz", "dim");

    ///https://arxiv.org/pdf/1606.04881.pdf 74
    value phi = BL_s_dyn + u_value;

    value pressure = bidx("pressure_in", ctx.uses_linear, false);
    value rho = bidx("rho_in", ctx.uses_linear, false);
    value rhoH = bidx("rhoH_in", ctx.uses_linear, false);
    value p0 = bidx("p0_in", ctx.uses_linear, false);

    tensor<value, 3> Si;

    for(int i=0; i < 3; i++)
    {
        Si.idx(i) = bidx("Si" + std::to_string(i) + "_in", ctx.uses_linear, false);;
    }

    tensor<value, 3> colour;

    for(int i=0; i < 3; i++)
    {
        colour.idx(i) = bidx("colour" + std::to_string(i) + "_in", ctx.uses_linear, false);;
    }

    value is_degenerate = rho < 0.0001f;

    value W2 = if_v(is_degenerate, 1.f, ((rhoH + pressure) / (rho + pressure)));

    ///https://arxiv.org/pdf/1606.04881.pdf (70)
    tensor<value, 3> u_upper;

    for(int k=0; k < 3; k++)
    {
        u_upper.idx(k) = divide_with_limit(Si.idx(k), rho + pressure, 0.f) * sqrt(W2);
    }

    ///conformal hydrodynamical quantities
    {
        auto flat = get_flat_metric<value, 3>();

        float Gamma = 2;

        /// https://arxiv.org/pdf/1606.04881.pdf (8), Yij = phi^4 * cYij
        ///in bssn, the conformal decomposition is chi * Yij = cYij
        ///or see initial conditions, its 1/12th Yij.det()
        metric<value, 3, 3> Yij = pow(phi, 4) * flat;

        value X = pow(Yij.det(), -1.f/3.f);

        metric<value, 3, 3> cY = X * Yij;

        tensor<value, 3> Si_lower = lower_index(Si, Yij, 0);

        tensor<value, 3> cSi_lower = Si_lower * chi_to_e_6phi(X);

        tensor<value, 3> u_lower = lower_index(u_upper, Yij, 0);

        ///https://arxiv.org/pdf/2012.13954.pdf (7)

        value u0 = sqrt(W2);
        value gA_u0 = gA * u0;

        gA_u0 = if_v(is_degenerate, 0.f, gA_u0);

        value p_star = p0 * gA * u0 * chi_to_e_6phi(X);

        /*ctx.add("D_p0", p0);
        ctx.add("D_gA", gA);
        ctx.add("D_u0", u0);
        ctx.add("D_chip", chi_to_e_6phi(X));
        ctx.add("D_X", X);
        ctx.add("D_phi", phi);
        ctx.add("D_u", u_value);
        ctx.add("D_DYN", BL_s_dyn);*/

        value littleW = p_star * gA * u0;

        value h = if_v(is_degenerate, 0.f, (rhoH + pressure) * chi_to_e_6phi(X) / littleW);

        value p0e = p0 * h - p0 + pressure;

        value e_star = pow(p0e, 1/Gamma) * gA * u0 * chi_to_e_6phi(X);

        /*ctx.add("D_rho", rho);
        ctx.add("D_rhoH", rhoH);
        ctx.add("D_conformal_rest_mass", unused_conformal_rest_mass);
        ctx.add("D_conformal_pressure", pressure_conformal);
        ctx.add("D_p_star", p_star);
        ctx.add("D_eps_p0", eps_p0);
        ctx.add("D_p0", p0);
        ctx.add("D_h", h);
        ctx.add("D_pressure", pressure);
        ctx.add("D_gA_u0", gA_u0);
        ctx.add("D_phi", phi);
        ctx.add("D_W2", W2);
        ctx.add("D_littlee", unused_littlee);
        ctx.add("D_eps", eps);*/

        ctx.add("build_p_star", p_star);
        ctx.add("build_e_star", e_star);

        for(int k=0; k < 3; k++)
        {
            ctx.add("build_sk" + std::to_string(k), Si_lower.idx(k));
        }

        ctx.add("build_cR", colour.x());
        ctx.add("build_cG", colour.y());
        ctx.add("build_cB", colour.z());
    }
}

#if 0
sandwich_result setup_sandwich_laplace(cl::context& clctx, cl::command_queue& cqueue, const std::vector<compact_object::data<float>>& cpu_holes, float scale, vec3i dim)
{
    cl::buffer u_arg(clctx);

    {
        laplace_data solve = setup_u_laplace(clctx, cpu_holes);

        u_arg = laplace_solver(clctx, cqueue, solve, calculate_scale(get_c_at_max(), dim), dim, 0.0001f);
    }

    tensor<value, 3> pos = {"ox", "oy", "oz"};

    metric<value, 3, 3> flat_metric;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            flat_metric.idx(i, j) = (i == j) ? 1 : 0;
        }
    }

    value u_value = dual_types::apply("buffer_index", "u_in", "ix", "iy", "iz", "dim");

    value BL_s_dyn = calculate_conformal_guess(pos, cpu_holes);

    value phi_dyn = BL_s_dyn + u_value;

    tensor<value, 3> gB;

    gB.idx(0) = bidx("gB0_in", false, false);
    gB.idx(1) = bidx("gB1_in", false, false);
    gB.idx(2) = bidx("gB2_in", false, false);

    equation_context ctx;
    ctx.order = 1;
    //ctx.always_directional_derivatives = true;

    equation_context precise;
    precise.always_directional_derivatives = true;
    precise.order = 1;

    value djbj_precalculate = 0;

    for(int j=0; j < 3; j++)
    {
        djbj_precalculate += diff1(precise, gB.idx(j), j);
    }

    sandwich_data sandwich(clctx);
    sandwich.u_arg = u_arg;
    sandwich.u_to_phi = phi_dyn;

    sandwich.djbj = djbj_precalculate;

    value djbj = bidx("djbj", false, false);

    value phi = bidx("phi", false, false);

    value gA_phi = bidx("gA_phi_in", false, false);

    value gA = gA_phi / phi;

    tensor<value, 3, 3> bcAij = calculate_bcAij(pos, cpu_holes);
    ///raised
    tensor<value, 3, 3> ibcAij = raise_both(bcAij, flat_metric, flat_metric.invert());

    value aij_aIJ = calculate_aij_aIJ(flat_metric, bcAij, cpu_holes);

    tensor<value, 3> djaphi;

    for(int j=0; j < 3; j++)
    {
        djaphi.idx(j) = diff1(ctx, gA * pow(phi, -6.f), j);;

        //djaphi.idx(j) = (phi * diff1(ctx, gA, j) - 6 * gA * diff1(ctx, phi, j)) / pow(phi, 7);
    }

    tensor<value, 3> gB_rhs;

    for(int i=0; i < 3; i++)
    {
        value p1 = -(1.f/3.f) * diff1(ctx, djbj, i);

        value p2 = 0;

        for(int j=0; j < 3; j++)
        {
            p2 += 2 * ibcAij.idx(i, j) * diff1(ctx, gA * pow(phi, -6.f), j);
        }

        ///value p3 = matter
        gB_rhs.idx(i) = p1 + p2;
    }

    sandwich.gB0_rhs = gB_rhs.idx(0);
    sandwich.gB1_rhs = gB_rhs.idx(1);
    sandwich.gB2_rhs = gB_rhs.idx(2);

    value gA_phi_rhs = 0;

    gA_phi_rhs = gA_phi * ((7.f/8.f) * pow(phi, -8.f) * aij_aIJ); ///todo: matter terms

    sandwich.gA_phi_rhs = gA_phi_rhs;

    sandwich_result result = sandwich_solver(clctx, cqueue, sandwich, scale, dim, 0.0001f);

    return result;
}
#endif // 0

#if 0
std::vector<float> calculate_adm_mass(const std::vector<black_hole<float>>& holes, cl::context& ctx, cl::command_queue& cqueue, float err = 0.0001f)
{
    std::vector<float> ret;

    vec3i dim = {281, 281, 281};

    laplace_data solve = setup_u_laplace(ctx, holes);

    cl::buffer u_arg = laplace_solver(ctx, cqueue, solve, calculate_scale(get_c_at_max(), dim), dim, err);

    for(int i=0; i < (int)holes.size(); i++)
    {
        ret.push_back(get_nonspinning_adm_mass(cqueue, i, holes, dim, calculate_scale(get_c_at_max(), dim), u_arg));
    }

    return ret;
}
#endif // 0

inline
initial_conditions get_bare_initial_conditions(cl::context& clctx, cl::command_queue& cqueue, float scale, std::vector<compact_object::data> objs, std::optional<particle_data>&& p_data_opt)
{
    initial_conditions ret;

    float bulge = 1;

    auto san_pos = [&](const tensor<float, 3>& in)
    {
        tensor<float, 3> scaled = round((in / scale) * bulge);

        return scaled * scale / bulge;
    };

    for(compact_object::data& obj : objs)
    {
        obj.position = san_pos(obj.position);
    }

    for(const compact_object::data& obj : objs)
    {
        if(obj.t == compact_object::NEUTRON_STAR)
        {
            ret.use_matter = true;
        }
    }

    ret.objs = objs;
    //ret.use_matter = true;

    if(p_data_opt.has_value())
    {
        ret.use_particles = true;
        ret.particles = std::move(p_data_opt.value());
    }

    return ret;
}

#if 0
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
#endif // 0

inline
initial_conditions setup_dynamic_initial_conditions(cl::context& clctx, cl::command_queue& cqueue, vec3f centre, float scale)
{
    initial_conditions ret;

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
    std::vector<compact_object::data> objects;
    std::optional<particle_data> data_opt;

    ///https://arxiv.org/pdf/gr-qc/0610128.pdf
    ///todo: revert the fact that I butchered this
    //#define PAPER_0610128
    #ifdef PAPER_0610128
    compact_object::data h1;
    h1.t = compact_object::BLACK_HOLE;
    h1.bare_mass = 0.483;
    h1.momentum = {0, 0.133, 0};
    h1.position = {-3.257, 0.f, 0.f};

    compact_object::data h2;
    h2.t = compact_object::BLACK_HOLE;
    h2.bare_mass = 0.483;
    h2.momentum = {0, -0.133, 0};
    h2.position = {3.257, 0.f, 0.f};

    objects = {h1, h2};
    #endif // PAPER_0610128

    ///https://arxiv.org/pdf/1507.00570.pdf
    //#define PAPER_1507
    #ifdef PAPER_1507
    compact_object::data h1;
    h1.t = compact_object::BLACK_HOLE;
    h1.bare_mass = 0.1764;
    h1.momentum = {0, 0.12616, 0};
    h1.position = {-2.966, 0.f, 0.f};
    h1.angular_momentum = {0, 0, 0.225};

    compact_object::data h2;
    h2.t = compact_object::BLACK_HOLE;
    h2.bare_mass = 0.1764;
    h2.momentum = {0, -0.12616, 0};
    h2.position = {2.966, 0.f, 0.f};
    h2.angular_momentum = {0, 0, 0.225};

    objects = {h1, h2};
    #endif

    /*compact_object::data h1;
    h1.t = compact_object::BLACK_HOLE;
    h1.bare_mass = 0.5;
    h1.momentum = {0, 0, 0};
    h1.position = {0, 0, 0};
    h1.angular_momentum = {0, 0, 0};

    objects = {h1};*/

    //#define SPINNING_SINGLE_NEUTRON
    #ifdef SPINNING_SINGLE_NEUTRON
    compact_object::data h1;
    h1.t = compact_object::NEUTRON_STAR;
    h1.bare_mass = 0.1;
    h1.angular_momentum = {0, 0, 0.01};
    h1.position = {-3,0,0};
    h1.matter.compactness = 0.08;

    compact_object::data h2;
    h2.t = compact_object::NEUTRON_STAR;
    h2.matter.compactness = 0.01;
    h2.bare_mass = 0.02;
    h2.position = {7, 0, 0};
    h2.matter.colour = {1, 0.4, 0};

    objects = {h1};
    #endif // SPINNING_SINGLE_NEUTRON

    //#define DOUBLE_SPINNING_NEUTRON
    #ifdef DOUBLE_SPINNING_NEUTRON
    compact_object::data h1;
    h1.t = compact_object::NEUTRON_STAR;
    h1.bare_mass = 0.075;
    h1.angular_momentum = {0, 0, 0.0075};
    h1.position = {-5,0,0};
    h1.matter.compactness = 0.06;

    compact_object::data h2;
    h2.t = compact_object::NEUTRON_STAR;
    h2.bare_mass = 0.075;
    h2.angular_momentum = {0, 0, -0.0075};
    h2.position = {5,0,0};
    h2.matter.compactness = 0.06;

    objects = {h1, h2};
    #endif // DOUBLE_SPINNING_NEUTRON

    //#define JET_CASE
    #ifdef JET_CASE
    compact_object::data h1;
    h1.t = compact_object::NEUTRON_STAR;
    h1.bare_mass = 0.15;
    h1.momentum = {0, 0.133 * 0.8 * 0.6, 0};
    h1.position = {-5.257, 0.f, 0.f};

    compact_object::data h2;
    h2.t = compact_object::BLACK_HOLE;
    h2.bare_mass = 0.4;
    h2.momentum = {0, -0.133 * 0.8 * 0.1, 0};
    h2.position = {2.257, 0.f, 0.f};

    objects = {h1, h2};
    #endif // JET_CASE

    //#define REALLYBIG
    #ifdef REALLYBIG
    compact_object::data h1;
    h1.t = compact_object::NEUTRON_STAR;
    h1.bare_mass = 0.15;
    h1.momentum = {0, 0.133 * 0.8 * 0.6, 0};
    h1.position = {-5.257, 0.f, 0.f};

    compact_object::data h2;
    h2.t = compact_object::NEUTRON_STAR;
    h2.bare_mass = 0.15;
    h2.momentum = {0, 00.133 * 0.8 * 0.6, 0};
    h2.position = {5.257, 0.f, 0.f};

    objects = {h1, h2};
    #endif // REALLYBIG

    //#define NEUTRON_DOUBLE_COLLAPSE
    #ifdef NEUTRON_DOUBLE_COLLAPSE
    compact_object::data<float> h1;
    h1.t = compact_object::NEUTRON_STAR;
    h1.bare_mass = 0.35;
    h1.momentum = {0, 0.133 * 0.8 * 0.6, 0};
    h1.position = {-4.257, 0.f, 0.f};

    compact_object::data<float> h2;
    h2.t = compact_object::NEUTRON_STAR;
    h2.bare_mass = 0.35;
    h2.momentum = {0, -0.133 * 0.8 * 0.6, 0};
    h2.position = {4.257, 0.f, 0.f};

    objects = {h1, h2};
    #endif

    //#define NEUTRON_BLACK_HOLE_MERGE
    #ifdef NEUTRON_BLACK_HOLE_MERGE
    compact_object::data h1;
    h1.t = compact_object::BLACK_HOLE;
    h1.bare_mass = 0.4;
    h1.momentum = {0, 0.133 * 0.8 * 0.02, 0};
    h1.position = {-3.257, 0.f, 0.f};

    compact_object::data h2;
    h2.t = compact_object::NEUTRON_STAR;
    h2.bare_mass = 0.03;
    h2.momentum = {0, -0.133 * 0.8 * 0.10, 0};
    h2.position = {4.257, 0.f, 0.f};

    objects = {h1, h2};
    #endif // NEUTRON_BLACK_HOLE_MERGE

    ///defined for a compactness of 0.02 sigh
    //#define GAS_CLOUD_BLACK_HOLE
    #ifdef GAS_CLOUD_BLACK_HOLE
    compact_object::data h1;
    h1.t = compact_object::BLACK_HOLE;
    h1.bare_mass = 0.5;
    h1.momentum = {0, 0.133 * 0.8 * 0.02, 0};
    h1.position = {-5.257, 3.f, 0.f};

    compact_object::data h2;
    h2.t = compact_object::NEUTRON_STAR;
    h2.matter_compactness = 0.02f;
    h2.bare_mass = 0.09;
    h2.momentum = {0, -0.133 * 0.8 * 0.20, 0};
    h2.position = {4.257, 3.f, 0.f};

    objects = {h1, h2};
    #endif // GAS_CLOUD_BLACK_HOLE

    ///this is an extremely cool matter case
    //#define NEUTRON_ACCRETION
    #ifdef NEUTRON_ACCRETION
    compact_object::data h1;
    h1.t = compact_object::NEUTRON_STAR;
    h1.bare_mass = 0.2;
    h1.matter.compactness = 0.08;
    h1.momentum = {0, 0.133 * 0.8 * 0.2, 0};
    h1.position = {-6.257, 0.f, 0.f};
    h1.matter.colour = {1, 1, 1};

    compact_object::data h2;
    h2.t = compact_object::NEUTRON_STAR;
    h2.matter.compactness = 0.025f;
    h2.bare_mass = 0.2;
    h2.momentum = {0, -0.133 * 0.8 * 0.20, 0};
    h2.position = {5.257, 0.f, 0.f};
    h2.matter.colour = {1, 0.4f, 0};

    objects = {h1, h2};
    #endif // NEUTRON_ACCRETION

    //#define N_BODY
    #ifdef N_BODY
    compact_object::data base;
    base.t = compact_object::NEUTRON_STAR;
    base.bare_mass = 0.0225f;
    base.matter.compactness = 0.01f;
    base.matter.colour = {1,1,1};

    tensor<float, 3> colours[9] = {{1,1,1}, {1,0,0}, {0,1,0}, {0, 0, 1}, {1, 0.4f, 0.f}, {1, 0, 1}, {0, 1, 1}, {1, 1, 0}, {0.4f, 1.f, 1.f}};

    std::minstd_rand rng(1234);

    for(int i=0; i < 20; i++)
    {
        for(int kk=0; kk < 1024; kk++)
        {
            rng();
        }

        vec3f pos = {rand_det_s(rng, 0.f, 1.f), rand_det_s(rng, 0.f, 1.f), rand_det_s(rng, 0.f, 1.f)};
        vec3f momentum = {rand_det_s(rng, 0.f, 1.f), rand_det_s(rng, 0.f, 1.f), rand_det_s(rng, 0.f, 1.f)};

        pos = (pos - 0.5f) * ((get_c_at_max()/2.3));

        momentum = (momentum - 0.5f) * 0.01f * 0.25f;

        compact_object::data h = base;
        h.position = {pos.x(), pos.y(), pos.z()};
        h.momentum = {momentum.x(), momentum.y(), momentum.z()};
        h.matter.colour = colours[i % 9];

        objects.push_back(h);
    }

    #endif // N_BODY

    //#define REGULAR_MERGE
    #ifdef REGULAR_MERGE
    compact_object::data h1;
    h1.t = compact_object::NEUTRON_STAR;
    h1.bare_mass = 0.075;
    h1.momentum = {0, 0.133 * 0.8 * 0.1, 0};
    h1.position = {-4.257, 0.f, 0.f};
    h1.matter.colour = {1, 0, 0};

    compact_object::data h2;
    h2.t = compact_object::NEUTRON_STAR;
    h2.bare_mass = 0.075;
    h2.momentum = {0, -0.133 * 0.8 * 0.1, 0};
    h2.position = {4.257, 0.f, 0.f};
    h2.matter.colour = {0, 1, 0};

    objects = {h1, h2};
    #endif // REGULAR_MERGE

    #ifdef MERGE_THEN_COLLAPSE
    compact_object::data h1;
    h1.t = compact_object::NEUTRON_STAR;
    h1.bare_mass = 0.2;
    h1.momentum = {0, 0.133 * 0.8 * 0.01, 0};
    h1.position = {-4.257, 0.f, 0.f};
    h1.matter.colour = {1, 0, 0};
    h1.matter.compactness = 0.08;

    compact_object::data h2;
    h2.t = compact_object::NEUTRON_STAR;
    h2.bare_mass = 0.2;
    h2.momentum = {0, -0.133 * 0.8 * 0.01, 0};
    h2.position = {4.257, 0.f, 0.f};
    h2.matter.colour = {0, 1, 0};
    h2.matter.compactness = 0.08;

    objects = {h1, h2};
    #endif // MERGE_THEN_COLLAPSE

    #define PARTICLE_TEST
    #ifdef PARTICLE_TEST

    {
        particle_data data;

        float start = -3.f;
        float fin = -18.f;

        float total_mass = 0.5f;

        for(int i=0; i < 40; i++)
        {
            float anglef = (float)i / 40;

            float angle = 2 * M_PI * anglef;

            vec3f pos = {cos(angle) * 7.f, sin(angle) * 7.f, 0};

            data.positions.push_back(pos);
            data.velocities.push_back({0,0,0});
            data.masses.push_back(total_mass / 40);
        }

        data_opt = std::move(data);
    }
    #endif

    return get_bare_initial_conditions(clctx, cqueue, scale, objects, std::move(data_opt));
    #endif // BARE_BLACK_HOLES

    //#define USE_ADM_HOLE
    #ifdef USE_ADM_HOLE
    std::vector<adm_black_hole> adm_holes;

    //#define TEST_CASE
    #ifdef TEST_CASE
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

    return get_adm_initial_conditions(clctx, cqueue, scale, adm_holes);
    #endif // USE_ADM_HOLE

    assert(false);
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
void get_initial_conditions_eqs(equation_context& ctx, const std::vector<compact_object::data>& holes)
{
    tensor<value, 3> pos = {"ox", "oy", "oz"};

    value bl_conformal = calculate_conformal_guess(pos, holes);
    value u = dual_types::apply("buffer_index", "u_value", "ix", "iy", "iz", "dim");

    vec2i linear_indices[6] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};

    std::array<int, 9> arg_table
    {
        0, 1, 2,
        1, 3, 4,
        2, 4, 5,
    };

    tensor<value, 3, 3> bcAij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            int index = arg_table[i * 3 + j];

            bcAij.idx(i, j) = bidx("bcAij" + std::to_string(index), false, false);
        }
    }

    metric<value, 3, 3> Yij = pow(bl_conformal + u, 4) * get_flat_metric<value, 3>();

    value Y = Yij.det();

    ///https://arxiv.org/pdf/gr-qc/0206072.pdf see 10
    ///https://arxiv.org/pdf/gr-qc/9810065.pdf, 11
    ///phi
    value conformal_factor = (1/12.f) * log(Y);

    ctx.pin(conformal_factor);

    ///https://indico.cern.ch/event/505595/contributions/1183661/attachments/1332828/2003830/sperhake.pdf the york-lichnerowicz split
    tensor<value, 3, 3> Aij = pow(bl_conformal + u, -2) * bcAij;

    value gA = 1;
    //value gA = 1/(pow(bl_conformal + u, 2));
    ///https://arxiv.org/pdf/1304.3937.pdf
    //value gA = 2/(1 + pow(bl_conformal + 1, 4));

    bssn::init(ctx, Yij, Aij, gA);
}

inline
value matter_X_1(const value& X)
{
    return max(X, 0.00f);

    value LX = clamp(X, value{0.f}, value{1.f});

    float cutoff_X = 0.45f;
    float value_at_min = 0.35f;

    ///when this is 1, X == absolute min
    ///so eg if X == 0, we get 1
    ///if X == cutoff, we get 0
    value X_frac_to_absolute_min = (cutoff_X - LX) / cutoff_X;

    value modified_X_value = X_frac_to_absolute_min * (value_at_min - cutoff_X) + cutoff_X;

    return if_v(X >= cutoff_X,
         X,
         modified_X_value);

    /*float min_X = 0.4f;
    float max_X = 0.2f;

    value extra = max(X, 1.f) - min_X;

    value interpolated = (extra / (1.f - min_X)) * (max_X - min_X) + min_X;

    value interp = dual_types::if_v(extra > 0, interpolated, X);

    return interp;*/
}

inline
value matter_X_2(const value& X)
{
    return max(X, 0.00f);
}

inline
value get_cacheable_W(equation_context& ctx, standard_arguments& args, matter& matt)
{
    inverse_metric<value, 3, 3> icY = args.cY.invert();

    value W = 0.5f;
    int iterations = 5;

    for(int i=0; i < iterations; i++)
    {
        ctx.pin(W);

        W = w_next(W, matt.p_star, matter_X_2(args.get_X()), icY, matt.cS, matt.Gamma, matt.e_star);
    }

    return W;
}

namespace hydrodynamics
{
    void build_intermediate_variables_derivatives(equation_context& ctx)
    {
        standard_arguments args(ctx);
        matter matt(ctx);

        ctx.is_derivative_free = true;

        value sW = get_cacheable_W(ctx, args, matt);

        ctx.pin(sW);

        value pressure = matt.gamma_eos_from_e_star(matter_X_2(args.get_X()), sW);

        ctx.add("init_pressure", pressure);
        ctx.add("init_W", sW);
    }

    void build_artificial_viscosity(equation_context& ctx)
    {
        standard_arguments args(ctx);
        matter matt(ctx);

        value sW = bidx("hW", ctx.uses_linear, false);

        inverse_metric<value, 3, 3> icY = args.cY.invert();

        value PQvis = matt.calculate_PQvis(ctx, args.gA, args.gB, icY, matter_X_2(args.get_X()), sW);

        ctx.add("init_artificial_viscosity", PQvis);
    }

    void build_equations(equation_context& ctx)
    {
        standard_arguments args(ctx);
        matter matt(ctx);

        inverse_metric<value, 3, 3> icY = args.cY.invert();

        value sW = bidx("hW", ctx.uses_linear, false);

        tensor<value, 3, 3> cSk_vi = matt.cSk_vi(icY, args.gA, args.gB, matter_X_2(args.get_X()), sW);

        tensor<value, 3> p_star_vi = matt.p_star_vi(icY, args.gA, args.gB, matter_X_2(args.get_X()), sW);
        tensor<value, 3> e_star_vi = matt.e_star_vi(icY, args.gA, args.gB, matter_X_2(args.get_X()), sW);

        value P = bidx("pressure", ctx.uses_linear, false);

        value lhs_dtp_star = 0;

        for(int i=0; i < 3; i++)
        {
            lhs_dtp_star += diff1(ctx, p_star_vi.idx(i), i);
        }

        value lhs_dte_star = 0;

        for(int i=0; i < 3; i++)
        {
            lhs_dte_star += diff1(ctx, e_star_vi.idx(i), i);
        }

        //#define IMMEDIATE_UPDATE
        #ifdef IMMEDIATE_UPDATE
        matter matt2 = matt;
        matt2.p_star = "fin_p_star";
        #else
        matter& matt2 = matt;
        #endif

        //value sW = get_cacheable_W(ctx, args, matt2);

        value rhs_dte_star = matt2.estar_vi_rhs(ctx, args.gA, args.gB, icY, matter_X_1(args.get_X()), sW);

        /*ctx.add("DBG_RHS_DTESTAR", rhs_dte_star);

        ctx.add("DBG_LHS_DTESTAR", lhs_dte_star);
        ctx.add("DBG_ESTARVI0", e_star_vi.idx(0));
        ctx.add("DBG_ESTARVI1", e_star_vi.idx(1));
        ctx.add("DBG_ESTARVI2", e_star_vi.idx(2));*/

        tensor<value, 3> lhs_dtSk;

        for(int k=0; k < 3; k++)
        {
            value sum = 0;

            for(int i=0; i < 3; i++)
            {
                sum += diff1(ctx, cSk_vi.idx(k, i), i);
            }

            lhs_dtSk.idx(k) = sum;
        }

        tensor<value, 3> rhs_dtSk = matt2.cSkvi_rhs(ctx, icY, args.gA, args.gB, matter_X_2(args.get_X()), args.get_dX(), P, sW);

        value dtp_star = -lhs_dtp_star;
        value dte_star = -lhs_dte_star + rhs_dte_star;

        ctx.add("lhs_dtsk0", -lhs_dtSk.idx(0));
        ctx.add("rhs_dtsk0", rhs_dtSk.idx(0));

        tensor<value, 3> dtSk = -lhs_dtSk + rhs_dtSk;

        ctx.add("init_dtp_star", dtp_star);
        ctx.add("init_dte_star", dte_star);

        for(int i=0; i < 3; i++)
        {
            ctx.add("init_dtSk" + std::to_string(i), dtSk.idx(i));
        }

        //ctx.add("e_star_p_limit", matt.p_star_below_e_star_threshold());
        //ctx.add("e_star_p_value", matt.e_star_clamped());
        ctx.add("p_star_max", matt.p_star_max());
    }

    void build_advection(equation_context& ctx)
    {
        standard_arguments args(ctx);
        matter matt(ctx);

        inverse_metric<value, 3, 3> icY = args.cY.invert();

        value sW = bidx("hW", ctx.uses_linear, false);

        tensor<value, 3> vi_upper = matt.get_v_upper(icY, args.gA, args.gB, matter_X_2(args.get_X()), sW);

        value quantity = max(bidx("quantity_in", ctx.uses_linear, false), 0.f);

        value fin = 0;

        for(int i=0; i < 3; i++)
        {
            value to_diff = quantity * vi_upper.idx(i);

            fin += diff1(ctx, to_diff, i);
        }

        /*value fin = 0;

        for(int i=0; i < 3; i++)
        {
            fin += vi_upper.idx(i) * diff1(ctx, quantity, i);
        }*/

        ctx.add("HYDRO_ADVECT", -fin);
    }
}


struct eularian_matter : matter_interop
{
    virtual value calculate_adm_S(equation_context& ctx, standard_arguments& args) override
    {
        matter matt(ctx);

        value W = get_cacheable_W(ctx, args, matt);

        ctx.pin(W);

        return matt.calculate_adm_S(args.cY, args.cY.invert(), args.get_X(), W);
    }

    virtual value calculate_adm_p(equation_context& ctx, standard_arguments& args) override
    {
        matter matt(ctx);

        value W = get_cacheable_W(ctx, args, matt);

        ctx.pin(W);

        return matt.calculate_adm_p(args.get_X(), W);
    }

    virtual tensor<value, 3, 3> calculate_adm_X_Sij(equation_context& ctx, standard_arguments& args) override
    {
        matter matt(ctx);

        value W = get_cacheable_W(ctx, args, matt);

        ctx.pin(W);

        return matt.calculate_adm_X_Sij(args.get_X(), W, args.cY);
    }

    virtual tensor<value, 3> calculate_adm_Si(equation_context& ctx, standard_arguments& args) override
    {
        matter matt(ctx);

        return matt.calculate_adm_Si(args.get_X());
    }
};


void build_sommerfeld_thin(equation_context& ctx)
{
    ctx.order = 1;
    ctx.always_directional_derivatives = true;
    ctx.use_precise_differentiation = true;
    standard_arguments args(ctx);

    tensor<value, 3> pos = {"ox", "oy", "oz"};

    value r = pos.length();

    auto sommerfeld = [&](const value& f, const value& f0, const value& v)
    {
        value sum = 0;

        for(int i=0; i < 3; i++)
        {
            sum += (v * pos.idx(i) / r) * diff1(ctx, f, i);
        }

        return -sum - v * (f - f0) / r;
    };

    value in = bidx("input", false, false);
    value asym = "asym";
    value v = "speed";

    value out = sommerfeld(in, asym, v);

    ctx.add("sommer_thin_out", out);
}

///algebraic_constraints
///https://arxiv.org/pdf/1507.00570.pdf says that the cY modification is bad
inline
void build_constraints(equation_context& ctx)
{
    standard_arguments args(ctx);

    metric<value, 3, 3> cY = args.cY;
    tensor<value, 3, 3> cA = args.cA;

    value det_cY_pow = pow(cY.det(), 1.f/3.f);

    ///it occurs to me here that the error is extremely non trivial
    ///and that rather than discarding it entirely, it could be applied to X as that is itself a scaling factor
    ///for cY
    metric<value, 3, 3> fixed_cY = cY / det_cY_pow;

    /*for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            fixed_cY.idx(i, j) = dual_types::clamp(fixed_cY.idx(i, j), value{-2}, value{2});
        }
    }*/

    /*fixed_cY.idx(0, 0) = max(fixed_cY.idx(0, 0), value{0.001f});
    fixed_cY.idx(1, 1) = max(fixed_cY.idx(1, 1), value{0.001f});
    fixed_cY.idx(2, 2) = max(fixed_cY.idx(2, 2), value{0.001f});*/

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
    tensor<value, 3, 3> fixed_cA = trace_free(cA, fixed_cY, fixed_cY.invert());

    ///this seems to work well (https://arxiv.org/pdf/0709.3559.pdf) (b.49), but needs testing with bbh which is why its disabled
    //tensor<value, 3, 3> fixed_cA = cA / det_cY_pow;
    #endif

    /*for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            fixed_cA.idx(i, j) = dual_types::clamp(fixed_cA.idx(i, j), value{-2}, value{2});
        }
    }*/

    for(int i=0; i < 6; i++)
    {
        vec2i idx = linear_indices[i];

        ctx.add("fix_cY" + std::to_string(i), fixed_cY.idx(idx.x(), idx.y()));
        ctx.add("fix_cA" + std::to_string(i), fixed_cA.idx(idx.x(), idx.y()));
    }

    ctx.add("CY_DET", det_cY_pow);
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

void build_intermediate_thin_directional(equation_context& ctx)
{
    ctx.always_directional_derivatives = true;
    ctx.use_precise_differentiation = true;

    standard_arguments args(ctx);

    value buffer = dual_types::apply("buffer_index", "buffer", "ix", "iy", "iz", "dim");

    value v1 = diff1(ctx, buffer, 0);
    value v2 = diff1(ctx, buffer, 1);
    value v3 = diff1(ctx, buffer, 2);

    ctx.add("init_buffer_intermediate0_directional", v1);
    ctx.add("init_buffer_intermediate1_directional", v2);
    ctx.add("init_buffer_intermediate2_directional", v3);
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
    tensor<value, 3, 3, 3> christoff2 = christoffel_symbols_2(ctx, met, inverse);

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

void build_momentum_constraint(matter_interop& interop, equation_context& ctx, bool use_matter)
{
    tensor<value, 3> Mi = bssn::calculate_momentum_constraint(interop, ctx, use_matter);

    #if defined(BETTERDAMP_DTCAIJ) || defined(DAMP_DTCAIJ) || defined(AIJ_SIGMA)
    #define CALCULATE_MOMENTUM_CONSTRAINT
    #endif // defined

    for(int i=0; i < 3; i++)
    {
        ctx.add("init_momentum" + std::to_string(i), Mi.idx(i));
    }
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
    ctx.order = 4;
    ctx.use_precise_differentiation = false;
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

    tensor<value, 3, 3, 3> christoff1 = christoffel_symbols_1(ctx, args.cY);
    tensor<value, 3, 3, 3> christoff2 = christoffel_symbols_2(ctx, args.cY, icY);

    ctx.pin(christoff1);
    ctx.pin(christoff2);

    ///NEEDS SCALING???
    vec<3, value> pos;
    pos.x() = "offset.x";
    pos.y() = "offset.y";
    pos.z() = "offset.z";

    tensor<value, 3, 3> xgARij = bssn::calculate_xgARij(ctx, args, icY, christoff1, christoff2);

    tensor<value, 3, 3> Rij = xgARij / (args.gA * args.get_X());

    ctx.pin(Rij);

    tensor<value, 3, 3> Kij = args.Kij;
    tensor<value, 3, 3> unpinned_Kij = Kij;

    ctx.pin(Kij);

    metric<value, 3, 3> Yij = args.Yij;

    ctx.pin(Yij);

    inverse_metric<value, 3, 3> iYij = args.iYij;
    ctx.pin(iYij);

    tensor<value, 3> dX = args.get_dX();

    //inverse_metric<value, 3, 3> iYij = Yij.invert();

    //ctx.pin(iYij);

    //auto christoff_Y = christoffel_symbols_2(Yij, iYij);

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
                    sum += -args.cY.idx(j, k) * icY.idx(i, m) * dX.idx(m);
                }

                christoff_Y.idx(i, j, k) = christoff2.idx(i, j, k) - (1 / (2 * args.get_X())) *
                (kronecker.idx(i, k) * dX.idx(j) +
                 kronecker.idx(i, j) * dX.idx(k) +
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

    value s = pos.length();

    ///https://arxiv.org/pdf/1606.02532.pdf (94)
    ///s is a scalar, so I think this is the covariant derivative?
    tensor<value, 3> s_j = {s.differentiate("offset.x"), s.differentiate("offset.y"), s.differentiate("offset.z")};

    value s_j_len = 0;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            s_j_len += iYij.idx(i, j) * s_j.idx(i) * s_j.idx(j);
        }
    }

    s_j_len = sqrt(s_j_len);

    tensor<value, 3> e_s;

    for(int i=0; i < 3; i++)
    {
        value sum = 0;

        for(int j=0; j < 3; j++)
        {
            sum += iYij.idx(i, j) * s_j.idx(j) / s_j_len;
        }

        e_s.idx(i) = sum;
    }


    ///https://arxiv.org/pdf/gr-qc/0610128.pdf
    ///https://arxiv.org/pdf/gr-qc/0104063.pdf 5.7. I already have code for doing this but lets stay exact

    vec<3, value> v1ai = {e_s.idx(0), e_s.idx(1), e_s.idx(2)};
    vec<3, value> v2ai = {-pos.y(), pos.x(), 0};

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

    auto [v1a, v2a, v3a] = orthonormalise(ctx, v1ai, v2ai, v3ai, Yij);

    ctx.pin(v1a);
    ctx.pin(v2a);
    ctx.pin(v3a);

    dual_types::complex<value> unit_i = dual_types::unit_i();

    tensor<dual_types::complex<value>, 4> mu;

    for(int i=1; i < 4; i++)
    {
        mu.idx(i) = (1.f/sqrt(2.f)) * (v2a[i - 1] + unit_i * v3a[i - 1]);
    }

    ///https://en.wikipedia.org/wiki/Newman%E2%80%93Penrose_formalism
    tensor<dual_types::complex<value>, 4> mu_dash;

    for(int i=0; i < 4; i++)
    {
        mu_dash.idx(i) = dual_types::conjugate(mu.idx(i));
    }

    tensor<dual_types::complex<value>, 4> mu_dash_p = tensor_project_upper(mu_dash, args.gA, args.gB);

    tensor<value, 3, 3, 3> raised_eijk = raise_index(raise_index(eijk_tensor, iYij, 1), iYij, 2);

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
                        ///this appears to be the specific term which is bad and full of reflections (!!!!)
                        ///so. If we have k_sum_2 *or* everything else, neither suffer from reflections
                        ///but both together seem to. Perhaps mu_dash_p is the issue?
                        k_sum_2 += unit_i * raised_eijk.idx(i, k, l) * cdKij.idx(l, j, k);
                    }

                    k_sum_1 += Kij.idx(i, k) * raise_index(Kij, iYij, 0).idx(k, j);
                }

                dual_types::complex<value> inner_sum = -Rij.idx(i, j) - args.K * Kij.idx(i, j) + k_sum_1 + k_sum_2;

                ///mu is a 4 vector, but we use it spatially
                ///this exposes the fact that i really runs from 1-4 instead of 0-3
                sum += inner_sum * mu_dash_p.idx(i + 1) * mu_dash_p.idx(j + 1);
            }
        }

        w4 = sum;
    }

    value length = pos.length();

    ///this... seems like I'm missing something
    value R = sqrt((4 * M_PI * length * length) / (4 * M_PI));

    w4 = R * w4;

    //std::cout << "MUr " << type_to_string(mu.idx(1).real) << std::endl;
    //std::cout << "MUi " << type_to_string(mu.idx(1).imaginary) << std::endl;

    ctx.add("w4_real", w4.real);
    ctx.add("w4_complex", w4.imaginary);
    //ctx.add("w4_debugr", mu.idx(1).real);
    //ctx.add("w4_debugi", mu.idx(1).imaginary);

    //vec<4, dual_types::complex<value>> mu = (1.f/sqrt(2)) * (thetau + i * phiu);
}
#endif // 0

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
    ctx.use_precise_differentiation = false;

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

    pixel_direction = pixel_direction.norm();

    metric<value, 4, 4> real_metric = calculate_real_metric(args.Yij, args.gA, args.gB);

    ctx.pin(real_metric);

    frame_basis basis = calculate_frame_basis(ctx, real_metric);

    vec<4, value> e0 = basis.v1;
    vec<4, value> e1 = basis.v2;
    vec<4, value> e2 = basis.v3;
    vec<4, value> e3 = basis.v4;

    ctx.pin(e0);
    ctx.pin(e1);
    ctx.pin(e2);
    ctx.pin(e3);

    vec<4, value> basis_x = e2;
    vec<4, value> basis_y = e3;
    vec<4, value> basis_z = e1;

    bool should_orient = true;

    if(should_orient)
    {
        tetrad tet = {e0, e1, e2, e3};
        inverse_tetrad itet = get_tetrad_inverse(tet);

        ctx.pin(itet.e[0]);
        ctx.pin(itet.e[1]);
        ctx.pin(itet.e[2]);
        ctx.pin(itet.e[3]);

        vec<4, value> cartesian_basis_x = {0, 1, 0, 0};
        vec<4, value> cartesian_basis_y = {0, 0, 1, 0};
        vec<4, value> cartesian_basis_z = {0, 0, 0, 1};

        vec<4, value> tE1 = coordinate_to_tetrad_basis(cartesian_basis_y, itet);
        vec<4, value> tE2 = coordinate_to_tetrad_basis(cartesian_basis_x, itet);
        vec<4, value> tE3 = coordinate_to_tetrad_basis(cartesian_basis_z, itet);

        ctx.pin(tE1);
        ctx.pin(tE2);
        ctx.pin(tE3);

        ortho_result result = orthonormalise(tE1.yzw(), tE2.yzw(), tE3.yzw());

        basis_x = {0, result.v2.x(), result.v2.y(), result.v2.z()};
        basis_y = {0, result.v1.x(), result.v1.y(), result.v1.z()};
        basis_z = {0, result.v3.x(), result.v3.y(), result.v3.z()};
        ///basis_t == e0
    }

    tetrad oriented = {e0, basis_x, basis_y, basis_z};

    /*
    {
        float4 observer_velocity = get_timelike_vector(cartesian_basis_speed, 1, e0, e1, e2, e3);

        float lorentz[16] = {};

        #ifndef GENERIC_BIG_METRIC
        float g_metric[4] = {};
        calculate_metric_generic(at_metric, g_metric, cfg);
        calculate_lorentz_boost(e0, observer_velocity, g_metric, lorentz);
        #else
        float g_metric_big[16] = {0};
        calculate_metric_generic_big(at_metric, g_metric_big, cfg);
        calculate_lorentz_boost_big(e0, observer_velocity, g_metric_big, lorentz);
        #endif // GENERIC_METRIC

        e0 = observer_velocity;
        e1 = tensor_contract(lorentz, e1);
        e2 = tensor_contract(lorentz, e2);
        e3 = tensor_contract(lorentz, e3);
    }
    */

    vec<4, value> pixel_x = pixel_direction.x() * oriented.e[1];
    vec<4, value> pixel_y = pixel_direction.y() * oriented.e[2];
    vec<4, value> pixel_z = pixel_direction.z() * oriented.e[3];
    vec<4, value> pixel_t = -oriented.e[0];

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

    tensor<value, 4> velocity_lower = lower_index(lightray_velocity_t, real_metric, 0);

    tensor<value, 3> adm_V_lower = {velocity_lower.idx(1), velocity_lower.idx(2), velocity_lower.idx(3)};

    tensor<value, 3> adm_V_higher = raise_index(adm_V_lower, args.Yij.invert(), 0);

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

///https://arxiv.org/pdf/1208.3927.pdf (28a)
void loop_geodesics(equation_context& ctx, vec3f dim)
{
    ctx.order = 1;
    ctx.use_precise_differentiation = false;

    standard_arguments args(ctx);

    ctx.pin(args.Kij);
    ctx.pin(args.Yij);

    float universe_length = (dim/2.f).max_elem();

    value scale = "scale";

    ctx.add("universe_size", universe_length * scale);

    //tensor<value, 3> X_upper = {"lp1", "lp2", "lp3"};
    tensor<value, 3> V_upper = {"V0", "V1", "V2"};

    inverse_metric<value, 3, 3> iYij = args.iYij;

    inverse_metric<value, 3, 3> icY = args.cY.invert();

    tensor<value, 3> dX = args.get_dX();

    tensor<value, 3, 3, 3> conformal_christoff2 = christoffel_symbols_2(ctx, args.cY, icY);

    tensor<value, 3, 3, 3> full_christoffel2 = get_full_christoffel2(args.get_X(), dX, args.cY, icY, conformal_christoff2);

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

            V_upper_diff.idx(i) += args.gA * V_upper.idx(j) * (V_upper.idx(i) * (dlog_gA - kjvk) + 2 * raise_index(args.Kij, iYij, 0).idx(i, j) - christoffel_sum)
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
    tensor<value, 3> p_lower = lower_index(V_upper, args.Yij, 0);

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

    ctx.add("GET_X_DS", args.get_X());

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

/*void build_hamiltonian_constraint(equation_context& ctx)
{
    standard_arguments args(ctx);

    ctx.add("HAMILTONIAN", calculate_hamiltonian(ctx, args));
}*/

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
    {
        float w = 0.5f;
        float w1 = 0.5f;

        for(int i=0; i < 50; i++)
        {
            w =              w_next_interior(w, 0.234f, 1.12f, 0.25f, 2.f, 0.1f);
            w1 = w_next_interior_nonregular(w1, 0.234f, 1.12f, 0.25f, 2.f, 0.1f);
        }

        assert(approx_equal(w, w1));

        printf("reg check %f %f\n", w, w1);
    }

    {
        float w = 0.5f;

        for(int i=0; i < 50; i++)
        {
            ///by the property that w = p*au0, perhaps set to 0 if p* < crit
            w = w_next_interior(w, 0.f, 0.f, 0.f, 2, 0.f);

            assert(isfinite(w));
        }
    }

    /*{
        for(float p1 = 0; p1 <= 1; p1 += 0.01f)
        {
            for(float p2 = 0; p2 <= 1; p2 += 0.01f)
            {
                for(float p3 = 0; p3 <= 1; p3 += 0.01f)
                {
                    for(float p4 = 0; p4 <= 1; p4 += 0.01f)
                    {
                        float w = 0.5f;

                        for(int i=0; i < 6; i++)
                        {
                            w = w_next_interior(w, p1, p2, p3, 2, p4);
                            assert(isfinite(w));
                        }
                    }
                }
            }
        }
    }*/

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

    std::string argument_string = "-I ./ -cl-std=CL2.0 -cl-mad-enable ";
    std::string hydro_argument_string = argument_string;

    ///must be a multiple of DIFFERENTIATION_WIDTH
    vec3i size = {255, 255, 255};
    //vec3i size = {250, 250, 250};
    //float c_at_max = 160;
    float c_at_max = get_c_at_max();
    float scale = calculate_scale(c_at_max, size);
    vec3f centre = {size.x()/2.f, size.y()/2.f, size.z()/2.f};

    initial_conditions holes = setup_dynamic_initial_conditions(clctx.ctx, clctx.cqueue, centre, scale);

    for(auto& obj : holes.objs)
    {
        tensor<float, 3> pos = world_to_voxel(obj.position, size, scale);

        printf("Voxel pos %f %f %f\n", pos.x(), pos.y(), pos.z());
    }

    cl::buffer u_arg(clctx.ctx);

    cl::program evolve_prog(clctx.ctx, "evolve_points.cl");
    evolve_prog.build(clctx.ctx, argument_string + "-DBORDER_WIDTH=" + std::to_string(BORDER_WIDTH) + " ");

    clctx.ctx.register_program(evolve_prog);

    evolution_points evolve_points = generate_evolution_points(clctx.ctx, clctx.cqueue, scale, size);

    //sandwich_result sandwich(clctx.ctx);

    matter_initial_vars matter_vars(clctx.ctx);

    auto u_thread = [c_at_max, scale, size, &clctx, &u_arg, &holes, &matter_vars]()
    {
        steady_timer u_time;

        cl::command_queue cqueue(clctx.ctx);

        superimposed_gpu_data super(clctx.ctx, cqueue, size);
        super.pull_all(clctx.ctx, cqueue, holes.objs, holes.particles, scale, size);

        u_arg = super.u_arg;

        matter_vars.bcAij = super.bcAij;
        matter_vars.superimposed_tov_phi = super.tov_phi;
        matter_vars.pressure_buf = super.pressure_buf;
        matter_vars.rho_buf = super.rho_buf;
        matter_vars.rhoH_buf = super.rhoH_buf;
        matter_vars.p0_buf = super.p0_buf;
        matter_vars.Si_buf = super.Si_buf;
        matter_vars.colour_buf = super.colour_buf;

        cqueue.block();

        printf("U time %f\n", u_time.get_elapsed_time_s());
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
    get_initial_conditions_eqs(ctx1, holes.objs);

    //eularian_matter interop;

    matter_interop* interop = new matter_interop();

    if(holes.use_matter)
    {
        interop = new eularian_matter();
    }

    if(holes.use_particles)
    {
        interop = new particle_matter_interop();
    }

    equation_context dtcY;
    bssn::build_cY(dtcY);

    equation_context dtcA;
    bssn::build_cA(*interop, dtcA, holes.use_matter || holes.use_particles);

    equation_context dtcGi;
    bssn::build_cGi(*interop, dtcGi, holes.use_matter || holes.use_particles);

    equation_context dtK;
    bssn::build_K(*interop, dtK, holes.use_matter || holes.use_particles);

    equation_context dtX;
    bssn::build_X(dtX);

    equation_context dtgA;
    bssn::build_gA(dtgA);

    equation_context dtgB;
    bssn::build_gB(dtgB);

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

    equation_context ctxdirectional;
    build_intermediate_thin_directional(ctxdirectional);

    equation_context ctx12;
    build_intermediate_thin_cY5(ctx12);

    equation_context ctx13;
    build_momentum_constraint(*interop, ctx13, holes.use_matter || holes.use_particles);

    equation_context ctx14;
    //build_hamiltonian_constraint(ctx14);

    equation_context ctxsommerthin;
    build_sommerfeld_thin(ctxsommerthin);

    ctx1.build(argument_string, 0);
    ctx4.build(argument_string, 3);
    ctx5.build(argument_string, 4);
    ctx6.build(argument_string, 5);
    ctx7.build(argument_string, 6);
    ctx10.build(argument_string, 9);
    ctx11.build(argument_string, 10);
    ctx12.build(argument_string, 11);
    ctx13.build(argument_string, 12);
    //ctx14.build(argument_string, "unused1");
    ctxdirectional.build(argument_string, "directional");
    ctxsommerthin.build(argument_string, "sommerthin");

    dtcY.build(argument_string, "tcy");
    dtcA.build(argument_string, "tca");
    dtcGi.build(argument_string, "tcgi");
    dtK.build(argument_string, "tk");
    dtX.build(argument_string, "tx");
    dtgA.build(argument_string, "tga");
    dtgB.build(argument_string, "tgb");

    argument_string += "-DBORDER_WIDTH=" + std::to_string(BORDER_WIDTH) + " ";
    hydro_argument_string += "-DBORDER_WIDTH=" + std::to_string(BORDER_WIDTH) + " ";

    if(holes.use_matter)
    {
        argument_string += "-DRENDER_MATTER ";
        argument_string += "-DSOMMER_MATTER ";
    }

    if(holes.use_particles)
    {
        argument_string += "-DTRACE_MATTER_P -DRENDER_MATTER_P ";
    }

    #ifdef USE_GBB
    argument_string += "-DUSE_GBB ";
    #endif // USE_GBB

    ///seems to make 0 difference to instability time
    #define USE_HALF_INTERMEDIATE
    #ifdef USE_HALF_INTERMEDIATE
    int intermediate_data_size = sizeof(cl_half);
    argument_string += "-DDERIV_PRECISION=half ";
    hydro_argument_string += "-DDERIV_PRECISION=half ";
    #else
    int intermediate_data_size = sizeof(cl_float);
    argument_string += "-DDERIV_PRECISION=float ";
    hydro_argument_string += "-DDERIV_PRECISION=float ";
    #endif

    {
        std::ofstream out("args.txt");
        out << argument_string;
    }

    std::cout << "Size " << argument_string.size() << std::endl;

    cpu_mesh_settings base_settings;

    #ifdef USE_HALF_INTERMEDIATE
    base_settings.use_half_intermediates = true;
    #else
    base_settings.use_half_intermediates = false;
    #endif // USE_HALF_INTERMEDIATE

    bool use_matter_colour = false;

    #ifdef USE_GBB
    base_settings.use_gBB = true;
    #endif // USE_GBB

    #ifdef CALCULATE_MOMENTUM_CONSTRAINT
    base_settings.calculate_momentum_constraint = true;
    #endif // CALCULATE_MOMENTUM_CONSTRAINT

    float gauge_wave_speed = sqrt(2.f);

    std::vector<buffer_descriptor> buffers = {
        {"cY0", "evolve_cY", cpu_mesh::dissipate_low, 1, 1},
        {"cY1", "evolve_cY", cpu_mesh::dissipate_low, 0, 1},
        {"cY2", "evolve_cY", cpu_mesh::dissipate_low, 0, 1},
        {"cY3", "evolve_cY", cpu_mesh::dissipate_low, 1, 1},
        {"cY4", "evolve_cY", cpu_mesh::dissipate_low, 0, 1},
        {"cY5", "evolve_cY", cpu_mesh::dissipate_low, 1, 1},

        {"cA0", "evolve_cA", cpu_mesh::dissipate_high, 0, 1},
        {"cA1", "evolve_cA", cpu_mesh::dissipate_high, 0, 1},
        {"cA2", "evolve_cA", cpu_mesh::dissipate_high, 0, 1},
        {"cA3", "evolve_cA", cpu_mesh::dissipate_high, 0, 1},
        {"cA4", "evolve_cA", cpu_mesh::dissipate_high, 0, 1},
        {"cA5", "evolve_cA", cpu_mesh::dissipate_high, 0, 1},

        {"cGi0", "evolve_cGi", cpu_mesh::dissipate_low, 0, 1},
        {"cGi1", "evolve_cGi", cpu_mesh::dissipate_low, 0, 1},
        {"cGi2", "evolve_cGi", cpu_mesh::dissipate_low, 0, 1},

        {"K", "evolve_K", cpu_mesh::dissipate_high, 0, 1},
        {"X", "evolve_X", cpu_mesh::dissipate_low, 1, 1},

        {"gA", "evolve_gA", cpu_mesh::dissipate_gauge, 1, gauge_wave_speed},
        {"gB0", "evolve_gB", cpu_mesh::dissipate_gauge, 0, gauge_wave_speed},
        {"gB1", "evolve_gB", cpu_mesh::dissipate_gauge, 0, gauge_wave_speed},
        {"gB2", "evolve_gB", cpu_mesh::dissipate_gauge, 0, gauge_wave_speed},
    };

    #ifdef USE_GBB
    buffers.push_back({"gBB0", "evolve_cGi", cpu_mesh::dissipate_gauge, 0, gauge_wave_speed});
    buffers.push_back({"gBB1", "evolve_cGi", cpu_mesh::dissipate_gauge, 0, gauge_wave_speed});
    buffers.push_back({"gBB2", "evolve_cGi", cpu_mesh::dissipate_gauge, 0, gauge_wave_speed});
    #endif // USE_GBB

    std::vector<plugin*> plugins;

    if(holes.use_matter)
    {
        eularian_hydrodynamics* hydro = new eularian_hydrodynamics(clctx.ctx);

        hydro->use_colour = use_matter_colour;

        plugins.push_back(hydro);
    }

    if(holes.use_particles)
    {
        particle_dynamics* particles = new particle_dynamics(clctx.ctx);

        particles->add_particles(std::move(holes.particles));

        plugins.push_back(particles);
    }

    for(plugin* p : plugins)
    {
        auto extra = p->get_buffers();

        for(auto& i : extra)
        {
            std::cout << "Added " << i.name << std::endl;

            buffers.push_back(i);
        }
    }

    if(use_matter_colour)
    {
        argument_string += "-DHAS_COLOUR ";
        hydro_argument_string += "-DHAS_COLOUR ";
    }

    {
        std::string generated_arglist = "#define GET_ARGLIST(a, p) ";

        for(const buffer_descriptor& desc : buffers)
        {
            generated_arglist += "a p##" + desc.name + ", ";
        }

        while(generated_arglist.back() == ',' || generated_arglist.back() == ' ')
            generated_arglist.pop_back();

        file::write("./generated_arglist.cl", generated_arglist, file::mode::TEXT);
    }

    cl::program prog(clctx.ctx, "cl.cl");
    prog.build(clctx.ctx, argument_string);

    bool joined = false;

    if(holes.use_matter)
    {
        printf("Begin hydro\n");

        equation_context hydro_intermediates;
        hydrodynamics::build_intermediate_variables_derivatives(hydro_intermediates);

        printf("Post interm\n");

        equation_context hydro_viscosity;
        hydrodynamics::build_artificial_viscosity(hydro_viscosity);

        printf("Post viscosity\n");

        equation_context hydro_final;
        hydrodynamics::build_equations(hydro_final);

        printf("Post main hydro equations\n");

        equation_context hydro_advect;
        hydrodynamics::build_advection(hydro_advect);

        printf("Post advect\n");

        equation_context build_hydro_quantities;
        construct_hydrodynamic_quantities(build_hydro_quantities, holes.objs);

        printf("End hydro\n");

        hydro_intermediates.build(hydro_argument_string, "hydrointermediates");
        hydro_viscosity.build(hydro_argument_string, "hydroviscosity");
        hydro_final.build(hydro_argument_string, "hydrofinal");
        hydro_advect.build(hydro_argument_string, "hydroadvect");
        build_hydro_quantities.build(hydro_argument_string, "hydroconvert");

        cl::program hydro_prog(clctx.ctx, "hydrodynamics.cl");
        hydro_prog.build(clctx.ctx, hydro_argument_string);

        async_u.join();
        joined = true;

        clctx.ctx.register_program(hydro_prog);
    }

    if(!joined)
        async_u.join();

    for(plugin* p : plugins)
    {
        eularian_hydrodynamics* ptr = dynamic_cast<eularian_hydrodynamics*>(p);

        if(ptr == nullptr)
            continue;

        ptr->grab_resources(matter_vars, u_arg);
    }

    #if 0
    for(int i=0; i < (int)holes.holes.size(); i++)
    {
        printf("Black hole test mass %f %i\n", get_nonspinning_adm_mass(clctx.cqueue, i, holes.holes, size, scale, u_arg), i);
    }
    #endif // 0

    ///this is not thread safe
    clctx.ctx.register_program(prog);

    texture_settings tsett;
    tsett.width = width;
    tsett.height = height;
    tsett.is_srgb = false;
    tsett.generate_mipmaps = false;

    texture tex;
    tex.load_from_memory(tsett, nullptr);

    cl::gl_rendertexture rtex{clctx.ctx};
    rtex.create_from_texture(tex.handle);

    cl::buffer ray_buffer{clctx.ctx};
    ray_buffer.alloc(sizeof(cl_float) * 20 * width * height);

    cl::buffer rays_terminated(clctx.ctx);
    rays_terminated.alloc(sizeof(cl_float) * 20 * width * height);

    cl::buffer ray_count{clctx.ctx};
    ray_count.alloc(sizeof(cl_int));

    cl::buffer ray_count_terminated(clctx.ctx);
    ray_count_terminated.alloc(sizeof(cl_int));

    cl::buffer texture_coordinates(clctx.ctx);
    texture_coordinates.alloc(sizeof(cl_float2) * width * height);

    cpu_mesh base_mesh(clctx.ctx, clctx.cqueue, {0,0,0}, size, base_settings, evolve_points, buffers, plugins);

    cl_float time_elapsed_s = 0;

    thin_intermediates_pool thin_pool;

    gravitational_wave_manager wave_manager(clctx.ctx, size, c_at_max, scale);

    base_mesh.init(clctx.ctx, clctx.cqueue, thin_pool, u_arg, matter_vars.bcAij);

    matter_vars = matter_initial_vars(clctx.ctx);
    u_arg = cl::buffer(clctx.ctx);

    std::vector<float> real_graph;
    std::vector<float> real_decomp;
    std::vector<float> imaginary_decomp;

    int steps = 0;

    bool run = false;
    bool should_render = false;

    vec3f camera_pos = {0, 0, -c_at_max/2.f + 1};
    quat camera_quat;
    camera_quat.load_from_axis_angle({1, 0, 0, 0});

    int rendering_method = 1;

    bool trapezoidal_init = false;

    bool pao = false;

    bool render_skipping = false;
    int skip_frames = 5;
    int current_skip_frame = 0;

    clctx.cqueue.block();

    std::cout << "Init time " << time_to_main.get_elapsed_time_s() << std::endl;

    mqueue.block();

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

        auto buffer_size = rtex.size<2>();

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

            tex.load_from_memory(new_sett, nullptr);

            rtex.create_from_texture(tex.handle);

            ray_buffer.alloc(sizeof(cl_float) * 20 * width * height);
            rays_terminated.alloc(sizeof(cl_float) * 20 * width * height);

            texture_coordinates.alloc(sizeof(cl_float2) * width * height);
        }

        rtex.acquire(clctx.cqueue);

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
                ImGui::PlotLines("w4      ", real_graph.data(), real_graph.size());
                ImGui::PopItemWidth();
            }

            ImGui::Checkbox("pao", &pao);

            if(ImGui::Button("Clear Waves"))
            {
                real_decomp.clear();
                imaginary_decomp.clear();
            }

            if(real_decomp.size() > 0)
            {
                ImGui::PlotLines("w4_l2_m2_re", real_decomp.data(), real_decomp.size(), 0, nullptr, FLT_MAX, FLT_MAX, ImVec2(400, 100));
            }

            if(imaginary_decomp.size() > 0)
            {
                ImGui::PlotLines("w4_l2_m2_im", real_decomp.data(), real_decomp.size(), 0, nullptr, FLT_MAX, FLT_MAX, ImVec2(400, 100));
            }

            for(plugin* p : plugins)
            {
                particle_dynamics* dyn = dynamic_cast<particle_dynamics*>(p);

                if(dyn == nullptr)
                    continue;

                /*ImGui::PlotLines("Initial Velocity", dyn->debug_velocities.data(), dyn->debug_velocities.size(), 0, nullptr, FLT_MAX, FLT_MAX, ImVec2(400, 100));

                ImGui::PlotLines("Real Mass", dyn->debug_real_mass.data(), dyn->debug_real_mass.size(), 0, nullptr, FLT_MAX, FLT_MAX, ImVec2(400, 100));
                ImGui::PlotLines("Analytic Mass", dyn->debug_analytic_mass.data(), dyn->debug_analytic_mass.size(), 0, nullptr, FLT_MAX, FLT_MAX, ImVec2(400, 100));*/
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
        float timestep = 0.04f;

        if(pao && time_elapsed_s > 250)
            step = false;

        if(step)
        {
            steps++;

            auto callback = [&](cl::managed_command_queue& mqueue, std::vector<cl::buffer>& bufs, std::vector<ref_counted_buffer>& intermediates)
            {
                wave_manager.issue_extraction(mqueue, bufs, intermediates, scale, clsize);

                if(!should_render)
                {
                    cl::args render;

                    for(auto& i : bufs)
                    {
                        render.push_back(i.as_device_read_only());
                    }

                    for(auto& i : intermediates)
                    {
                        render.push_back(i.as_device_read_only());
                    }

                    //render.push_back(bssnok_datas[which_data]);
                    render.push_back(scale);
                    render.push_back(clsize);
                    render.push_back(rtex);

                    mqueue.exec("render", render, {size.x(), size.y()}, {16, 16});
                }
            };

            base_mesh.full_step(clctx.ctx, clctx.cqueue, mqueue, timestep, thin_pool, callback);

            time_elapsed_s += timestep;
        }

        {
            std::vector<dual_types::complex<float>> values = wave_manager.process();

            for(dual_types::complex<float> v : values)
            {
                real_decomp.push_back(v.real);
                imaginary_decomp.push_back(v.imaginary);
            }
        }

        if(should_render || snap)
        {
            if(rendering_method == 0)
            {
                cl::args render_args;

                auto buffers = base_mesh.data[0].buffers;

                for(auto& i : buffers)
                {
                    render_args.push_back(i.buf.as_device_read_only());
                }

                cl_float3 ccamera_pos = {camera_pos.x(), camera_pos.y(), camera_pos.z()};
                cl_float4 ccamera_quat = {camera_quat.q.x(), camera_quat.q.y(), camera_quat.q.z(), camera_quat.q.w()};

                render_args.push_back(scale);
                render_args.push_back(ccamera_pos);
                render_args.push_back(ccamera_quat);
                render_args.push_back(clsize);
                render_args.push_back(rtex);

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

                    init_args.push_back(ray_buffer);
                    init_args.push_back(ray_count);

                    for(auto& i : base_mesh.data[0].buffers)
                    {
                        init_args.push_back(i.buf.as_device_read_only());
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

                    render_args.push_back(ray_buffer);
                    render_args.push_back(rays_terminated);

                    for(auto& i : base_mesh.data[0].buffers)
                    {
                        render_args.push_back(i.buf.as_device_read_only());
                    }

                    cl_int use_colour = use_matter_colour;

                    render_args.push_back(use_colour);

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

                    for(auto& i : base_mesh.data[0].buffers)
                    {
                        texture_args.push_back(i.buf.as_device_read_only());
                    }

                    texture_args.push_back(scale);
                    texture_args.push_back(clsize);

                    clctx.cqueue.exec("calculate_adm_texture_coordinates", texture_args, {width, height}, {8, 8});
                }

                {
                    cl::args render_args;
                    render_args.push_back(rays_terminated.as_device_read_only());
                    render_args.push_back(ray_count_terminated.as_device_read_only());
                    render_args.push_back(rtex);

                    for(auto& i : base_mesh.data[0].buffers)
                    {
                        render_args.push_back(i.buf.as_device_read_only());
                    }

                    render_args.push_back(scale);
                    render_args.push_back(clsize);
                    render_args.push_back(width);
                    render_args.push_back(height);
                    render_args.push_back(background_mipped);
                    render_args.push_back(texture_coordinates);
                    render_args.push_back(sam);
                    render_args.push_back(ccamera_pos);

                    clctx.cqueue.exec("render_rays", render_args, {width * height}, {128});
                }
            }
        }

        /*if(rendering_method == 2 && snap)
        {
            cl_float3 ccamera_pos = {camera_pos.x(), camera_pos.y(), camera_pos.z()};
            cl_float4 ccamera_quat = {camera_quat.q.x(), camera_quat.q.y(), camera_quat.q.z(), camera_quat.q.w()};

            cl::args init_args;

            auto buffers = base_mesh.data[0].buffers;

            for(auto& i : buffers)
            {
                init_args.push_back(i.buf);
            }

            init_args.push_back(scale);
            init_args.push_back(ccamera_pos);
            init_args.push_back(ccamera_quat);
            init_args.push_back(clsize);
            init_args.push_back(rtex);
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

            auto buffers = base_mesh.data[0].buffers;

            for(auto& i : buffers)
            {
                step_args.push_back(i.buf);
            }

            step_args.push_back(scale);
            step_args.push_back(ccamera_pos);
            step_args.push_back(ccamera_quat);
            step_args.push_back(clsize);
            step_args.push_back(rtex);
            step_args.push_back(ray_buffer);
            step_args.push_back(timestep);

            clctx.cqueue.exec("step_accurate_rays", step_args, {width * height}, {128});
        }*/

        rtex.unacquire(clctx.cqueue);

        {
            ImDrawList* lst = ImGui::GetBackgroundDrawList();

            ImVec2 screen_pos = ImGui::GetMainViewport()->Pos;

            ImVec2 tl = {0,0};
            ImVec2 br = {rtex.size<2>().x(), rtex.size<2>().y()};

            if(win.get_render_settings().viewports)
            {
                tl.x += screen_pos.x;
                tl.y += screen_pos.y;

                br.x += screen_pos.x;
                br.y += screen_pos.y;
            }

            lst->AddImage((void*)rtex.texture_id, tl, br, ImVec2(0, 0), ImVec2(1.f, 1.f));
        }

        win.display();

        if(frametime.get_elapsed_time_s() > 10)
            return 0;

        printf("Time: %f\n", frametime.restart() * 1000.);
    }
}
