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
#include "legendre_weights.h"
#include "legendre_nodes.h"
#include <fstream>
#include <imgui/misc/freetype/imgui_freetype.h>
#include <vec/tensor.hpp>

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
https://www.researchgate.net/publication/51952973_Numerical_stability_of_the_Z4c_formulation_of_general_relativity ccz4 equations

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
*/

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

        for(auto& i : aliases)
        {
            if(dual_types::equivalent(v, i.first))
            {
                v = i.second;
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

    void alias(value& concrete, value& alias)
    {
        for(auto& i : aliases)
        {
            if(dual_types::equivalent(concrete, i.first))
            {
                concrete = i.second;
                return;
            }
        }

        aliases.push_back({concrete, alias});
    }

    void add(const std::string& name, const value& v)
    {
        values.push_back({name, v});
    }

    void build(std::string& argument_string, int idx)
    {
        for(auto& i : values)
        {
            std::string str = "-D" + i.first + "=" + type_to_string(i.second) + " ";

            argument_string += str;
        }

        if(temporaries.size() == 0)
        {
            argument_string += "-DTEMPORARIES" + std::to_string(idx) + "=DUMMY ";
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

        argument_string += "-DTEMPORARIES" + std::to_string(idx) + "=" + temporary_string + " ";
    }
};

//#define SYMMETRY_BOUNDARY
#define BORDER_WIDTH 4

inline
std::tuple<std::string, std::string, bool> decompose_variable(std::string str)
{
    std::string buffer;
    std::string val;

    if(str.ends_with(")]"))
    {
        if(!str.ends_with("[IDX(ix,iy,iz)]"))
        {
            std::cout << "Got bad string " << str << std::endl;
            assert(false);
        }
    }

    std::array variables
    {
        "cY0",
        "cY1",
        "cY2",
        "cY3",
        "cY4",
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
        "theta",

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

        "momentum0",
        "momentum1",
        "momentum2",
    };

    bool uses_extension = false;

    if(str.starts_with("v->"))
    {
        buffer = "in";

        val = str;

        val.erase(val.begin());
        val.erase(val.begin());
        val.erase(val.begin());

        uses_extension = true;
    }
    else if(str.starts_with("v."))
    {
        buffer = "in";

        val = str;

        val.erase(val.begin());
        val.erase(val.begin());

        uses_extension = true;
    }
    else if(str.starts_with("ik."))
    {
        buffer = "temp_in";

        val = str;

        val.erase(val.begin());
        val.erase(val.begin());
        val.erase(val.begin());

        uses_extension = true;
    }

    else if(str.starts_with("buffer_read_linear"))
    {
        std::string_view sview = str;
        sview.remove_prefix(strlen("buffer_read_linear("));

        int len = sview.find_first_of(',');

        assert(len != std::string_view::npos);

        buffer = std::string(sview.begin(), sview.begin() + len);
        val = buffer;
    }

    else if(str.starts_with("buffer"))
    {
        buffer = "buffer";
        val = "buffer";
    }
    else
    {
        bool any_found = false;

        for(auto& i : variables)
        {
            if(str.starts_with(i))
            {
                buffer = i;
                val = buffer;
                any_found = true;
            }
        }

        if(!any_found)
        {
            std::cout << "input " << str << std::endl;

            assert(false);
        }
    }

    return {buffer, val, uses_extension};
}

template<int elements = 5>
struct differentiation_context
{
    std::array<value, elements> vars;

    std::array<value, elements> xs;
    std::array<value, elements> ys;
    std::array<value, elements> zs;

    differentiation_context(equation_context& ctx, const value& in, int idx, const vec<3, value>& idx_offset, bool should_pin = true, bool linear_interpolation = false)
    {
        std::vector<std::string> variables = in.get_all_variables();

        value cp = in;

        auto index_raw = [](const value& x, const value& y, const value& z)
        {
            return "IDX(" + type_to_string(x, true) + "," + type_to_string(y, true) + "," + type_to_string(z, true) + ")";
        };

        auto fetch_linear = [](const std::string& buffer, const value& x, const value& y, const value& z)
        {
            return "buffer_read_linear(" + buffer + ",(float3)(" + type_to_string(x, false) + "," + type_to_string(y, false) + "," + type_to_string(z, false) + "),dim)";
        };

        auto index_buffer = [](const std::string& variable, const std::string& buffer, const std::string& with_what)
        {
            return buffer + "[" + with_what + "]." + variable;
        };

        auto index_without_extension = [](const std::string& buffer, const std::string& with_what)
        {
            return buffer + "[" + with_what + "]";
        };

        auto index = [&ctx, index_buffer, index_without_extension, index_raw, fetch_linear, linear_interpolation](const std::string& val, const std::string& buffer, bool uses_extension, const value& x, const value& y, const value& z)
        {
            value v;

            if(linear_interpolation)
            {
                if(uses_extension)
                {
                    std::cout << "Uses extension " << buffer << std::endl;
                }

                assert(!uses_extension);

                v = value(fetch_linear(buffer, x, y, z));
            }

            else
            {
                if(uses_extension)
                    v = value(index_buffer(val, buffer, index_raw(x, y, z)));
                else
                    v = value(index_without_extension(buffer, index_raw(x, y, z)));
            }

            //ctx.pin(v);

            return v;
        };

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
                xs[i] += offset + idx_offset.x();
            if(idx == 1)
                ys[i] += offset + idx_offset.y();
            if(idx == 2)
                zs[i] += offset + idx_offset.z();
        }

        /*std::map<std::string, std::string> substitutions1;
        std::map<std::string, std::string> substitutions2;

        for(auto& i : variables)
        {
            std::tuple<std::string, std::string, bool> decomp = decompose_variable(i);

            value to_sub1 = index(std::get<1>(decomp), std::get<0>(decomp), std::get<2>(decomp), x1, y1, z1);
            value to_sub2 = index(std::get<1>(decomp), std::get<0>(decomp), std::get<2>(decomp), x2, y2, z2);

            substitutions1[i] = type_to_string(to_sub1);
            substitutions2[i] = type_to_string(to_sub2);
        }*/

        std::array<std::map<std::string, std::string>, elements> substitutions;

        for(auto& i : variables)
        {
            std::tuple<std::string, std::string, bool> decomp = decompose_variable(i);

            for(int kk=0; kk < elements; kk++)
            {
                value to_sub = index(std::get<1>(decomp), std::get<0>(decomp), std::get<2>(decomp), xs[kk], ys[kk], zs[kk]);

                if(should_pin)
                {
                    //ctx.pin(to_sub);
                }

                substitutions[kk][i] = type_to_string(to_sub);
            }
        }

        for(int i=0; i < elements; i++)
        {
            vars[i] = cp;
            vars[i].substitute(substitutions[i]);
        }

        ///todo: pin the individual variables?
        /*if(should_pin)
        {
            for(auto& i : vars)
            {
                ctx.pin(i);
            }
        }*/
    }
};

value get_distance(const vec<3, value>& p1, const vec<3, value>& p2)
{
    return "get_distance(" + type_to_string(p1.x(), true) + "," + type_to_string(p1.y(), true) + "," + type_to_string(p1.z(), true) + "," + type_to_string(p2.x(), true) + "," + type_to_string(p2.y(), true) + "," + type_to_string(p2.z(), true) + ",dim,scale)";
}

value get_scale_distance(equation_context& ctx, const value& in, const vec<3, value>& offset, int idx, int which)
{
    differentiation_context<3> dctx(ctx, in, idx, offset, false);

    value final_command;

    value ix = "ix";
    value iy = "iy";
    value iz = "iz";

    ix += offset.x();
    iy += offset.y();
    iz += offset.z();

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

value first_derivative(equation_context& ctx, const value& in, const vec<3, value>& offset, int idx)
{
    differentiation_context<3> dctx(ctx, in, idx, offset, false);

    value h = get_scale_distance(ctx, in, offset, idx, 0);
    value k = get_scale_distance(ctx, in, offset, idx, 1);

    ///f(x + h) - f(x - k)
    value final_command = (dctx.vars[2] - dctx.vars[0]) / (h + k);

    return final_command;
}

value second_derivative(equation_context& ctx, const value& in, const vec<3, value>& offset, int idx)
{
    differentiation_context<5> dctx(ctx, in, idx, offset, false);

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

/*value third_derivative(equation_context& ctx, const value& in, const vec<3, value>& offset, int idx)
{
    value h = get_scale_distance(ctx, in, offset, idx, 0);
    value k = get_scale_distance(ctx, in, offset, idx, 1);

    value right = second_derivative(ctx, in, get_idx_offset(idx) + offset, idx);
    value left = second_derivative(ctx, in, -get_idx_offset(idx) + offset, idx);

    return (right - left) / (h + k);
}*/

value fourth_derivative(equation_context& ctx, const value& in, const vec<3, value>& offset, int idx)
{
    differentiation_context<5> dctx(ctx, in, idx, offset, false);

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

    vec<3, value> offset = {"0", "0", "0"};

    //value h = get_scale_distance(ctx, in, offset, idx, 0);
    //value k = get_scale_distance(ctx, in, offset, idx, 1);

    ///todo: fix this
    //value effective_scale = (h + k) / 2.f;

    ///https://en.wikipedia.org/wiki/Finite_difference_coefficient according to wikipedia, this is the 6th derivative with 2nd order accuracy. I am confused, but at least I know where it came from
    value scale = "scale";

    //#define FOURTH
    #ifdef FOURTH
    differentiation_context<5> dctx(ctx, in, idx, {"0", "0", "0"}, false);
    //value stencil = -(1 / (16.f * effective_scale)) * (dctx.vars[0] - 4 * dctx.vars[1] + 6 * dctx.vars[2] - 4 * dctx.vars[3] + dctx.vars[4]);

    value stencil = (-1 / 16.f) * pow(effective_scale, 3.f) * fourth_derivative(ctx, in, offset, idx);

    #endif // FOURTH

    #define SIXTH
    #ifdef SIXTH
    differentiation_context<7> dctx(ctx, in, idx, {"0", "0", "0"}, false);
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

std::string bidx(const std::string& buf, bool interpolate)
{
    if(interpolate)
        return "buffer_read_linear(" + buf + ",(float3)(fx,fy,fz),dim)";
    else
        return buf + "[IDX(ix,iy,iz)]";
}

struct standard_arguments
{
    value gA;
    tensor<value, 3> gB;

    unit_metric<value, 3, 3> cY;
    tensor<value, 3, 3> cA;

    value X;
    value K;

    value clamped_X;

    tensor<value, 3> cGi;

    metric<value, 3, 3> Yij;
    tensor<value, 3, 3> Kij;

    tensor<value, 3> momentum_constraint;

    value theta;

    standard_arguments(bool interpolate)
    {
        gA.make_value(bidx("gA", interpolate));

        gB.idx(0).make_value(bidx("gB0", interpolate));
        gB.idx(1).make_value(bidx("gB1", interpolate));
        gB.idx(2).make_value(bidx("gB2", interpolate));

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

        cY.idx(2, 2) = (1 + cY.idx(1, 1) * cY.idx(0, 2) * cY.idx(0, 2) - 2 * cY.idx(0, 1) * cY.idx(1, 2) * cY.idx(0, 2) + cY.idx(0, 0) * cY.idx(1, 2) * cY.idx(1, 2)) / (cY.idx(0, 0) * cY.idx(1, 1) - cY.idx(0, 1) * cY.idx(0, 1));

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

        X.make_value(bidx("X", interpolate));
        K.make_value(bidx("K", interpolate));

        cGi.idx(0).make_value(bidx("cGi0", interpolate));
        cGi.idx(1).make_value(bidx("cGi1", interpolate));
        cGi.idx(2).make_value(bidx("cGi2", interpolate));

        clamped_X = max(X, 0.01f);

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                Yij.idx(i, j) = cY.idx(i, j) / clamped_X;
            }
        }

        tensor<value, 3, 3> Aij = cA / clamped_X;

        Kij = Aij + Yij.to_tensor() * (K / 3.f);

        momentum_constraint.idx(0).make_value(bidx("momentum0", interpolate));
        momentum_constraint.idx(1).make_value(bidx("momentum1", interpolate));
        momentum_constraint.idx(2).make_value(bidx("momentum2", interpolate));

        theta.make_value(bidx("theta", interpolate));
    }
};

#if 0
void build_kreiss_oliger_dissipate(equation_context& ctx)
{
    vec2i linear_indices[6] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};

    standard_arguments args(false);

    float dissipate_low = 0.15;
    float dissipate_high = 0.35;

    for(int i=0; i < 6; i++)
    {
        vec2i idx = linear_indices[i];

        ctx.add("k_cYij" + std::to_string(i), dissipate_low * kreiss_oliger_dissipate(ctx, args.cY.idx(idx.x(), idx.y())));
    }

    ctx.add("k_X", dissipate_high * kreiss_oliger_dissipate(ctx, args.X));

    for(int i=0; i < 6; i++)
    {
        vec2i idx = linear_indices[i];

        ctx.add("k_cAij" + std::to_string(i), dissipate_high * kreiss_oliger_dissipate(ctx, args.cA.idx(idx.x(), idx.y())));
    }

    ctx.add("k_K", dissipate_high * kreiss_oliger_dissipate(ctx, args.K));

    for(int i=0; i < 3; i++)
    {
        ctx.add("k_cGi" + std::to_string(i), dissipate_high * kreiss_oliger_dissipate(ctx, args.cGi.idx(i)));
    }

    ctx.add("k_gA", dissipate_high * kreiss_oliger_dissipate(ctx, args.gA));

    for(int i=0; i < 3; i++)
    {
        ctx.add("k_gB" + std::to_string(i), dissipate_high * kreiss_oliger_dissipate(ctx, args.gB.idx(i)));
    }
}
#endif // 0

void build_kreiss_oliger_dissipate_singular(equation_context& ctx)
{
    value buf = "buffer[IDX(ix,iy,iz)]";

    value coeff = "coefficient";

    ctx.add("KREISS_DISSIPATE_SINGULAR", coeff * kreiss_oliger_dissipate(ctx, buf));
}

template<int order = 1>
value hacky_differentiate(equation_context& ctx, const value& in, int idx, bool pin = true, bool linear = false)
{
    differentiation_context dctx(ctx, in, idx, {"0", "0", "0"}, true, linear);

    std::array<value, 5> vars = dctx.vars;

    value scale = "scale";

    value final_command;

    if(order == 1)
    {
        final_command = (vars[3] - vars[1]) / (2 * scale);
    }
    else if(order == 2)
    {
        final_command = (-vars[4] + 8 * vars[3] - 8 * vars[1] + vars[0]) / (12 * scale);
    }

    static_assert(order == 1 || order == 2);

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

    if(pin)
    {
        //ctx.pin(final_command);
    }

    return final_command;
}

tensor<value, 3> tensor_derivative(equation_context& ctx, const value& in)
{
    tensor<value, 3> ret;

    for(int i=0; i < 3; i++)
    {
        ret.idx(i) = hacky_differentiate(ctx, in, i);
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
            ret.idx(i, j) = hacky_differentiate(ctx, in.idx(j), i);
        }
    }

    return ret;
}

///B^i * Di whatever
value upwind_differentiate(equation_context& ctx, const value& prefix, const value& in, int idx, bool pin = true)
{
    /*differentiation_context dctx(ctx, in, idx);

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

    if(pin)
    {
        ctx.pin(final_command);
    }

    return final_command;*/

    return prefix * hacky_differentiate(ctx, in, idx, pin);

    /*differentiation_context<7> dctx(ctx, in, idx);

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
                //sum = sum + B.idx(k) * hacky_differentiate(ctx, mT.idx(i, j), k);
                sum = sum + upwind_differentiate(ctx, B.idx(k), mT.idx(i, j), k);
                sum = sum + mT.idx(i, k) * hacky_differentiate(ctx, B.idx(k), j);
                sum = sum + mT.idx(j, k) * hacky_differentiate(ctx, B.idx(k), i);
                sum = sum - (2.f/3.f) * mT.idx(i, j) * hacky_differentiate(ctx, B.idx(k), k);
            }

            lie.idx(i, j) = sum;
        }
    }

    return lie;
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

/*template<typename T, int N>
inline
tensor<T, N> gpu_covariant_derivative_scalar(equation_context& ctx, const T& in)
{
    tensor<T, N> ret;

    for(int i=0; i < N; i++)
    {
        ret.idx(i) = hacky_differentiate(ctx, in, i);
    }

    return ret;
}

template<typename T, int N>
inline
tensor<T, N> gpu_high_covariant_derivative_scalar(equation_context& ctx, const T& in, const metric<T, N, N>& met, const inverse_metric<T, N, N>& inverse)
{
    tensor<T, N> deriv_low = gpu_covariant_derivative_scalar<T, N>(ctx, in);

    tensor<T, N> ret;

    for(int i=0; i < N; i++)
    {
        T sum = 0;

        for(int p=0; p < N; p++)
        {
            sum = sum + inverse.idx(i, p) * deriv_low.idx(p);
        }

        ret.idx(i) = sum;
    }

    return ret;
}*/

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

            lac.idx(a, c) = hacky_differentiate(ctx, v_in.idx(a), c) - sum;
        }
    }

    return lac;
}

template<typename T, int N>
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
}

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

//https://arxiv.org/pdf/0709.3559.pdf b.48??
///the algebraic constraints are an extremely good candidate for simulation instability
template<typename T, int N>
inline
tensor<T, N, N> gpu_trace_free_cAij(const tensor<T, N, N>& mT, const metric<T, N, N>& met, const inverse_metric<T, N, N>& inverse)
{
    tensor<T, N, N> TF;

    for(int i=0; i < N; i++)
    {
        for(int j=0; j < N; j++)
        {
            value sum = 0;

            for(int l=0; l < N; l++)
            {
                for(int m=0; m < N; m++)
                {
                    sum += mT.idx(l, m) * inverse.idx(i, l) * inverse.idx(j, m);
                }
            }

            TF.idx(i, j) = mT.idx(i, j) - (1.f/3.f) * sum;
        }
    }

    return TF;
}


template<typename T, int N>
inline
tensor<T, N, N, N> gpu_christoffel_symbols_2(equation_context& ctx, const metric<T, N, N>& met, const inverse_metric<T, N, N>& inverse, bool linear = false)
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

                    local = local + hacky_differentiate(ctx, met.idx(m, k), l, true, linear);
                    local = local + hacky_differentiate(ctx, met.idx(m, l), k, true, linear);
                    local = local - hacky_differentiate(ctx, met.idx(k, l), m, true, linear);

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
                christoff.idx(c, a, b) = 0.5f * (hacky_differentiate(ctx, met.idx(c, a), b) + hacky_differentiate(ctx, met.idx(c, b), a) - hacky_differentiate(ctx, met.idx(a, b), c));
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

///this does not return the same kind of conformal cAij as bssn uses, need to reconstruct Kij!
tensor<value, 3, 3> calculate_bcAij(const vec<3, value>& pos, const std::vector<float>& black_hole_m, const std::vector<vec3f>& black_hole_pos, const std::vector<vec3f>& black_hole_velocity)
{
    tensor<value, 3, 3> bcAij;

    metric<value, 3, 3> flat;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            flat.idx(i, j) = (i == j) ? 1 : 0;
        }
    }

    for(int bh_idx = 0; bh_idx < (int)black_hole_pos.size(); bh_idx++)
    {
        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                vec3f bhpos = black_hole_pos[bh_idx];
                vec3f momentum = black_hole_velocity[bh_idx] * black_hole_m[bh_idx];
                tensor<value, 3> momentum_tensor = {momentum.x(), momentum.y(), momentum.z()};

                vec<3, value> vri = {bhpos.x(), bhpos.y(), bhpos.z()};

                value ra = (pos - vri).length();
                vec<3, value> nia = (pos - vri) / ra;

                tensor<value, 3> momentum_lower = lower_index(momentum_tensor, flat);
                tensor<value, 3> nia_lower = lower_index(tensor<value, 3>{nia.x(), nia.y(), nia.z()}, flat);

                bcAij.idx(i, j) += (3 / (2.f * ra * ra)) * (momentum_lower.idx(i) * nia_lower.idx(j) + momentum_lower.idx(j) * nia_lower.idx(i) - (flat.idx(i, j) - nia_lower.idx(i) * nia_lower.idx(j)) * sum_multiply(momentum_tensor, nia_lower));
            }
        }
    }

    return bcAij;
}

inline
void setup_initial_conditions(equation_context& ctx, vec3f centre, float scale)
{
    vec<3, value> pos;

    pos[0].make_value("ox");
    pos[1].make_value("oy");
    pos[2].make_value("oz");

    float bulge = 1;

    auto san_black_hole_pos = [&](vec3f in)
    {
        float s1 = in.x();
        float s2 = in.y();
        float s3 = in.z();

        vec3f scaled = round((in / scale) * bulge);

        vec3f offsets = {0.5f, 0.5f, 0.5f};

        auto get_sign = [](float in)
        {
            return in >= 0 ? 1 : -1;
        };

        offsets.x() *= get_sign(s1);
        offsets.y() *= get_sign(s2);
        offsets.z() *= get_sign(s3);

        std::cout << "Black hole at voxel " << scaled + centre + offsets << std::endl;

        return scaled * scale / bulge + offsets * scale / bulge;
    };

    ///https://arxiv.org/pdf/gr-qc/0505055.pdf
    //std::vector<vec3f> black_hole_pos{san_black_hole_pos({0, -1.1515 * 0.5f, 0}), san_black_hole_pos({0, 1.1515 * 0.5f, 0})};
    //std::vector<vec3f> black_hole_pos{san_black_hole_pos({-3.1515, 0, 0}), san_black_hole_pos({3.1515, 0, 0})};
    //std::vector<vec3f> black_hole_pos{san_black_hole_pos({5, 0, 0})};
    //std::vector<float> black_hole_m{0.5f, 0.5f};
    //std::vector<vec3f> black_hole_velocity{{0, 0, 0.025}, {0, 0, -0.025}}; ///pick better velocities

    //std::vector<vec3f> black_hole_pos{san_black_hole_pos({-2.1515f, 0, 0}), san_black_hole_pos({2.1515f, 0, 0})};
    //std::vector<vec3f> black_hole_pos{san_black_hole_pos({-2.5f - 0, 0, 0}), san_black_hole_pos({2.5f + 0, 0, 0})};
    //std::vector<vec3f> black_hole_velocity{{0, 0, -0.05}, {0, 0, 0.05}};

    /*std::vector<float> black_hole_m{0.45f, 0.45f};
    std::vector<vec3f> black_hole_pos{san_black_hole_pos({-3.1515f, 0, 0}), san_black_hole_pos({3.1515f, 0, 0})};
    std::vector<vec3f> black_hole_velocity{{0, 0, 0.335/8}, {0, 0, -0.335/8}};*/

    ///https://arxiv.org/pdf/1205.5111v1.pdf under binary black hole with punctures
    std::vector<float> black_hole_m{0.463, 0.47};
    std::vector<vec3f> black_hole_pos{san_black_hole_pos({-3.516, 0, 0}), san_black_hole_pos({3.516, 0, 0})};
    //std::vector<vec3f> black_hole_velocity{{0, 0, 0}, {0, 0, 0}};
    std::vector<vec3f> black_hole_velocity{{0, 0, -0.258 * 0.55f}, {0, 0, 0.258 * 0.55f}};
    //std::vector<vec3f> black_hole_velocity{{0, 0, 0.5f * -0.258/black_hole_m[0]}, {0, 0, 0.5f * 0.258/black_hole_m[1]}};

    //std::vector<vec3f> black_hole_velocity{{0,0,0.000025}, {0,0,-0.000025}};

    metric<value, 3, 3> flat_metric;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            flat_metric.idx(i, j) = (i == j) ? 1 : 0;
        }
    }

    tensor<value, 3, 3> bcAij = calculate_bcAij(pos, black_hole_m, black_hole_pos, black_hole_velocity);

    //https://arxiv.org/pdf/gr-qc/9703066.pdf (8)
    //value BL_a = 0;

    value BL_s = 0;

    for(int i=0; i < (int)black_hole_m.size(); i++)
    {
        float Mi = black_hole_m[i];
        vec3f ri = black_hole_pos[i];

        vec<3, value> vri = {ri.x(), ri.y(), ri.z()};

        value dist = (pos - vri).length();

        BL_s += Mi / (2 * dist);
    }

    ctx.add("init_BL_val", BL_s);

    value aij_aIJ = 0;

    tensor<value, 3, 3> ibcAij = raise_both(bcAij, flat_metric, flat_metric.invert());

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            aij_aIJ += ibcAij.idx(i, j) * bcAij.idx(i, j);
        }
    }

    ctx.add("init_aij_aIJ", aij_aIJ);

    vec2i linear_indices[6] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};

    for(int i=0; i < 6; i++)
    {
        ctx.add("init_bcA" + std::to_string(i), bcAij.idx(linear_indices[i].x(), linear_indices[i].y()));
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
    value u = "u_value[IDX(ix,iy,iz)]";

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

    value gA = 1;
    //value gA = 1/(pow(bl_conformal + u, 2));
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

    ctx.add("init_theta", 0);

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
inline
void build_constraints(equation_context& ctx)
{
    standard_arguments args(false);

    unit_metric<value, 3, 3> cY = args.cY;
    tensor<value, 3, 3> cA = args.cA;

    /*value det_cY_pow = pow(cY.det(), 1.f/3.f);

    det_cY_pow = 1;
    ctx.pin(det_cY_pow);

    /// / det_cY_pow
    metric<value, 3, 3> fixed_cY = cY / det_cY_pow;*/

    ///sucks
    //tensor<value, 3, 3> fixed_cA = gpu_trace_free_cAij(cA, fixed_cY, fixed_cY.invert());

    /*tensor<value, 3, 3> fixed_cA = cA;

    ///https://arxiv.org/pdf/0709.3559.pdf b.49
    fixed_cA = fixed_cA / det_cY_pow;*/

    //tensor<value, 3, 3> fixed_cA = gpu_trace_free(cA, cY, cY.invert());

    inverse_metric<value, 3, 3> icY = cY.invert();

    tensor<value, 3, 3> raised_cA = raise_second_index(cA, cY, icY);

    tensor<value, 3, 3> fixed_cA = cA;

    fixed_cA.idx(1, 1) = -(raised_cA.idx(0, 0) + raised_cA.idx(2, 2) + cA.idx(0, 1) * icY.idx(0, 1) + cA.idx(1, 2) * icY.idx(1, 2)) / icY.idx(1, 1);

    vec2i linear_indices[6] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};

    for(int i=0; i < 6; i++)
    {
        vec2i idx = linear_indices[i];

        //ctx.add("fix_cY" + std::to_string(i), fixed_cY.idx(idx.x(), idx.y()));
        ctx.add("fix_cA" + std::to_string(i), fixed_cA.idx(idx.x(), idx.y()));
    }

    ctx.add("NO_CAIJYY", 1);
}

void build_intermediate_thin(equation_context& ctx)
{
    standard_arguments args(false);

    value buffer = "buffer[IDX(ix,iy,iz)]";

    value v1 = hacky_differentiate(ctx, buffer, 0);
    value v2 = hacky_differentiate(ctx, buffer, 1);
    value v3 = hacky_differentiate(ctx, buffer, 2);

    ctx.add("init_buffer_intermediate0", v1);
    ctx.add("init_buffer_intermediate1", v2);
    ctx.add("init_buffer_intermediate2", v3);
}

void build_intermediate_thin_cY5(equation_context& ctx)
{
    standard_arguments args(false);

    for(int k=0; k < 3; k++)
    {
        ctx.add("init_cY5_intermediate" + std::to_string(k), hacky_differentiate(ctx, args.cY.idx(2, 2), k, false));
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

                ret.idx(c, a, b) = hacky_differentiate(ctx, mT.idx(a, b), c) + sum;
            }
        }
    }

    return ret;
}

void build_momentum_constraint(equation_context& ctx)
{
    standard_arguments args(false);

    inverse_metric<value, 3, 3> icY = args.cY.invert();
    auto unpinned_icY = icY;
    ctx.pin(icY);

    value X_recip = 0.f;

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
    }

    tensor<value, 3> Mi;

    for(int i=0; i < 3; i++)
    {
        Mi.idx(i) = 0;
    }

    //#define DAMP_DTCAIJ
    #ifdef DAMP_DTCAIJ
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

        value s2 = -(2.f/3.f) * hacky_differentiate(ctx, args.K, i);

        value s3 = 0;

        for(int m=0; m < 3; m++)
        {
            s3 += -(3.f/2.f) * mixed_cAij.idx(m, i) * hacky_differentiate(ctx, args.X, m) * X_recip;
        }

        Mi.idx(i) = dual_if(args.X <= 0.001f,
        []()
        {
            return 0.f;
        },
        [&]()
        {
            return s1 + s2 + s3;
        });
    }
    #endif // 0

    /*tensor<value, 3> Mi;

    tensor<value, 3, 3> second_cAij = raise_second_index(args.cA, args.cY, unpinned_icY);

    for(int i=0; i < 3; i++)
    {
        value s1 = 0;

        for(int j=0; j < 3; j++)
        {
            s1 += hacky_differentiate(ctx, second_cAij.idx(i, j), j);
        }

        value s2 = 0;

        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                s2 += -0.5f * icY.idx(j, k) * hacky_differentiate(ctx, args.cA.idx(j, k), i);
            }
        }

        value s3 = 0;

        for(int j=0; j < 3; j++)
        {
            s3 += -0.25f * 6 * X_recip * hacky_differentiate(ctx, args.X, j) * second_cAij.idx(i, j);
        }

        value s4 = -(2.f/3.f) * hacky_differentiate(ctx, args.K, i);

        Mi.idx(i) = s1 + s2 + s3 + s4;
    }*/

    for(int i=0; i < 3; i++)
    {
        ctx.add("init_momentum" + std::to_string(i), Mi.idx(i));
    }
}

///https://arxiv.org/pdf/gr-qc/0206072.pdf on stability, they recompute cGi where it does nto hae a derivative
///todo: X: This is think is why we're getting nans. Half done
///todo: fisheye - half done
///todo: better differentiation
///todo: Enforce the algebraic constraints in this paper: https://arxiv.org/pdf/gr-qc/0505055.pdf. 3.21 and 3.22 - done
///todo: if I use a mirror boundary condition, it'd simulate an infinite grid of black hole pairs colliding
///they would however all be relatively far away from each other, so this may turn out fairly acceptably
///todo: I think I can cut down on the memory consumption by precalculating the necessary derivatives
///todo: Removing phi was never the issue with the numerical stability
///todo: With the current full double buffering scheme, all equations could have their own kernel
#if 0
inline
void build_eqs(equation_context& ctx)
{
    standard_arguments args(false);

    int index_table[3][3] = {{0, 1, 2},
                             {1, 3, 4},
                             {2, 4, 5}};

    vec2i linear_indices[6] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};

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

    unit_metric<value, 3, 3> cY = args.cY;

    inverse_metric<value, 3, 3> icY = cY.invert();
    inverse_metric<value, 3, 3> unpinned_icY = cY.invert();

    ctx.pin(icY);

    tensor<value, 3, 3> cA = args.cA;

    auto unpinned_cA = cA;
    //ctx.pin(cA);

    ///the christoffel symbol
    tensor<value, 3> cGi = args.cGi;

    tensor<value, 3> digA;
    digA.idx(0).make_value("digA0[IDX(ix,iy,iz)]");
    digA.idx(1).make_value("digA1[IDX(ix,iy,iz)]");
    digA.idx(2).make_value("digA2[IDX(ix,iy,iz)]");

    tensor<value, 3> dX;
    dX.idx(0).make_value("dX0[IDX(ix,iy,iz)]");
    dX.idx(1).make_value("dX1[IDX(ix,iy,iz)]");
    dX.idx(2).make_value("dX2[IDX(ix,iy,iz)]");

    tensor<value, 3, 3> digB;

    ///derivative
    for(int i=0; i < 3; i++)
    {
        ///value
        for(int j=0; j < 3; j++)
        {
            int idx = i + j * 3;

            std::string name = "digB" + std::to_string(idx) + "[IDX(ix,iy,iz)]";

            digB.idx(i, j).make_value(name);
        }
    }

    value gA = args.gA;

    tensor<value, 3> gB = args.gB;

    #ifdef USE_GBB
    tensor<value, 3> gBB;
    gBB.idx(0).make_value("gBB0[IDX(ix,iy,iz)]");
    gBB.idx(1).make_value("gBB1[IDX(ix,iy,iz)]");
    gBB.idx(2).make_value("gBB2[IDX(ix,iy,iz)]");
    #endif // USE_GBB

    value X = args.X;
    value K = args.K;

    tensor<value, 3, 3, 3> dcYij;

    for(int k=0; k < 3; k++)
    {
        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                int symmetric_index = index_table[i][j];

                int final_index = k + symmetric_index * 3;

                std::string name = "dcYij" + std::to_string(final_index) + "[IDX(ix,iy,iz)]";

                dcYij.idx(k, i, j) = name;
            }
        }
    }

    ///dcgA alias
    for(int i=0; i < 3; i++)
    {
        value v = hacky_differentiate(ctx, gA, i, false);

        ctx.alias(v, digA.idx(i));
    }

    ///dcgB alias
    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value v = hacky_differentiate(ctx, gB.idx(j), i, false);

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
                value v = hacky_differentiate(ctx, cY.idx(i, j), k, false);

                ctx.alias(v, dcYij.idx(k, i, j));
            }
        }
    }

    for(int k=0; k < 3; k++)
    {
        for(int i=0; i < 6; i++)
        {
            vec2i idx = linear_indices[i];

            hacky_differentiate(ctx, cY.idx(idx.x(), idx.y()), k);
        }

        for(int i=0; i < 6; i++)
        {
            hacky_differentiate(ctx, "cA" + std::to_string(i), k);
        }

        for(int i=0; i < 3; i++)
        {
            hacky_differentiate(ctx, "cGi" + std::to_string(i), k);
        }

        hacky_differentiate(ctx, "K", k);
        hacky_differentiate(ctx, "X", k);
    }

    /*for(int k=0; k < 3; k++)
    {
        for(int i=0; i < 3 * 6; i++)
        {
            hacky_differentiate(ctx, "ik.dcYij[" + std::to_string(i) + "]", k);
        }

        for(int i=0; i < 3; i++)
        {
            hacky_differentiate(ctx, "ik.digA[" + std::to_string(i) + "]", k);
        }

        for(int i=0; i < 3; i++)
        {
            hacky_differentiate(ctx, "ik.digB[" + std::to_string(i) + "]", k);
        }

        for(int i=0; i < 3; i++)
        {
            hacky_differentiate(ctx, "ik.dX[" + std::to_string(i) + "]", k);
        }
    }*/

    tensor<value, 3, 3, 3> christoff1 = gpu_christoffel_symbols_1(ctx, cY);
    tensor<value, 3, 3, 3> christoff2 = gpu_christoffel_symbols_2(ctx, cY, icY);

    ///https://arxiv.org/pdf/1205.5111v1.pdf just after 34
    ///this is currently the same as derived_cGi, but they're used differently
    tensor<value, 3> cGi_G;

    for(int i=0; i < 3; i++)
    {
        value sum = 0;

        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                sum += icY.idx(j, k) * christoff2.idx(i, j, k);
            }
        }

        cGi_G.idx(i) = sum;
    }

    ///https://arxiv.org/pdf/1205.5111v1.pdf 34
    tensor<value, 3> bigGi;

    for(int i=0; i < 3; i++)
    {
        bigGi.idx(i) = cGi.idx(i) - cGi_G.idx(i);
    }

    tensor<value, 3> bigGi_lower = lower_index(bigGi, cY);

    tensor<value, 3> derived_cGi;

    ///https://arxiv.org/pdf/gr-qc/0206072.pdf page 4
    ///or https://arxiv.org/pdf/gr-qc/0511048.pdf after 7 actually
    /*for(int i=0; i < 3; i++)
    {
        value sum = 0;

        for(int j=0; j < 3; j++)
        {
            sum = sum - hacky_differentiate(ctx, unpinned_icY.idx(i, j), j);
        }

        derived_cGi.idx(i) = sum;
    }*/

    #define USE_DERIVED_CGI
    #ifdef USE_DERIVED_CGI
    for(int i=0; i < 3; i++)
    {
        value sum = 0;

        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                sum += icY.idx(j, k) * christoff2.idx(i, j, k);
            }
        }

        derived_cGi.idx(i) = sum;
    }
    #else
    derived_cGi = cGi;
    #endif

    ctx.add("derived0", derived_cGi.idx(0));
    ctx.add("derived1", derived_cGi.idx(1));
    ctx.add("derived2", derived_cGi.idx(2));

    /*tensor<value, 3, 3, 3> cGijk;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                int symmetric_index = index_table[j][k];

                int final_index = i * 6 + symmetric_index;

                std::string name = "ik.christoffel[" + std::to_string(final_index) + "]";

                cGijk.idx(i, j, k) = name;
            }
        }
    }*/

    /*tensor<value, 3, 3, 3, 3> dcGijk;
    ///coordinate derivative direction
    for(int i=0; i < 3; i++)
    {
        ///upper index
        for(int j=0; j < 3; j++)
        {
            ///two symmetric lower indices
            for(int k=0; k < 3; k++)
            {
                for(int l=0; l < 3; l++)
                {
                    int symmetric_index = index_table[k][l];

                    int final_index = i * 3 * 6 + j * 6 + symmetric_index;

                    std::string name = "dcGijk[" + std::to_string(final_index) + "]";

                    dcGijk.idx(i, j, k, l).make_value(name);
                }
            }
        }
    }*/

    tensor<value, 3> gB_lower = lower_index(gB, cY);

    tensor<value, 3> linear_dB;

    for(int i=0; i < 3; i++)
    {
        linear_dB.idx(i) = hacky_differentiate(ctx, gB.idx(i), i);
    }

    tensor<value, 3, 3> lie_cYij = gpu_lie_derivative_weight(ctx, gB, cY);

    ///https://arxiv.org/pdf/gr-qc/0511048.pdf (1)
    ///https://scholarworks.rit.edu/cgi/viewcontent.cgi?article=11286&context=theses 3.66
    tensor<value, 3, 3> dtcYij = -2 * gA * cA + lie_cYij;

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

            dtcYij.idx(i, j) += -(1.f/5.f) * cY.idx(i, j) * sum_multiply(gB, bigGi_lower);
        }
    }
    #endif // USE_DTCYIJ_MODIFICATION

    value dtX = (2.f/3.f) * X * (gA * K - sum(linear_dB)) + sum(tensor_upwind(ctx, gB, X));

    ///ok use the proper form
    tensor<value, 3, 3> cRij;

    ///https://en.wikipedia.org/wiki/Ricci_curvature#Definition_via_local_coordinates_on_a_smooth_manifold
    /*for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value sum = 0;

            for(int a=0; a < 3; a++)
            {
                sum = sum + dcGijk.idx(a, a, i, j);
            }

            value sum2 = 0;

            for(int a=0; a < 3; a++)
            {
                sum2 = sum2 + dcGijk.idx(i, a, a, j);
            }

            value sum3 = 0;

            for(int a=0; a < 3; a++)
            {
                for(int b=0; b < 3; b++)
                {
                    sum3 = sum3 + cGijk.idx(a, a, b) * cGijk.idx(b, i, j) - cGijk.idx(a, i, b) * cGijk.idx(b, a, j);
                }
            }

            cRij.idx(i, j) = sum - sum2 + sum3;
        }
    }*/

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value s1 = 0;

            for(int l=0; l < 3; l++)
            {
                for(int m=0; m < 3; m++)
                {
                    s1 = s1 + -0.5f * icY.idx(l, m) * hacky_differentiate(ctx, dcYij.idx(m, i, j), l);
                }
            }

            value s2 = 0;

            for(int k=0; k < 3; k++)
            {
                s2 = s2 + 0.5f * (cY.idx(k, i) * hacky_differentiate(ctx, cGi.idx(k), j) + cY.idx(k, j) * hacky_differentiate(ctx, cGi.idx(k), i));
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

    ///a / X
    value gA_X = 0;

    {
        float min_X = 0.001;

        gA_X = dual_if(X <= min_X,
        [&]()
        {
            ///linearly interpolate to 0
            value value_at_min = gA / min_X;

            return value_at_min / min_X;
        },
        [&]()
        {
            return gA / X;
        });
    }

    /*tensor<value, 3, 3> xgADiDjphi;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value p1 = -gA * hacky_differentiate(ctx, dX.idx(j), i);

            value p2 = gA_X * dX.idx(i) * dX.idx(j);

            value p3s = 0;

            for(int k=0; k < 3; k++)
            {
                p3s = p3s + christoff2.idx(k, i, j) * -dX.idx(k);
            }

            xgADiDjphi.idx(i, j) = 0.25f * (p1 + p2 - gA * p3s);
        }
    }

    ctx.pin(xgADiDjphi);

    tensor<value, 3, 3> xgARphiij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value s1XgA = -2 * xgADiDjphi.idx(i, j);

            value s2XgA = 0;

            for(int k=0; k < 3; k++)
            {
                for(int s=0; s < 3; s++)
                {
                    s2XgA = s2XgA + icY.idx(k, s) * xgADiDjphi.idx(s, k);
                }
            }

            s2XgA = -2 * cY.idx(i, j) * s2XgA;

            value s3XgA = 4 * (1.f/16.f) * gA_X * dX.idx(i) * dX.idx(j);

            value s4XgA = 0;

            for(int k=0; k < 3; k++)
            {
                for(int s=0; s < 3; s++)
                {
                    s4XgA = s4XgA + icY.idx(k, s) * (1/16.f) * gA_X * dX.idx(s) * dX.idx(k);
                }
            }

            s4XgA = -4 * cY.idx(i, j) * s4XgA;

            xgARphiij.idx(i, j) = s1XgA + s2XgA + s3XgA + s4XgA;
        }
    }*/

    ///https://indico.cern.ch/event/505595/contributions/1183661/attachments/1332828/2003830/sperhake.pdf
    tensor<value, 3, 3> xgARphiij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value sum = 0;

            for(int m=0; m < 3; m++)
            {
                for(int n=0; n < 3; n++)
                {
                    sum += gA * (cY.idx(i, j) / 2.f) * icY.idx(m, n) * gpu_covariant_derivative_low_vec(ctx, dX, cY, icY).idx(n, m);
                    sum += (cY.idx(i, j) / 2.f) * gA_X * -(3.f/2.f) * icY.idx(m, n) * dX.idx(m) * dX.idx(n);
                }
            }

            value p2 = (1/2.f) * (gA * gpu_covariant_derivative_low_vec(ctx, dX, cY, icY).idx(j, i) - gA_X * (1/2.f) * dX.idx(i) * dX.idx(j));

            xgARphiij.idx(i, j) = sum + p2;
        }
    }

    tensor<value, 3, 3> xgARij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            xgARij.idx(i, j) = xgARphiij.idx(i, j) + X * gA * cRij.idx(i, j);

            ctx.pin(xgARij.idx(i, j));
        }
    }

    tensor<value, 3, 3> Xdidja;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value Xderiv = X * gpu_covariant_derivative_low_vec(ctx, digA, cY, icY).idx(j, i);

            value s2 = 0.5f * (hacky_differentiate(ctx, X, i) * hacky_differentiate(ctx, gA, j) + hacky_differentiate(ctx, X, j) * hacky_differentiate(ctx, gA, i));

            value s3 = 0;

            for(int m=0; m < 3; m++)
            {
                for(int n=0; n < 3; n++)
                {
                    value v = -0.5f * cY.idx(i, j) * icY.idx(m, n) * hacky_differentiate(ctx, X, m) * hacky_differentiate(ctx, gA, n);

                    s3 += v;
                }
            }

            Xdidja.idx(i, j) = Xderiv + s2 + s3;
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

    ///not sure dtcaij is correct, need to investigate
    tensor<value, 3, 3> dtcAij;

    ///https://indico.cern.ch/event/505595/contributions/1183661/attachments/1332828/2003830/sperhake.pdf replaced with definition under bssn aux
    tensor<value, 3, 3> with_trace = -Xdidja + xgARij;

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
            value trace_free_part = gpu_trace_free(with_trace, cY, icY).idx(i, j);

            value p1 = trace_free_part;

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
            float Ka = 0.05f;

            dtcAij.idx(i, j) += Ka * gA * 0.5f *
                                                (gpu_covariant_derivative_low_vec(ctx, args.momentum_constraint, cY, icY).idx(i, j)
                                                 + gpu_covariant_derivative_low_vec(ctx, args.momentum_constraint, cY, icY).idx(j, i));
            #endif // DAMP_DTCAIJ
        }
    }

    tensor<value, 3, 3> icAij = raise_both(cA, cY, icY);

    value dtK = sum(tensor_upwind(ctx, gB, K)) - sum_multiply(icY.to_tensor(), Xdidja) + gA * (sum_multiply(icAij, cA) + (1/3.f) * K * K);

    ///these seem to suffer from oscillations
    tensor<value, 3> dtcGi;

    ///https://arxiv.org/pdf/gr-qc/0511048.pdf
    ///could likely eliminate the dphi term

    #ifdef SIMPLE_CHRISTOFFEL
    for(int i=0; i < 3; i++)
    {
        value sum = 0;

        for(int j=0; j < 3; j++)
        {
            value s1 = 0;

            for(int k=0; k < 3; k++)
            {
                s1 = s1 + icY.idx(j, k) * hacky_differentiate(ctx, digB.idx(k, i), j);
            }

            value s2 = 0;

            for(int k=0; k < 3; k++)
            {
                s2 = s2 + (1.f/3.f) * icY.idx(i, j) * hacky_differentiate(ctx, digB.idx(k, k), j);
            }

            value s3 = upwind_differentiate(ctx, gB.idx(j), cGi.idx(i), j);

            value s4 = -derived_cGi.idx(j) * hacky_differentiate(ctx, gB.idx(i), j);

            value s5 = (2.f/3.f) * derived_cGi.idx(i) * hacky_differentiate(ctx, gB.idx(j), j);

            value s6 = -2 * icAij.idx(i, j) * hacky_differentiate(ctx, gA, j);

            value s7 = 0;

            {
                value s8 = 0;

                for(int k=0; k < 3; k++)
                {
                    s8 = s8 + christoff2.idx(i, j, k) * icAij.idx(j, k);
                }

                value s9 = (-1/4.f) * gA_X * 6 * icAij.idx(i, j) * hacky_differentiate(ctx, X, j);

                value s10 = -(2.f/3.f) * icY.idx(i, j) * hacky_differentiate(ctx, K, j);

                s7 = 2 * (gA * s8 + s9 + gA * s10);
            }


            sum = sum + s1 + s2 + s3 + s4 + s5 + s6 + s7;
        }

        dtcGi.idx(i) = sum;
    }
    #endif // SIMPLE_CHRISTOFFEL

    ///https://arxiv.org/pdf/1205.5111v1.pdf 49
    ///made it to 58 with this
    #define CHRISTOFFEL_49
    #ifdef CHRISTOFFEL_49
    tensor<value, 3, 3> littlekij = unpinned_icY.to_tensor() * K;

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

        value s2 = 0;

        for(int j=0; j < 3; j++)
        {
            s2 += 2 * gA * -(2.f/3.f) * hacky_differentiate(ctx, littlekij.idx(i, j), j);
        }

        value s3 = 0;

        for(int j=0; j < 3; j++)
        {
            s3 += 2 * (-1.f/4.f) * gA_X * 6 * icAij.idx(i, j) * hacky_differentiate(ctx, X, j);
        }

        value s4 = 0;

        for(int j=0; j < 3; j++)
        {
            s4 += -2 * icAij.idx(i, j) * hacky_differentiate(ctx, gA, j);
        }

        value s5 = 0;

        for(int j=0; j < 3; j++)
        {
            s5 += upwind_differentiate(ctx, gB.idx(j), cGi.idx(i), j);
        }

        value s6 = 0;

        for(int j=0; j < 3; j++)
        {
            s6 += -derived_cGi.idx(j) * digB.idx(j, i);
        }

        value s7 = 0;

        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                s7 += icY.idx(j, k) * hacky_differentiate(ctx, digB.idx(k, i), j);
            }
        }

        value s8 = 0;

        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                s8 += (1.f/3.f) * icY.idx(i, j) * hacky_differentiate(ctx, digB.idx(k, k), j);
            }
        }

        value s9 = 0;

        for(int k=0; k < 3; k++)
        {
            s9 += (2.f/3.f) * digB.idx(k, k) * derived_cGi.idx(i);
        }

        ///this is the only instanced of derived_cGi that might want to be regular cGi
        value s10 = (2.f/3.f) * -2 * gA * K * derived_cGi.idx(i);

        dtcGi.idx(i) = s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10;

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
            bkk += digB.idx(k, k);
        }

        float E = 1;

        value lambdai = (2.f/3.f) * (bkk - 2 * gA * K)
                        - digB.idx(i, i)
                        - (2.f/5.f) * gA * raise_second_index(cA, cY, icY).idx(i, i);

        dtcGi.idx(i) += -(1 + E) * step(lambdai) * lambdai * bigGi.idx(i);
        #endif // EQ_50
    }
    #endif // CHRISTOFFEL_49

    ///https://arxiv.org/pdf/1410.8607.pdf
    ///https://arxiv.org/pdf/gr-qc/0210050.pdf (88)
    /*
    //auto f_a = 8 / (3 * gA * (3 - gA));
    auto f_a = 2 / gA;

    value dtgA = -gA * gA * f_a * K + lie_derivative(ctx, gB, gA);*/

    ///so -gA * gA * f_a * K with f_a = 8 / (3 * gA * (3 - gA))
    ///-gA * f_a * K with f_a = 8 / (3 * (3 - gA)) = 8/(9 - 3 * gA)
    /*auto f_a_reduced = 8 / (3 * (3 - gA));
    value dtgA = -gA * f_a_reduced * K + lie_derivative(ctx, gB, gA);*/

    value dtgA = lie_derivative(ctx, gB, gA) - 2 * gA * K;

    #ifndef USE_GBB
    ///https://arxiv.org/pdf/gr-qc/0605030.pdf 26
    ///todo: remove this
    tensor<value, 3> bjdjbi;

    for(int i=0; i < 3; i++)
    {
        value v = 0;

        for(int j=0; j < 3; j++)
        {
           v += upwind_differentiate(ctx, gB.idx(j), gB.idx(i), j);
        }

        bjdjbi.idx(i) = v;
    }

    float N = 2;

    tensor<value, 3> dtgB = (3.f/4.f) * derived_cGi + bjdjbi - N * gB;

    tensor<value, 3> dtgBB;
    dtgBB.idx(0) = 0;
    dtgBB.idx(1) = 0;
    dtgBB.idx(2) = 0;

    #else

    tensor<value, 3> dtgB;
    tensor<value, 3> dtgBB;

    ///https://arxiv.org/pdf/gr-qc/0511048.pdf (11)
    for(int i=0; i < 3; i++)
    {
        dtgB.idx(i) = gBB.idx(i);
    }

    for(int i=0; i < 3; i++)
    {
        float N = 2;

        dtgBB.idx(i) = (3.f/4.f) * dtcGi.idx(i) - N * gBB.idx(i);
    }

    #endif // USE_GBB

    /*value scalar_curvature = 0;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            scalar_curvature = scalar_curvature + iYij.idx(i, j) * Rij.idx(i, j);
        }
    }*/

    for(int i=0; i < 6; i++)
    {
        std::string name = "dtcYij" + std::to_string(i);

        vec2i idx = linear_indices[i];

        ctx.add(name, dtcYij.idx(idx.x(), idx.y()));
    }

    ctx.add("dtX", dtX);

    for(int i=0; i < 6; i++)
    {
        std::string name = "dtcAij" + std::to_string(i);

        vec2i idx = linear_indices[i];

        ctx.add(name, dtcAij.idx(idx.x(), idx.y()));
    }

    ctx.add("dtK", dtK);

    for(int i=0; i < 3; i++)
    {
        std::string name = "dtcGi" + std::to_string(i);

        ctx.add(name, dtcGi.idx(i));
    }

    ctx.add("dtgA", dtgA);

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

    ctx.add("dttheta", 0);

    //ctx.add("scalar_curvature", scalar_curvature);
}
#endif // 0

value advect(equation_context& ctx, const tensor<value, 3>& gB, const value& in)
{
    value ret = 0;

    for(int i=0; i < 3; i++)
    {
        ret += gB.idx(i) * hacky_differentiate(ctx, in, i);
    }

    return ret;
}

tensor<value, 3> advect(equation_context& ctx, const tensor<value, 3>& gB, const tensor<value, 3>& in)
{
    tensor<value, 3> ret;

    for(int i=0; i < 3; i++)
    {
        value sum = 0;

        for(int k=0; k < 3; k++)
        {
            sum += gB.idx(k) * hacky_differentiate(ctx, in.idx(i), k);
        }

        ret.idx(i) = sum;
    }

    return ret;
}

tensor<value, 3, 3> advect(equation_context& ctx, const tensor<value, 3>& gB, const tensor<value, 3, 3>& in)
{
    tensor<value, 3, 3> ret;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value sum = 0;

            for(int k=0; k < 3; k++)
            {
                sum += gB.idx(k) * hacky_differentiate(ctx, in.idx(i, j), k);
            }

            ret.idx(i, j) = sum;
        }
    }

    return ret;
}

inline
void build_eqs(equation_context& ctx)
{
    standard_arguments args(false);

    int index_table[3][3] = {{0, 1, 2},
                             {1, 3, 4},
                             {2, 4, 5}};

    vec2i linear_indices[6] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};

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

    unit_metric<value, 3, 3> cY = args.cY;

    inverse_metric<value, 3, 3> icY = cY.invert();
    inverse_metric<value, 3, 3> unpinned_icY = cY.invert();

    ctx.pin(icY);

    tensor<value, 3, 3> cA = args.cA;
    tensor<value, 3, 3> icA = raise_both(cA, cY, icY);

    auto unpinned_cA = cA;
    //ctx.pin(cA);

    ///the christoffel symbol
    tensor<value, 3> cGi = args.cGi;

    tensor<value, 3> digA;
    digA.idx(0).make_value("digA0[IDX(ix,iy,iz)]");
    digA.idx(1).make_value("digA1[IDX(ix,iy,iz)]");
    digA.idx(2).make_value("digA2[IDX(ix,iy,iz)]");

    tensor<value, 3> dX;
    dX.idx(0).make_value("dX0[IDX(ix,iy,iz)]");
    dX.idx(1).make_value("dX1[IDX(ix,iy,iz)]");
    dX.idx(2).make_value("dX2[IDX(ix,iy,iz)]");

    tensor<value, 3, 3> digB;

    ///derivative
    for(int i=0; i < 3; i++)
    {
        ///value
        for(int j=0; j < 3; j++)
        {
            int idx = i + j * 3;

            std::string name = "digB" + std::to_string(idx) + "[IDX(ix,iy,iz)]";

            digB.idx(i, j).make_value(name);
        }
    }

    value gA = args.gA;

    tensor<value, 3> gB = args.gB;

    #ifdef USE_GBB
    tensor<value, 3> gBB;
    gBB.idx(0).make_value("gBB0[IDX(ix,iy,iz)]");
    gBB.idx(1).make_value("gBB1[IDX(ix,iy,iz)]");
    gBB.idx(2).make_value("gBB2[IDX(ix,iy,iz)]");
    #endif // USE_GBB

    value X = args.X;
    value K = args.K;

    tensor<value, 3, 3, 3> dcYij;

    for(int k=0; k < 3; k++)
    {
        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                int symmetric_index = index_table[i][j];

                int final_index = k + symmetric_index * 3;

                std::string name = "dcYij" + std::to_string(final_index) + "[IDX(ix,iy,iz)]";

                dcYij.idx(k, i, j) = name;
            }
        }
    }

    ///dcgA alias
    for(int i=0; i < 3; i++)
    {
        value v = hacky_differentiate(ctx, gA, i, false);

        ctx.alias(v, digA.idx(i));
    }

    ///dcgB alias
    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value v = hacky_differentiate(ctx, gB.idx(j), i, false);

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
                value v = hacky_differentiate(ctx, cY.idx(i, j), k, false);

                ctx.alias(v, dcYij.idx(k, i, j));
            }
        }
    }

    for(int k=0; k < 3; k++)
    {
        for(int i=0; i < 6; i++)
        {
            vec2i idx = linear_indices[i];

            hacky_differentiate(ctx, cY.idx(idx.x(), idx.y()), k);
        }

        for(int i=0; i < 6; i++)
        {
            hacky_differentiate(ctx, "cA" + std::to_string(i), k);
        }

        for(int i=0; i < 3; i++)
        {
            hacky_differentiate(ctx, "cGi" + std::to_string(i), k);
        }

        hacky_differentiate(ctx, "K", k);
        hacky_differentiate(ctx, "X", k);
        hacky_differentiate(ctx, "theta", k);
    }

    tensor<value, 3, 3, 3> christoff1 = gpu_christoffel_symbols_1(ctx, cY);
    //tensor<value, 3, 3, 3> christoff2 = gpu_christoffel_symbols_2(ctx, cY, icY);

    ///https://arxiv.org/pdf/1810.12346.pdf 2.17
    tensor<value, 3, 3, 3> christoff2;

    for(int k=0; k < 3; k++)
    {
        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                value sum = 0;

                for(int l=0; l < 3; l++)
                {
                    sum += icY.idx(k, l) * christoff1.idx(l, i, j);
                }

                christoff2.idx(k, i, j) = sum;
            }
        }
    }

    ctx.pin(christoff2);

    tensor<value, 3> derived_cGi;

    for(int i=0; i < 3; i++)
    {
        value sum = 0;

        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                sum += icY.idx(j, k) * christoff2.idx(i, j, k);
            }
        }

        derived_cGi.idx(i) = sum;
    }

    value theta = args.theta;

    tensor<value, 3, 3> aIJ = raise_both(cA, cY, icY);

    value aij_aIJ = 0;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            aij_aIJ += cA.idx(i, j) * aIJ.idx(i, j);
        }
    }

    tensor<value, 3> cZ = 0.5f * (cGi - derived_cGi);

    float s = 1;
    float c = 1;
    float littlek = 0.01f;

    tensor<value, 3> ccGi;

    ///https://arxiv.org/pdf/1810.12346.pdf 2.15
    ccGi = c * cGi + (1 - c) * derived_cGi;

    ///TODO: FIXY FIX THIS IS ONLY POTENTIALLY RIGHT FOR s=1 c=1
    tensor<value, 3> sccGi = cGi;

    tensor<value, 3, 3> hat_Rij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value p1 = 0;

            for(int k=0; k < 3; k++)
            {
                for(int l=0; l < 3; l++)
                {
                    p1 += -0.5f * icY.idx(k, l) * hacky_differentiate(ctx, dcYij.idx(l, i, j), k);
                }
            }

            value p2 = 0;

            for(int k=0; k < 3; k++)
            {
                p2 += 0.5f * (cY.idx(k, i) * hacky_differentiate(ctx, cGi.idx(k), j) + cY.idx(k, j) * hacky_differentiate(ctx, cGi.idx(k), i));
            }

            value p3 = 0;

            for(int k=0; k < 3; k++)
            {
                p3 += 0.5f * (christoff1.idx(i, j, k) * ccGi.idx(k) + christoff1.idx(j, i, k) * ccGi.idx(k));
            }

            value p4 = 0;

            for(int k=0; k < 3; k++)
            {
                for(int l=0; l < 3; l++)
                {
                    for(int m=0; m < 3; m++)
                    {
                        p4 += icY.idx(k, l) *
                        (christoff2.idx(m, k, i) * christoff1.idx(m, l, j)
                        + 0.5f * 2 * (christoff2.idx(m, k, i) * christoff1.idx(j, m, l) + christoff2.idx(m, k, j) * christoff1.idx(i, m, l)));
                    }
                }
            }

            value s1 = X * (p1 + p2 + p3 + p4);


            value p5 = hacky_differentiate(ctx, dX.idx(j), i);

            value p6 = -0.5f * (1/args.clamped_X) * dX.idx(i) * dX.idx(j);

            value p7 = 0;

            for(int k=0; k < 3; k++)
            {
                for(int l=0; l < 3; l++)
                {
                    p7 += cY.idx(i, j) * icY.idx(k, l) * (hacky_differentiate(ctx, dX.idx(l), k) - (3.f/2.f) * (1/args.clamped_X) * dX.idx(k) * dX.idx(l));
                }
            }

            value p8 = 0;

            for(int k=0; k < 3; k++)
            {
                p8 += -christoff2.idx(k, i, j) * dX.idx(k);
            }

            value p9 = 0;

            for(int k=0; k < 3; k++)
            {
                p9 += -cY.idx(i, j) * ccGi.idx(k) * dX.idx(k);
            }

            value s2 = 0.5f * (p5 + p6 + p7 + p8 + p9);

            hat_Rij.idx(i, j) = s1 + s2;
        }
    }

    value icY_hatRij = 0;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            icY_hatRij += icY.idx(i, j) * hat_Rij.idx(i, j);
        }
    }

    value Ziterms = 0;

    for(int i=0; i < 3; i++)
    {
        Ziterms += cZ.idx(i) * (gA * dX.idx(i) * X * digA.idx(i));
    }

    value Dttheta = gA * ((1.f/3.f) * K * K - (1 - (4.f/3.f) * s) * K * theta - littlek * theta - (2.f/3.f) * s * theta * theta - 0.5f * aij_aIJ + 0.5f * icY_hatRij) +
                    Ziterms;

    value dttheta = Dttheta + advect(ctx, gB, theta);


    value dibi = 0;

    for(int i=0; i < 3; i++)
    {
        dibi += hacky_differentiate(ctx, gB.idx(i), i);
    }

    value DtX = (2.f/3.f) * X * (gA * (K + 2 * s * theta) - dibi);

    value dtX = DtX + advect(ctx, gB, X);

    value DtK = 0;

    {
        value p1 = gA * (
                         (1 - (2.f/3.f) * s) * K * K
                         - 2 * (1.f - (5.f/3.f) * s) * K * theta
                         - (3 - 4 * s) * littlek * theta
                         + (4.f/3.f) * s * theta * theta
                         + s * aij_aIJ
                         + 2 * (1 - s) * c * sum_multiply(cZ, dX));

        value p2 = X * sum_multiply(sccGi, digA);

        value p3 = 0;

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                p3 += icY.idx(i, j) *
                (-X * hacky_differentiate(ctx, digA.idx(j), i)
                 + 0.5f * dX.idx(i) * digA.idx(j)
                 + gA * ((1 - s) * hat_Rij.idx(i, j)));
            }
        }

        DtK = p1 + p2 + p3;
    }

    value dtK = DtK + advect(ctx, gB, K);

    tensor<value, 3> DtcGi;

    for(int i=0; i < 3; i++)
    {
        value p1 = 0;

        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                p1 += christoff2.idx(i, j, k) * icA.idx(j, k);
            }
        }

        value p2 = 0;

        for(int j=0; j < 3; j++)
        {
            p2 += -(3.f/2.f) * icA.idx(i, j) * (1/args.clamped_X) * dX.idx(j);
        }

        value p3 = 0;

        for(int j=0; j < 3; j++)
        {
            p3 += icY.idx(i, j) * ((1 - (4.f/3.f) * s) * hacky_differentiate(ctx, theta, j) - (2.f/3.f) * hacky_differentiate(ctx, K, j));
        }

        value p4 = -c * ((2.f/3.f) * K + (4.f/3.f) * s * theta + littlek) * cZ.idx(i);

        value s1 = 2 * gA * (p1 + p2 + p3 + p4);

        value p5 = 0;

        for(int j=0; j < 3; j++)
        {
            p5 += -2 * c * theta * icY.idx(i, j) * digA.idx(j);
        }

        value p6 = 0;

        for(int j=0; j < 3; j++)
        {
            p6 += -2.f * icA.idx(i, j) * digA.idx(j);
        }

        value p7 = 0;

        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                p7 += icY.idx(j, k) * hacky_differentiate(ctx, digB.idx(k, i), j);
            }
        }

        value p8 = 0;

        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                p8 += (1.f/3.f) * icY.idx(i, j) * hacky_differentiate(ctx, digB.idx(k, k), j);
            }
        }

        value p9 = 0;

        for(int j=0; j < 3; j++)
        {
            p9 += -ccGi.idx(j) * digB.idx(j, i);
        }

        value p10 = 0;

        for(int j=0; j < 3; j++)
        {
            p10 += (2.f/3.f) * ccGi.idx(i) * digB.idx(j, j);
        }

        DtcGi.idx(i) = s1 + p5 + p6 + p7 + p8 + p9 + p10;
    }

    tensor<value, 3> dtcGi = DtcGi + advect(ctx, gB, cGi);

    tensor<value, 3, 3> DtcYij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value p1 = -2 * gA * cA.idx(i, j);

            value p2 = 0;

            for(int k=0; k < 3; k++)
            {
                p2 += cY.idx(i, k) * digB.idx(j, k) + cY.idx(j, k) * digB.idx(i, k) - (2.f/3.f) * cY.idx(i, j) * digB.idx(k, k);
            }

            DtcYij.idx(i, j) = p1 + p2;
        }
    }

    tensor<value, 3, 3> dtcYij = DtcYij + advect(ctx, gB, cY.to_tensor());

    tensor<value, 3, 3> inner_trace;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value inner_1 = -hacky_differentiate(ctx, digA.idx(j), i);

            value inner_2 = 0;

            for(int k=0; k < 3; k++)
            {
                inner_2 += christoff2.idx(k, i, j) * digA.idx(k);
            }

            value isum1 = X * (inner_1 + inner_2);

            value isum2 = -0.5f * (dX.idx(i) * digA.idx(j) + dX.idx(j) * digA.idx(i));

            value isum3 = 0;

            for(int k=0; k < 3; k++)
            {
                isum3 += 2 * c * gA * cZ.idx(k) * 0.5f * (cY.idx(k, i) * dX.idx(j) + cY.idx(k, j) * dX.idx(i));
            }

            value isum4 = gA * hat_Rij.idx(i, j);

            inner_trace.idx(i, j) = isum1 + isum2 + isum3 + isum4;
        }
    }

    tensor<value, 3, 3> DtcAij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value p1 = 0;

            for(int k=0; k < 3; k++)
            {
                for(int l=0; l < 3; l++)
                {
                    p1 += -2 * icY.idx(k, l) * cA.idx(i, k) * cA.idx(j, l);
                }
            }

            value p2 = (K - 2 * (1 - s) * theta) * cA.idx(i, j);

            value s1 = gA * (p1 + p2);

            value p3 = 0;

            for(int k=0; k < 3; k++)
            {
                p3 += cA.idx(i, k) * digB.idx(j, k) + cA.idx(j, k) * digB.idx(i, k) - (2.f/3.f) * cA.idx(i, j) * digB.idx(k, k);
            }

            value s2 = p3;

            value s3 = gpu_trace_free(inner_trace, cY, icY).idx(i, j);

            DtcAij.idx(i, j) = s1 + s2 + s3;
        }
    }

    tensor<value, 3, 3> dtcAij = DtcAij + advect(ctx, gB, cA);

    ///https://arxiv.org/pdf/1410.8607.pdf
    ///https://arxiv.org/pdf/gr-qc/0210050.pdf (88)
    /*
    //auto f_a = 8 / (3 * gA * (3 - gA));
    auto f_a = 2 / gA;

    value dtgA = -gA * gA * f_a * K + lie_derivative(ctx, gB, gA);*/

    ///so -gA * gA * f_a * K with f_a = 8 / (3 * gA * (3 - gA))
    ///-gA * f_a * K with f_a = 8 / (3 * (3 - gA)) = 8/(9 - 3 * gA)
    /*auto f_a_reduced = 8 / (3 * (3 - gA));
    value dtgA = -gA * f_a_reduced * K + lie_derivative(ctx, gB, gA);*/

    value dtgA = lie_derivative(ctx, gB, gA) - 2 * gA * K;

    ///https://arxiv.org/pdf/gr-qc/0605030.pdf 26
    ///todo: remove this
    tensor<value, 3> bjdjbi;

    for(int i=0; i < 3; i++)
    {
        value v = 0;

        for(int j=0; j < 3; j++)
        {
           v += upwind_differentiate(ctx, gB.idx(j), gB.idx(i), j);
        }

        bjdjbi.idx(i) = v;
    }

    float N = 2;

    tensor<value, 3> dtgB = (3.f/4.f) * derived_cGi + bjdjbi - N * gB;

    tensor<value, 3> dtgBB;
    dtgBB.idx(0) = 0;
    dtgBB.idx(1) = 0;
    dtgBB.idx(2) = 0;

    for(int i=0; i < 6; i++)
    {
        std::string name = "dtcYij" + std::to_string(i);

        vec2i idx = linear_indices[i];

        ctx.add(name, dtcYij.idx(idx.x(), idx.y()));
    }

    ctx.add("dtX", dtX);

    for(int i=0; i < 6; i++)
    {
        std::string name = "dtcAij" + std::to_string(i);

        vec2i idx = linear_indices[i];

        ctx.add(name, dtcAij.idx(idx.x(), idx.y()));
    }

    ctx.add("dtK", dtK);

    for(int i=0; i < 3; i++)
    {
        std::string name = "dtcGi" + std::to_string(i);

        ctx.add(name, dtcGi.idx(i));
    }

    ctx.add("dtgA", dtgA);

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

    ctx.add("dttheta", dttheta);

    ctx.add("debug_p1", 0);
    ctx.add("debug_p2", 0);
    ctx.add("debug_p3", 0);

    //ctx.add("scalar_curvature", scalar_curvature);
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

inline
int64_t factorial(int i)
{
    if(i == 0)
        return 1;

    return i * factorial(i - 1);
}

inline
dual_types::complex<float> expi(float val)
{
    return dual_types::complex<float>(cos(val), sin(val));
}

///https://arxiv.org/pdf/1906.03877.pdf 8
///aha!
///https://arxiv.org/pdf/0709.0093.pdf
///at last! A non horrible reference and non gpl reference for negative spin!
///this is where the cactus code comes from as well
template<typename T>
inline
dual_types::complex<T> sYlm(int negative_s, int l, int m, T theta, T phi)
{
    int s = negative_s;

    auto dlms = [&]()
    {
        int k1 = std::max(0, m - s);
        int k2 = std::min(l + m, l - s);

        T sum = 0;

        for(int k=k1; k <= k2; k++)
        {
            float cp1 = (double)(pow(-1, k) * sqrt((double)(factorial(l + m) * factorial(l - m) * factorial(l + s) * factorial(l - s)))) /
                        ((double)(factorial(l + m - k) * factorial(l - s - k) * factorial(k) * factorial(k + s - m)));

            assert(isfinite(cp1));

            T cp2 = pow(cos(theta/2.f), 2 * l + m - s - 2 * k);
            T cp3 = pow(sin(theta/2.f), 2 * k + s - m);

            sum = sum + cp1 * cp2 * cp3;
        }

        return sum;
    };

    T coeff = pow(-1, s) * sqrt((2 * l + 1) / (4 * M_PI));

    return coeff * dlms() * expi(m * phi);
}

///https://pomax.github.io/bezierinfo/legendre-gauss.html
///https://cbeentjes.github.io/files/Ramblings/QuadratureSphere.pdf
///http://homepage.divms.uiowa.edu/~atkinson/papers/SphereQuad1982.pdf

template<typename T>
inline
auto integrate(float lowerbound, float upperbound, const T& f_x, int n)
{
    using variable_type = decltype(f_x(0.f));

    variable_type sum = 0;

    std::vector<float> weights = get_legendre_weights(n);
    std::vector<float> nodes = get_legendre_nodes(n);

    for(int i=0; i < n; i++)
    {
        float wi = weights[i];
        float xi = nodes[i];

        float final_val = ((upperbound - lowerbound)/2.f) * xi + (upperbound + lowerbound) / 2.f;

        auto func_eval = wi * f_x(final_val);

        sum = sum + func_eval;
    }

    return ((upperbound - lowerbound) / 2.f) * sum;
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
    for(int i=0; i < n; i++)
    {
        variable_type lsum = 0;
        float xi = nodes[i];

        ///theta
        for(int j=0; j < n; j++)
        {
            float w = weights[j];
            float xj = nodes[j];

            float final_valphi = ((iupper - ilower)/2.f) * xi + (iupper + ilower) / 2.f;
            float final_valtheta = ((jupper - jlower)/2.f) * xj + (jupper + jlower) / 2.f;

            auto func_eval = w * f_theta_phi(final_valtheta, final_valphi);

            //printf("VPhi %f %f\n", final_valtheta, final_valphi);
            //printf("Gval %f\n", func_eval.real);

            lsum = lsum + func_eval;
        }

        sum = sum + weights[i] * lsum;
    }

    return ((iupper - ilower) / 2.f) * ((jupper - jlower) / 2.f) * sum;
}

///this isn't correct at all. The integration might be fine, but we can't take the spherical harmonics of a constant
inline
dual_types::complex<float> get_harmonic(const dual_types::complex<float>& value, int l, int m)
{
    auto func = [&](float theta, float phi)
    {
        dual_types::complex<float> harmonic = sYlm(2, l, m, theta, phi);

        //printf("Hreal %f\n", harmonic.real);

        return value * harmonic;
    };

    int n = 16;

    dual_types::complex<float> harmonic = spherical_integrate(func, n);

    return harmonic;
}

///https://scc.ustc.edu.cn/zlsc/sugon/intel/ipp/ipp_manual/IPPM/ippm_ch9/ch9_SHT.htm this states you can approximate
///a spherical harmonic transform integral with simple summation
///assumes unigrid
#if 0
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

    int index_table[3][3] = {{0, 1, 2},
                             {1, 3, 4},
                             {2, 4, 5}};

    unit_metric<value, 3, 3> cY;

    cY.idx(0, 0).make_value("cY0[IDX(ix,iy,iz)]"); cY.idx(0, 1).make_value("cY1[IDX(ix,iy,iz)]"); cY.idx(0, 2).make_value("cY2[IDX(ix,iy,iz)]");
    cY.idx(1, 0).make_value("cY1[IDX(ix,iy,iz)]"); cY.idx(1, 1).make_value("cY3[IDX(ix,iy,iz)]"); cY.idx(1, 2).make_value("cY4[IDX(ix,iy,iz)]");
    cY.idx(2, 0).make_value("cY2[IDX(ix,iy,iz)]"); cY.idx(2, 1).make_value("cY4[IDX(ix,iy,iz)]"); cY.idx(2, 2).make_value("cY5[IDX(ix,iy,iz)]");

    inverse_metric<value, 3, 3> icY = cY.invert();

    ctx.pin(icY);

    tensor<value, 3, 3> cA;

    cA.idx(0, 0).make_value("cA0[IDX(ix,iy,iz)]"); cA.idx(0, 1).make_value("cA1[IDX(ix,iy,iz)]"); cA.idx(0, 2).make_value("cA2[IDX(ix,iy,iz)]");
    cA.idx(1, 0).make_value("cA1[IDX(ix,iy,iz)]"); cA.idx(1, 1).make_value("cA3[IDX(ix,iy,iz)]"); cA.idx(1, 2).make_value("cA4[IDX(ix,iy,iz)]");
    cA.idx(2, 0).make_value("cA2[IDX(ix,iy,iz)]"); cA.idx(2, 1).make_value("cA4[IDX(ix,iy,iz)]"); cA.idx(2, 2).make_value("cA5[IDX(ix,iy,iz)]");

    tensor<value, 3> cGi;
    cGi.idx(0).make_value("cGi0[IDX(ix,iy,iz)]");
    cGi.idx(1).make_value("cGi1[IDX(ix,iy,iz)]");
    cGi.idx(2).make_value("cGi2[IDX(ix,iy,iz)]");

    tensor<value, 3> digA;
    digA.idx(0).make_value("ik.digA[0]");
    digA.idx(1).make_value("ik.digA[1]");
    digA.idx(2).make_value("ik.digA[2]");

    tensor<value, 3> dX;
    dX.idx(0).make_value("ik.dX[0]");
    dX.idx(1).make_value("ik.dX[1]");
    dX.idx(2).make_value("ik.dX[2]");

    tensor<value, 3, 3> digB;

    ///derivative
    for(int i=0; i < 3; i++)
    {
        ///value
        for(int j=0; j < 3; j++)
        {
            int idx = i * 3 + j;

            std::string name = "ik.digB[" + std::to_string(idx) + "]";

            digB.idx(i, j).make_value(name);
        }
    }

    value gA;
    gA.make_value("gA[IDX(ix,iy,iz)]");

    tensor<value, 3> gB;
    gB.idx(0).make_value("gB0[IDX(ix,iy,iz)]");
    gB.idx(1).make_value("gB1[IDX(ix,iy,iz)]");
    gB.idx(2).make_value("gB2[IDX(ix,iy,iz)]");

    #ifdef USE_GBB
    tensor<value, 3> gBB;
    gBB.idx(0).make_value("v.gBB0");
    gBB.idx(1).make_value("v.gBB1");
    gBB.idx(2).make_value("v.gBB2");
    #endif // USE_GBB

    value X;
    X.make_value("X[IDX(ix,iy,iz)]");

    value K;
    K.make_value("K[IDX(ix,iy,iz)]");

    tensor<value, 3, 3, 3> dcYij;

    for(int k=0; k < 3; k++)
    {
        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                int symmetric_index = index_table[i][j];

                int final_index = k * 6 + symmetric_index;

                std::string name = "ik.dcYij[" + std::to_string(final_index) + "]";

                dcYij.idx(k, i, j) = name;
            }
        }
    }

    ///dcgA alias
    for(int i=0; i < 3; i++)
    {
        value v = hacky_differentiate(ctx, gA, i, false);

        ctx.alias(v, digA.idx(i));
    }

    ///dcgB alias
    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value v = hacky_differentiate(ctx, gB.idx(j), i, false);

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
                value v = hacky_differentiate(ctx, cY.idx(i, j), k, false);

                ctx.alias(v, dcYij.idx(k, i, j));
            }
        }
    }

    for(int k=0; k < 3; k++)
    {
        for(int i=0; i < 6; i++)
        {
            hacky_differentiate(ctx, "cY" + std::to_string(i), k);
        }

        for(int i=0; i < 6; i++)
        {
            hacky_differentiate(ctx, "cA" + std::to_string(i), k);
        }

        for(int i=0; i < 3; i++)
        {
            hacky_differentiate(ctx, "cGi" + std::to_string(i), k);
        }

        hacky_differentiate(ctx, "K", k);
        hacky_differentiate(ctx, "X", k);
    }

    for(int k=0; k < 3; k++)
    {
        for(int i=0; i < 3 * 6; i++)
        {
            hacky_differentiate(ctx, "ik.dcYij[" + std::to_string(i) + "]", k);
        }

        for(int i=0; i < 3; i++)
        {
            hacky_differentiate(ctx, "ik.digA[" + std::to_string(i) + "]", k);
        }

        for(int i=0; i < 3; i++)
        {
            hacky_differentiate(ctx, "ik.digB[" + std::to_string(i) + "]", k);
        }

        for(int i=0; i < 3; i++)
        {
            hacky_differentiate(ctx, "ik.dX[" + std::to_string(i) + "]", k);
        }
    }

    tensor<value, 3, 3, 3> christoff1 = gpu_christoffel_symbols_1(ctx, cY);
    tensor<value, 3, 3, 3> christoff2 = gpu_christoffel_symbols_2(ctx, cY, icY);

    tensor<value, 3> derived_cGi;

    ///https://arxiv.org/pdf/gr-qc/0206072.pdf page 4
    ///or https://arxiv.org/pdf/gr-qc/0511048.pdf after 7 actually
    for(int i=0; i < 3; i++)
    {
        value sum = 0;

        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                sum = sum + icY.idx(j, k) * christoff2.idx(i, j, k);
            }
        }

        derived_cGi.idx(i) = sum;
    }

    ///NEEDS SCALING???
    vec<3, value> pos;
    pos.x() = "offset.x";
    pos.y() = "offset.y";
    pos.z() = "offset.z";

    tensor<value, 3, 3> cRij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value s1 = 0;

            for(int l=0; l < 3; l++)
            {
                for(int m=0; m < 3; m++)
                {
                    s1 = s1 + -0.5f * icY.idx(l, m) * hacky_differentiate(ctx, dcYij.idx(m, i, j), l);
                }
            }

            value s2 = 0;

            for(int k=0; k < 3; k++)
            {
                s2 = s2 + 0.5f * (cY.idx(k, i) * hacky_differentiate(ctx, cGi.idx(k), j) + cY.idx(k, j) * hacky_differentiate(ctx, cGi.idx(k), i));
            }

            value s3 = 0;

            ///could factor out cGi
            for(int k=0; k < 3; k++)
            {
                s3 = s3 + 0.5f * (derived_cGi.idx(k) * christoff1.idx(i, j, k) + derived_cGi.idx(k) * christoff1.idx(j, i, k));
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

    ctx.pin(cRij);

    tensor<value, 3, 3> Rphiij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value s1 = -2 * gpu_covariant_derivative_low_vec(ctx, dphi, cY, icY).idx(j, i);

            value s2 = 0;

            for(int l=0; l < 3; l++)
            {
                s2 = s2 + gpu_high_covariant_derivative_vec(ctx, dphi, cY, icY).idx(l, l);
            }

            s2 = -2 * cY.idx(i, j) * s2;

            value s3 = 4 * (dphi.idx(i)) * (dphi.idx(j));

            value s4 = 0;

            for(int l=0; l < 3; l++)
            {
                s4 = s4 + raise_index(dphi, cY, icY).idx(l) * dphi.idx(l);
            }

            s4 = -4 * cY.idx(i, j) * s4;

            Rphiij.idx(i, j) = s1 + s2 + s3 + s4;
        }
    }

    ctx.pin(Rphiij);

    tensor<value, 3, 3> Rij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            Rij.idx(i, j) = Rphiij.idx(i, j) + cRij.idx(i, j);

        }
    }

    ctx.pin(Rij);

    tensor<value, 3, 3> Kij;

    ///https://arxiv.org/pdf/gr-qc/0505055.pdf 3.6
    ///https://arxiv.org/pdf/gr-qc/0511048.pdf note that these equations are the same (just before (1)), its because they're using the non conformal metric
    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            Kij.idx(i, j) = (1/X) * (cA.idx(i, j) + (1.f/3.f) * cY.idx(i, j) * K);
        }
    }

    metric<value, 3, 3> Yij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            Yij.idx(i, j) = cY.idx(i, j) / X;
        }
    }

    inverse_metric<value, 3, 3> iYij = Yij.invert();

    ctx.pin(iYij);

    //auto christoff_Y = gpu_christoffel_symbols_2(ctx, Yij, iYij);

    tensor<value, 3, 3, 3> xcChristoff;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                value lsum = 0;

                for(int l=0; l < 3; l++)
                {
                    lsum = lsum - cY.idx(i, j) * icY.idx(k, l) * hacky_differentiate(ctx, X, l);
                }

                xcChristoff.idx(k, i, j) = X * christoff2.idx(k, i, j) - 0.5f * (kronecker.idx(k,i) * hacky_differentiate(ctx, X, j) + kronecker.idx(k, j) * hacky_differentiate(ctx, X, i) + lsum);

                ctx.pin(xcChristoff.idx(k, i, j));
            }
        }
    }

    tensor<value, 3, 3, 3> christoff_Y;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                christoff_Y.idx(i, j, k) = xcChristoff.idx(i, j, k) / X;
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
                value deriv = hacky_differentiate(ctx, Kij.idx(a, b), c);

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

    ///can raise and lower the indices... its a regular tensor bizarrely
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

    tensor<value, 3, 3, 3> eijk_tensor;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                eijk_tensor.idx(i, j, k) = sqrt(Yij.det()) * eijk.idx(i, j, k);
            }
        }
    }

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

    vec<3, value> v1a = {-pos.y(), pos.x(), 0};
    vec<3, value> v2a = {pos.x(), pos.y(), pos.z()};
    vec<3, value> v3a;

    for(int a=0; a < 3; a++)
    {
        value sum = 0;

        for(int d=0; d < 3; d++)
        {
            for(int c=0; c < 3; c++)
            {
                for(int b=0; b < 3; b++)
                {
                    sum = sum + iYij.idx(a, d) * eijk.idx(d, b, c) * v1a[b] * v2a[c];
                }
            }
        }

        v3a[a] = pow(Yij.det(), 0.5f) * sum;
    }

    auto v_idx = [&](int idx)
    {
        if(idx == 0)
            return v1a;

        if(idx == 1)
            return v2a;

        if(idx == 2)
            return v3a;

        assert(false);
    };

    auto wij = [&](int i, int j)
    {
        auto v_i = v_idx(i);
        auto v_j = v_idx(j);

        value sum = 0;

        for(int a=0; a < 3; a++)
        {
            for(int b=0; b < 3; b++)
            {
                sum = sum + v_i[a] * v_j[b] * Yij.idx(a, b);
            }
        }

        return sum;
    };

    ///https://arxiv.org/pdf/gr-qc/0104063.pdf 5.7. I already have code for doing this but lets stay exact
    v1a = v1a / sqrt(wij(0, 0));
    v2a = (v2a - v1a * wij(0, 1)) / (wij(1, 1));
    v3a = (v3a - v1a * wij(0, 2) - v2a * wij(1, 2)) / sqrt(wij(2, 2));

    vec<4, value> thetau = {0, v3a[0], v3a[1], v3a[2]};
    vec<4, value> phiu = {0, v1a[0], v1a[1], v1a[2]};

    dual_types::complex<value> unit_i = dual_types::unit_i();

    tensor<dual_types::complex<value>, 4> mu;

    for(int i=0; i < 4; i++)
    {
        mu.idx(i) = dual_types::complex<value>(1.f/sqrt(2.f)) * (thetau[i] + unit_i * phiu[i]);
    }

    tensor<value, 3, 3, 3> raised_eijk = raise_index_generic(raise_index_generic(eijk_tensor, iYij, 1), iYij, 2);

    dual_types::complex<value> w4;

    {
        dual_types::complex<value> sum(0.f);

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                dual_types::complex<value> k_sum_1(0.f);
                dual_types::complex<value> k_sum_2(0.f);

                for(int k=0; k < 3; k++)
                {
                    dual_types::complex<value> rhs_sum(0.f);

                    for(int l=0; l < 3; l++)
                    {
                        rhs_sum = rhs_sum + unit_i * raised_eijk.idx(i, k, l) * cdKij.idx(l, j, k);
                    }

                    k_sum_2 = k_sum_2 + rhs_sum;

                    k_sum_1 = k_sum_1 + Kij.idx(i, k) * raise_index_generic(Kij, iYij, 0).idx(k, j);
                }

                dual_types::complex<value> inner_sum = -Rij.idx(i, j) - K * Kij.idx(i, j) + k_sum_1 + k_sum_2;

                ///mu is a 4 vector, but we use it spatially
                ///this exposes the fact that i really runs from 1-4 instead of 0-3
                sum = sum + inner_sum * mu.idx(i + 1) * mu.idx(j + 1);
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

struct frame_basis
{
    vec<4, value> v1;
    vec<4, value> v2;
    vec<4, value> v3;
    vec<4, value> v4;
};

value dot_product(const vec<4, value>& u, const vec<4, value>& v, const metric<value, 4, 4>& met)
{
    tensor<value, 4> as_tensor;

    for(int i=0; i < 4; i++)
    {
        as_tensor.idx(i) = u[i];
    }

    auto lowered_as_tensor = lower_index(as_tensor, met);

    vec<4, value> lowered = {lowered_as_tensor.idx(0), lowered_as_tensor.idx(1), lowered_as_tensor.idx(2), lowered_as_tensor.idx(3)};

    return dot(lowered, v);
}

vec<4, value> gram_proj(const vec<4, value>& u, const vec<4, value>& v, const metric<value, 4, 4>& met)
{
    value top = dot_product(u, v, met);

    value bottom = dot_product(u, u, met);

    return (top / bottom) * u;
}

vec<4, value> normalize_big_metric(const vec<4, value>& in, const metric<value, 4, 4>& met)
{
    value dot = dot_product(in, in, met);

    return in / sqrt(fabs(dot));
}

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

tensor<value, 4> get_adm_hypersurface_normal(const value& gA, const tensor<value, 3>& gB)
{
    return {1/gA, -gB.idx(0)/gA, -gB.idx(1)/gA, -gB.idx(2)/gA};
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
    standard_arguments args(true);

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

    pixel_direction = pixel_direction.norm();

    vec<3, value> basis3_x = {basis_x.y(), basis_x.z(), basis_x.w()};
    vec<3, value> basis3_y = {basis_y.y(), basis_y.z(), basis_y.w()};
    vec<3, value> basis3_z = {basis_z.y(), basis_z.z(), basis_z.w()};

    pixel_direction = unrotate_vector(basis3_x.norm(), basis3_y.norm(), basis3_z.norm(),  pixel_direction);

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
    standard_arguments args(true);

    /*ctx.pin(args.gA);
    ctx.pin(args.gB);
    ctx.pin(args.cY);
    ctx.pin(args.X);
    ctx.pin(args.Yij);*/

    //ctx.pin(args.Yij);

    auto unpinned_Yij = args.Yij;

    ///upper index, aka contravariant
    vec<4, value> loop_lightray_velocity = {"lv0", "lv1", "lv2", "lv3"};
    vec<4, value> loop_lightray_position = {"lp0", "lp1", "lp2", "lp3"};

    constexpr int order = 1;

    tensor<value, 3, 3> digB;

    ///derivative
    for(int i=0; i < 3; i++)
    {
        ///index
        for(int j=0; j < 3; j++)
        {
            digB.idx(i, j) = hacky_differentiate<order>(ctx, args.gB.idx(j), i, true, true);
        }
    }

    tensor<value, 3> digA;

    for(int i=0; i < 3; i++)
    {
        digA.idx(i) = hacky_differentiate<order>(ctx, args.gA, i, true, true);
    }

    float universe_length = (dim/2.f).max_elem();

    value scale = "scale";

    ctx.add("universe_size", universe_length * scale);

    tensor<value, 3> X_upper = {"lp1", "lp2", "lp3"};
    tensor<value, 3> V_upper = {"V0", "V1", "V2"};

    /*inverse_metric<value, 3, 3> iYij = args.Yij.invert();

    ctx.pin(iYij);*/

    inverse_metric<value, 3, 3> iYij = args.X * args.cY.invert();

    inverse_metric<value, 3, 3> icY = args.cY.invert();

    tensor<value, 3, 3, 3> conformal_christoff2 = gpu_christoffel_symbols_2(ctx, args.cY, icY, true);

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
                    sm += icY.idx(i, m) * hacky_differentiate<order>(ctx, args.X, m, true, true);
                }

                full_christoffel2.idx(i, j, k) = conformal_christoff2.idx(i, j, k) -
                                                 (1.f/(2.f * max(args.X, 0.001f))) * (kronecker_ik * hacky_differentiate<order>(ctx, args.X, j, true, true) + kronecker_ij * hacky_differentiate<order>(ctx, args.X, k, true, true) - args.cY.idx(j, k) * sm);
            }
        }
    }

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

            V_upper_diff.idx(i) += args.gA * V_upper.idx(j) * (V_upper.idx(i) * (hacky_differentiate<order>(ctx, log(max(args.gA, 0.0001f)), j, true, true) - kjvk) + 2 * raise_index(args.Kij, args.Yij, iYij).idx(i, j) - christoffel_sum)
                                   - iYij.idx(i, j) * hacky_differentiate<order>(ctx, args.gA, j, true, true) - V_upper.idx(j) * hacky_differentiate<order>(ctx, args.gB.idx(i), j, true, true);

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
            p0 += (1/args.gA) * sqrt(iYij.idx(i, j) * p_lower.idx(i) * p_lower.idx(j));
        }
    }

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
        value s1 = -args.gA * hacky_differentiate(ctx, args.gA, i, true, true) * p0 * p0;

        value s2 = 0;

        for(int k=0; k < 3; k++)
        {
            s2 += hacky_differentiate(ctx, args.gB.idx(k), i, true, true) * p_lower.idx(k) * p0;
        }

        value s3 = 0;

        for(int l=0; l < 3; l++)
        {
            for(int m=0; m < 3; m++)
            {
                s3 += -0.5f * hacky_differentiate(ctx, iYij.idx(l, m), i, true, true) * p_lower.idx(l) * p_lower.idx(m);
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

struct lightray
{
    cl_float4 pos;
    cl_float4 vel;
    cl_int x, y;
};

std::pair<cl::buffer, int> generate_sponge_points(cl::context& ctx, cl::command_queue& cqueue, float scale, vec3i size)
{
    cl::buffer points(ctx);
    cl::buffer real_count(ctx);

    points.alloc(size.x() * size.y() * size.z() * sizeof(cl_ushort4));
    real_count.alloc(sizeof(cl_int));
    real_count.set_to_zero(cqueue);

    vec<4, cl_int> clsize = {size.x(), size.y(), size.z(), 0};

    cl::args args;
    args.push_back(points);
    args.push_back(real_count);
    args.push_back(scale);
    args.push_back(clsize);

    cqueue.exec("generate_sponge_points", args, {size.x(),  size.y(),  size.z()}, {8, 8, 1});

    std::vector<cl_ushort4> cpu_points = points.read<cl_ushort4>(cqueue);

    printf("Original sponge points %i\n", cpu_points.size());

    cl_int count = real_count.read<cl_int>(cqueue).at(0);

    assert(count > 0);

    cpu_points.resize(count);

    std::sort(cpu_points.begin(), cpu_points.end(), [](const cl_ushort4& p1, const cl_ushort4& p2)
    {
        return std::tie(p1.s[2], p1.s[1], p1.s[0]) < std::tie(p2.s[2], p2.s[1], p2.s[0]);
    });

    cl::buffer real(ctx);
    real.alloc(cpu_points.size() * sizeof(cl_ushort4));
    real.write(cqueue, cpu_points);

    printf("Sponge point reduction %i\n", count);

    return {real, count};
}

std::tuple<cl::buffer, int, cl::buffer, int> generate_evolution_points(cl::context& ctx, cl::command_queue& cqueue, float scale, vec3i size)
{
    cl::buffer evolved_points(ctx);
    cl::buffer evolved_count(ctx);

    cl::buffer non_evolved_points(ctx);
    cl::buffer non_evolved_count(ctx);

    evolved_points.alloc(size.x() * size.y() * size.z() * sizeof(cl_ushort4));
    non_evolved_points.alloc(size.x() * size.y() * size.z() * sizeof(cl_ushort4));

    evolved_count.alloc(sizeof(cl_int));
    non_evolved_count.alloc(sizeof(cl_int));

    evolved_count.set_to_zero(cqueue);
    non_evolved_count.set_to_zero(cqueue);

    vec<4, cl_int> clsize = {size.x(), size.y(), size.z(), 0};

    cl::args args;
    args.push_back(evolved_points);
    args.push_back(evolved_count);
    args.push_back(non_evolved_points);
    args.push_back(non_evolved_count);
    args.push_back(scale);
    args.push_back(clsize);

    cqueue.exec("generate_evolution_points", args, {size.x(),  size.y(),  size.z()}, {8, 8, 1});

    std::vector<cl_ushort4> cpu_points = evolved_points.read<cl_ushort4>(cqueue);
    std::vector<cl_ushort4> cpu_non_points = non_evolved_points.read<cl_ushort4>(cqueue);

    printf("Original evolve points %i\n", cpu_points.size());

    cl_int count = evolved_count.read<cl_int>(cqueue).at(0);
    cl_int non_count = non_evolved_count.read<cl_int>(cqueue).at(0);

    assert(count > 0);

    cpu_points.resize(count);
    cpu_non_points.resize(non_count);

    std::sort(cpu_points.begin(), cpu_points.end(), [](const cl_ushort4& p1, const cl_ushort4& p2)
    {
        return std::tie(p1.s[2], p1.s[1], p1.s[0]) < std::tie(p2.s[2], p2.s[1], p2.s[0]);
    });

    std::sort(cpu_non_points.begin(), cpu_non_points.end(), [](const cl_ushort4& p1, const cl_ushort4& p2)
    {
        return std::tie(p1.s[2], p1.s[1], p1.s[0]) < std::tie(p2.s[2], p2.s[1], p2.s[0]);
    });

    cl::buffer real(ctx);
    real.alloc(cpu_points.size() * sizeof(cl_ushort4));
    real.write(cqueue, cpu_points);

    cl::buffer non(ctx);
    non.alloc(cpu_non_points.size() * sizeof(cl_ushort4));
    non.write(cqueue, cpu_non_points);

    printf("Evolve point reduction %i\n", count);

    return {real, count, non, non_count};
}

struct buffer_set
{
    #ifndef USE_GBB
    static constexpr int buffer_count = 11+9+1;
    #else
    static constexpr int buffer_count = 12 + 9 + 3 + 1;
    #endif

    std::vector<cl::buffer> buffers;

    buffer_set(cl::context& ctx, vec3i size)
    {
        for(int kk=0; kk < buffer_count; kk++)
        {
            buffers.emplace_back(ctx);
            buffers.back().alloc(size.x() * size.y() * size.z() * sizeof(cl_float));
        }
    }
};

///it seems like basically i need numerical dissipation of some form
///if i didn't evolve where sponge = 1, would be massively faster
int main()
{
    //return 0;

    /*for(float v = 0; v <= M_PI * 10; v += 0.01)
    {
        auto harm = get_harmonic(v, 2, 0);

        printf("HARM %f\n", harm.real);
    }*/

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

    std::string argument_string = "-O3 -cl-std=CL2.0 ";

    ///the simulation domain is this * 2
    int current_simulation_boundary = 5;
    ///must be a multiple of DIFFERENTIATION_WIDTH
    vec3i size = {300, 300, 300};
    //vec3i size = {250, 250, 250};
    //float c_at_max = 160;
    float c_at_max = 65;
    float scale = c_at_max / (size.largest_elem());
    vec3f centre = {size.x()/2, size.y()/2, size.z()/2};

    equation_context setup_initial;
    setup_initial_conditions(setup_initial, centre, scale);

    equation_context ctx1;
    get_initial_conditions_eqs(ctx1, centre, scale);

    equation_context ctx3;
    build_eqs(ctx3);

    equation_context ctx4;
    build_constraints(ctx4);

    //equation_context ctx5;
    //extract_waveforms(ctx5);

    equation_context ctx6;
    process_geodesics(ctx6);

    equation_context ctx7;
    loop_geodesics(ctx7, {size.x(), size.y(), size.z()});

    /*equation_context ctx8;
    build_kreiss_oliger_dissipate(ctx8);*/

    equation_context ctx10;
    build_kreiss_oliger_dissipate_singular(ctx10);

    equation_context ctx11;
    build_intermediate_thin(ctx11);

    equation_context ctx12;
    build_intermediate_thin_cY5(ctx12);

    equation_context ctx13;
    build_momentum_constraint(ctx13);

    /*for(auto& i : ctx.values)
    {
        std::string str = "-D" + i.first + "=" + type_to_string(i.second) + " ";

        argument_string += str;
    }

    argument_string += "-DTEMP_COUNT=" + std::to_string(ctx.temporaries.size()) + " ";

    std::string temporary_string;

    for(auto& i : ctx.temporaries)
    {
        temporary_string += type_to_string(i.second) + ",";
    }

    argument_string += "-DTEMPORARIES=" + temporary_string + " ";*/

    ctx1.build(argument_string, 0);
    //ctx2.build(argument_string, 1);
    ctx3.build(argument_string, 2);
    ctx4.build(argument_string, 3);
    //ctx5.build(argument_string, 4);
    ctx6.build(argument_string, 5);
    ctx7.build(argument_string, 6);
    //ctx8.build(argument_string, 7);
    setup_initial.build(argument_string, 8);
    ctx10.build(argument_string, 9);
    ctx11.build(argument_string, 10);
    ctx12.build(argument_string, 11);
    ctx13.build(argument_string, 12);

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

    std::cout << "ARGS " << argument_string << std::endl;

    {
        std::ofstream out("args.txt");
        out << argument_string;
    }

    cl::program prog(clctx.ctx, "cl.cl");
    prog.build(clctx.ctx, argument_string);

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

    cl::buffer ray_buffer(clctx.ctx);
    ray_buffer.alloc(sizeof(lightray) * width * height);

    int which_data = 0;

    std::array<buffer_set, 2> generic_data{buffer_set(clctx.ctx, size), buffer_set(clctx.ctx, size)};

    std::array<std::string, buffer_set::buffer_count> buffer_names
    {
        "cY0", "cY1", "cY2", "cY3", "cY4",
        "cA0", "cA1", "cA2", "cA3", "cA4", "cA5",
        "cGi0", "cGi1", "cGi2",
        "K", "X", "gA", "gB0", "gB1", "gB2",
        "theta",
    };

    auto buffer_to_index = [&](const std::string& name)
    {
        for(int idx = 0; idx < buffer_set::buffer_count; idx++)
        {
            if(buffer_names[idx] == name)
                return idx;
        }

        assert(false);
    };

    float dissipate_low = 0.4;
    float dissipate_high = 0.4;
    float dissipate_gauge = 0.1;

    /*std::array<float, buffer_count> dissipation_coefficients
    {
        dissipate_low, dissipate_low, dissipate_low, dissipate_low, dissipate_low, //cY
        dissipate_high, dissipate_high, dissipate_high, dissipate_high, dissipate_high, dissipate_high, //cA
        dissipate_high, dissipate_high, dissipate_high, //cGi
        dissipate_high, //K
        dissipate_high, //X
        dissipate_high, //gA
        dissipate_high, dissipate_high, dissipate_high //gB
    };*/

    std::array<float, buffer_set::buffer_count> dissipation_coefficients
    {
        dissipate_low, dissipate_low, dissipate_low, dissipate_low, dissipate_low, //cY
        dissipate_high, dissipate_high, dissipate_high, 0, dissipate_high, dissipate_high, //cA
        dissipate_low, dissipate_low, dissipate_low, //cGi
        dissipate_high, //K
        dissipate_low, //X
        dissipate_gauge, //gA
        dissipate_gauge, dissipate_gauge, dissipate_gauge, //gB
        dissipate_low
    };

    std::array<cl::buffer, 2> u_args{clctx.ctx, clctx.ctx};
    u_args[0].alloc(size.x() * size.y() * size.z() * sizeof(cl_float));
    u_args[1].alloc(size.x() * size.y() * size.z() * sizeof(cl_float));

    int which_u_args = 0;

    vec<4, cl_int> clsize = {size.x(), size.y(), size.z(), 0};

    cl_float time_elapsed_s = 0;

    cl::args initial_u_args;
    initial_u_args.push_back(u_args[0]);
    initial_u_args.push_back(clsize);

    clctx.cqueue.exec("setup_u_offset", initial_u_args, {size.x(), size.y(), size.z()}, {8, 8, 1});

    cl::args initial_u_args2;
    initial_u_args2.push_back(u_args[1]);
    initial_u_args2.push_back(clsize);

    clctx.cqueue.exec("setup_u_offset", initial_u_args2, {size.x(), size.y(), size.z()}, {8, 8, 1});

    ///I need to do this properly, where it keeps iterating until it converges
    #ifndef GPU_PROFILE
    for(int i=0; i < 5000; i++)
    #else
    for(int i=0; i < 1000; i++)
    #endif
    {
        cl::args iterate_u_args;
        iterate_u_args.push_back(u_args[which_u_args]);
        iterate_u_args.push_back(u_args[(which_u_args + 1) % 2]);
        iterate_u_args.push_back(scale);
        iterate_u_args.push_back(clsize);

        clctx.cqueue.exec("iterative_u_solve", iterate_u_args, {size.x(), size.y(), size.z()}, {8, 8, 1});

        which_u_args = (which_u_args + 1) % 2;
    }

    u_args[(which_u_args + 1) % 2].native_mem_object.release();

    auto [sponge_positions, sponge_positions_count] = generate_sponge_points(clctx.ctx, clctx.cqueue, scale, size);
    auto [evolution_positions, evolution_positions_count, non_evolution_positions, non_evolution_positions_count] = generate_evolution_points(clctx.ctx, clctx.cqueue, scale, size);

    clctx.cqueue.block();

    std::vector<cl::buffer> thin_intermediates;

    constexpr int thin_intermediate_buffer_count = 18 + 3 + 9 + 3;

    for(int i = 0; i < thin_intermediate_buffer_count; i++)
    {
        thin_intermediates.emplace_back(clctx.ctx);
        thin_intermediates.back().alloc(size.x() * size.y() * size.z() * intermediate_data_size);
    }

    std::array<cl::buffer, 3> momentum_constraint{clctx.ctx, clctx.ctx, clctx.ctx};

    for(auto& i : momentum_constraint)
    {
        i.alloc(size.x() * size.y() * size.z() * sizeof(cl_float));
    }

    cl::buffer waveform(clctx.ctx);
    waveform.alloc(sizeof(cl_float2));

    {
        cl::args init;

        for(auto& i : generic_data[0].buffers)
        {
            init.push_back(i);
        }

        init.push_back(u_args[which_u_args]);
        init.push_back(scale);
        init.push_back(clsize);

        clctx.cqueue.exec("calculate_initial_conditions", init, {size.x(), size.y(), size.z()}, {8, 8, 1});
    }

    {
        cl::args init;

        for(auto& i : generic_data[1].buffers)
        {
            init.push_back(i);
        }

        init.push_back(u_args[which_u_args]);
        init.push_back(scale);
        init.push_back(clsize);

        clctx.cqueue.exec("calculate_initial_conditions", init, {size.x(), size.y(), size.z()}, {8, 8, 1});
    }

    std::vector<cl::read_info<cl_float2>> read_data;

    std::vector<float> real_graph;
    std::vector<float> real_decomp;

    int which_texture = 0;
    int steps = 0;

    bool run = false;
    bool should_render = true;

    vec3f camera_pos = {0, c_at_max/2.f - 1, 0};
    //vec3f camera_pos = {175,200,175};
    quat camera_quat;
    camera_quat.load_from_axis_angle({1, 0, 0, M_PI/2});

    std::optional<cl::event> last_event;

    int rendering_method = 0;

    while(!win.should_close())
    {
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

            texture_settings new_sett;
            new_sett.width = width;
            new_sett.height = height;
            new_sett.is_srgb = false;
            new_sett.generate_mipmaps = false;

            tex[0].load_from_memory(new_sett, nullptr);
            tex[1].load_from_memory(new_sett, nullptr);

            rtex[0].create_from_texture(tex[0].handle);
            rtex[1].create_from_texture(tex[1].handle);

            ray_buffer.alloc(4 * 10 * width * height);
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

            if(real_graph.size() > 0)
            {
                ImGui::PushItemWidth(400);
                ImGui::PlotLines("w4      ", &real_graph[0], real_graph.size());
                ImGui::PopItemWidth();
            }

            /*if(real_decomp.size() > 0)
            {
                ImGui::PushItemWidth(400);
                ImGui::PlotLines("w4_l2_m0", &real_decomp[0], real_decomp.size());
                ImGui::PopItemWidth();
            }*/

            ImGui::End();

        if(run)
            step = true;

        cl::args render;

        for(auto& i : generic_data[which_data].buffers)
        {
            render.push_back(i);
        }

        //render.push_back(bssnok_datas[which_data]);
        render.push_back(scale);
        render.push_back(clsize);
        render.push_back(rtex[which_texture]);
        render.push_back(time_elapsed_s);

        clctx.cqueue.exec("render", render, {size.x(), size.y()}, {16, 16});

        float timestep = 0.02/2;

        if(steps < 20)
           timestep = 0.001;

        if(steps < 10)
            timestep = 0.0001;

        if(step)
        {
            steps++;

            auto step = [&](auto& generic_in, auto& generic_out, float current_timestep)
            {
                {
                    auto differentiate = [&](const std::string& name, cl::buffer& out1, cl::buffer& out2, cl::buffer& out3)
                    {
                        if(name != "cY5")
                        {
                            int idx = buffer_to_index(name);

                            cl::args thin;
                            thin.push_back(evolution_positions);
                            thin.push_back(evolution_positions_count);
                            thin.push_back(generic_in[idx]);
                            thin.push_back(out1);
                            thin.push_back(out2);
                            thin.push_back(out3);
                            thin.push_back(scale);
                            thin.push_back(clsize);

                            clctx.cqueue.exec("calculate_intermediate_data_thin", thin, {evolution_positions_count}, {128});
                        }
                        else
                        {
                            int idx0 = buffer_to_index("cY0");
                            int idx1 = buffer_to_index("cY1");
                            int idx2 = buffer_to_index("cY2");
                            int idx3 = buffer_to_index("cY3");
                            int idx4 = buffer_to_index("cY4");

                            cl::args thin;
                            thin.push_back(evolution_positions);
                            thin.push_back(evolution_positions_count);
                            thin.push_back(generic_in[idx0]);
                            thin.push_back(generic_in[idx1]);
                            thin.push_back(generic_in[idx2]);
                            thin.push_back(generic_in[idx3]);
                            thin.push_back(generic_in[idx4]);

                            thin.push_back(out1);
                            thin.push_back(out2);
                            thin.push_back(out3);
                            thin.push_back(scale);
                            thin.push_back(clsize);

                            clctx.cqueue.exec("calculate_intermediate_data_thin_cY5", thin, {evolution_positions_count}, {128});
                        }
                    };

                    std::array buffers = {"cY0", "cY1", "cY2", "cY3", "cY4", "cY5",
                                          "gA", "gB0", "gB1", "gB2", "X"};

                    for(int idx = 0; idx < (int)buffers.size(); idx++)
                    {
                        int i1 = idx * 3 + 0;
                        int i2 = idx * 3 + 1;
                        int i3 = idx * 3 + 2;

                        differentiate(buffers[idx], thin_intermediates[i1], thin_intermediates[i2], thin_intermediates[i3]);
                    }
                }

                {
                    cl::args momentum_args;

                    for(auto& i : generic_in)
                    {
                        momentum_args.push_back(i);
                    }

                    for(auto& i : momentum_constraint)
                    {
                        momentum_args.push_back(i);
                    }

                    momentum_args.push_back(scale);
                    momentum_args.push_back(clsize);
                    momentum_args.push_back(time_elapsed_s);

                    clctx.cqueue.exec("calculate_momentum_constraint", momentum_args, {size.x(), size.y(), size.z()}, {64, 1, 1});
                }

                cl::args a1;

                a1.push_back(evolution_positions);
                a1.push_back(evolution_positions_count);

                for(auto& i : generic_in)
                {
                    a1.push_back(i);
                }

                for(auto& i : generic_out)
                {
                    a1.push_back(i);
                }

                for(auto& i : momentum_constraint)
                {
                    a1.push_back(i);
                }

                for(auto& i : thin_intermediates)
                {
                    a1.push_back(i);
                }

                a1.push_back(scale);
                a1.push_back(clsize);
                a1.push_back(current_timestep);
                a1.push_back(time_elapsed_s);
                a1.push_back(current_simulation_boundary);

                clctx.cqueue.exec("evolve", a1, {evolution_positions_count}, {128});
            };

            step(generic_data[which_data].buffers, generic_data[(which_data + 1) % 2].buffers, timestep);

            {
                cl::args constraints;

                constraints.push_back(evolution_positions);
                constraints.push_back(evolution_positions_count);

                for(auto& i : generic_data[(which_data + 1) % 2].buffers)
                {
                    constraints.push_back(i);
                }

                constraints.push_back(scale);
                constraints.push_back(clsize);

                clctx.cqueue.exec("enforce_algebraic_constraints", constraints, {evolution_positions_count}, {128});
            }

            {
                for(int i=0; i < buffer_set::buffer_count; i++)
                {
                    cl::args diss;

                    diss.push_back(evolution_positions);
                    diss.push_back(evolution_positions_count);

                    diss.push_back(generic_data[which_data].buffers[i]);
                    diss.push_back(generic_data[(which_data + 1) % 2].buffers[i]);

                    float coeff = dissipation_coefficients[i];

                    diss.push_back(coeff);
                    diss.push_back(scale);
                    diss.push_back(clsize);
                    diss.push_back(timestep);

                    if(coeff == 0)
                        continue;

                    clctx.cqueue.exec("dissipate_single", diss, {evolution_positions_count}, {128});
                }
            }

            which_data = (which_data + 1) % 2;

            {
                cl::args cleaner;
                cleaner.push_back(sponge_positions);
                cleaner.push_back(sponge_positions_count);

                for(auto& i : generic_data[which_data].buffers)
                {
                    cleaner.push_back(i);
                }

                //cleaner.push_back(bssnok_datas[which_data]);
                cleaner.push_back(u_args[which_u_args]);
                cleaner.push_back(scale);
                cleaner.push_back(clsize);
                cleaner.push_back(time_elapsed_s);
                cleaner.push_back(timestep);

                clctx.cqueue.exec("clean_data", cleaner, {sponge_positions_count}, {256});
            }

            float r_extract = c_at_max/4;

            //printf("OFF %f\n", r_extract/scale);

            /*cl_int4 pos = {clsize.x()/2, clsize.y()/2 + r_extract / scale, clsize.z()/2, 0};

            cl::args waveform_args;

            for(auto& i : generic_data[which_data].buffers)
            {
                waveform_args.push_back(i);
            }

            //waveform_args.push_back(bssnok_datas[which_data]);
            waveform_args.push_back(scale);
            waveform_args.push_back(clsize);
            waveform_args.push_back(intermediate);
            waveform_args.push_back(pos);
            waveform_args.push_back(waveform);

            clctx.cqueue.exec("extract_waveform", waveform_args, {size.x(), size.y(), size.z()}, {128, 1, 1});

            cl::read_info<cl_float2> data = waveform.read_async<cl_float2>(clctx.cqueue, 1);

            data.evt.block();

            cl_float2 val = *data.data;

            data.consume();

            dual_types::complex<float> w4 = {val.s[0], val.s[1]};

            real_graph.push_back(val.s[0]);

            float harmonic = get_harmonic(w4, 2, 0).real;

            //printf("Harm %f\n", harmonic);

            real_decomp.push_back(harmonic);*/

            time_elapsed_s += timestep;
            current_simulation_boundary += DIFFERENTIATION_WIDTH;

            current_simulation_boundary = clamp(current_simulation_boundary, 0, size.x()/2);
        }

        if(rendering_method == 0 || rendering_method == 1)
        {
            cl::args render_args;

            for(auto& i : generic_data[which_data].buffers)
            {
                render_args.push_back(i);
            }

            cl_float3 ccamera_pos = {camera_pos.x(), camera_pos.y(), camera_pos.z()};
            cl_float4 ccamera_quat = {camera_quat.q.x(), camera_quat.q.y(), camera_quat.q.z(), camera_quat.q.w()};

            render_args.push_back(scale);
            render_args.push_back(ccamera_pos);
            render_args.push_back(ccamera_quat);
            render_args.push_back(clsize);
            render_args.push_back(rtex[which_texture]);

            //assert(render_args.arg_list.size() == 29);

            if(should_render || snap)
            {
                if(rendering_method == 0)
                    clctx.cqueue.exec("trace_metric", render_args, {width, height}, {16, 16});
                else
                    clctx.cqueue.exec("trace_rays", render_args, {width, height}, {16, 16});
            }
        }

        if(rendering_method == 2 && snap)
        {
            cl_float3 ccamera_pos = {camera_pos.x(), camera_pos.y(), camera_pos.z()};
            cl_float4 ccamera_quat = {camera_quat.q.x(), camera_quat.q.y(), camera_quat.q.z(), camera_quat.q.w()};

            cl::args init_args;

            for(auto& i : generic_data[which_data].buffers)
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

            for(auto& i : generic_data[which_data].buffers)
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

        ///todo: get rid of this
        clctx.cqueue.block();
        glFinish();

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
        glFinish();
    }
}
