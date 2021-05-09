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
*/

///notes:
///off centre black hole results in distortion, probably due to boundary conditions contaminating things
///this is odd. Maybe don't boundary condition shift and lapse?

//#define USE_GBB

///all conformal variables are explicitly labelled
/*struct bssnok_data
{
    //6 units of cY
    //6 units of cA

    cl_float cGi0, cGi1, cGi2;

    cl_float K;
    cl_float X;

    cl_float gA;
    cl_float gB0;
    cl_float gB1;
    cl_float gB2;

    #ifdef USE_GBB
    cl_float gBB0;
    cl_float gBB1;
    cl_float gBB2;
    #endif // USE_GBB
};
///total size = 21

*/

///https://scholarworks.rit.edu/cgi/viewcontent.cgi?article=11286&context=theses
///38.2
struct lightray
{
    cl_float3 x; ///position?
    cl_float3 V; ///lower i, this is some sort of spatial component
    cl_float T; ///proper time
};

struct intermediate_bssnok_data
{
    cl_float dcYij[3 * 6];
    cl_float digA[6];
    cl_float digB[3*3];
    cl_float dX[3];
};

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
            argument_string += "-DTEMPORARIES" + std::to_string(idx) + "==pv0 ";
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
#define BORDER_WIDTH 6

inline
std::tuple<std::string, std::string, bool> decompose_variable(std::string str)
{
    std::string buffer;
    std::string val;

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

    else if(str == "buffer")
    {
        buffer = "buffer";
        val = "buffer";
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
    else if(str.starts_with("cY0"))
    {
        buffer = "cY0";
        val = buffer;
    }
    else if(str.starts_with("cY1"))
    {
        buffer = "cY1";
        val = buffer;
    }
    else if(str.starts_with("cY2"))
    {
        buffer = "cY2";
        val = buffer;
    }
    else if(str.starts_with("cY3"))
    {
        buffer = "cY3";
        val = buffer;
    }
    else if(str.starts_with("cY4"))
    {
        buffer = "cY4";
        val = buffer;
    }
    else if(str.starts_with("cY5"))
    {
        buffer = "cY5";
        val = buffer;
    }

    else if(str.starts_with("cA0"))
    {
        buffer = "cA0";
        val = buffer;
    }
    else if(str.starts_with("cA1"))
    {
        buffer = "cA1";
        val = buffer;
    }
    else if(str.starts_with("cA2"))
    {
        buffer = "cA2";
        val = buffer;
    }
    else if(str.starts_with("cA3"))
    {
        buffer = "cA3";
        val = buffer;
    }
    else if(str.starts_with("cA4"))
    {
        buffer = "cA4";
        val = buffer;
    }
    else if(str.starts_with("cA5"))
    {
        buffer = "cA5";
        val = buffer;
    }
    else if(str.starts_with("cGi0"))
    {
        buffer = "cGi0";
        val = buffer;
    }
    else if(str.starts_with("cGi1"))
    {
        buffer = "cGi1";
        val = buffer;
    }
    else if(str.starts_with("cGi2"))
    {
        buffer = "cGi2";
        val = buffer;
    }
    else if(str.starts_with("X"))
    {
        buffer = "X";
        val = buffer;
    }
    else if(str.starts_with("K"))
    {
        buffer = "K";
        val = buffer;
    }
    else if(str.starts_with("gA"))
    {
        buffer = "gA";
        val = buffer;
    }
    else if(str.starts_with("gB0"))
    {
        buffer = "gB0";
        val = buffer;
    }
    else if(str.starts_with("gB1"))
    {
        buffer = "gB1";
        val = buffer;
    }
    else if(str.starts_with("gB2"))
    {
        buffer = "gB2";
        val = buffer;
    }
    else
    {
        std::cout << "input " << str << std::endl;

        assert(false);
    }

    return {buffer, val, uses_extension};
}

template<int elements = 5>
struct differentiation_context
{
    std::array<value, elements> vars;

    std::array<std::string, elements> xs;
    std::array<std::string, elements> ys;
    std::array<std::string, elements> zs;

    differentiation_context(equation_context& ctx, const value& in, int idx, bool should_pin = true, bool linear_interpolation = false)
    {
        std::vector<std::string> variables = in.get_all_variables();

        value cp = in;

        auto index_raw = [](const std::string& x, const std::string& y, const std::string& z)
        {
            return "IDX(" + x + "," + y + "," + z + ")";
        };

        auto fetch_linear = [](const std::string& buffer, const std::string& x, const std::string& y, const std::string& z)
        {
            return "buffer_read_linear(" + buffer + ",(float3)(" + x + "," + y + "," + z + "),dim)";
        };

        auto index_buffer = [](const std::string& variable, const std::string& buffer, const std::string& with_what)
        {
            return buffer + "[" + with_what + "]." + variable;
        };

        auto index_without_extension = [](const std::string& buffer, const std::string& with_what)
        {
            return buffer + "[" + with_what + "]";
        };

        auto index = [&ctx, index_buffer, index_without_extension, index_raw, fetch_linear, linear_interpolation](const std::string& val, const std::string& buffer, bool uses_extension, const std::string& x, const std::string& y, const std::string& z)
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

            if(offset == 0)
                continue;

            if(offset > 0)
            {
                if(idx == 0)
                    xs[i] += "+" + std::to_string(offset);
                if(idx == 1)
                    ys[i] += "+" + std::to_string(offset);
                if(idx == 2)
                    zs[i] += "+" + std::to_string(offset);
            }
            else
            {
                if(idx == 0)
                    xs[i] += std::to_string(offset);
                if(idx == 1)
                    ys[i] += std::to_string(offset);
                if(idx == 2)
                    zs[i] += std::to_string(offset);
            }
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

                substitutions[kk][i] = type_to_string(to_sub);
            }
        }

        for(int i=0; i < elements; i++)
        {
            vars[i] = cp;
            vars[i].substitute(substitutions[i]);
        }

        if(should_pin)
        {
            ctx.pin(vars[4]);
            ctx.pin(vars[3]);
            ctx.pin(vars[2]);
            ctx.pin(vars[1]);
            ctx.pin(vars[0]);
        }
    }
};

#define DIFFERENTIATION_WIDTH 2

///https://hal.archives-ouvertes.fr/hal-00569776/document this paper implies you simply sum the directions
value kreiss_oliger_dissipate_dir(equation_context& ctx, const value& in, int idx)
{
    //differentiation_context<7> dctx(ctx, in, idx, false);
    differentiation_context<5> dctx(ctx, in, idx, false);

    int d = 2;

    ///todo: test lower value again
    //float dissipate = 0.25f/16.f;

    float dissipate = 0.25f/1.5f;

    value scale = "scale";

    value stencil = -(dissipate / (16.f * scale)) * (dctx.vars[0] - 4 * dctx.vars[1] + 6 * dctx.vars[2] - 4 * dctx.vars[3] + dctx.vars[4]);

    //value stencil = (dissipate / (64 * scale)) * (dctx.vars[0] - 6 * dctx.vars[1] + 15 * dctx.vars[2] - 20 * dctx.vars[3] + 15 * dctx.vars[4] - 6 * dctx.vars[5] + dctx.vars[6]);

    return stencil;
}

value kreiss_oliger_dissipate(equation_context& ctx, const value& in)
{
    value fin = 0;

    for(int i=0; i < 3; i++)
    {
        fin = fin + kreiss_oliger_dissipate_dir(ctx, in, i);
    }

    return fin;
}

void build_kreiss_oliger_dissipate(equation_context& ctx)
{
    value v = "buffer";
    ctx.add("KREISS_OLIGER_DISSIPATE", kreiss_oliger_dissipate(ctx, v));

    //value z = 0;
    //ctx.add("KREISS_OLIGER_DISSIPATE", z);
}

template<int order = 1>
value hacky_differentiate(equation_context& ctx, const value& in, int idx, bool pin = true, bool linear = false)
{
    differentiation_context dctx(ctx, in, idx, true, linear);

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
        ctx.pin(final_command);
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

    //value u_n = (dctx.vars[2] - dctx.vars[1]) / scale;
    //value u_p = (dctx.vars[3] - dctx.vars[2]) / scale;

    value u_n = (3 * dctx.vars[2] - 4 * dctx.vars[1] + dctx.vars[0]) / (2 * scale);
    value u_p = (-dctx.vars[4] + 4 * dctx.vars[3] - 3 * dctx.vars[2]) / (2 * scale);

    ///- here probably isn't right
    ///neither is correct, this is fundamentally wrong somewhere
    value final_command = (a_p * u_n + a_n * u_p);

    if(pin)
    {
        ctx.pin(final_command);
    }

    return final_command;*/

    return prefix * hacky_differentiate(ctx, in, idx, pin);
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
    return sum_multiply(gB, tensor_derivative(ctx, variable));
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

template<typename T, int N>
inline
tensor<T, N, N> gpu_lie_derivative_weight(equation_context& ctx, const tensor<T, N>& B, const tensor<T, N, N>& mT)
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

                    local = local + hacky_differentiate(ctx, met.idx(m, k), l);
                    local = local + hacky_differentiate(ctx, met.idx(m, l), k);
                    local = local - hacky_differentiate(ctx, met.idx(k, l), m);

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

tensor<value, 3, 3> calculate_icAij(const vec<3, value>& pos, const std::vector<float>& black_hole_m, const std::vector<vec3f>& black_hole_pos, const std::vector<vec3f>& black_hole_velocity)
{
    metric<value, 3, 3> cYij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            cYij.idx(i, j) = (i == j) ? 1 : 0;
        }
    }

    tensor<value, 3, 3> icAij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            icAij.idx(i, j) = 0;
        }
    }

    ///https://arxiv.org/pdf/1610.03805.pdf 126
    for(int bh_idx = 0; bh_idx < (int)black_hole_pos.size(); bh_idx++)
    {
        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                vec3f bhpos = black_hole_pos[bh_idx];
                vec3f momentum = black_hole_velocity[bh_idx] * black_hole_m[bh_idx];

                vec<3, value> vri = {bhpos.x(), bhpos.y(), bhpos.z()};

                value ra = (pos - vri).length();
                vec<3, value> nia = (pos - vri) / ra;
                tensor<value, 3> nia_lower = lower_index({nia.x(), nia.y(), nia.z()}, cYij);

                icAij.idx(i, j) += (3 / (2.f * ra * ra)) * (momentum[i] * nia[j] + momentum[j] * nia[i] - (cYij.invert().idx(i, j) - nia[i] * nia[j]) * sum_multiply({momentum.x(), momentum.y(), momentum.z()}, nia_lower));
            }
        }
    }

    return icAij;
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
    std::vector<vec3f> black_hole_pos{san_black_hole_pos({-3.1515, 0, 0}), san_black_hole_pos({3.1515, 0, 0})};
    //std::vector<vec3f> black_hole_pos{san_black_hole_pos({5, 0, 0})};
    std::vector<float> black_hole_m{0.5f, 0.5f};
    //std::vector<float> black_hole_m{0.5f, 0.5f};
    //std::vector<vec3f> black_hole_velocity{{0, 0, 0.25}, {0, 0, -0.25}}; ///pick better velocities

    std::vector<vec3f> black_hole_velocity{{0,0,0.0025}, {0,0,-0.0025}};

    metric<value, 3, 3> flat_metric;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            flat_metric.idx(i, j) = (i == j) ? 1 : 0;
        }
    }

    tensor<value, 3, 3> icAij = calculate_icAij(pos, black_hole_m, black_hole_pos, black_hole_velocity);

    tensor<value, 3, 3> cAij = lower_both(icAij, flat_metric);

    //https://arxiv.org/pdf/gr-qc/9703066.pdf (8)
    //value BL_a = 0;

    value BL_c;

    for(int i=0; i < (int)black_hole_m.size(); i++)
    {
        float Mi = black_hole_m[i];
        vec3f ri = black_hole_pos[i];

        vec<3, value> vri = {ri.x(), ri.y(), ri.z()};

        value dist = (pos - vri).length();

        value iva = Mi / (2 * dist);

        BL_c += iva;
    }

    value BL_a = 1/BL_c;

    ctx.add("init_BL_a", BL_a);

    value aij_aIJ = 0;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            aij_aIJ += icAij.idx(i, j) * cAij.idx(i, j);
        }
    }

    ctx.add("init_aij_aIJ", aij_aIJ);

    vec2i linear_indices[6] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};

    for(int i=0; i < 6; i++)
    {
        ctx.add("init_cA" + std::to_string(i), cAij.idx(linear_indices[i].x(), linear_indices[i].y()));
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

    value gA = 1;
    //value gA = 1/(BL_conformal * BL_conformal);
    value gB0 = 0;
    value gB1 = 0;
    value gB2 = 0;

    tensor<value, 3> cGi;
    value K = 0;

    ///https://arxiv.org/pdf/gr-qc/0206072.pdf (58)

    value X = exp(-4 * conformal_factor);

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

inline
void build_constraints(equation_context& ctx)
{
    unit_metric<value, 3, 3> fixed_cY;
    unit_metric<value, 3, 3> cY;

    cY.idx(0, 0).make_value("cY0[IDX(ix,iy,iz)]"); cY.idx(0, 1).make_value("cY1[IDX(ix,iy,iz)]"); cY.idx(0, 2).make_value("cY2[IDX(ix,iy,iz)]");
    cY.idx(1, 0).make_value("cY1[IDX(ix,iy,iz)]"); cY.idx(1, 1).make_value("cY3[IDX(ix,iy,iz)]"); cY.idx(1, 2).make_value("cY4[IDX(ix,iy,iz)]");
    cY.idx(2, 0).make_value("cY2[IDX(ix,iy,iz)]"); cY.idx(2, 1).make_value("cY4[IDX(ix,iy,iz)]"); cY.idx(2, 2).make_value("cY5[IDX(ix,iy,iz)]");

    value det_cY_pow = pow(cY.det(), 1.f/3.f);

    ctx.pin(det_cY_pow);

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            fixed_cY.idx(i, j) = cY.idx(i, j) / det_cY_pow;
        }
    }

    tensor<value, 3, 3> cA;

    cA.idx(0, 0).make_value("cA0[IDX(ix,iy,iz)]"); cA.idx(0, 1).make_value("cA1[IDX(ix,iy,iz)]"); cA.idx(0, 2).make_value("cA2[IDX(ix,iy,iz)]");
    cA.idx(1, 0).make_value("cA1[IDX(ix,iy,iz)]"); cA.idx(1, 1).make_value("cA3[IDX(ix,iy,iz)]"); cA.idx(1, 2).make_value("cA4[IDX(ix,iy,iz)]");
    cA.idx(2, 0).make_value("cA2[IDX(ix,iy,iz)]"); cA.idx(2, 1).make_value("cA4[IDX(ix,iy,iz)]"); cA.idx(2, 2).make_value("cA5[IDX(ix,iy,iz)]");

    tensor<value, 3, 3> fixed_cA = gpu_trace_free(cA, fixed_cY, fixed_cY.invert());

    vec2i linear_indices[6] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};

    for(int i=0; i < 6; i++)
    {
        vec2i idx = linear_indices[i];

        ctx.add("fix_cY" + std::to_string(i), fixed_cY.idx(idx.x(), idx.y()));
        ctx.add("fix_cA" + std::to_string(i), fixed_cA.idx(idx.x(), idx.y()));
    }
}

inline
void build_intermediate(equation_context& ctx)
{
    int index_table[3][3] = {{0, 1, 2},
                             {1, 3, 4},
                             {2, 4, 5}};

    vec2i linear_indices[6] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};

    unit_metric<value, 3, 3> cY;

    cY.idx(0, 0).make_value("cY0[IDX(ix,iy,iz)]"); cY.idx(0, 1).make_value("cY1[IDX(ix,iy,iz)]"); cY.idx(0, 2).make_value("cY2[IDX(ix,iy,iz)]");
    cY.idx(1, 0).make_value("cY1[IDX(ix,iy,iz)]"); cY.idx(1, 1).make_value("cY3[IDX(ix,iy,iz)]"); cY.idx(1, 2).make_value("cY4[IDX(ix,iy,iz)]");
    cY.idx(2, 0).make_value("cY2[IDX(ix,iy,iz)]"); cY.idx(2, 1).make_value("cY4[IDX(ix,iy,iz)]"); cY.idx(2, 2).make_value("cY5[IDX(ix,iy,iz)]");

    inverse_metric<value, 3, 3> icY = cY.invert();

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            ctx.pin(icY.idx(i, j));
        }
    }

    value gA;
    gA.make_value("gA[IDX(ix,iy,iz)]");

    value X;
    X.make_value("X[IDX(ix,iy,iz)]");

    tensor<value, 3> gB;
    gB.idx(0).make_value("gB0[IDX(ix,iy,iz)]");
    gB.idx(1).make_value("gB1[IDX(ix,iy,iz)]");
    gB.idx(2).make_value("gB2[IDX(ix,iy,iz)]");

    //tensor<value, 3, 3, 3> christoff = gpu_christoffel_symbols_2(ctx, cY, icY);

    tensor<value, 3> digA;

    for(int i=0; i < 3; i++)
    {
        digA.idx(i) = hacky_differentiate(ctx, gA, i);
    }

    tensor<value, 3, 3> digB;

    ///derivative
    for(int i=0; i < 3; i++)
    {
        ///index
        for(int j=0; j < 3; j++)
        {
            digB.idx(i, j) = hacky_differentiate(ctx, gB.idx(j), i);
        }
    }

    /*for(int k=0; k < 3; k++)
    {
        for(int i=0; i < 6; i++)
        {
            vec2i idx = linear_indices[i];

            int linear_idx = k * 6 + i;

            ctx.add("init_christoffel" + std::to_string(linear_idx), christoff.idx(k, idx.x(), idx.y()));
        }
    }*/

    for(int k=0; k < 3; k++)
    {
        for(int i=0; i < 6; i++)
        {
            value diff = hacky_differentiate(ctx, "cY" + std::to_string(i) + "[IDX(ix,iy,iz)]", k);

            int linear_idx = k * 6 + i;

            ctx.add("init_dcYij" + std::to_string(linear_idx), diff);
        }
    }

    for(int i=0; i < 3; i++)
    {
        ctx.add("init_digA" + std::to_string(i), digA.idx(i));
    }

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            int linear_idx = i * 3 + j;

            ctx.add("init_digB" + std::to_string(linear_idx), digB.idx(i, j));
        }
    }

    for(int i=0; i < 3; i++)
    {
        ctx.add("init_dX" + std::to_string(i), hacky_differentiate(ctx, X, i));
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
inline
void build_eqs(equation_context& ctx)
{
    int index_table[3][3] = {{0, 1, 2},
                             {1, 3, 4},
                             {2, 4, 5}};

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

    unit_metric<value, 3, 3> cY;

    cY.idx(0, 0).make_value("cY0[IDX(ix,iy,iz)]"); cY.idx(0, 1).make_value("cY1[IDX(ix,iy,iz)]"); cY.idx(0, 2).make_value("cY2[IDX(ix,iy,iz)]");
    cY.idx(1, 0).make_value("cY1[IDX(ix,iy,iz)]"); cY.idx(1, 1).make_value("cY3[IDX(ix,iy,iz)]"); cY.idx(1, 2).make_value("cY4[IDX(ix,iy,iz)]");
    cY.idx(2, 0).make_value("cY2[IDX(ix,iy,iz)]"); cY.idx(2, 1).make_value("cY4[IDX(ix,iy,iz)]"); cY.idx(2, 2).make_value("cY5[IDX(ix,iy,iz)]");

    inverse_metric<value, 3, 3> icY = cY.invert();
    inverse_metric<value, 3, 3> unpinned_icY = cY.invert();

    ctx.pin(icY);

    tensor<value, 3, 3> cA;

    cA.idx(0, 0).make_value("cA0[IDX(ix,iy,iz)]"); cA.idx(0, 1).make_value("cA1[IDX(ix,iy,iz)]"); cA.idx(0, 2).make_value("cA2[IDX(ix,iy,iz)]");
    cA.idx(1, 0).make_value("cA1[IDX(ix,iy,iz)]"); cA.idx(1, 1).make_value("cA3[IDX(ix,iy,iz)]"); cA.idx(1, 2).make_value("cA4[IDX(ix,iy,iz)]");
    cA.idx(2, 0).make_value("cA2[IDX(ix,iy,iz)]"); cA.idx(2, 1).make_value("cA4[IDX(ix,iy,iz)]"); cA.idx(2, 2).make_value("cA5[IDX(ix,iy,iz)]");

    ///the christoffel symbol
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
    gBB.idx(0).make_value("gBB0[IDX(ix,iy,iz)]");
    gBB.idx(1).make_value("gBB1[IDX(ix,iy,iz)]");
    gBB.idx(2).make_value("gBB2[IDX(ix,iy,iz)]");
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
    /*for(int i=0; i < 3; i++)
    {
        value sum = 0;

        for(int j=0; j < 3; j++)
        {
            sum = sum - hacky_differentiate(ctx, unpinned_icY.idx(i, j), j);
        }

        derived_cGi.idx(i) = sum;
    }*/

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

    tensor<value, 3> linear_dB;

    for(int i=0; i < 3; i++)
    {
        linear_dB.idx(i) = hacky_differentiate(ctx, gB.idx(i), i);
    }

    tensor<value, 3, 3> lie_cYij = gpu_lie_derivative_weight(ctx, gB, cY);

    ///https://arxiv.org/pdf/gr-qc/0511048.pdf (1)
    ///https://scholarworks.rit.edu/cgi/viewcontent.cgi?article=11286&context=theses 3.66
    tensor<value, 3, 3> dtcYij = -2 * gA * cA + lie_cYij;

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

    ///a / X
    value gA_X = 0;

    {
        float min_X = 0.001;

        gA_X = dual_if(X <= min_X,
        [&]()
        {
            ///linearly interpolate to 0
            value value_at_min = gA / min_X;

            return value_at_min * (X / min_X);
        },
        [&]()
        {
            return gA / X;
        });
    }

    ///X * christoffel symbols of the conformal metric
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

    tensor<value, 3, 3> xgADiDjphi;

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
    }

    //ctx.add("debug_val", Rphiij.idx(i, j));

    //ctx.add("debug_val", gpu_trace(cA, cY, icY));

    tensor<value, 3, 3> xgARij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            xgARij.idx(i, j) = xgARphiij.idx(i, j) + X * gA * cRij.idx(i, j);

            ctx.pin(xgARij.idx(i, j));
        }
    }

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

    tensor<value, 3, 3> with_trace;

    ///todo: Investigate this, there's a good chance dtcAij is whats broken
    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            ///Moved X inside the trace free bracket and expanded it
            //value trace_free_interior_old = -X * gpu_covariant_derivative_low_vec(ctx, digA, Yij, iYij).idx(j, i);

            value trace_free_interior_1 = X * hacky_differentiate(ctx, digA.idx(j), i);

            value reduced = 0;

            for(int b=0; b < 3; b++)
            {
                reduced = reduced + xcChristoff.idx(b, i, j) * digA.idx(b);
            }

            trace_free_interior_1 = trace_free_interior_1 - reduced;

            value trace_free_interior_2 = xgARij.idx(i, j);

            with_trace.idx(i, j) = -trace_free_interior_1 + trace_free_interior_2;
        }
    }

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
            value trace_free_part = gpu_trace_free(with_trace, cY, icY).idx(i, j);

            value p1 = trace_free_part;

            value p2 = gA * (K * cA.idx(i, j) - 2 * sum);

            value p3 = gpu_lie_derivative_weight(ctx, gB, cA).idx(i, j);

            if(i == 0 && j == 0)
            {
                ctx.add("debug_p1", p1);
                ctx.add("debug_p2", p2);
                ctx.add("debug_p3", p3);
            }

            /*value sanitised = dual_types::dual_if(X <= 0.000001, []()
            {
                return 0;
            },
            [&]()
            {
                return dual_types::dual_if(isfinite(p1),
                [&]()
                {
                    return p1;
                },
                []()
                {
                    return 0;
                });
            });*/

            dtcAij.idx(i, j) = p1 + p2 + p3;
        }
    }

    tensor<value, 3, 3> icAij = raise_both(cA, cY, icY);

    value dtK = 0;

    {
        value sum1 = 0;

        value christoffel_sum = 0;

        for(int k=0; k < 3; k++)
        {
            christoffel_sum = christoffel_sum - X * derived_cGi.idx(k) * digA.idx(k);
        }

        for(int i=0; i < 3; i++)
        {
            value s1 = 0;

            for(int j=0; j < 3; j++)
            {
                ///pull X out
                s1 = s1 + X * icY.idx(i, j) * hacky_differentiate(ctx, digA.idx(j), i);
            }

            value s3 = 0;

            for(int j=0; j < 3; j++)
            {
                s3 = s3 + 2 * (-1.f/4.f) * icY.idx(i, j) * hacky_differentiate(ctx, X, i) * digA.idx(j);
            }


            sum1 = sum1 + s1 + s3;

            //sum1 = sum1 + gpu_high_covariant_derivative_vec(ctx, digA, Yij, iYij).idx(i, i);
        }

        sum1 = sum1 + christoffel_sum;

        value sum2 = 0;

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                sum2 = sum2 + cA.idx(i, j) * icAij.idx(i, j);
            }
        }

        sum2 = gA * (sum2 + (1.f/3.f) * K * K);

        value sum3 = 0;

        for(int i=0; i < 3; i++)
        {
            //sum3 = sum3 + gB.idx(i) * hacky_differentiate(ctx, K, i);
            sum3 = sum3 + upwind_differentiate(ctx, gB.idx(i), K, i);
        }

        dtK = -sum1 + sum2 + sum3;
    }

    ///these seem to suffer from oscillations
    tensor<value, 3> dtcGi;

    ///https://arxiv.org/pdf/gr-qc/0511048.pdf
    ///could likely eliminate the dphi term

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

            /*if(i == 0 && j == 0)
            {
                ctx.add("debug_val", hacky_differentiate(ctx, digB.idx(0, 0), 0));
            }*/

            value s2 = 0;

            for(int k=0; k < 3; k++)
            {
                s2 = s2 + (1.f/3.f) * icY.idx(i, j) * hacky_differentiate(ctx, digB.idx(k, k), j);
            }

            //value s3 = gB.idx(j) * hacky_differentiate(ctx, cGi.idx(i), j);

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

                /*value s9 = 6 * icAij.idx(i, j) * dphi.idx(j);

                s9 = dual_if(isfinite(s9), [&]()
                {
                    return s9;
                },
                []()
                {
                    return 0;
                });*/

                value s9 = (-1/4.f) * gA_X * 6 * icAij.idx(i, j) * hacky_differentiate(ctx, X, j);

                value s10 = -(2.f/3.f) * icY.idx(i, j) * hacky_differentiate(ctx, K, j);

                s7 = 2 * (gA * s8 + s9 + gA * s10);
            }


            sum = sum + s1 + s2 + s3 + s4 + s5 + s6 + s7;
        }

        dtcGi.idx(i) = sum;
    }

    value dtgA = -2 * gA * K + lie_derivative(ctx, gB, gA);

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

    vec2i linear_indices[6] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};

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

    ti1 = raise_index(ti1, met, met.invert());
    ti2 = raise_index(ti2, met, met.invert());
    ti3 = raise_index(ti3, met, met.invert());
    ti4 = raise_index(ti4, met, met.invert());

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
    value X;

    metric<value, 3, 3> Yij;

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

        X.make_value(bidx("X", true));

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                Yij.idx(i, j) = cY.idx(i, j) / X;
            }
        }
    }
};

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

void loop_geodesics(equation_context& ctx, vec3f dim)
{
    standard_arguments args(true);

    /*ctx.pin(args.gA);
    ctx.pin(args.gB);
    ctx.pin(args.cY);
    ctx.pin(args.X);
    ctx.pin(args.Yij);*/

    //ctx.pin(args.Yij);

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
    tensor<value, 3> V_lower = lower_index(V_upper, args.Yij);

    //ctx.pin(V_lower);

    value WH;

    {
        value WH_sum_inner = 0;

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                WH_sum_inner += V_upper.idx(i) * V_upper.idx(j) * args.Yij.idx(i, j);
            }
        }

        WH = sqrt(1 + WH_sum_inner);
    }

    ctx.pin(WH);

    ctx.add("debug_wh", WH);

    tensor<value, 3> dx;

    for(int i=0; i < 3; i++)
    {
        dx.idx(i) = -args.gB.idx(i) + (args.gA / WH) * V_upper.idx(i);
    }

    value dTdt = args.gA / WH;

    tensor<value, 3> dVi_l;

    for(int i=0; i < 3; i++)
    {
        value p1 = -WH * digA.idx(i);

        value p2 = 0;

        for(int j=0; j < 3; j++)
        {
            p2 += V_lower.idx(j) * digB.idx(i, j);
        }

        value p3 = 0;

        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                p3 += 0.5f * V_upper.idx(j) * V_upper.idx(k) * hacky_differentiate<order>(ctx, args.Yij.idx(j, k), i, true, true);
            }
        }

        dVi_l.idx(i) = p1 + p2 + p3;
    }

    tensor<value, 3> V_lower_next;
    //tensor<value, 3> V_lower_next = V_lower + dVi_l * step;

    for(int i=0; i < 3; i++)
    {
        V_lower_next.idx(i) = V_lower.idx(i) + dVi_l.idx(i);
    }

    tensor<value, 3> V_upper_next = raise_index(V_lower_next, args.Yij, args.Yij.invert());

    tensor<value, 3> V_upper_diff;

    for(int i=0; i <3; i++)
    {
        V_upper_diff.idx(i) = V_upper_next.idx(i) - V_upper.idx(i);
    }

    ctx.add("V0Diff", V_upper_diff.idx(0));
    ctx.add("V1Diff", V_upper_diff.idx(1));
    ctx.add("V2Diff", V_upper_diff.idx(2));

    ctx.add("X0Diff", dx.idx(0));
    ctx.add("X1Diff", dx.idx(1));
    ctx.add("X2Diff", dx.idx(2));

    /*ctx.add("V0N_d", V_upper_next.idx(0));
    ctx.add("V1N_d", V_upper_next.idx(1));
    ctx.add("V2N_d", V_upper_next.idx(2));

    tensor<value, 3> X_next;

    for(int i=0; i < 3; i++)
    {
        X_next.idx(i) = X_upper.idx(i) + dx.idx(i);
    }

    //ctx.add("DTN", )

    ctx.add("X0N_d", X_next.idx(0));
    ctx.add("X1N_d", X_next.idx(1));
    ctx.add("X2N_d", X_next.idx(2));*/

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

///it seems like basically i need numerical dissipation of some form
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
    int current_simulation_boundary = 1024;
    ///must be a multiple of DIFFERENTIATION_WIDTH
    vec3i size = {300, 300, 300};
    //vec3i size = {250, 250, 250};
    float c_at_max = 90;
    //float c_at_max = 45;
    float scale = c_at_max / (size.largest_elem());
    vec3f centre = {size.x()/2, size.y()/2, size.z()/2};

    equation_context setup_initial;
    setup_initial_conditions(setup_initial, centre, scale);

    equation_context ctx1;
    get_initial_conditions_eqs(ctx1, centre, scale);

    equation_context ctx2;
    build_intermediate(ctx2);

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

    equation_context ctx8;
    build_kreiss_oliger_dissipate(ctx8);


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
    ctx2.build(argument_string, 1);
    ctx3.build(argument_string, 2);
    ctx4.build(argument_string, 3);
    //ctx5.build(argument_string, 4);
    ctx6.build(argument_string, 5);
    ctx7.build(argument_string, 6);
    ctx8.build(argument_string, 7);
    setup_initial.build(argument_string, 8);

    argument_string += "-DBORDER_WIDTH=" + std::to_string(BORDER_WIDTH) + " ";

    #ifdef USE_GBB
    argument_string += "-DUSE_GBB ";
    #endif // USE_GBB

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

    int which_data = 0;

    std::array<std::vector<cl::buffer>, 2> generic_data;

    /*std::array<bool, 21> redundant_buffers
    {
        true, true, true, true, true, true, //dtcYij makes differentiating this unnecessary
        false, false, false, false, false, false, ///cA
        false, false, false, ///cGi
        false, ///K
        false, ///X
        true, true, true, true, ///gA, gB0, gB1, gB2,
    };*/

    #ifndef USE_GBB
    constexpr int buffer_count = 12+9;
    #else
    constexpr int buffer_count = 12 + 9 + 3;
    #endif

    std::array<bool, buffer_count> redundant_buffers;

    for(int idx=0; idx < 2; idx++)
    {
        for(int kk=0; kk < buffer_count; kk++)
        {
            if(redundant_buffers[kk] && idx == 1)
            {
                generic_data[idx].push_back(generic_data[0][kk]);
            }
            else
            {
                generic_data[idx].emplace_back(clctx.ctx);
                generic_data[idx].back().alloc(size.x() * size.y() * size.z() * sizeof(cl_float));
            }
        }
    }

    cl::buffer u_1(clctx.ctx);
    cl::buffer u_2(clctx.ctx);

    u_1.alloc(size.x() * size.y() * size.z() * sizeof(cl_float));
    u_2.alloc(size.x() * size.y() * size.z() * sizeof(cl_float));

    cl::buffer intermediate(clctx.ctx);
    intermediate.alloc(size.x() * size.y() * size.z() * sizeof(intermediate_bssnok_data));

    cl::buffer waveform(clctx.ctx);
    waveform.alloc(sizeof(cl_float2));

    /*std::vector<bssnok_data> cpu_data;

    for(int z=0; z < size.z(); z++)
    {
        for(int y=0; y < size.y(); y++)
        {
            for(int x=0; x < size.x(); x++)
            {
                vec3f pos = {x, y, z};

                cpu_data.push_back(get_conditions(pos, centre, scale));
            }
        }
    }

    bssnok_datas[0].write(clctx.cqueue, cpu_data);*/

    /*bssnok_data test_init = get_conditions({50, 50, 50}, centre, scale);

    std::cout << "TEST0 " << test_init.cY0 << std::endl;
    std::cout << "TEST1 " << test_init.cY1 << std::endl;
    std::cout << "TEST2 " << test_init.cY2 << std::endl;
    std::cout << "TEST3 " << test_init.cY3 << std::endl;
    std::cout << "TEST4 " << test_init.cY4 << std::endl;
    std::cout << "TEST5 " << test_init.cY5 << std::endl;
    std::cout << "TESTb0 " << test_init.gB0 << std::endl;
    std::cout << "TESTX " << test_init.X << std::endl;
    std::cout << "TESTgA " << test_init.gA << std::endl;*/

    vec<4, cl_int> clsize = {size.x(), size.y(), size.z(), 0};
    cl_float time_elapsed_s = 0;

    cl::args initial_u_args;
    initial_u_args.push_back(u_1);
    initial_u_args.push_back(clsize);

    clctx.cqueue.exec("setup_u_offset", initial_u_args, {size.x(), size.y(), size.z()}, {8, 8, 1});

    cl::args initial_u_args2;
    initial_u_args2.push_back(u_2);
    initial_u_args2.push_back(clsize);

    clctx.cqueue.exec("setup_u_offset", initial_u_args2, {size.x(), size.y(), size.z()}, {8, 8, 1});

    for(int i=0; i < 10000; i++)
    {
        cl::args interate_u_args;
        interate_u_args.push_back(u_1);
        interate_u_args.push_back(u_2);
        interate_u_args.push_back(scale);
        interate_u_args.push_back(clsize);

        clctx.cqueue.exec("iterative_u_solve", interate_u_args, {size.x(), size.y(), size.z()}, {8, 8, 1});

        std::swap(u_1, u_2);
    }

    {
        cl::args init;

        for(auto& i : generic_data[0])
        {
            init.push_back(i);
        }

        init.push_back(u_2);
        init.push_back(scale);
        init.push_back(clsize);

        clctx.cqueue.exec("calculate_initial_conditions", init, {size.x(), size.y(), size.z()}, {8, 8, 1});
    }

    {
        cl::args init;

        for(auto& i : generic_data[1])
        {
            init.push_back(i);
        }

        init.push_back(u_2);
        init.push_back(scale);
        init.push_back(clsize);

        clctx.cqueue.exec("calculate_initial_conditions", init, {size.x(), size.y(), size.z()}, {8, 8, 1});
    }

    cl::args initial_clean;

    for(auto& i : generic_data[0])
    {
        initial_clean.push_back(i);
    }

    //initial_clean.push_back(bssnok_datas[0]);
    initial_clean.push_back(u_2);
    initial_clean.push_back(intermediate);
    initial_clean.push_back(scale);
    initial_clean.push_back(clsize);
    initial_clean.push_back(time_elapsed_s);

    clctx.cqueue.exec("clean_data", initial_clean, {size.x(), size.y(), size.z()}, {8, 8, 1});

    cl::args initial_constraints;

    for(auto& i : generic_data[0])
    {
        initial_constraints.push_back(i);
    }

    //initial_constraints.push_back(bssnok_datas[0]);
    initial_constraints.push_back(scale);
    initial_constraints.push_back(clsize);

    clctx.cqueue.exec("enforce_algebraic_constraints", initial_constraints, {size.x(), size.y(), size.z()}, {8, 8, 1});

    cl::args fl2;

    for(auto& i : generic_data[0])
    {
        fl2.push_back(i);
    }

    //fl2.push_back(bssnok_datas[0]);
    fl2.push_back(scale);
    fl2.push_back(clsize);
    fl2.push_back(intermediate);

    clctx.cqueue.exec("calculate_intermediate_data", fl2, {size.x(), size.y(), size.z()}, {8, 8, 1});

    std::vector<cl::read_info<cl_float2>> read_data;

    std::vector<float> real_graph;
    std::vector<float> real_decomp;

    //clctx.cqueue.exec("clean_data", initial_clean, {size.x(), size.y(), size.z()}, {8, 8, 1});

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
            width = buffer_size.x();
            height = buffer_size.y();

            texture_settings new_sett;
            new_sett.width = width;
            new_sett.height = height;
            new_sett.is_srgb = false;

            tex[0].load_from_memory(new_sett, nullptr);
            tex[1].load_from_memory(new_sett, nullptr);

            rtex[0].create_from_texture(tex[0].handle);
            rtex[1].create_from_texture(tex[1].handle);
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

        for(auto& i : generic_data[which_data])
        {
            render.push_back(i);
        }

        //render.push_back(bssnok_datas[which_data]);
        render.push_back(scale);
        render.push_back(clsize);
        render.push_back(intermediate);
        render.push_back(rtex[which_texture]);
        render.push_back(time_elapsed_s);

        clctx.cqueue.exec("render", render, {size.x(), size.y()}, {16, 16});

        if(step)
        {
            float timestep = 0.01;

            if(steps < 200)
               timestep = 0.001;

            if(steps < 10)
                timestep = 0.0001;

            steps++;

            cl::args a1;

            for(auto& i : generic_data[which_data])
            {
                a1.push_back(i);
            }

            for(auto& i : generic_data[(which_data + 1) % 2])
            {
                a1.push_back(i);
            }

            //a1.push_back(bssnok_datas[which_data]);
            //a1.push_back(bssnok_datas[(which_data + 1) % 2]);
            a1.push_back(scale);
            a1.push_back(clsize);
            a1.push_back(intermediate);
            a1.push_back(timestep);
            a1.push_back(time_elapsed_s);
            a1.push_back(current_simulation_boundary);

            clctx.cqueue.exec("evolve", a1, {size.x(), size.y(), size.z()}, {128, 1, 1});

            which_data = (which_data + 1) % 2;

            {
                cl::args cleaner;

                for(auto& i : generic_data[which_data])
                {
                    cleaner.push_back(i);
                }

                //cleaner.push_back(bssnok_datas[which_data]);
                cleaner.push_back(u_2);
                cleaner.push_back(intermediate);
                cleaner.push_back(scale);
                cleaner.push_back(clsize);

                clctx.cqueue.exec("clean_data", cleaner, {size.x(), size.y(), size.z()}, {128, 1, 1});
            }

            cl::args fl3;

            for(auto& i : generic_data[which_data])
            {
                fl3.push_back(i);
            }

            //fl3.push_back(bssnok_datas[which_data]);
            fl3.push_back(scale);
            fl3.push_back(clsize);
            fl3.push_back(intermediate);

            clctx.cqueue.exec("calculate_intermediate_data", fl3, {size.x(), size.y(), size.z()}, {128, 1, 1});

            cl::args constraints;

            for(auto& i : generic_data[which_data])
            {
                constraints.push_back(i);
            }

            //constraints.push_back(bssnok_datas[which_data]);
            constraints.push_back(scale);
            constraints.push_back(clsize);

            clctx.cqueue.exec("enforce_algebraic_constraints", constraints, {size.x(), size.y(), size.z()}, {128, 1, 1});

            float r_extract = c_at_max/4;

            //printf("OFF %f\n", r_extract/scale);

            cl_int4 pos = {clsize.x()/2, clsize.y()/2 + r_extract / scale, clsize.z()/2, 0};

            /*cl::args waveform_args;

            for(auto& i : generic_data[which_data])
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

        {
            cl::args render_args;

            for(auto& i : generic_data[which_data])
            {
                render_args.push_back(i);
            }

            float fwidth = width;
            float fheight = height;

            cl_float3 ccamera_pos = {camera_pos.x(), camera_pos.y(), camera_pos.z()};
            cl_float4 ccamera_quat = {camera_quat.q.x(), camera_quat.q.y(), camera_quat.q.z(), camera_quat.q.w()};

            render_args.push_back(scale);
            render_args.push_back(intermediate);
            render_args.push_back(ccamera_pos);
            render_args.push_back(ccamera_quat);
            render_args.push_back(fwidth);
            render_args.push_back(fheight);
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

        cl::event next_event = rtex[which_texture].unacquire(clctx.cqueue);

        if(last_event.has_value())
            last_event.value().block();

        last_event = next_event;

        ///todo: get rid of this
        clctx.cqueue.block();

        {
            ImDrawList* lst = ImGui::GetBackgroundDrawList();

            ImVec2 screen_pos = ImGui::GetMainViewport()->Pos;

            ImVec2 tl = {0,0};
            ImVec2 br = {win.get_window_size().x(),win.get_window_size().y()};

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
    }
}
