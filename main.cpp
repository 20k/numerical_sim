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
*/

//#define USE_GBB

///all conformal variables are explicitly labelled
struct bssnok_data
{
    /**
    conformal
    [0, 1, 2,
     X, 3, 4,
     X, X, 5]
    */
    cl_float cY0, cY1, cY2, cY3, cY4, cY5;

    /**
    conformal
    [0, 1, 2,
     X, 3, 4,
     X, X, 5]
    */
    cl_float cA0, cA1, cA2, cA3, cA4, cA5;

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

struct intermediate_bssnok_data
{
    cl_float dcYij[3 * 6];
    cl_float digA[6];
    cl_float digB[3*3];
    //cl_float phi;
    cl_float dphi[3];
};

template<typename T, typename U, int N, size_t M>
inline
tensor<T, N, N> gpu_lie_derivative_weight_arbitrary(const tensor<T, N>& B, const tensor<T, N, N>& mT, float weight, const std::array<U, M>& variables)
{
    tensor<T, N, N> lie;

    for(int i=0; i < N; i++)
    {
        for(int j=0; j < N; j++)
        {
            T sum = 0;

            for(int k=0; k < N; k++)
            {
                sum = sum + B.idx(k) * mT.idx(i, j).differentiate(variables[k]);
                sum = sum + mT.idx(i, k) * B.idx(k).differentiate(variables[j]);
                sum = sum + mT.idx(k, j) * B.idx(k).differentiate(variables[i]);
                sum = sum + weight * mT.idx(i, j) * B.idx(k).differentiate(variables[k]);
            }

            lie.idx(i, j) = sum;
        }
    }

    return lie;
}

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

value f_r(value r)
{
    auto interpolating_polynomial = [](value x)
    {
        ///https://www.wolframalpha.com/input/?i=InterpolatingPolynomial%5B%7B%7B%7B0%7D%2C+0%2C+0%2C+0%7D%2C+%7B%7B1%7D%2C+1%2C+0%2C+0%7D%7D%2C+%7Bx%7D%5D
        ///(1 + (-3 + 6 (-1 + x)) (-1 + x)) x^3

        return (1 + (-3 + 6 * (-1 + x)) * (-1 + x)) * x * x * x;
    };

    value r_max = 0.8;
    value r_min = 0.2;

    r = max(min(r, r_max), r_min);

    value scaled = (r - r_min) / (r_max - r_min);

    return interpolating_polynomial(scaled);
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

        std::string name = "pv[" + std::to_string(temporaries.size()) + "]";

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
            argument_string += "-DTEMP_COUNT" + std::to_string(idx) + "=1 -DTEMPORARIES={0} ";
            return;
        }

        argument_string += "-DTEMP_COUNT" + std::to_string(idx) + "=" + std::to_string(temporaries.size()) + " ";

        std::string temporary_string;

        for(auto& i : temporaries)
        {
            temporary_string += type_to_string(i.second) + ",";
        }

        if(temporary_string.size() > 0 && temporary_string.back() == ',')
            temporary_string.pop_back();

        argument_string += "-DTEMPORARIES" + std::to_string(idx) + "=" + temporary_string + " ";
    }
};

//#define SYMMETRY_BOUNDARY
#define BORDER_WIDTH 2

///todo: I know for a fact that clang is too silly to optimise out the memory lookups
value hacky_differentiate(equation_context& ctx, const value& in, int idx, bool pin = true)
{
    assert(in.is_value());

    assert(in.value_payload.value().starts_with("v->") || in.value_payload.value().starts_with("v.") || in.value_payload.value().starts_with("ik."));

    std::string buffer;
    std::string val;

    if(in.value_payload.value().starts_with("v->"))
    {
        buffer = "in";

        val = in.value_payload.value();

        val.erase(val.begin());
        val.erase(val.begin());
        val.erase(val.begin());
    }
    else if(in.value_payload.value().starts_with("v."))
    {
        buffer = "in";

        val = in.value_payload.value();

        val.erase(val.begin());
        val.erase(val.begin());
    }
    else if(in.value_payload.value().starts_with("ik."))
    {
        buffer = "temp_in";

        val = in.value_payload.value();

        val.erase(val.begin());
        val.erase(val.begin());
        val.erase(val.begin());
    }

    std::string dimx = "dim.x";
    std::string dimy = "dim.y";
    std::string dimz = "dim.z";

    std::string x = "x";
    std::string y = "y";
    std::string z = "z";

    std::string xp1 = "x+1";
    std::string yp1 = "y+1";
    std::string zp1 = "z+1";

    std::string xp2 = "x+2";
    std::string yp2 = "y+2";
    std::string zp2 = "z+2";

    std::string xm1 = "x-1";
    std::string ym1 = "y-1";
    std::string zm1 = "z-1";

    std::string xm2 = "x-2";
    std::string ym2 = "y-2";
    std::string zm2 = "z-2";

    #ifdef SYMMETRY_BOUNDARY
    xp1 = "(" + xp1 + ")%dim.x";
    yp1 = "(" + yp1 + ")%dim.y";
    zp1 = "(" + zp1 + ")%dim.z";

    xp2 = "(" + xp2 + ")%dim.x";
    yp2 = "(" + yp2 + ")%dim.y";
    zp2 = "(" + zp2 + ")%dim.z";

    xm1 = "abs(" + xm1 + ")";
    ym1 = "abs(" + ym1 + ")";
    zm1 = "abs(" + zm1 + ")";

    xm2 = "abs(" + xm2 + ")";
    ym2 = "abs(" + ym2 + ")";
    zm2 = "abs(" + zm2 + ")";
    #endif // SYMMETRY_BOUNDARY

    auto index_raw = [](const std::string& x, const std::string& y, const std::string& z)
    {
        return "IDX(" + x + "," + y + "," + z + ")";
    };

    auto index_buffer = [](const std::string& variable, const std::string& buffer, const std::string& with_what)
    {
        return buffer + "[" + with_what + "]." + variable;
    };

    auto finite_difference = [](const std::string& upper, const std::string& lower)
    {
        return "finite_difference(" + upper + "," + lower + ",scale)";
    };

    auto index = [&](const std::string& x, const std::string& y, const std::string& z)
    {
        value v = value(index_buffer(val, buffer, index_raw(x, y, z)));

        ctx.pin(v);

        return v;
    };

    value scale = "scale";

    value final_command;

    if(idx == 0)
    {
        //final_command = (-index(xp2, y, z) + 8 * index(xp1, y, z) - 8 * index(xm1, y, z) + index(xm2, y, z)) / (12 * scale);

        final_command = (index(xp1, y, z) - index(xm1, y, z)) / (2 * scale);
    }

    if(idx == 1)
    {
        //final_command = (-index(x, yp2, z) + 8 * index(x, yp1, z) - 8 * index(x, ym1, z) + index(x, ym2, z)) / (12 * scale);

        final_command = (index(x, yp1, z) - index(x, ym1, z)) / (2 * scale);
    }

    if(idx == 2)
    {
        //final_command = (-index(x, y, zp2) + 8 * index(x, y, zp1) - 8 * index(x, y, zm1) + index(x, y, zm2)) / (12 * scale);

        final_command = (index(x, y, zp1) - index(x, y, zm1)) / (2 * scale);
    }

    if(pin)
    {
        ctx.pin(final_command);
    }

    return final_command;
}

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
                sum = sum + B.idx(k) * hacky_differentiate(ctx, mT.idx(i, j), k);
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
tensor<value, 3, 3> raise_index(const tensor<value, 3, 3>& mT, const metric<value, 3, 3>& met, const inverse_metric<value, 3, 3>& inverse)
{
    tensor<value, 3, 3> ret;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value sum = 0;

            for(int k=0; k < 3; k++)
            {
                sum = sum + inverse.idx(i, k) * mT.idx(k, j);
                //sum = sum + mT.idx(i, k) * inverse.idx(k, j);
            }

            ret.idx(i, j) = sum;
        }
    }

    return ret;
}


tensor<value, 3> raise_index(const tensor<value, 3>& mT, const metric<value, 3, 3>& met, const inverse_metric<value, 3, 3>& inverse)
{
    tensor<value, 3> ret;

    for(int i=0; i < 3; i++)
    {
        value sum = 0;

        for(int j=0; j < 3; j++)
        {
            sum = sum + inverse.idx(i, j) * mT.idx(j);
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
    vec<3, value> pos;

    pos[0].make_value("ox");
    pos[1].make_value("oy");
    pos[2].make_value("oz");

    //#define DEBUG
    #ifdef DEBUG
    pos[0].make_value("20");
    pos[1].make_value("125");
    pos[2].make_value("125");
    #endif // DEBUG

    std::array<std::string, 3> variables = {"ox", "oy", "oz"};

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


    value BL_conformal = 1;

    //vec<3, value> vcentre = {centre.x(), centre.y(), centre.z()};

    //value r = (pos - vcentre).length() * scale;

    value r = pos.length() * scale;

    //std::cout << "REQ " << type_to_string(r) << std::endl;

    r = max(r, 0.01f);

   // std::cout << "FR " << r.substitute("x", 20).substitute("y", 125).substitute("z", 125).get_constant() << std::endl;

    auto san_black_hole_pos = [&](vec3f in)
    {
        return floor(in / scale) + (vec3f){0.5, 0.5, 0.5};
    };

    ///https://arxiv.org/pdf/gr-qc/0505055.pdf
    //std::vector<vec3f> black_hole_pos{san_black_hole_pos({0, -1.1515 * 0.5f, 0}), san_black_hole_pos({0, 1.1515 * 0.5f, 0})};
    std::vector<vec3f> black_hole_pos{san_black_hole_pos({-1.1515 * 0.5f, -0.01, -0.01}), san_black_hole_pos({1.1515 * 0.5f, 0.01, 0.01})};
    std::vector<float> black_hole_m{0.5f, 0.5f};
    std::vector<vec3f> black_hole_velocity{{0, 0.5, 0}, {0, -0.5, 0}}; ///pick better velocities
    //std::vector<float> black_hole_m{0.1f, 0.1f};
    //std::vector<float> black_hole_m{1, 1};

    //std::vector<vec3f> black_hole_pos{{0,0,0}};
    //std::vector<float> black_hole_m{1};
    //std::vector<vec3f> black_hole_velocity{{0, 0.4, 0}};

    ///3.57 https://scholarworks.rit.edu/cgi/viewcontent.cgi?article=11286&context=theses
    ///todo: not sure this is correctly done, check r - ri, and what coordinate r really is
    ///todo pt 2: Try sponging to schwarzschild
    for(int i=0; i < (int)black_hole_m.size(); i++)
    {
        float Mi = black_hole_m[i];
        vec3f ri = black_hole_pos[i];

        vec<3, value> vri = {ri.x(), ri.y(), ri.z()};

        value dist = (pos - vri).length() * scale;

        dist = max(dist, 0.001f);

        BL_conformal = BL_conformal + Mi / (2 * dist);
    }

    ///ok so: I'm pretty sure this is correct
    metric<value, 3, 3> yij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            float u = 0.0;

            ///https://arxiv.org/pdf/gr-qc/0511048.pdf
            yij.idx(i, j) = pow(BL_conformal + u, 4) * kronecker.idx(i, j);

            /*if(i == j)
            {
                yij.idx(i, j) = f_r(r) * yij.idx(i, j) + (1 - f_r(r)) * 9999;
            }
            else
            {
                yij.idx(i, j) = f_r(r) * yij.idx(i, j);
            }*/
        }
    }

    ///https://arxiv.org/pdf/gr-qc/9810065.pdf, 11
    value Y = yij.det();

    inverse_metric<value, 3, 3> iyij = yij.invert();

    ///phi
    value conformal_factor = (1/12.f) * log(Y);

    ctx.pin(conformal_factor);

    metric<value, 3, 3> cyij;

    ///checked, cyij is correct
    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            cyij.idx(i, j) = exp(-4 * conformal_factor) * yij.idx(i, j);
        }
    }

    inverse_metric<value, 3, 3> icY = cyij.invert();

    ///calculate icAij from https://arxiv.org/pdf/gr-qc/0206072.pdf (58)
    tensor<value, 3, 3> icAij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value bh_sum = 0;

            /*for(int bh_idx = 0; bh_idx < (int)black_hole_pos.size(); bh_idx++)
            {
                float Mi = black_hole_m[bh_idx];
                vec3f ri = black_hole_pos[bh_idx];

                vec<3, value> vri = {ri.x(), ri.y(), ri.z()};

                value dist = (pos - vri).length() * scale;

                dist = max(dist, 0.1f);

                {
                    vec3f P = black_hole_velocity[bh_idx] * black_hole_m[bh_idx];
                    vec<3, value> normal = (vec<3, value>){0, 0, -1} / dist;

                    value lsum = 0;

                    for(int k=0; k < 3; k++)
                    {
                        for(int l=0; l < 3; l++)
                        {
                            lsum = lsum + yij.idx(k, l) * normal[k] * P[l];
                        }
                    }

                    bh_sum = bh_sum + (3.f / (2.f * dist * dist)) * (normal[i] * P[j] + normal[j] * P[i] - (iyij.idx(i, j) - normal[i] * normal[j]) * lsum);
                }
            }*/

            icAij.idx(i, j) = bh_sum;
        }
    }

    /*value gA = 1/BL_conformal;
    value gB0 = 1/BL_conformal;
    value gB1 = 1/BL_conformal;
    value gB2 = 1/BL_conformal;*/

    value gA = 1/(BL_conformal * BL_conformal);
    value gB0 = 0;
    value gB1 = 0;
    value gB2 = 0;

    /*value gA = 1;
    value gB0 = 0;
    value gB1 = 0;
    value gB2 = 0;*/

    #if 0
    tensor<value, 3> norm;
    norm.idx(0) = -gB0 / gA;
    norm.idx(1) = -gB1 / gA;
    norm.idx(2) = -gB2 / gA;

    tensor<value, 3, 3> nearly_Kij = gpu_lie_derivative_weight_arbitrary(norm, yij, 0, variables);

    tensor<value, 3, 3> Kij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            Kij.idx(i, j) = nearly_Kij.idx(i, j) / -2.f;

            //Kij.idx(i, j) = f_r(r) * Kij.idx(i, j);
        }
    }

    value K = gpu_trace(Kij, yij, yij.invert());

    tensor<value, 3, 3> Aij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            Aij.idx(i, j) = Kij.idx(i, j) - (1.f/3.f) * yij.idx(i, j) * K;
        }
    }

    tensor<value, 3, 3> cAij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            cAij.idx(i, j) = exp(-4 * conformal_factor) * Aij.idx(i, j);
        }
    }


    ///https://arxiv.org/pdf/gr-qc/9810065.pdf (21)

    inverse_metric<value, 3, 3> inverse_cyij = cyij.invert();

    tensor<value, 3> cGi;

    for(int i=0; i < 3; i++)
    {
        value sum = 0;

        for(int j=0; j < 3; j++)
        {
            sum = sum + inverse_cyij.idx(i, j).differentiate(variables[j]);
        }

        cGi.idx(i) = -sum;
    }
    #endif // 0

    //tensor<value, 3, 3> iYij = yij.invert();

    tensor<value, 3, 3> cAij = lower_both(icAij, cyij);
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

        for(int j=0; j < 3; j++)
        {
            cAij.idx(i, j) = 0;
        }
    }

    K = 0;
    #endif // OLDFLAT

    for(int i=0; i < 6; i++)
    {
        vec2i index = linear_indices[i];

        std::string y_name = "init_cY" + std::to_string(i);
        std::string a_name = "init_cA" + std::to_string(i);

        ctx.add(y_name, cyij.idx(index.x(), index.y()));
        ctx.add(a_name, cAij.idx(index.x(), index.y()));
    }

    ctx.add("init_cGi0", cGi.idx(0));
    ctx.add("init_cGi1", cGi.idx(1));
    ctx.add("init_cGi2", cGi.idx(2));

    ctx.add("init_K", K);
    ctx.add("init_X", X);

    ctx.add("init_bl_conformal", BL_conformal);
    ctx.add("init_conformal_factor", conformal_factor);

    ctx.add("init_gA", gA);
    ctx.add("init_gB0", gB0);
    ctx.add("init_gB1", gB1);
    ctx.add("init_gB2", gB2);

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

    cY.idx(0, 0).make_value("v.cY0"); cY.idx(0, 1).make_value("v.cY1"); cY.idx(0, 2).make_value("v.cY2");
    cY.idx(1, 0).make_value("v.cY1"); cY.idx(1, 1).make_value("v.cY3"); cY.idx(1, 2).make_value("v.cY4");
    cY.idx(2, 0).make_value("v.cY2"); cY.idx(2, 1).make_value("v.cY4"); cY.idx(2, 2).make_value("v.cY5");

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

    cA.idx(0, 0).make_value("v.cA0"); cA.idx(0, 1).make_value("v.cA1"); cA.idx(0, 2).make_value("v.cA2");
    cA.idx(1, 0).make_value("v.cA1"); cA.idx(1, 1).make_value("v.cA3"); cA.idx(1, 2).make_value("v.cA4");
    cA.idx(2, 0).make_value("v.cA2"); cA.idx(2, 1).make_value("v.cA4"); cA.idx(2, 2).make_value("v.cA5");

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

    cY.idx(0, 0).make_value("v.cY0"); cY.idx(0, 1).make_value("v.cY1"); cY.idx(0, 2).make_value("v.cY2");
    cY.idx(1, 0).make_value("v.cY1"); cY.idx(1, 1).make_value("v.cY3"); cY.idx(1, 2).make_value("v.cY4");
    cY.idx(2, 0).make_value("v.cY2"); cY.idx(2, 1).make_value("v.cY4"); cY.idx(2, 2).make_value("v.cY5");

    inverse_metric<value, 3, 3> icY = cY.invert();

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            ctx.pin(icY.idx(i, j));
        }
    }

    value gA;
    gA.make_value("v.gA");

    value X;
    X.make_value("v.X");

    tensor<value, 3> gB;
    gB.idx(0).make_value("v.gB0");
    gB.idx(1).make_value("v.gB1");
    gB.idx(2).make_value("v.gB2");

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

    ///or 0.25f * log(1.f/v.X);
    auto X_to_phi = [](value X)
    {
        return -0.25f * log(X);
    };

    value phi = X_to_phi(X);

    ctx.pin(phi);

    tensor<value, 3> dphi;

    for(int i=0; i < 3; i++)
    {
        value dX = hacky_differentiate(ctx, X, i);

        dphi.idx(i) = -dX / (4 * X);
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
            value diff = hacky_differentiate(ctx, "v.cY" + std::to_string(i), k);

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

    //ctx.add("init_phi", phi);

    ctx.add("init_dphi0", dphi.idx(0));
    ctx.add("init_dphi1", dphi.idx(1));
    ctx.add("init_dphi2", dphi.idx(2));
}

///https://arxiv.org/pdf/gr-qc/0206072.pdf on stability, they recompute cGi where it does nto hae a derivative
///todo: X: This is think is why we're getting nans. Half done
///todo: fisheye - half done
///todo: better differentiation
///todo: Enforce the algebraic constraints in this paper: https://arxiv.org/pdf/gr-qc/0505055.pdf. 3.21 and 3.22 - done
///todo: if I use a mirror boundary condition, it'd simulate an infinite grid of black hole pairs colliding
///they would however all be relatively far away from each other, so this may turn out fairly acceptably
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

    cY.idx(0, 0).make_value("v->cY0"); cY.idx(0, 1).make_value("v->cY1"); cY.idx(0, 2).make_value("v->cY2");
    cY.idx(1, 0).make_value("v->cY1"); cY.idx(1, 1).make_value("v->cY3"); cY.idx(1, 2).make_value("v->cY4");
    cY.idx(2, 0).make_value("v->cY2"); cY.idx(2, 1).make_value("v->cY4"); cY.idx(2, 2).make_value("v->cY5");

    inverse_metric<value, 3, 3> icY = cY.invert();

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            ctx.pin(icY.idx(i, j));
        }
    }

    tensor<value, 3, 3> cA;

    cA.idx(0, 0).make_value("v->cA0"); cA.idx(0, 1).make_value("v->cA1"); cA.idx(0, 2).make_value("v->cA2");
    cA.idx(1, 0).make_value("v->cA1"); cA.idx(1, 1).make_value("v->cA3"); cA.idx(1, 2).make_value("v->cA4");
    cA.idx(2, 0).make_value("v->cA2"); cA.idx(2, 1).make_value("v->cA4"); cA.idx(2, 2).make_value("v->cA5");

    ///the christoffel symbol
    tensor<value, 3> cGi;
    cGi.idx(0).make_value("v->cGi0");
    cGi.idx(1).make_value("v->cGi1");
    cGi.idx(2).make_value("v->cGi2");

    tensor<value, 3> digA;
    digA.idx(0).make_value("ik.digA[0]");
    digA.idx(1).make_value("ik.digA[1]");
    digA.idx(2).make_value("ik.digA[2]");

    tensor<value, 3> dphi;
    dphi.idx(0).make_value("ik.dphi[0]");
    dphi.idx(1).make_value("ik.dphi[1]");
    dphi.idx(2).make_value("ik.dphi[2]");

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
    gA.make_value("v->gA");

    tensor<value, 3> gB;
    gB.idx(0).make_value("v->gB0");
    gB.idx(1).make_value("v->gB1");
    gB.idx(2).make_value("v->gB2");

    #ifdef USE_GBB
    tensor<value, 3> gBB;
    gBB.idx(0).make_value("v.gBB0");
    gBB.idx(1).make_value("v.gBB1");
    gBB.idx(2).make_value("v.gBB2");
    #endif // USE_GBB

    value X;
    X.make_value("v->X");

    value K;
    K.make_value("v->K");

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
            hacky_differentiate(ctx, "v->cY" + std::to_string(i), k);
        }

        for(int i=0; i < 6; i++)
        {
            hacky_differentiate(ctx, "v->cA" + std::to_string(i), k);
        }

        for(int i=0; i < 3; i++)
        {
            hacky_differentiate(ctx, "v->cGi" + std::to_string(i), k);
        }

        hacky_differentiate(ctx, "v->K", k);
        hacky_differentiate(ctx, "v->X", k);
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
            hacky_differentiate(ctx, "ik.dphi[" + std::to_string(i) + "]", k);
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

    tensor<value, 3, 3> lie_cYij = gpu_lie_derivative_weight(ctx, gB, cY);

    tensor<value, 3, 3> dtcYij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            ///https://arxiv.org/pdf/gr-qc/0511048.pdf (1)
            dtcYij.idx(i, j) = -2 * gA * cA.idx(i, j) + lie_cYij.idx(i, j);

            ///https://scholarworks.rit.edu/cgi/viewcontent.cgi?article=11286&context=theses 3.66
            //dtcYij.idx(i, j) = -2 * gA + lie_cYij.idx(i, j);

            /*if(i == 0 && j == 0)
            {
                ctx.add("debug_val", lie_cYij.idx(i, j));
            }*/
        }
    }

    //ctx.add("debug_val", cY.det());

    value dtX = 0;

    {
        value s1 = 0;
        value s2 = 0;

        for(int i=0; i < 3; i++)
        {
            s1 = s1 + hacky_differentiate(ctx, gB.idx(i), i);
            s2 = s2 + gB.idx(i) * hacky_differentiate(ctx, X, i);
        }

        dtX = (2.f/3.f) * X * (gA * K - s1) + s2;
    }

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

    //ctx.add("debug_val", Rphiij.idx(i, j));

    //ctx.add("debug_val", gpu_trace(cA, cY, icY));

    tensor<value, 3, 3> Rij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            Rij.idx(i, j) = Rphiij.idx(i, j) + cRij.idx(i, j);

            ctx.pin(Rij.idx(i, j));
        }
    }

    ///recover Yij from X and cYij
    ///https://arxiv.org/pdf/gr-qc/0511048.pdf
    ///https://arxiv.org/pdf/gr-qc/9810065.pdf
    ///X = exp(-4 phi)
    ///consider trying to eliminate via https://arxiv.org/pdf/gr-qc/0206072.pdf (27). I think this is what you're meant to do
    ///to eliminate the dependency on the non conformal metric entirely. This would improve stability quite significantly
    ///near the puncture
    metric<value, 3, 3> Yij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            Yij.idx(i, j) = cY.idx(i, j) / X;
        }
    }

    //ctx.add("debug_val", derived_cGi.idx(0));

    //ctx.add("debug_val", Yij.idx(0, 1));

    inverse_metric<value, 3, 3> iYij = Yij.invert();

    ///Aki G^kj
    tensor<value, 3, 3> mixed_cAij = raise_index(cA, cY, icY);

    tensor<value, 3, 3> dtcAij;

    tensor<value, 3, 3> with_trace;


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

            value trace_free_interior_2 = X * gA * Rij.idx(i, j);

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
                sum = cA.idx(i, k) * mixed_cAij.idx(k, j);
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
            value trace_free_part = gpu_trace_free(with_trace, cY, icY).idx(i, j);

            value p1 = trace_free_part;

            value p2 = gA * (K * cA.idx(i, j) - 2 * sum);

            value p3 = gpu_lie_derivative_weight(ctx, gB, cA).idx(i, j);

            value sanitised = dual_types::dual_if(X <= 0.000001, []()
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
            });

            dtcAij.idx(i, j) = sanitised + p2 + p3;
        }
    }

    tensor<value, 3, 3> icAij = raise_both(cA, cY, icY);

    value dtK = 0;

    {
        value sum1 = 0;

        for(int i=0; i < 3; i++)
        {
            value christoffel_sum = 0;

            for(int k=0; k < 3; k++)
            {
                christoffel_sum = christoffel_sum - X * derived_cGi.idx(k) * digA.idx(k);
            }

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


            sum1 = sum1 + s1 + christoffel_sum + s3;

            //sum1 = sum1 + gpu_high_covariant_derivative_vec(ctx, digA, Yij, iYij).idx(i, i);
        }

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
            sum3 = sum3 + gB.idx(i) * hacky_differentiate(ctx, K, i);
        }

        dtK = -sum1 + sum2 + sum3;
    }

    ///a / X
    value gA_X = 0;

    {
        float min_X = 0.00001;

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

            value s2 = 0;

            for(int k=0; k < 3; k++)
            {
                s2 = s2 + (1.f/3.f) * icY.idx(i, j) * hacky_differentiate(ctx, digB.idx(k, k), j);
            }

            value s3 = gB.idx(j) * hacky_differentiate(ctx, cGi.idx(i), j);

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

    ///DAMP?
    for(int i=0; i < 3; i++)
    {
        //dtcGi.idx(i) = dtcGi.idx(i) + (derived_cGi.idx(i) - cGi.idx(i)) * 10;
    }

    value dtgA = -2 * gA * K;

    for(int i=0; i < 3; i++)
    {
        dtgA = dtgA + gB.idx(i) * hacky_differentiate(ctx, gA, i);
    }

    //ctx.add("debug_val", dtgA);

    #ifndef USE_GBB
    tensor<value, 3> dtgB;

    /*
    ///https://arxiv.org/pdf/1404.6523.pdf (4)
    for(int i=0; i < 3; i++)
    {
        float N = 1.375;

        dtgB.idx(i) = (3.f/4.f) * derived_cGi.idx(i) - N * gB.idx(i);
    }*/

    ///https://arxiv.org/pdf/gr-qc/0605030.pdf 26

    for(int i=0; i < 3; i++)
    {
        float N = 2;
        //float N = 0.1;

        value sum = 0;

        for(int j=0; j < 3; j++)
        {
            sum = gB.idx(j) * hacky_differentiate(ctx, gB.idx(i), j);
        }

        dtgB.idx(i) = (3.f/4.f) * derived_cGi.idx(i) + sum - N * gB.idx(i);
    }

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

    value scalar_curvature = 0;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            scalar_curvature = scalar_curvature + iYij.idx(i, j) * Rij.idx(i, j);
        }
    }

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

    ctx.add("scalar_curvature", scalar_curvature);
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
}

template<typename T, int U, int... N>
inline
tensor<T, N...> raise_index_generic(const tensor<T, N...>& mT, const inverse_metric<T, U, U>& met, int index)
{
    tensor<T, N...> ret;

    return raise_index_impl(mT, met, index);
}

///assumes unigrid
inline
void extract_waveforms(equation_context& ctx)
{
    int index_table[3][3] = {{0, 1, 2},
                             {1, 3, 4},
                             {2, 4, 5}};

    unit_metric<value, 3, 3> cY;

    cY.idx(0, 0).make_value("v->cY0"); cY.idx(0, 1).make_value("v->cY1"); cY.idx(0, 2).make_value("v->cY2");
    cY.idx(1, 0).make_value("v->cY1"); cY.idx(1, 1).make_value("v->cY3"); cY.idx(1, 2).make_value("v->cY4");
    cY.idx(2, 0).make_value("v->cY2"); cY.idx(2, 1).make_value("v->cY4"); cY.idx(2, 2).make_value("v->cY5");

    inverse_metric<value, 3, 3> icY = cY.invert();

    ctx.pin(icY);

    tensor<value, 3, 3> cA;

    cA.idx(0, 0).make_value("v->cA0"); cA.idx(0, 1).make_value("v->cA1"); cA.idx(0, 2).make_value("v->cA2");
    cA.idx(1, 0).make_value("v->cA1"); cA.idx(1, 1).make_value("v->cA3"); cA.idx(1, 2).make_value("v->cA4");
    cA.idx(2, 0).make_value("v->cA2"); cA.idx(2, 1).make_value("v->cA4"); cA.idx(2, 2).make_value("v->cA5");

    tensor<value, 3> cGi;
    cGi.idx(0).make_value("v->cGi0");
    cGi.idx(1).make_value("v->cGi1");
    cGi.idx(2).make_value("v->cGi2");

    tensor<value, 3> digA;
    digA.idx(0).make_value("ik.digA[0]");
    digA.idx(1).make_value("ik.digA[1]");
    digA.idx(2).make_value("ik.digA[2]");

    tensor<value, 3> dphi;
    dphi.idx(0).make_value("ik.dphi[0]");
    dphi.idx(1).make_value("ik.dphi[1]");
    dphi.idx(2).make_value("ik.dphi[2]");

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
    gA.make_value("v->gA");

    tensor<value, 3> gB;
    gB.idx(0).make_value("v->gB0");
    gB.idx(1).make_value("v->gB1");
    gB.idx(2).make_value("v->gB2");

    #ifdef USE_GBB
    tensor<value, 3> gBB;
    gBB.idx(0).make_value("v.gBB0");
    gBB.idx(1).make_value("v.gBB1");
    gBB.idx(2).make_value("v.gBB2");
    #endif // USE_GBB

    value X;
    X.make_value("v->X");

    value K;
    K.make_value("v->K");

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
            hacky_differentiate(ctx, "v->cY" + std::to_string(i), k);
        }

        for(int i=0; i < 6; i++)
        {
            hacky_differentiate(ctx, "v->cA" + std::to_string(i), k);
        }

        for(int i=0; i < 3; i++)
        {
            hacky_differentiate(ctx, "v->cGi" + std::to_string(i), k);
        }

        hacky_differentiate(ctx, "v->K", k);
        hacky_differentiate(ctx, "v->X", k);
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
            hacky_differentiate(ctx, "ik.dphi[" + std::to_string(i) + "]", k);
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

    tensor<value, 3, 3> Rij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            Rij.idx(i, j) = Rphiij.idx(i, j) + cRij.idx(i, j);

            ctx.pin(Rij.idx(i, j));
        }
    }

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
    v3a = (v3a - v1a * wij(0, 2) - v2a * wij(1, 2)) / sqrt(wij(3, 3));

    vec<4, value> thetau = {0, v3a[0], v3a[1], v3a[2]};
    vec<4, value> phiu = {0, v1a[0], v1a[1], v1a[2]};

    dual_types::complex<value> unit_i = dual_types::unit_i();

    tensor<dual_types::complex<value>, 4> mu;

    for(int i=0; i < 4; i++)
    {
        mu.idx(i) = dual_types::complex<value>(1.f/sqrt(2.f)) * (thetau[i] + unit_i * phiu[i]);
    }

    tensor<value, 3, 3, 3> raised_eijk = raise_index_generic(raise_index_generic(eijk_tensor, iYij, 1), iYij, 2);



    //vec<4, dual_types::complex<value>> mu = (1.f/sqrt(2)) * (thetau + i * phiu);
}

float fisheye(float r)
{
    float a = 3;
    float r0 = 5.5f * 0.5f;
    float s = 1.2f * 0.5f;

    float other = 0;

    ///https://arxiv.org/pdf/gr-qc/0505055.pdf 5.5
    float R_r = (s / (2 * r * tanh(r0/s))) * log(cosh((r + r0)/s)/cosh((r - r0)/s));

    float r_phys = r * (a + (1 - a) * R_r);

    return r_phys;
}

int main()
{
    for(float r = 0; r <= 100; r += 1)
    {
        printf("Fish %f %f\n", r, fisheye(r));
    }

    //return 0;

    int width = 1422;
    int height = 800;

    render_settings sett;
    sett.width = width;
    sett.height = height;
    sett.opencl = true;
    sett.no_double_buffer = true;

    render_window win(sett, "Geodesics");

    assert(win.clctx);

    opencl_context& clctx = *win.clctx;

    std::string argument_string = "-O3 -cl-std=CL2.2 ";

    vec3i size = {280, 280, 280};
    //vec3i size = {250, 250, 250};
    float c_at_max = 40;
    float scale = c_at_max / size.largest_elem();
    vec3f centre = {size.x()/2, size.y()/2, size.z()/2};

    equation_context ctx1;
    get_initial_conditions_eqs(ctx1, centre, scale);

    equation_context ctx2;
    build_intermediate(ctx2);

    equation_context ctx3;
    build_eqs(ctx3);

    equation_context ctx4;
    build_constraints(ctx4);

    equation_context ctx5;
    extract_waveforms(ctx5);

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
    ctx5.build(argument_string, 4);

    argument_string += "-DBORDER_WIDTH=" + std::to_string(BORDER_WIDTH) + " ";

    std::cout << "ARGS " << argument_string << std::endl;

    cl::program prog(clctx.ctx, "cl.cl");
    prog.build(clctx.ctx, argument_string);

    clctx.ctx.register_program(prog);

    texture_settings tsett;
    tsett.width = width;
    tsett.height = height;
    tsett.is_srgb = false;

    std::array<texture, 2> tex;
    tex[0].load_from_memory(tsett, nullptr);
    tex[1].load_from_memory(tsett, nullptr);

    std::array<cl::gl_rendertexture, 2> rtex{clctx.ctx, clctx.ctx};
    rtex[0].create_from_texture(tex[0].handle);
    rtex[1].create_from_texture(tex[1].handle);

    std::array<cl::buffer, 2> bssnok_datas{clctx.ctx, clctx.ctx};
    int which_data = 0;

    bssnok_datas[0].alloc(size.x() * size.y() * size.z() * sizeof(bssnok_data));
    bssnok_datas[1].alloc(size.x() * size.y() * size.z() * sizeof(bssnok_data));

    cl::buffer intermediate(clctx.ctx);
    intermediate.alloc(size.x() * size.y() * size.z() * sizeof(intermediate_bssnok_data));

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

    cl::args init;
    init.push_back(bssnok_datas[0]);
    init.push_back(scale);
    init.push_back(clsize);

    clctx.cqueue.exec("calculate_initial_conditions", init, {size.x(), size.y(), size.z()}, {8, 8, 1});

    cl::args initial_clean;
    initial_clean.push_back(bssnok_datas[0]);
    initial_clean.push_back(intermediate);
    initial_clean.push_back(scale);
    initial_clean.push_back(clsize);
    initial_clean.push_back(time_elapsed_s);

    clctx.cqueue.exec("clean_data", initial_clean, {size.x(), size.y(), size.z()}, {8, 8, 1});

    cl::args initial_constraints;
    initial_constraints.push_back(bssnok_datas[0]);
    initial_constraints.push_back(scale);
    initial_constraints.push_back(clsize);

    clctx.cqueue.exec("enforce_algebraic_constraints", initial_constraints, {size.x(), size.y(), size.z()}, {8, 8, 1});

    cl::args fl2;
    fl2.push_back(bssnok_datas[0]);
    fl2.push_back(scale);
    fl2.push_back(clsize);
    fl2.push_back(intermediate);

    clctx.cqueue.exec("calculate_intermediate_data", fl2, {size.x(), size.y(), size.z()}, {8, 8, 1});

    //clctx.cqueue.exec("clean_data", initial_clean, {size.x(), size.y(), size.z()}, {8, 8, 1});

    int which_buffer = 0;
    int steps = 0;

    bool run = false;

    while(!win.should_close())
    {
        win.poll();

        auto buffer_size = rtex[which_buffer].size<2>();

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

        glFinish();

        rtex[which_buffer].acquire(clctx.cqueue);

        bool step = false;

            ImGui::Begin("Test Window", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

            if(ImGui::Button("Step"))
                step = true;

            ImGui::Checkbox("Run", &run);

            ImGui::Text("Time: %f\n", time_elapsed_s);

            ImGui::End();

        if(run)
            step = true;

        cl::args render;
        render.push_back(bssnok_datas[which_data]);
        render.push_back(scale);
        render.push_back(clsize);
        render.push_back(intermediate);
        render.push_back(rtex[which_buffer]);
        render.push_back(time_elapsed_s);

        clctx.cqueue.exec("render", render, {size.x(), size.y()}, {16, 16});

        if(step)
        {
            float timestep = 0.01;

            if(steps < 10)
                timestep = 0.001;

            steps++;

            cl::args a1;
            a1.push_back(bssnok_datas[which_data]);
            a1.push_back(bssnok_datas[(which_data + 1) % 2]);
            a1.push_back(scale);
            a1.push_back(clsize);
            a1.push_back(intermediate);
            a1.push_back(timestep);
            a1.push_back(time_elapsed_s);

            clctx.cqueue.exec("evolve", a1, {size.x(), size.y(), size.z()}, {128, 1, 1});

            which_data = (which_data + 1) % 2;

            {
                cl::args cleaner;
                cleaner.push_back(bssnok_datas[which_data]);
                cleaner.push_back(intermediate);
                cleaner.push_back(scale);
                cleaner.push_back(clsize);

                clctx.cqueue.exec("clean_data", cleaner, {size.x(), size.y(), size.z()}, {128, 1, 1});
            }

            cl::args fl3;
            fl3.push_back(bssnok_datas[which_data]);
            fl3.push_back(scale);
            fl3.push_back(clsize);
            fl3.push_back(intermediate);

            clctx.cqueue.exec("calculate_intermediate_data", fl3, {size.x(), size.y(), size.z()}, {128, 1, 1});

            cl::args constraints;
            constraints.push_back(bssnok_datas[0]);
            constraints.push_back(scale);
            constraints.push_back(clsize);

            clctx.cqueue.exec("enforce_algebraic_constraints", constraints, {size.x(), size.y(), size.z()}, {128, 1, 1});


            time_elapsed_s += timestep;
        }

        clctx.cqueue.flush();

        rtex[which_buffer].unacquire(clctx.cqueue);

        which_buffer = (which_buffer + 1) % 2;

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

            lst->AddImage((void*)rtex[which_buffer].texture_id, tl, br, ImVec2(0, 0), ImVec2(1.f, 1.f));
        }

        win.display();
    }
}
