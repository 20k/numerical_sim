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
};

struct intermediate_bssnok_data
{
    cl_float christoffel[3 * 6];
    cl_float digA[6];
    cl_float digB[3*3];
    cl_float phi;
    cl_float Yij[6];
};

#if 1
bssnok_data get_conditions(vec3f pos, vec3f centre, float scale)
{
    tensor<float, 3, 3> kronecker;

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

    auto interpolating_polynomial = [](float x)
    {
        ///https://www.wolframalpha.com/input/?i=InterpolatingPolynomial%5B%7B%7B%7B0%7D%2C+0%2C+0%2C+0%7D%2C+%7B%7B1%7D%2C+1%2C+0%2C+0%7D%7D%2C+%7Bx%7D%5D

        ///(1 + (-3 + 6 (-1 + x)) (-1 + x)) x^3

        return (1 + (-3 + 6 * (-1 + x)) * (-1 + x)) * x * x * x;
    };

    ///I could fix this by improving the dual library to allow for algebraic substitution
    float BL_conformal = 1;

    float r = (pos - centre).length() * scale;

    if(r < 0.01)
        r = 0.01;

    //value vr("r");

    std::vector<float> black_hole_r{0};
    std::vector<float> black_hole_m{1};

    ///3.57 https://scholarworks.rit.edu/cgi/viewcontent.cgi?article=11286&context=theses
    ///todo: not sure this is correctly done, check r - ri, and what coordinate r really is
    for(int i=0; i < (int)black_hole_r.size(); i++)
    {
        float Mi = black_hole_m[i];
        float ri = black_hole_r[i];

        BL_conformal = BL_conformal + Mi / (2 * fabs(r - ri));
    }

    auto f_r = [&](float r)
    {
        float r_max = 0.8;
        float r_min = 0.2;

        if(r <= r_min)
            return 0.f;

        if(r >= r_max)
            return 1.f;

        float r_scaled = (r - r_min) / (r_max - r_min);

        return interpolating_polynomial(r_scaled);
    };

    ///todo: Stuff
    ///https://scholarworks.rit.edu/cgi/viewcontent.cgi?article=11286&context=theses 5.17
    tensor<float, 3, 3> yij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            yij.idx(i, j) = pow(BL_conformal, 4) * kronecker.idx(i, j);

            if(i == j)
            {
                yij.idx(i, j) = f_r(r) * yij.idx(i, j) + (1 - f_r(r)) * 99999;
            }
            else
            {
                yij.idx(i, j) = f_r(r) * yij.idx(i, j);
            }
        }
    }
    ///https://arxiv.org/pdf/gr-qc/9810065.pdf, 11
    float Y = yij.det();

    ///https://scholarworks.rit.edu/cgi/viewcontent.cgi?article=11286&context=theses pre 3.65
    float conformal_factor = (1/12.f) * log(Y);

    tensor<float, 3, 3> cyij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            cyij.idx(i, j) = exp(-4 * conformal_factor) * yij.idx(i, j);
        }
    }

    ///determinant of cyij is 1
    ///
    {
        float cY = cyij.det();
        //float real_value = cY.get_constant();
        float real_value = cY;

        //std::cout << "REAL " << real_value << std::endl;
    }

    float real_conformal = conformal_factor;

    ///so, via 3.47, Kij = Aij + (1/3) Yij K
    ///via 3.55, Kij = 0 for the initial conditions
    ///therefore Aij = 0
    ///cAij = exp(-4 phi) Aij
    ///therefore cAij = 0?

    ///no, none of this is right, they're talking about two separate variables
    ///once we spatially vary the conformal factor, the metric is no longer globally flat, so there are
    ///christoffal symbols
    tensor<float, 3, 3> cAij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            cAij.idx(i, j) = 0;
        }
    }

    float X = exp(-4 * real_conformal);

    std::string v1 = "x";
    std::string v2 = "y";
    std::string v3 = "z";

    ///3.59 says the christoffel symbols are 0 in cartesian

    tensor<float, 3, 3, 3> christoff;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                christoff.idx(i, j, k) = 0;
            }
        }
    }

    tensor<float, 3, 3> inverse_cYij = cyij.invert();
    vec3f cGi = {0,0,0};

    ///https://arxiv.org/pdf/gr-qc/9810065.pdf (21)
    ///aka cy^jk cGijk

    for(int i=0; i < 3; i++)
    {
        float sum = 0;

        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                sum += inverse_cYij.idx(j, k) * christoff.idx(i, j, k);
            }
        }

        cGi[i] = sum;
    }

    for(int i=0; i < 3; i++)
    {
        if(cGi[i] != 0)
        {
            std::cout << cGi[i] << std::endl;
            std::cout << "y " << Y << std::endl;
        }

        assert(cGi[i] == 0);
    }

    /*auto iv_cYij = cyij.invert();

    tensor<float, 4> cGi;

    for(int i=0; i < 4; i++)
    {
        cGi = iv_cYij.differentiate()
    }*/

    ///Kij = Aij + (1/3) yij K, where K is trace
    ///in the initial data, Kij = 0
    ///Which means K = 0
    ///which means that Aij.. is 0?

    bssnok_data ret;

    ret.cA0 = cAij.idx(0, 0);
    ret.cA1 = cAij.idx(0, 1);
    ret.cA2 = cAij.idx(0, 2);
    ret.cA3 = cAij.idx(1, 1);
    ret.cA4 = cAij.idx(1, 2);
    ret.cA5 = cAij.idx(2, 2);

    ret.cY0 = cyij.idx(0, 0);
    ret.cY1 = cyij.idx(0, 1);
    ret.cY2 = cyij.idx(0, 2);
    ret.cY3 = cyij.idx(1, 1);
    ret.cY4 = cyij.idx(1, 2);
    ret.cY5 = cyij.idx(2, 2);

    ret.cGi0 = cGi[0];
    ret.cGi1 = cGi[1];
    ret.cGi2 = cGi[2];

    ///https://scholarworks.rit.edu/cgi/viewcontent.cgi?article=11286&context=theses
    ///5.17: Kij is processed with the weighting function, which strongly implies that its not just fine to ditch it

    ret.K = 0;
    ret.X = X;

    ///https://arxiv.org/pdf/1404.6523.pdf section A, initial data
    ret.gA = 1/BL_conformal;
    ret.gB0 = 1/BL_conformal;
    ret.gB1 = 1/BL_conformal;
    ret.gB2 = 1/BL_conformal;

    return ret;
}
#endif // 0

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
T gpu_trace(const tensor<T, N, N>& mT, const tensor<T, N, N>& metric)
{
    tensor<T, N, N> inverse = metric.invert();

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

inline
std::vector<std::pair<std::string, std::string>>
get_initial_conditions_eqs(vec3f centre, float scale)
{
    vec<3, value> pos;

    pos[0].make_value("x");
    pos[1].make_value("y");
    pos[2].make_value("z");

    //#define DEBUG
    #ifdef DEBUG
    pos[0].make_value("20");
    pos[1].make_value("125");
    pos[2].make_value("125");
    #endif // DEBUG

    std::array<std::string, 3> variables = {"x", "y", "z"};

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

    vec<3, value> vcentre = {centre.x(), centre.y(), centre.z()};

    value r = (pos - vcentre).length() * scale;

    //std::cout << "REQ " << type_to_string(r) << std::endl;

    r = max(r, 0.01f);

   // std::cout << "FR " << r.substitute("x", 20).substitute("y", 125).substitute("z", 125).get_constant() << std::endl;

    std::vector<float> black_hole_r{0};
    std::vector<float> black_hole_m{1};

    ///3.57 https://scholarworks.rit.edu/cgi/viewcontent.cgi?article=11286&context=theses
    ///todo: not sure this is correctly done, check r - ri, and what coordinate r really is
    for(int i=0; i < (int)black_hole_r.size(); i++)
    {
        float Mi = black_hole_m[i];
        float ri = black_hole_r[i];

        BL_conformal = BL_conformal + Mi / (2 * fabs(r - ri));
    }

    ///ok so: I'm pretty sure this is correct
    tensor<value, 3, 3> yij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            yij.idx(i, j) = pow(BL_conformal, 4) * kronecker.idx(i, j);

            if(i == j)
            {
                yij.idx(i, j) = f_r(r) * yij.idx(i, j) + (1 - f_r(r)) * 99999;
            }
            else
            {
                yij.idx(i, j) = f_r(r) * yij.idx(i, j);
            }
        }
    }

    //std::cout << "YIJ " << yij.idx(0, 0).substitute("x", 20).substitute("y", 125).substitute("z", 125).get_constant() << std::endl;

    ///https://arxiv.org/pdf/gr-qc/9810065.pdf, 11
    value Y = yij.det();

    ///phi
    value conformal_factor = (1/12.f) * log(Y);
    value conformal_factor_concrete;
    conformal_factor_concrete.make_value("conformal_factor");

    tensor<value, 3, 3> cyij;

    ///checked, cyij is correct
    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            cyij.idx(i, j) = exp(-4 * conformal_factor) * yij.idx(i, j);
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

    tensor<value, 3> norm;
    norm.idx(0) = -gB0 / gA;
    norm.idx(1) = -gB1 / gA;
    norm.idx(2) = -gB2 / gA;

    //std::cout << "n0 " << norm.idx(0).substitute("x", 20).substitute("y", 125).substitute("z", 125).get_constant() << std::endl;
    //std::cout << "y0 " << yij.idx(0, 0).substitute("x", 20).substitute("y", 125).substitute("z", 125).get_constant() << std::endl;

    tensor<value, 3, 3> nearly_Kij = gpu_lie_derivative_weight_arbitrary(norm, yij, -1, variables);

    //std::cout << "ni0 " << nearly_Kij.idx(0, 0).substitute("x", 20).substitute("y", 125).substitute("z", 125).get_constant() << std::endl;

    tensor<value, 3, 3> Kij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            Kij.idx(i, j) = nearly_Kij.idx(i, j) / -2.f;

            Kij.idx(i, j) = f_r(r) * Kij.idx(i, j);
        }
    }

    value K = gpu_trace(Kij, yij);

    //std::cout << "K " << K.substitute("x", 20).substitute("y", 125).substitute("z", 125).get_constant() << std::endl;
    //std::cout << "Kij " << Kij.idx(0, 0).substitute("x", 20).substitute("y", 125).substitute("z", 125).get_constant() << std::endl;

    tensor<value, 3, 3> Aij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            Aij.idx(i, j) = Kij.idx(i, j) - (1.f/3.f) * yij.idx(i, j) * K;
        }
    }

    //std::cout << "AIJ0 " << (Aij.idx(0, 0).substitute("x", 20).substitute("y", 125).substitute("z", 125)).get_constant() << std::endl;
    //std::cout << "TFAIJ0 " << (gpu_trace(Aij, yij).substitute("x", 20).substitute("y", 125).substitute("z", 125).get_constant()) << std::endl;
    //assert(false);

    tensor<value, 3, 3> cAij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            cAij.idx(i, j) = exp(-4 * conformal_factor) * Aij.idx(i, j);
        }
    }


    ///https://arxiv.org/pdf/gr-qc/9810065.pdf (21)

    tensor<value, 3, 3> inverse_cyij = cyij.invert();

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

    value X = exp(-4 * conformal_factor_concrete);

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

    std::vector<std::pair<std::string, std::string>> equations;

    for(int i=0; i < 6; i++)
    {
        vec2i index = linear_indices[i];

        std::string y_name = "init_cY" + std::to_string(i);
        std::string a_name = "init_cA" + std::to_string(i);

        equations.push_back({y_name, type_to_string(cyij.idx(index.x(), index.y()))});
        equations.push_back({a_name, type_to_string(cAij.idx(index.x(), index.y()))});
    }

    equations.push_back({"init_cGi0", type_to_string(cGi.idx(0))});
    equations.push_back({"init_cGi1", type_to_string(cGi.idx(1))});
    equations.push_back({"init_cGi2", type_to_string(cGi.idx(2))});

    equations.push_back({"init_K", type_to_string(K)});
    equations.push_back({"init_X", type_to_string(X)});

    equations.push_back({"init_bl_conformal", type_to_string(BL_conformal)});
    equations.push_back({"init_conformal_factor", type_to_string(conformal_factor)});

    /*equations.push_back({"init_gA", type_to_string(gA)});
    equations.push_back({"init_gB0", type_to_string(gB0)});
    equations.push_back({"init_gB1", type_to_string(gB1)});
    equations.push_back({"init_gB2", type_to_string(gB2)});*/

    //equations.push_back({"init_det", type_to_string(cyij.det())});

    return equations;
}

///todo: I know for a fact that clang is too silly to optimise out the memory lookups
value hacky_differentiate(value in, int idx)
{
    assert(in.is_value());

    if(in.value_payload.value().starts_with("v."))
    {
        std::string nstr = in.value_payload.value();

        nstr.erase(nstr.begin());
        nstr.erase(nstr.begin());

        value ret;
        ret.make_value("nan");

        if(idx == 0)
            ret.make_value("DIFFX(" + nstr + ")");
        if(idx == 1)
            ret.make_value("DIFFY(" + nstr + ")");
        if(idx == 2)
            ret.make_value("DIFFZ(" + nstr + ")");

        return ret;
    }
    else if(in.value_payload.value().starts_with("ik."))
    {
        std::string nstr = in.value_payload.value();

        nstr.erase(nstr.begin());
        nstr.erase(nstr.begin());
        nstr.erase(nstr.begin());

        value ret;
        ret.make_value("nan");

        if(idx == 0)
            ret.make_value("INTERMEDIATE_DIFFX(" + nstr + ")");
        if(idx == 1)
            ret.make_value("INTERMEDIATE_DIFFY(" + nstr + ")");
        if(idx == 2)
            ret.make_value("INTERMEDIATE_DIFFZ(" + nstr + ")");

        return ret;
    }

    assert(false);
}

template<typename T, int N>
inline
tensor<T, N, N> gpu_lie_derivative_weight(const tensor<T, N>& B, const tensor<T, N, N>& mT)
{
    tensor<T, N, N> lie;

    for(int i=0; i < N; i++)
    {
        for(int j=0; j < N; j++)
        {
            T sum = 0;

            for(int k=0; k < N; k++)
            {
                sum = sum + B.idx(k) * hacky_differentiate(mT.idx(i, j), k);
                sum = sum + mT.idx(i, k) * hacky_differentiate(B.idx(k), j);
                sum = sum + mT.idx(k, j) * hacky_differentiate(B.idx(k), i);
                sum = sum - (2.f/3.f) * mT.idx(i, j) * hacky_differentiate(B.idx(k), k);
            }

            lie.idx(i, j) = sum;
        }
    }

    return lie;
}

///mT symmetric?
tensor<value, 3, 3> raise_index(const tensor<value, 3, 3>& mT, const tensor<value, 3, 3>& metric)
{
    tensor<value, 3, 3> inverse = metric.invert();

    tensor<value, 3, 3> ret;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value sum = 0;

            for(int k=0; k < 3; k++)
            {
                sum = sum + mT.idx(i, k) * inverse.idx(k, j);
            }

            ret.idx(i, j) = sum;
        }
    }

    return ret;
}


template<typename T, int N>
inline
tensor<T, N> gpu_covariant_derivative_scalar(const T& in)
{
    tensor<T, N> ret;

    for(int i=0; i < N; i++)
    {
        ret.idx(i) = hacky_differentiate(in.differentiate, i);
    }

    return ret;
}

template<typename T, typename U, int N>
inline
tensor<T, N, N> gpu_high_covariant_derivative_scalar(const T& in, const tensor<T, N, N>& metric)
{
    tensor<T, N, N> iv_metric = metric.invert();

    tensor<T, N> deriv_low = gpu_covariant_derivative_scalar(in);

    tensor<T, N> ret;

    for(int i=0; i < N; i++)
    {
        T sum = 0;

        for(int p=0; p < N; p++)
        {
            sum += iv_metric.idx(i, p) * deriv_low.idx(p);
        }

        ret.idx(i) = sum;
    }

    return ret;
}

template<typename T, int N>
inline
tensor<T, N, N, N> gpu_high_covariant_derivative_vec(const tensor<T, N>& in, const tensor<T, N, N>& metric)
{
    tensor<T, N, N> iv_metric = metric.invert();
}

///https://en.wikipedia.org/wiki/Covariant_derivative#Covariant_derivative_by_field_type
template<typename T, int N>
inline
tensor<T, N, N> gpu_covariant_derivative_low_vec(const tensor<T, N>& v_in, const tensor<T, N, N>& metric, const tensor<T, N, N, N>& christoff)
{
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

            lac.idx(a, c) = hacky_differentiate(v_in.idx(a), c) - sum;
        }
    }

    return lac;
}

template<typename T, int N>
inline
tensor<T, N, N> gpu_trace_free(const tensor<T, N, N>& mT, const tensor<T, N, N>& metric)
{
    tensor<T, N, N> inverse = metric.invert();

    tensor<T, N, N> TF;
    T t = gpu_trace(mT, metric);

    for(int i=0; i < N; i++)
    {
        for(int j=0; j < N; j++)
        {
            TF.idx(i, j) = mT.idx(i, j) - (1/3.f) * metric.idx(i, j) * t;
        }
    }

    return TF;
}

template<typename T, int N>
inline
tensor<T, N, N, N> gpu_christoffel_symbols_2(const tensor<T, N, N>& metric)
{
    tensor<T, N, N, N> christoff;
    tensor<T, N, N> inverted = metric.invert();

    for(int i=0; i < N; i++)
    {
        for(int k=0; k < N; k++)
        {
            for(int l=0; l < N; l++)
            {
                T sum = 0;

                for(int m=0; m < N; m++)
                {
                    sum = sum + inverted.idx(i, m) * hacky_differentiate(metric.idx(m, k), l);
                    sum = sum + inverted.idx(i, m) * hacky_differentiate(metric.idx(m, l), k);
                    sum = sum - inverted.idx(i, m) * hacky_differentiate(metric.idx(k, l), m);
                }

                christoff.idx(i, k, l) = 0.5 * sum;
            }
        }
    }

    return christoff;
}

template<typename T, int N>
inline
tensor<T, N, N> raise_both(const tensor<T, N, N>& mT, const tensor<T, N, N>& metric)
{
    tensor<T, N, N> inverse = metric.invert();

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

inline
std::vector<std::pair<std::string, std::string>>
build_eqs()
{
    int index_table[3][3] = {{0, 1, 2},
                             {1, 3, 4},
                             {2, 4, 5}};

    tensor<value, 3, 3> cY;

    cY.idx(0, 0).make_value("v.cY0"); cY.idx(0, 1).make_value("v.cY1"); cY.idx(0, 2).make_value("v.cY2");
    cY.idx(1, 0).make_value("v.cY1"); cY.idx(1, 1).make_value("v.cY3"); cY.idx(1, 2).make_value("v.cY4");
    cY.idx(2, 0).make_value("v.cY2"); cY.idx(2, 1).make_value("v.cY4"); cY.idx(2, 2).make_value("v.cY5");

    tensor<value, 3, 3> icY = cY.invert();

    tensor<value, 3, 3> cA;

    cA.idx(0, 0).make_value("v.cA0"); cA.idx(0, 1).make_value("v.cA1"); cA.idx(0, 2).make_value("v.cA2");
    cA.idx(1, 0).make_value("v.cA1"); cA.idx(1, 1).make_value("v.cA3"); cA.idx(1, 2).make_value("v.cA4");
    cA.idx(2, 0).make_value("v.cA2"); cA.idx(2, 1).make_value("v.cA4"); cA.idx(2, 2).make_value("v.cA5");

    ///the christoffel symbol
    tensor<value, 3> cGi;
    cGi.idx(0).make_value("v.cGi0");
    cGi.idx(1).make_value("v.cGi1");
    cGi.idx(2).make_value("v.cGi2");

    tensor<value, 3> digA;
    digA.idx(0).make_value("ik.digA[0]");
    digA.idx(1).make_value("ik.digA[1]");
    digA.idx(2).make_value("ik.digA[2]");

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
    gA.make_value("v.gA");

    tensor<value, 3> gB;
    gB.idx(0).make_value("v.gB0");
    gB.idx(1).make_value("v.gB1");
    gB.idx(2).make_value("v.gB2");

    value X;
    X.make_value("v.X");

    value K;
    K.make_value("v.K");

    value phi;
    phi.make_value("ik.phi");

    tensor<value, 3, 3, 3> cGijk;

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
    }

    tensor<value, 3, 3, 3, 3> dcGijk;
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
    }

    tensor<value, 3, 3> lie_cYij = gpu_lie_derivative_weight(gB, cY);

    tensor<value, 3, 3> dtcYij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            dtcYij.idx(i, j) = -2 * gA + lie_cYij.idx(i, j);
        }
    }

    value dtX = 0;

    {
        value s1 = 0;
        value s2 = 0;

        for(int i=0; i < 3; i++)
        {
            s1 = s1 + hacky_differentiate(gB.idx(i), i);
            s2 = s2 + gB.idx(i) * hacky_differentiate(X, i);
        }

        dtX = (2.f/3.f) * X * (gA * K - s1) + s2;
    }

    tensor<value, 3, 3> Rij;

    ///https://en.wikipedia.org/wiki/Ricci_curvature#Definition_via_local_coordinates_on_a_smooth_manifold
    for(int i=0; i < 3; i++)
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

            Rij.idx(i, j) = sum - sum2 + sum3;
        }
    }

    ///recover Yij from X and cYij
    ///https://arxiv.org/pdf/gr-qc/0511048.pdf
    ///https://arxiv.org/pdf/gr-qc/9810065.pdf
    ///X = exp(-4 phi)
    ///consider trying to eliminate
    tensor<value, 3, 3> Yij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            int linear_idx = index_table[i][j];

            std::string name = "ik.Yij[" + std::to_string(linear_idx) + "]";

            Yij.idx(i, j).make_value(name);
        }
    }

    tensor<value, 3, 3> iYij = Yij.invert();

    tensor<value, 3, 3, 3> christoff_Yij = gpu_christoffel_symbols_2(Yij);

    ///Aki G^kj
    tensor<value, 3, 3> mixed_cAij = raise_index(cA, cY);

    tensor<value, 3, 3> dtcAij;

    tensor<value, 3, 3> with_trace;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value trace_free_interior_1 = -gpu_covariant_derivative_low_vec(digA, Yij, christoff_Yij).idx(j, i);
            value trace_free_interior_2 = gA * Rij.idx(i, j);

            with_trace.idx(i, j) = trace_free_interior_1 + trace_free_interior_2;
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

            value trace_free_part = gpu_trace_free(with_trace, Yij).idx(i, j);

            value p1 = X * trace_free_part;

            value p2 = gA * (K * cA.idx(i, j) - 2 * sum);

            value p3 = gpu_lie_derivative_weight(gB, cA).idx(i, j);

            dtcAij.idx(i, j) = p1 + p2 + p3;
        }
    }

    //tensor<value, 3, 3> icAij = cA.invert();

    tensor<value, 3, 3> icAij = raise_both(cA, cY);

    value dtK = 0;

    {
        value sum1 = 0;

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                sum1 = sum1 + iYij.idx(i, j) * gpu_covariant_derivative_low_vec(digA, Yij, christoff_Yij).idx(i, j);
            }
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
            sum3 = sum3 + gB.idx(i) * hacky_differentiate(K, i);
        }

        dtK = -sum1 + sum2 + sum3;
    }

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
                s1 = s1 + icY.idx(j, k) * hacky_differentiate(digB.idx(k, i), j);
            }

            value s2 = 0;

            for(int k=0; k < 3; k++)
            {
                s2 = s2 + (1.f/3.f) * icY.idx(i, j) * hacky_differentiate(digB.idx(k, k), j);
            }

            value s3 = gB.idx(j) * hacky_differentiate(cGi.idx(i), j);

            value s4 = -cGi.idx(j) * hacky_differentiate(gB.idx(i), j);

            value s5 = (2.f/3.f) * cGi.idx(i) * hacky_differentiate(gB.idx(j), j);

            value s6 = -2 * icAij.idx(i, j) * hacky_differentiate(gA, j);

            value s7 = 0;

            {
                value s8 = 0;

                for(int k=0; k < 3; k++)
                {
                    s8 = s8 + cGijk.idx(i, j, k) * icAij.idx(j, k);
                }

                value s9 = 6 * icAij.idx(i, j) * hacky_differentiate(phi, j);

                value s10 = -(2.f/3.f) * icY.idx(i, j) * hacky_differentiate(K, j);

                s7 = 2 * gA * (s8 + s9 + s10);
            }


            sum = sum + s1 + s2 + s3 + s4 + s5 + s6 + s7;
        }

        dtcGi.idx(i) = sum;
    }

    value dtgA = -2 * gA * K;

    for(int i=0; i < 3; i++)
    {
        dtgA = dtgA + gB.idx(i) * hacky_differentiate(gA, i);
    }

    tensor<value, 3> dtgB;

    ///https://arxiv.org/pdf/1404.6523.pdf (4)
    for(int i=0; i < 3; i++)
    {
        float N = 1.375;

        dtgB.idx(i) = (3.f/4.f) * cGi.idx(i) - N * gB.idx(i);
    }

    value scalar_curvature = 0;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            scalar_curvature = scalar_curvature + iYij.idx(i, j) * Rij.idx(i, j);
        }
    }

    std::vector<std::pair<std::string, std::string>> equations;

    vec2i linear_indices[6] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};

    for(int i=0; i < 6; i++)
    {
        std::string name = "dtcYij" + std::to_string(i);

        vec2i idx = linear_indices[i];

        equations.push_back({name, type_to_string(dtcYij.idx(idx.x(), idx.y()))});
    }

    equations.push_back({"dtX", type_to_string(dtX)});

    for(int i=0; i < 6; i++)
    {
        std::string name = "dtcAij" + std::to_string(i);

        vec2i idx = linear_indices[i];

        equations.push_back({name, type_to_string(dtcAij.idx(idx.x(), idx.y()))});
    }

    equations.push_back({"dtK", type_to_string(dtK)});

    for(int i=0; i < 3; i++)
    {
        std::string name = "dtcGi" + std::to_string(i);

        equations.push_back({name, type_to_string(dtcGi.idx(i))});
    }

    equations.push_back({"dtgA", type_to_string(dtgA)});

    for(int i=0; i < 3; i++)
    {
        std::string name = "dtgB" + std::to_string(i);

        equations.push_back({name, type_to_string(dtgB.idx(i))});
    }

    equations.push_back({"scalar_curvature", type_to_string(scalar_curvature)});

    return equations;
}

int main()
{
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

    std::vector<std::pair<std::string, std::string>> equations = build_eqs();

    vec3i size = {250, 250, 250};
    float c_at_max = 8;
    float scale = c_at_max / size.largest_elem();
    vec3f centre = {size.x()/2, size.y()/2, size.z()/2};

    std::vector<std::pair<std::string, std::string>> equations2 = get_initial_conditions_eqs(centre, scale);

    for(auto& i : equations)
    {
        std::string str = "-D" + i.first + "=" + i.second + " ";

        argument_string += str;
    }

    for(auto& i : equations2)
    {
        std::string str = "-D" + i.first + "=" + i.second + " ";

        argument_string += str;
    }

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

    cl::args init;
    init.push_back(bssnok_datas[0]);
    init.push_back(scale);
    init.push_back(clsize);

    clctx.cqueue.exec("calculate_initial_conditions", init, {size.x(), size.y(), size.z()}, {8, 8, 1});

    cl::args fl2;
    fl2.push_back(bssnok_datas[0]);
    fl2.push_back(scale);
    fl2.push_back(clsize);
    fl2.push_back(intermediate);

    clctx.cqueue.exec("calculate_intermediate_data", fl2, {size.x(), size.y(), size.z()}, {8, 8, 1});

    int which_buffer = 0;

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

            ImGui::End();

        cl::args render;
        render.push_back(bssnok_datas[which_data]);
        render.push_back(scale);
        render.push_back(clsize);
        render.push_back(intermediate);
        render.push_back(rtex[which_buffer]);

        clctx.cqueue.exec("render", render, {size.x(), size.y()}, {16, 16});

        if(step)
        {
            cl::args fl;
            fl.push_back(bssnok_datas[which_data]);
            fl.push_back(scale);
            fl.push_back(clsize);
            fl.push_back(intermediate);

            clctx.cqueue.exec("calculate_intermediate_data", fl, {size.x(), size.y(), size.z()}, {8, 8, 1});

            float timestep = 0.00001;

            cl::args a1;
            a1.push_back(bssnok_datas[which_data]);
            a1.push_back(bssnok_datas[(which_data + 1) % 2]);
            a1.push_back(scale);
            a1.push_back(clsize);
            a1.push_back(intermediate);
            a1.push_back(timestep);

            clctx.cqueue.exec("evolve", a1, {size.x(), size.y(), size.z()}, {8, 8, 1});

            which_data = (which_data + 1) % 2;
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
