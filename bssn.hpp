#ifndef BSSN_HPP_INCLUDED
#define BSSN_HPP_INCLUDED

#include <geodesic/dual_value.hpp>
#include <vec/tensor.hpp>
#include "tensor_algebra.hpp"
#include "equation_context.hpp"

inline
value as_float3(const value& x, const value& y, const value& z)
{
    return dual_types::apply("(float3)", x, y, z);
}

inline
value bidx(const std::string& buf, bool interpolate, bool is_derivative)
{
    if(interpolate)
    {
        if(is_derivative)
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
        if(is_derivative)
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

    value X_impl;
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
    tensor<value, 3> dX_impl;
    tensor<value, 3> dX_calc;

    tensor<value, 3> bigGi;
    tensor<value, 3> derived_cGi; ///undifferentiated cGi. Poor naming
    tensor<value, 3> always_derived_cGi; ///always calculated from metric

    tensor<value, 3, 3, 3> christoff2;

    value get_X()
    {
        return X_impl;
    }

    tensor<value, 3> get_dX()
    {
        return dX_calc;
    }

    standard_arguments(equation_context& ctx)
    {
        bool interpolate = ctx.uses_linear;

        gA = bidx("gA", interpolate, false);

        gA = max(gA, 0.f);
        //gA = max(gA, 0.00001f);

        gB.idx(0) = bidx("gB0", interpolate, false);
        gB.idx(1) = bidx("gB1", interpolate, false);
        gB.idx(2) = bidx("gB2", interpolate, false);

        gBB.idx(0) = bidx("gBB0", interpolate, false);
        gBB.idx(1) = bidx("gBB1", interpolate, false);
        gBB.idx(2) = bidx("gBB2", interpolate, false);

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

                cY.idx(i, j) = bidx("cY" + std::to_string(index), interpolate, false);
            }
        }

        //cY.idx(2, 2) = (1 + cY.idx(1, 1) * cY.idx(0, 2) * cY.idx(0, 2) - 2 * cY.idx(0, 1) * cY.idx(1, 2) * cY.idx(0, 2) + cY.idx(0, 0) * cY.idx(1, 2) * cY.idx(1, 2)) / (cY.idx(0, 0) * cY.idx(1, 1) - cY.idx(0, 1) * cY.idx(0, 1));

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                int index = arg_table[i * 3 + j];

                cA.idx(i, j) = bidx("cA" + std::to_string(index), interpolate, false);
            }
        }

        inverse_metric<value, 3, 3> icY = cY.invert();

        //tensor<value, 3, 3> raised_cAij = raise_index(cA, cY, icY);

        //cA.idx(1, 1) = -(raised_cAij.idx(0, 0) + raised_cAij.idx(2, 2) + cA.idx(0, 1) * icY.idx(0, 1) + cA.idx(1, 2) * icY.idx(1, 2)) / (icY.idx(1, 1));

        X_impl = max(bidx("X", interpolate, false), 0);
        K = bidx("K", interpolate, false);

        //X = max(X, 0.0001f);

        gA_X = gA / max(X_impl, 0.001f);

        cGi.idx(0) = bidx("cGi0", interpolate, false);
        cGi.idx(1) = bidx("cGi1", interpolate, false);
        cGi.idx(2) = bidx("cGi2", interpolate, false);

        Yij = cY / max(get_X(), 0.001f);
        iYij = get_X() * icY;

        tensor<value, 3, 3> Aij = cA / max(get_X(), 0.001f);

        Kij = Aij + Yij.to_tensor() * (K / 3.f);

        momentum_constraint.idx(0) = bidx("momentum0", interpolate, false);
        momentum_constraint.idx(1) = bidx("momentum1", interpolate, false);
        momentum_constraint.idx(2) = bidx("momentum2", interpolate, false);

        for(int k=0; k < 3; k++)
        {
            for(int i=0; i < 3; i++)
            {
                for(int j=0; j < 3; j++)
                {
                    int symmetric_index = index_table[i][j];

                    int final_index = k + symmetric_index * 3;

                    dcYij.idx(k, i, j) = bidx("dcYij" + std::to_string(final_index), interpolate, true);
                }
            }
        }

        digA.idx(0) = bidx("digA0", interpolate, true);
        digA.idx(1) = bidx("digA1", interpolate, true);
        digA.idx(2) = bidx("digA2", interpolate, true);

        dX_impl.idx(0) = bidx("dX0", interpolate, true);
        dX_impl.idx(1) = bidx("dX1", interpolate, true);
        dX_impl.idx(2) = bidx("dX2", interpolate, true);

        for(int i=0; i < 3; i++)
        {
            dX_calc.idx(i) = diff1(ctx, X_impl, i);
        }

        ///derivative
        for(int i=0; i < 3; i++)
        {
            ///value
            for(int j=0; j < 3; j++)
            {
                int idx = i + j * 3;

                digB.idx(i, j)  = bidx("digB" + std::to_string(idx), interpolate, true);
            }
        }

        /*tensor<value, 3, 3, 3> lchristoff2 = christoffel_symbols_2(ctx, cY, icY);

        tensor<value, 3> cGi_G;

        for(int i=0; i < 3; i++)
        {
            value sum = 0;

            for(int j=0; j < 3; j++)
            {
                for(int k=0; k < 3; k++)
                {
                    sum += pinned_icY.idx(j, k) * lchristoff2.idx(i, j, k);
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

        always_derived_cGi = cGi_G;

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
        tensor<value, 3> bigGi_lower = lower_index(bigGi, cY, 0);

        tensor<value, 3, 3, 3> raw_christoff2 = christoffel_symbols_2(ctx, cY, icY);

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
        christoff2 = christoffel_symbols_2(ctx, cY, icY);
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

struct matter_interop
{
    virtual value               calculate_adm_S(equation_context& ctx, standard_arguments& bssn_args){assert(false); return value();};
    virtual value               calculate_adm_p(equation_context& ctx, standard_arguments& bssn_args){assert(false); return value();};
    virtual tensor<value, 3, 3> calculate_adm_X_Sij(equation_context& ctx, standard_arguments& bssn_args){assert(false); return tensor<value, 3, 3>();};
    virtual tensor<value, 3>    calculate_adm_Si(equation_context& ctx, standard_arguments& bssn_args){assert(false); return tensor<value, 3>();};
};

namespace bssn
{
    tensor<value, 3, 3> calculate_xgARij(equation_context& ctx, standard_arguments& args, const inverse_metric<value, 3, 3>& icY, const tensor<value, 3, 3, 3>& christoff1, const tensor<value, 3, 3, 3>& christoff2);

    void init(equation_context& ctx, const metric<value, 3, 3>& Yij, const tensor<value, 3, 3>& Aij, const value& gA);

    tensor<value, 3> calculate_momentum_constraint(matter_interop& interop, equation_context& ctx, bool use_matter);
    value calculate_hamiltonian_constraint(matter_interop& interop, equation_context& ctx, bool use_matter);

    void build_cY(equation_context& ctx);
    void build_cA(matter_interop& interop, equation_context& ctx, bool use_matter);
    void build_cGi(matter_interop& interop, equation_context& ctx, bool use_matter);
    void build_K(matter_interop& interop, equation_context& ctx, bool use_matter);
    void build_X(equation_context& ctx);
    void build_gA(equation_context& ctx);
    void build_gB(equation_context& ctx);
}

#endif // BSSN_HPP_INCLUDED
