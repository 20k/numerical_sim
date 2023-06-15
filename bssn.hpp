#ifndef BSSN_HPP_INCLUDED
#define BSSN_HPP_INCLUDED

#include <geodesic/dual_value.hpp>
#include <vec/tensor.hpp>
#include "tensor_algebra.hpp"
#include "equation_context.hpp"
#include <toolkit/opencl.hpp>
#include "single_source.hpp"

///must be 4 because of damping
#define BORDER_WIDTH 4

inline
value as_float3(const value& x, const value& y, const value& z)
{
    return dual_types::apply(value("(float3)"), x, y, z);
}

struct argument_pack
{
    std::array<std::string, 6> cY;
    std::array<std::string, 6> cA;
    std::array<std::string, 3> cGi;

    std::string K;
    std::string X;
    std::string gA;
    std::array<std::string, 3> gB;
    std::array<std::string, 3> gBB;

    std::array<std::string, 3> momentum;
    std::array<std::string, 3> digA;
    std::array<std::string, 9> digB;
    std::array<std::string, 18> dcYij;
    std::array<std::string, 3> dX;

    std::array<std::string, 3> pos;
    std::string dim;

    argument_pack()
    {
        for(int i=0; i < 6; i++)
            cY[i] = "cY" + std::to_string(i);

        for(int i=0; i < 6; i++)
            cA[i] = "cA" + std::to_string(i);

        for(int i=0; i < 3; i++)
            cGi[i] = "cGi" + std::to_string(i);

        K = "K";
        X = "X";
        gA = "gA";

        for(int i=0; i < 3; i++)
        {
            gB[i] = "gB" + std::to_string(i);
            gBB[i] = "gBB" + std::to_string(i);
            momentum[i] = "momentum" + std::to_string(i);
            digA[i] = "digA" + std::to_string(i);
            dX[i] = "dX" + std::to_string(i);
        }

        for(int i=0; i < 9; i++)
            digB[i] = "digB" + std::to_string(i);

        for(int i=0; i < 18; i++)
            dcYij[i] = "dcYij" + std::to_string(i);

        pos = {"ix", "iy", "iz"};
        dim = "dim";
    }
};

inline
value bidx(const std::string& buf, bool interpolate, bool is_derivative, const argument_pack& pack = argument_pack())
{
    if(interpolate)
    {
        if(is_derivative)
        {
            return dual_types::apply(value("buffer_read_linearh"), buf, as_float3("fx", "fy", "fz"), "dim");
        }
        else
        {
            return dual_types::apply(value("buffer_read_linear"), buf, as_float3("fx", "fy", "fz"), "dim");
        }
    }
    else
    {
        if(is_derivative)
        {
            return dual_types::apply(value("buffer_indexh"), buf, pack.pos[0], pack.pos[1], pack.pos[2], pack.dim);
        }
        else
        {
            return dual_types::apply(value("buffer_index"), buf, pack.pos[0], pack.pos[1], pack.pos[2], pack.dim);
        }
    }
}

struct base_bssn_args
{
    std::vector<single_source::impl::input> buffers;
};

struct base_utility_args
{
    std::vector<single_source::impl::input> buffers;
};

#define USE_W
//#define BETTERDAMP_DTCAIJ
#define DAMP_C
//#define USE_GBB
//#define DAMP_DTCAIJ

#define USE_HALF_INTERMEDIATE

struct standard_arguments
{
    value gA;
    tensor<value, 3> gB;
    tensor<value, 3> gBB;

    #ifndef DAMP_C
    unit_metric<value, 3, 3> cY;
    #else
    metric<value, 3, 3> cY;
    #endif

    tensor<value, 3, 3> cA;

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

    #ifndef USE_W
    value X_impl;
    tensor<value, 3> dX_impl;
    tensor<value, 3> dX_calc;
    #else
    value W_impl;
    tensor<value, 3> dW_impl;
    tensor<value, 3> dW_calc;
    #endif

    tensor<value, 3> bigGi;
    tensor<value, 3> derived_cGi; ///undifferentiated cGi. Poor naming
    tensor<value, 3> always_derived_cGi; ///always calculated from metric

    tensor<value, 3, 3, 3> christoff2;

    value get_X()
    {
        #ifndef USE_W
        return X_impl;
        #else
        return W_impl * W_impl;
        #endif
    }

    tensor<value, 3> get_dX()
    {
        #ifndef USE_W
        return dX_calc;
        #else
        return 2 * W_impl * dW_calc;
        #endif
    }

    standard_arguments(equation_context& ctx, const argument_pack& pack = argument_pack())
    {
        bool interpolate = ctx.uses_linear;

        gA = bidx(pack.gA, interpolate, false);

        gA = max(gA, 0.f);
        //gA = max(gA, 0.00001f);

        gB.idx(0) = bidx(pack.gB[0], interpolate, false);
        gB.idx(1) = bidx(pack.gB[1], interpolate, false);
        gB.idx(2) = bidx(pack.gB[2], interpolate, false);

        gBB.idx(0) = bidx(pack.gBB[0], interpolate, false);
        gBB.idx(1) = bidx(pack.gBB[1], interpolate, false);
        gBB.idx(2) = bidx(pack.gBB[2], interpolate, false);

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

                cY.idx(i, j) = bidx(pack.cY[index], interpolate, false);
            }
        }

        //cY.idx(2, 2) = (1 + cY.idx(1, 1) * cY.idx(0, 2) * cY.idx(0, 2) - 2 * cY.idx(0, 1) * cY.idx(1, 2) * cY.idx(0, 2) + cY.idx(0, 0) * cY.idx(1, 2) * cY.idx(1, 2)) / (cY.idx(0, 0) * cY.idx(1, 1) - cY.idx(0, 1) * cY.idx(0, 1));

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                int index = arg_table[i * 3 + j];

                cA.idx(i, j) = bidx(pack.cA[index], interpolate, false);
            }
        }

        inverse_metric<value, 3, 3> icY = cY.invert();

        //tensor<value, 3, 3> raised_cAij = raise_index(cA, cY, icY);

        //cA.idx(1, 1) = -(raised_cAij.idx(0, 0) + raised_cAij.idx(2, 2) + cA.idx(0, 1) * icY.idx(0, 1) + cA.idx(1, 2) * icY.idx(1, 2)) / (icY.idx(1, 1));

        #ifndef USE_W
        X_impl = max(bidx(pack.X, interpolate, false), 0);
        #else
        W_impl = bidx(pack.X, interpolate, false);
        #endif

        K = bidx(pack.K, interpolate, false);

        //X = max(X, 0.0001f);

        gA_X = gA / max(get_X(), 0.001f);

        cGi.idx(0) = bidx(pack.cGi[0], interpolate, false);
        cGi.idx(1) = bidx(pack.cGi[1], interpolate, false);
        cGi.idx(2) = bidx(pack.cGi[2], interpolate, false);

        Yij = cY / max(get_X(), 0.001f);
        iYij = get_X() * icY;

        tensor<value, 3, 3> Aij = cA / max(get_X(), 0.001f);

        Kij = Aij + Yij.to_tensor() * (K / 3.f);

        momentum_constraint.idx(0) = bidx(pack.momentum[0], interpolate, false);
        momentum_constraint.idx(1) = bidx(pack.momentum[1], interpolate, false);
        momentum_constraint.idx(2) = bidx(pack.momentum[2], interpolate, false);

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

        digA.idx(0) = bidx(pack.digA[0], interpolate, true);
        digA.idx(1) = bidx(pack.digA[1], interpolate, true);
        digA.idx(2) = bidx(pack.digA[2], interpolate, true);

        #ifndef USE_W
        dX_impl.idx(0) = bidx(pack.dX[0], interpolate, true);
        dX_impl.idx(1) = bidx(pack.dX[1], interpolate, true);
        dX_impl.idx(2) = bidx(pack.dX[2], interpolate, true);

        for(int i=0; i < 3; i++)
        {
            dX_calc.idx(i) = diff1(ctx, X_impl, i);
        }
        #else
        dW_impl.idx(0) = bidx(pack.dX[0], interpolate, true);
        dW_impl.idx(1) = bidx(pack.dX[1], interpolate, true);
        dW_impl.idx(2) = bidx(pack.dX[2], interpolate, true);

        for(int i=0; i < 3; i++)
        {
            dW_calc.idx(i) = diff1(ctx, W_impl, i);
        }
        #endif

        ///derivative
        for(int i=0; i < 3; i++)
        {
            ///value
            for(int j=0; j < 3; j++)
            {
                int idx = i + j * 3;

                digB.idx(i, j)  = bidx(pack.digB[idx], interpolate, true);
            }
        }

        #define CGIG_FROM_DERIVED_CHRISTOFFEL
        #ifdef CGIG_FROM_DERIVED_CHRISTOFFEL
        #ifdef CGIG_RECALC
        tensor<value, 3, 3, 3> lchristoff2 = christoffel_symbols_2(ctx, cY, icY);
        #else
        tensor<value, 3, 3, 3> lchristoff2 = christoffel_symbols_2(icY, dcYij);
        #endif // CGIG_RECALC

        tensor<value, 3> cGi_G;

        for(int i=0; i < 3; i++)
        {
            value sum = 0;

            for(int j=0; j < 3; j++)
            {
                for(int k=0; k < 3; k++)
                {
                    sum += icY.idx(j, k) * lchristoff2.idx(i, j, k);
                }
            }

            cGi_G.idx(i) = sum;
        }
        #endif

        ///best performing
        //#define CGIG_FROM_ICY
        #ifdef CGIG_FROM_ICY
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
        #endif

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

///matter interop needs to be moved to plugins
struct matter_interop
{
    virtual value               calculate_adm_S(equation_context& ctx, standard_arguments& bssn_args){return 0;};
    virtual value               calculate_adm_p(equation_context& ctx, standard_arguments& bssn_args){return 0;};
    virtual tensor<value, 3, 3> calculate_adm_X_Sij(equation_context& ctx, standard_arguments& bssn_args){return {0,0,0,0,0,0,0,0,0};};
    virtual tensor<value, 3>    calculate_adm_Si(equation_context& ctx, standard_arguments& bssn_args){return {0,0,0};};

    virtual ~matter_interop(){}
};

struct matter_meta_interop : matter_interop
{
    std::vector<matter_interop*> sub_interop;

    virtual value               calculate_adm_S(equation_context& ctx, standard_arguments& bssn_args) override;
    virtual value               calculate_adm_p(equation_context& ctx, standard_arguments& bssn_args) override;
    virtual tensor<value, 3, 3> calculate_adm_X_Sij(equation_context& ctx, standard_arguments& bssn_args) override;
    virtual tensor<value, 3>    calculate_adm_Si(equation_context& ctx, standard_arguments& bssn_args) override;
};

namespace bssn
{
    tensor<value, 3, 3> calculate_xgARij(equation_context& ctx, standard_arguments& args, const inverse_metric<value, 3, 3>& icY, const tensor<value, 3, 3, 3>& christoff1, const tensor<value, 3, 3, 3>& christoff2);

    void init(equation_context& ctx, const metric<value, 3, 3>& Yij, const tensor<value, 3, 3>& Aij, const value& gA);

    tensor<value, 3> calculate_momentum_constraint(matter_interop& interop, equation_context& ctx, bool use_matter);
    value calculate_hamiltonian_constraint(matter_interop& interop, equation_context& ctx, bool use_matter);

    void build_cY(cl::context& clctx, matter_interop& interop, bool use_matter, base_bssn_args& bssn_args, base_utility_args& utility_args);
    void build_cA(matter_interop& interop, equation_context& ctx, bool use_matter);
    void build_cGi(matter_interop& interop, equation_context& ctx, bool use_matter);
    void build_K(matter_interop& interop, equation_context& ctx, bool use_matter);
    void build_X(equation_context& ctx);
    void build_gA(equation_context& ctx);
    void build_gB(equation_context& ctx);
}

#endif // BSSN_HPP_INCLUDED
