#ifndef TENSOR_ALGEBRA_HPP_INCLUDED
#define TENSOR_ALGEBRA_HPP_INCLUDED

#include <vec/tensor.hpp>
#include <geodesic/dual_value.hpp>
#include <assert.h>

struct differentiator
{
    virtual value diff1(const value& in, int idx){assert(false); return value{0};};
    virtual value diff2(const value& in, int idx, int idy, const value& dx, const value& dy){assert(false); return value{0};};
};

inline
value diff1(differentiator& ctx, const value& in, int idx)
{
    return ctx.diff1(in, idx);
}

inline
value diff2(differentiator& ctx, const value& in, int idx, int idy, const value& dx, const value& dy)
{
    return ctx.diff2(in, idx, idy, dx, dy);
}

///https://arxiv.org/pdf/gr-qc/9810065.pdf
template<typename T, int N>
inline
T trace(const tensor<T, N, N>& mT, const inverse_metric<T, N, N>& inverse)
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

template<typename T, int N>
inline
tensor<T, N, N> trace_free(const tensor<T, N, N>& mT, const metric<T, N, N>& met, const inverse_metric<T, N, N>& inverse)
{
    tensor<T, N, N> TF;
    T t = trace(mT, inverse);

    for(int i=0; i < N; i++)
    {
        for(int j=0; j < N; j++)
        {
            TF.idx(i, j) = mT.idx(i, j) - (1/3.f) * met.idx(i, j) * t;
        }
    }

    return TF;
}

///B^i * Di whatever
inline
value upwind_differentiate(differentiator& ctx, const value& prefix, const value& in, int idx, bool pin = true)
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

    return prefix * ctx.diff1(in, idx);

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

inline
tensor<value, 3> tensor_upwind(differentiator& ctx, const tensor<value, 3>& prefix, const value& in)
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
T lie_derivative(differentiator& ctx, const tensor<T, N>& gB, const T& variable)
{
    ///https://en.wikipedia.org/wiki/Lie_derivative#Coordinate_expressions
    return sum(tensor_upwind(ctx, gB, variable));
}

template<typename T, int N, typename S>
inline
tensor<T, N, N> lie_derivative_weight(differentiator& ctx, const tensor<T, N>& B, const S& mT)
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

inline
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


template<typename T, int N>
inline
tensor<T, N, N, N> christoffel_symbols_2(differentiator& ctx, const metric<T, N, N>& met, const inverse_metric<T, N, N>& inverse)
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
tensor<T, N, N, N> christoffel_symbols_1(differentiator& ctx, const metric<T, N, N>& met)
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

///https://en.wikipedia.org/wiki/Covariant_derivative#Covariant_derivative_by_field_type
///for the tensor DcDa, this returns idx(a, c)
template<typename T, int N>
inline
tensor<T, N, N> covariant_derivative_low_vec(differentiator& ctx, const tensor<T, N>& v_in, const metric<T, N, N>& met, const inverse_metric<T, N, N>& inverse)
{
    auto christoff = christoffel_symbols_2(ctx, met, inverse);

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
tensor<T, N, N> double_covariant_derivative(differentiator& ctx, const T& in, const tensor<T, N>& first_derivatives,
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

template<typename T, int N>
inline
tensor<T, N, N> raise_both(const tensor<T, N, N>& mT, const inverse_metric<T, N, N>& inverse)
{
    return raise_index(raise_index(mT, inverse, 0), inverse, 1);
}

template<typename T, int N>
inline
tensor<T, N, N> lower_both(const tensor<T, N, N>& mT, const metric<T, N, N>& met)
{
    return lower_index(lower_index(mT, met, 0), met, 1);
}


#endif // TENSOR_ALGEBRA_HPP_INCLUDED
