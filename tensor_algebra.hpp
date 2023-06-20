#ifndef TENSOR_ALGEBRA_HPP_INCLUDED
#define TENSOR_ALGEBRA_HPP_INCLUDED

#include <vec/tensor.hpp>
#include <geodesic/dual_value.hpp>
#include <assert.h>
#include "equation_context.hpp"
#include "differentiator.hpp"

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

                christoff.idx(i, k, l) = 0.5f * sum;
            }
        }
    }

    return christoff;
}

///derivatives are pass in like dk met ij
template<typename T, int N>
inline
tensor<T, N, N, N> christoffel_symbols_2(const inverse_metric<T, N, N>& inverse, const tensor<value, 3, 3, 3>& derivatives)
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

                    local += derivatives.idx(l, m, k);
                    local += derivatives.idx(k, m, l);
                    local += -derivatives.idx(m, k, l);

                    sum += local * inverse.idx(i, m);
                }

                christoff.idx(i, k, l) = 0.5f * sum;
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

///https://en.wikipedia.org/wiki/Covariant_derivative#Covariant_derivative_by_field_type
///for the tensor DcDa, this returns idx(a, c)
template<typename T, int N>
inline
tensor<T, N, N> covariant_derivative_low_vec(differentiator& ctx, const tensor<T, N>& v_in, const tensor<T, N, N, N>& christoff2)
{
    tensor<T, N, N> lac;

    for(int a=0; a < N; a++)
    {
        for(int c=0; c < N; c++)
        {
            T sum = 0;

            for(int b=0; b < N; b++)
            {
                sum = sum + christoff2.idx(b, c, a) * v_in.idx(b);
            }

            lac.idx(a, c) = diff1(ctx, v_in.idx(a), c) - sum;
        }
    }

    return lac;
}

///https://en.wikipedia.org/wiki/Covariant_derivative#Covariant_derivative_by_field_type
///for the tensor DbDa, this returns idx(a, b)
template<typename T, int N>
inline
tensor<T, N, N> covariant_derivative_high_vec(differentiator& ctx, const tensor<T, N>& v_in, const tensor<T, N, N, N>& christoff2)
{
    tensor<T, N, N> lab;

    for(int a=0; a < N; a++)
    {
        for(int b=0; b < N; b++)
        {
            T sum = 0;

            for(int c=0; c < N; c++)
            {
                sum = sum + christoff2.idx(a, b, c) * v_in.idx(c);
            }

            lab.idx(a, b) = diff1(ctx, v_in.idx(a), b) + sum;
        }
    }

    return lab;
}

///https://en.wikipedia.org/wiki/Covariant_derivative#Covariant_derivative_by_field_type
///for the tensor DbDa, this returns idx(a, b)
///derivatives come in in the format derivatives[deriv, root]. Yes this is confusing
template<typename T, int N>
inline
tensor<T, N, N> covariant_derivative_high_vec(differentiator& ctx, const tensor<T, N>& v_in, const tensor<T, N, N>& derivatives, const tensor<T, N, N, N>& christoff2)
{
    tensor<T, N, N> lab;

    for(int a=0; a < N; a++)
    {
        for(int b=0; b < N; b++)
        {
            T sum = 0;

            for(int c=0; c < N; c++)
            {
                sum = sum + christoff2.idx(a, b, c) * v_in.idx(c);
            }

            lab.idx(a, b) = derivatives[b, a] + sum;
        }
    }

    return lab;
}

template<typename T, int N>
inline
tensor<T, N, N> double_covariant_derivative(differentiator& ctx, const T& in, const tensor<T, N>& first_derivatives,
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

template<typename T, int N>
inline
unit_metric<T, N, N> get_flat_metric()
{
    unit_metric<T, N, N> ret;

    for(int i=0; i < N; i++)
    {
        for(int j=0; j < N; j++)
        {
            ret.idx(i, j) = i == j ? 1 : 0;
        }
    }

    return ret;
}

template<typename T>
inline
tensor<T, 3, 3, 3> get_full_christoffel2(const value& X, const tensor<value, 3>& dX, const metric<T, 3, 3>& cY, const inverse_metric<T, 3, 3>& icY, tensor<T, 3, 3, 3>& christoff2)
{
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
                    sm += icY.idx(i, m) * dX.idx(m);
                }

                full_christoffel2.idx(i, j, k) = christoff2.idx(i, j, k) -
                                                 (1.f/(2.f * X)) * (kronecker_ik * dX.idx(j) + kronecker_ij * dX.idx(k) - cY.idx(j, k) * sm);
            }
        }
    }

    return full_christoffel2;
}

///all the tetrad/frame basis stuff needs a rework, its from when I didn't know what was going on
///and poorly ported from the raytracer
template<int N>
value dot_product(const vec<N, value>& u, const vec<N, value>& v, const metric<value, N, N>& met)
{
    tensor<value, N> as_tensor;

    for(int i=0; i < N; i++)
    {
        as_tensor.idx(i) = u[i];
    }

    auto lowered_as_tensor = lower_index(as_tensor, met, 0);

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

struct frame_basis
{
    vec<4, value> v1;
    vec<4, value> v2;
    vec<4, value> v3;
    vec<4, value> v4;
};

inline
frame_basis calculate_frame_basis(equation_context& ctx, const metric<value, 4, 4>& met)
{
    vec<4, value> i1 = {1.f, 0.f, 0.f, 0.f};
    vec<4, value> i2 = {0.f, 1.f, 0.f, 0.f};
    vec<4, value> i3 = {0.f, 0.f, 1.f, 0.f};
    vec<4, value> i4 = {0.f, 0.f, 0.f, 1.f};

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


///https://arxiv.org/pdf/1503.08455.pdf (8)
inline
metric<value, 4, 4> calculate_real_metric(const metric<value, 3, 3>& adm, const value& gA, const tensor<value, 3>& gB)
{
    tensor<value, 3> lower_gB = lower_index(gB, adm, 0);

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

template<typename T>
inline
void matrix_inverse(const T m[16], T invOut[16])
{
    T inv[16], det;

    inv[0] = m[5]  * m[10] * m[15] -
             m[5]  * m[11] * m[14] -
             m[9]  * m[6]  * m[15] +
             m[9]  * m[7]  * m[14] +
             m[13] * m[6]  * m[11] -
             m[13] * m[7]  * m[10];

    inv[4] = -m[4]  * m[10] * m[15] +
              m[4]  * m[11] * m[14] +
              m[8]  * m[6]  * m[15] -
              m[8]  * m[7]  * m[14] -
              m[12] * m[6]  * m[11] +
              m[12] * m[7]  * m[10];

    inv[8] = m[4]  * m[9] * m[15] -
             m[4]  * m[11] * m[13] -
             m[8]  * m[5] * m[15] +
             m[8]  * m[7] * m[13] +
             m[12] * m[5] * m[11] -
             m[12] * m[7] * m[9];

    inv[12] = -m[4]  * m[9] * m[14] +
               m[4]  * m[10] * m[13] +
               m[8]  * m[5] * m[14] -
               m[8]  * m[6] * m[13] -
               m[12] * m[5] * m[10] +
               m[12] * m[6] * m[9];

    inv[1] = -m[1]  * m[10] * m[15] +
              m[1]  * m[11] * m[14] +
              m[9]  * m[2] * m[15] -
              m[9]  * m[3] * m[14] -
              m[13] * m[2] * m[11] +
              m[13] * m[3] * m[10];

    inv[5] = m[0]  * m[10] * m[15] -
             m[0]  * m[11] * m[14] -
             m[8]  * m[2] * m[15] +
             m[8]  * m[3] * m[14] +
             m[12] * m[2] * m[11] -
             m[12] * m[3] * m[10];

    inv[9] = -m[0]  * m[9] * m[15] +
              m[0]  * m[11] * m[13] +
              m[8]  * m[1] * m[15] -
              m[8]  * m[3] * m[13] -
              m[12] * m[1] * m[11] +
              m[12] * m[3] * m[9];

    inv[13] = m[0]  * m[9] * m[14] -
              m[0]  * m[10] * m[13] -
              m[8]  * m[1] * m[14] +
              m[8]  * m[2] * m[13] +
              m[12] * m[1] * m[10] -
              m[12] * m[2] * m[9];

    inv[2] = m[1]  * m[6] * m[15] -
             m[1]  * m[7] * m[14] -
             m[5]  * m[2] * m[15] +
             m[5]  * m[3] * m[14] +
             m[13] * m[2] * m[7] -
             m[13] * m[3] * m[6];

    inv[6] = -m[0]  * m[6] * m[15] +
              m[0]  * m[7] * m[14] +
              m[4]  * m[2] * m[15] -
              m[4]  * m[3] * m[14] -
              m[12] * m[2] * m[7] +
              m[12] * m[3] * m[6];

    inv[10] = m[0]  * m[5] * m[15] -
              m[0]  * m[7] * m[13] -
              m[4]  * m[1] * m[15] +
              m[4]  * m[3] * m[13] +
              m[12] * m[1] * m[7] -
              m[12] * m[3] * m[5];

    inv[14] = -m[0]  * m[5] * m[14] +
               m[0]  * m[6] * m[13] +
               m[4]  * m[1] * m[14] -
               m[4]  * m[2] * m[13] -
               m[12] * m[1] * m[6] +
               m[12] * m[2] * m[5];

    inv[3] = -m[1] * m[6] * m[11] +
              m[1] * m[7] * m[10] +
              m[5] * m[2] * m[11] -
              m[5] * m[3] * m[10] -
              m[9] * m[2] * m[7] +
              m[9] * m[3] * m[6];

    inv[7] = m[0] * m[6] * m[11] -
             m[0] * m[7] * m[10] -
             m[4] * m[2] * m[11] +
             m[4] * m[3] * m[10] +
             m[8] * m[2] * m[7] -
             m[8] * m[3] * m[6];

    inv[11] = -m[0] * m[5] * m[11] +
               m[0] * m[7] * m[9] +
               m[4] * m[1] * m[11] -
               m[4] * m[3] * m[9] -
               m[8] * m[1] * m[7] +
               m[8] * m[3] * m[5];

    inv[15] = m[0] * m[5] * m[10] -
              m[0] * m[6] * m[9] -
              m[4] * m[1] * m[10] +
              m[4] * m[2] * m[9] +
              m[8] * m[1] * m[6] -
              m[8] * m[2] * m[5];

    det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

    det = T(1.0f) / det;

    for(int i = 0; i < 16; i++)
        invOut[i] = inv[i] * det;
}


struct tetrad
{
    std::array<vec<4, value>, 4> e;
};

struct inverse_tetrad
{
    std::array<vec<4, value>, 4> e;
};

inline
inverse_tetrad get_tetrad_inverse(const tetrad& in)
{
    /*float m[16] = {e0_hi.x, e0_hi.y, e0_hi.z, e0_hi.w,
                   e1_hi.x, e1_hi.y, e1_hi.z, e1_hi.w,
                   e2_hi.x, e2_hi.y, e2_hi.z, e2_hi.w,
                   e3_hi.x, e3_hi.y, e3_hi.z, e3_hi.w};*/

    value m[16] = {in.e[0].x(), in.e[1].x(), in.e[2].x(), in.e[3].x(),
                   in.e[0].y(), in.e[1].y(), in.e[2].y(), in.e[3].y(),
                   in.e[0].z(), in.e[1].z(), in.e[2].z(), in.e[3].z(),
                   in.e[0].w(), in.e[1].w(), in.e[2].w(), in.e[3].w()};

    value inv[16] = {0};

    matrix_inverse(m, inv);

    inverse_tetrad out;
    out.e[0] = {inv[0 * 4 + 0], inv[0 * 4 + 1], inv[0 * 4 + 2], inv[0 * 4 + 3]};
    out.e[1] = {inv[1 * 4 + 0], inv[1 * 4 + 1], inv[1 * 4 + 2], inv[1 * 4 + 3]};
    out.e[2] = {inv[2 * 4 + 0], inv[2 * 4 + 1], inv[2 * 4 + 2], inv[2 * 4 + 3]};
    out.e[3] = {inv[3 * 4 + 0], inv[3 * 4 + 1], inv[3 * 4 + 2], inv[3 * 4 + 3]};

    return out;
}

/// e upper i, lower mu, which must be inverse of tetrad to coordinate basis vectors
inline
vec<4, value> coordinate_to_tetrad_basis(const vec<4, value>& vec_up, const inverse_tetrad& e_lo)
{
    vec<4, value> ret;

    ret.x() = dot(e_lo.e[0], vec_up);
    ret.y() = dot(e_lo.e[1], vec_up);
    ret.z() = dot(e_lo.e[2], vec_up);
    ret.w() = dot(e_lo.e[3], vec_up);

    return ret;
}

///so. The hi tetrads are the one we get out of gram schmidt
///so this is lower i, upper mu, against a vec with upper i
inline
vec<4, value> tetrad_to_coordinate_basis(const vec<4, value>& vec_up, const tetrad& e_hi)
{
    return vec_up.x() * e_hi.e[0] + vec_up.y() * e_hi.e[1] + vec_up.z() * e_hi.e[2] + vec_up.w() * e_hi.e[3];
}

struct ortho_result
{
    vec<3, value> v1, v2, v3;
};

inline
vec<3, value> project(const vec<3, value>& u, const vec<3, value>& v)
{
    return (dot(u, v) / dot(u, u)) * u;
}

inline
ortho_result orthonormalise(const vec<3, value>& i1, const vec<3, value>& i2, const vec<3, value>& i3)
{
    vec<3, value> u1 = i1;
    vec<3, value> u2 = i2;
    vec<3, value> u3 = i3;

    u2 = u2 - project(u1, u2);

    u3 = u3 - project(u1, u3);
    u3 = u3 - project(u2, u3);

    struct ortho_result result;
    result.v1 = u1.norm();
    result.v2 = u2.norm();
    result.v3 = u3.norm();

    return result;
};


///https://arxiv.org/pdf/0904.4184.pdf 1.4.18
///forms the velocity of a timelike geodesic
inline
vec<4, value> get_timelike_vector(const vec<3, value>& cartesian_basis_speed, float time_direction,
                                  const tetrad& tet)
{
    value v = cartesian_basis_speed.length();
    value Y = 1 / sqrt(1 - v*v);

    value B = v;

    value psi = B * Y;

    vec<4, value> bT = time_direction * Y * tet.e[0];

    vec<3, value> dir = cartesian_basis_speed / max(v, 0.000001f);

    vec<4, value> bX = psi * dir.x() * tet.e[1];
    vec<4, value> bY = psi * dir.y() * tet.e[2];
    vec<4, value> bZ = psi * dir.z() * tet.e[3];

    return bT + bX + bY + bZ;
}

///todo: this is all just crap
template<typename T, int N>
inline
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
inline
value dot_metric(const tensor<value, N>& v1_upper, const tensor<value, N>& v2_upper, const metric<value, N, N>& met)
{
    return dot(v1_upper, lower_index(v2_upper, met, 0));
}


///https://scholarworks.rit.edu/cgi/viewcontent.cgi?article=11286&context=theses (3.33)
inline
tensor<value, 4> get_adm_hypersurface_normal_raised(const value& gA, const tensor<value, 3>& gB)
{
    return {1/gA, -gB.idx(0)/gA, -gB.idx(1)/gA, -gB.idx(2)/gA};
}

inline
tensor<value, 4> get_adm_hypersurface_normal_lowered(const value& gA)
{
    return {-gA, 0, 0, 0};
}

#endif // TENSOR_ALGEBRA_HPP_INCLUDED
