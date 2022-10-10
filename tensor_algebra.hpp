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

#endif // TENSOR_ALGEBRA_HPP_INCLUDED
