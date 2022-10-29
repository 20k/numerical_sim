#ifndef DIFFERENTIATOR_HPP_INCLUDED
#define DIFFERENTIATOR_HPP_INCLUDED

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

#endif // DIFFERENTIATOR_HPP_INCLUDED
