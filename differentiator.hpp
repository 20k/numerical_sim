#ifndef DIFFERENTIATOR_HPP_INCLUDED
#define DIFFERENTIATOR_HPP_INCLUDED

#include <geodesic/dual_value.hpp>
#include <vec/tensor.hpp>

struct differentiator
{
    std::optional<std::array<std::string, 3>> position_override;
    //std::optional<value> scale_override;

    virtual value diff1(const value& in, int idx){assert(false); return value{0.f};};
    //virtual value diff1(const buffer<value, 3>& in, int idx, const v3i& where, const value& scale){assert(false); return value{0.f};};
    virtual value diff2(const value& in, int idx, int idy, const value& dx, const value& dy){assert(false); return value{0.f};};

    virtual ~differentiator(){}
};

inline
value diff1(differentiator& ctx, const value& in, int idx)
{
    return ctx.diff1(in, idx);
}

/*inline
value diff1(differentiator& ctx, const buffer<value, 3>& in, int idx, const v3i& where, const value& scale)
{
    return ctx.diff1(in, idx, where, scale);
}*/

inline
value diff2(differentiator& ctx, const value& in, int idx, int idy, const value& dx, const value& dy)
{
    return ctx.diff2(in, idx, idy, dx, dy);
}

#endif // DIFFERENTIATOR_HPP_INCLUDED
