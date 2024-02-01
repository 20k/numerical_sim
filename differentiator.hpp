#ifndef DIFFERENTIATOR_HPP_INCLUDED
#define DIFFERENTIATOR_HPP_INCLUDED

#include <vec/value.hpp>
#include <vec/tensor.hpp>

struct differentiator
{
    std::optional<std::array<std::string, 3>> position_override;
    std::vector<std::string> ignored_variables;
    bool uses_linear = false;
    bool use_precise_differentiation = true;
    bool always_directional_derivatives = false;
    bool is_derivative_free = false;
    bool dynamic_drop = false;
    int order = 2;
    std::optional<vec3i> fixed_dim;
};

struct equation_context;

value diff1(differentiator& ctx, const value& in, int idx);
value diff2(equation_context& ctx, const value& in, int idx, int idy, const value& first_x, const value& first_y);
value upwind(differentiator& ctx, const value& prefix, const value& in, int idx);

value diffnth(differentiator& ctx, const value& in, int idx, int nth, const value& scale);
value diffnth(differentiator& ctx, const value& in, int idx, const value_i& nth, const value& scale);

value_i get_maximum_differentiation_derivative(const value_i& order);

#endif // DIFFERENTIATOR_HPP_INCLUDED
