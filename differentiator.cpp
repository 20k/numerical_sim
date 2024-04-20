#include "differentiator.hpp"
#include "bitflags.cl"
#include "equation_context.hpp"
#include "util.hpp"

template<int elements = 5>
struct differentiation_context
{
    std::array<value, elements> vars;

    differentiation_context(differentiator& ctx, const value& in, int idx, bool linear_interpolation = false)
    {
        std::array<int, elements> offx = {};
        std::array<int, elements> offy = {};
        std::array<int, elements> offz = {};

        for(int i=0; i < elements; i++)
        {
            int offset = i - (elements - 1)/2;

            if(idx == 0)
                offx[i] = offset;
            if(idx == 1)
                offy[i] = offset;
            if(idx == 2)
                offz[i] = offset;
        }

        ///for each element, ie x-2, x-1, x, x+1, x+2
        for(int i=0; i < elements; i++)
        {
            ///assign to the original element, ie x
            vars[i] = in;

            ///then recurse, and substitute all variables "x" with "x-2"
            vars[i].recurse_arguments([&i, &offx, &offy, &offz](value& v)
            {
                if(v.type == dual_types::ops::BRACKET2 || v.type == dual_types::ops::BRACKET_LINEAR)
                {
                    auto get_substitution = [&i, &offx, &offy, &offz]<typename T>(const value& v, const T& tag)
                    {
                        assert(v.args.size() == 7);

                        value buf = v.args[0];

                        value_base<T> old_x = v.args[1].assert_as<T>();
                        value_base<T> old_y = v.args[2].assert_as<T>();
                        value_base<T> old_z = v.args[3].assert_as<T>();

                        value_i old_dx = v.args[4].assert_as<int>();
                        value_i old_dy = v.args[5].assert_as<int>();
                        value_i old_dz = v.args[6].assert_as<int>();

                        value_base<T> next_x = old_x + offx[i];
                        value_base<T> next_y = old_y + offy[i];
                        value_base<T> next_z = old_z + offz[i];

                        if(v.original_type == dual_types::name_type(float16()))
                        {
                            return dual_types::make_op<float16>(v.type, buf, next_x, next_y, next_z, old_dx, old_dy, old_dz).template reinterpret_as<value>();
                        }
                        else if(v.original_type == dual_types::name_type(float()))
                        {
                            return dual_types::make_op<float>(v.type, buf, next_x, next_y, next_z, old_dx, old_dy, old_dz);
                        }
                        else
                            assert(false);

                        return value();
                    };

                    value out;

                    if(v.type == dual_types::ops::BRACKET2)
                        out = get_substitution(v, int());
                    else if(v.type == dual_types::ops::BRACKET_LINEAR)
                        out = get_substitution(v, float());
                    else
                        assert(false);

                    v = out;
                }
            });
        }

        #define DETECT_INCORRECT_DIFFERENTIATION
        #ifdef DETECT_INCORRECT_DIFFERENTIATION
        std::array<value_v, 3> root_variables;

        if(linear_interpolation)
        {
            root_variables = {value("fx").as_generic(), value("fy").as_generic(), value("fz").as_generic()};
        }
        else
        {
            root_variables = {value_i("ix").as_generic(), value_i("iy").as_generic(), value_i("iz").as_generic()};
        }

        if(ctx.position_override.has_value())
        {
            auto val = ctx.position_override.value();

            root_variables = {val[0], val[1], val[2]};
        }

        std::vector<value> indexed_variables;

        in.recurse_arguments([&indexed_variables, linear_interpolation](const value& v)
        {
            if(v.type == dual_types::ops::BRACKET2 || v.type == dual_types::ops::BRACKET_LINEAR)
            {
                indexed_variables.push_back(v);
            }
        });

        std::vector<std::string> variables = in.get_all_variables();

        for(auto& v : variables)
        {
            if(v == "dim" || v == "dim.x" || v == "dim.y" || v == "dim.z" || v == "scale" || v == type_to_string(root_variables[0]) || v == type_to_string(root_variables[1]) || v == type_to_string(root_variables[2]) || v == "index")
                continue;

            bool skip = false;

            for(const auto& i : ctx.ignored_variables)
            {
                if(v == i)
                {
                    skip = true;
                    break;
                }
            }

            if(skip)
                continue;

            bool found = false;

            for(auto& o : indexed_variables)
            {
                if(v == type_to_string(o.args[0]))
                {
                    found = true;
                    break;
                }
            }

            if(v.starts_with("(float3)"))
                continue;

            if(!found)
            {
                std::cout << "Could not find " << v << std::endl;
                std::cout << "Root expression: " << type_to_string(in) << std::endl;

                assert(false);
            }
        }

        if(indexed_variables.size() == 0)
        {
            std::cout << "No variables found\n";

            std::cout << "WHAT? " << type_to_string(in) << std::endl;
        }

        assert(indexed_variables.size() > 0);
        #endif // DETECT_INCORRECT_DIFFERENTIATION
    }
};

///with 2nd order accuracy
value_i get_maximum_differentiation_derivative(const value_i& order)
{
    std::array<value_i, 6> distances = {
        (int)D_GTE_WIDTH_1,
        (int)D_GTE_WIDTH_2,
        (int)D_GTE_WIDTH_3,
        (int)D_GTE_WIDTH_4,
        (int)D_GTE_WIDTH_5,
        (int)D_GTE_WIDTH_6,
    };

    std::array<value_i, 6> values = {
        2,
        4,
        6,
        8,
        10,
        12,
    };

    value_i last = values[0];

    for(int i=0; i < (int)values.size(); i++)
    {
        last = if_v((order & distances[i]) > 0, values[i], last);
    }

    return last;
}

value diff1_interior(differentiator& ctx, const value& in, int idx, int order, int direction)
{
    value scale = "scale";

    if(direction != 0)
        assert(order == 1);

    if(order == 1)
    {
        differentiation_context<5> dctx(ctx, in, idx, ctx.uses_linear);
        std::array<value, 5> vars = dctx.vars;

        if(direction == 0)
            return (vars[3] - vars[1]) / (2 * scale);

        if(direction == 1)
            return (vars[3] - vars[2]) / scale;

        if(direction == -1)
            return -(vars[1] - vars[2]) / scale;
    }
    else if(order == 2)
    {
        differentiation_context dctx(ctx, in, idx, ctx.uses_linear);
        std::array<value, 5> vars = dctx.vars;

        value p1 = -vars[4] + vars[0];
        value p2 = 8 * (vars[3] - vars[1]);

        return (p1 + p2) / (12 * scale);

        //return (-vars[4] + 8 * vars[3] - 8 * vars[1] + vars[0]) / (12 * scale);
    }
    else if(order == 3)
    {
        differentiation_context<7> dctx(ctx, in, idx, ctx.uses_linear);
        std::array<value, 7> vars = dctx.vars;

        return (-(1/60.f) * vars[0] + (3/20.f) * vars[1] - (3/4.f) * vars[2] + 0 * vars[3] + (3/4.f) * vars[4] - (3/20.f) * vars[5] + (1/60.f) * vars[6]) / scale;
    }
    else if(order == 4)
    {
        differentiation_context<9> dctx(ctx, in, idx, ctx.uses_linear);
        std::array<value, 9> vars = dctx.vars;

        return ((1/280.f) * vars[0] - (4/105.f) * vars[1] + (1/5.f) * vars[2] - (4/5.f) * vars[3] + (4/5.f) * vars[5] - (1/5.f) * vars[6] + (4/105.f) * vars[7] - (1/280.f) * vars[8]) / scale;
    }

    assert(false);
    return 0;
}

value diffnth(differentiator& ctx, const value& in, int idx, int nth, const value& scale)
{
    ///1 with accuracy 2
    if(nth == 1)
    {
        assert(false);

        differentiation_context<3> dctx(ctx, in, idx, ctx.uses_linear);
        auto vars = dctx.vars;

        return (vars[2] - vars[0]) / (2 * scale);
    }

    ///2 with accuracy 2
    if(nth == 2)
    {
        differentiation_context<3> dctx(ctx, in, idx, ctx.uses_linear);
        auto vars = dctx.vars;

        value p1 = vars[0] + vars[2];
        value p2 = -2 * vars[1];

        return (p1 + p2) / pow(scale, 2);

        //return (vars[0] - 2 * vars[1] + vars[2]) / pow(scale, 2);
    }

    ///3 with accuracy 2
    if(nth == 3)
    {
        assert(false);

        differentiation_context<5> dctx(ctx, in, idx, ctx.uses_linear);
        auto vars = dctx.vars;

        return (-0.5f * vars[0] + vars[1] + -vars[3] + 0.5f * vars[4]) / pow(scale, 3);
    }

    ///4 with accuracy 2
    if(nth == 4)
    {
        differentiation_context<5> dctx(ctx, in, idx, ctx.uses_linear);
        auto vars = dctx.vars;

        value p1 = vars[0] + vars[4];
        value p2 = -4 * (vars[1] + vars[3]);
        value p3 = 6 * vars[2];

        return (p1 + p2 + p3) / pow(scale, 4);

        //return (vars[0] - 4 * vars[1] + 6 * vars[2] - 4 * vars[3] + vars[4]) / pow(scale, 4);
    }

    ///5 with accuracy 2
    if(nth == 5)
    {
        assert(false);

        differentiation_context<7> dctx(ctx, in, idx, ctx.uses_linear);
        auto vars = dctx.vars;

        return (-0.5f * vars[0] + 2 * vars[1] - (5.f/2.f) * vars[2] + 0 * vars[3] + (5.f/2.f) * vars[4] - 2 * vars[5] + 0.5f * vars[6]) / pow(scale, 5);
    }

    ///6 with accuracy 2
    if(nth == 6)
    {
        differentiation_context<7> dctx(ctx, in, idx, ctx.uses_linear);
        auto vars = dctx.vars;

        value p1 = vars[0] + vars[6];
        value p2 = -6 * (vars[1] + vars[5]);
        value p3 = 15 * (vars[2] + vars[4]);
        value p4 = -20 * vars[3];

        return (p1 + p2 + p3 + p4) / pow(scale, 6);

        //return (1 * vars[0] - 6 * vars[1] + 15 * vars[2] - 20 * vars[3] + 15 * vars[4] - 6 * vars[5] + 1 * vars[6]) / pow(scale, 6);
    }

    assert(false);
}

value diffnth(differentiator& ctx, const value& in, int idx, const value_i& nth, const value& scale)
{
    value last = 0;

    for(int i=2; i <= 6; i+=2)
    {
        last = if_v(nth == i, diffnth(ctx, in, idx, i, scale), last);
    }

    return last;
}

value_i satisfies_order(int order, value_i found_order)
{
    int which_check = D_FULL;

    if(order == 1)
    {
        which_check = D_GTE_WIDTH_1;
    }
    else if(order == 2)
    {
        which_check = D_GTE_WIDTH_2;
    }
    else if(order == 3)
    {
        which_check = D_GTE_WIDTH_3;
    }
    else if(order == 4)
    {
        which_check = D_GTE_WIDTH_4;
    }

    return (found_order & which_check) > 0;
}


value diff1(differentiator& ctx, const value& in, int idx)
{
    //ctx.use_precise_differentiation = false;

    assert(!ctx.is_derivative_free);

    if(!ctx.use_precise_differentiation)
    {
        assert(!ctx.always_directional_derivatives);

        return diff1_interior(ctx, in, idx, ctx.order, 0);
    }
    else
    {
        ///in my case, order corresponds to width


        //value_us d_low = (uint16_t)D_LOW;
        value_i d_only_px = (int)D_ONLY_PX;
        value_i d_only_py = (int)D_ONLY_PY;
        value_i d_only_pz = (int)D_ONLY_PZ;
        value_i d_both_px = (int)D_BOTH_PX;
        value_i d_both_py = (int)D_BOTH_PY;
        value_i d_both_pz = (int)D_BOTH_PZ;

        value_i directional_single = 0;
        value_i directional_both = 0;

        if(idx == 0)
        {
            directional_single = d_only_px;
            directional_both = d_both_px;
        }
        else if(idx == 1)
        {
            directional_single = d_only_py;
            directional_both = d_both_py;
        }
        else if(idx == 2)
        {
            directional_single = d_only_pz;
            directional_both = d_both_pz;
        }

        value_i order = "order";

        value_i is_high_order = satisfies_order(ctx.order, order);

        value_i is_forward = (order & directional_single) > 0;
        value_i is_bidi = (order & directional_both) > 0;

        value regular_d = diff1_interior(ctx, in, idx, ctx.order, 0);
        value low_d = diff1_interior(ctx, in, idx, 1, 0);

        value forward_d = diff1_interior(ctx, in, idx, 1, 1);
        value back_d = diff1_interior(ctx, in, idx, 1, -1);

        value selected_directional = dual_types::if_v(is_forward, forward_d, back_d);

        value selected_full = dual_types::if_v(is_bidi, low_d, selected_directional);

        //#define DYNAMIC_DROP
        #ifdef DYNAMIC_DROP
        if(ctx.dynamic_drop)
        {
            value X = "X[index]";

            #ifdef USE_W
            X = X*X;
            #endif

            is_high_order = if_v(X < 0.03f, value_i{0}, is_high_order);
        }
        #endif // DYNAMIC_DROP

        if(ctx.always_directional_derivatives)
            return selected_full;
        else
            return dual_types::if_v(is_high_order, regular_d, low_d);
    }
}

value diff2(equation_context& ctx, const value& in, int idx, int idy, const value& first_x, const value& first_y)
{
    #define SYMMETRIC_DERIVATIVES
    #ifdef SYMMETRIC_DERIVATIVES
    if(idx < idy)
    {
        ///we're using first_y, so alias unconditionally
        ctx.alias(diff1(ctx, in, idy), first_y);

        return diff1(ctx, first_y, idx);
    }
    else
    {
        ctx.alias(diff1(ctx, in, idx), first_x);

        return diff1(ctx, first_x, idy);
    }
    #endif // SYMMETRIC_DERIVATIVES

    return diff1(ctx, first_x, idy);
}

value upwind(differentiator& ctx, const value& prefix, const value& in, int idx)
{
    //#define USE_UPWIND_STENCILS
    #ifdef USE_UPWIND_STENCILS
    value_i is_high_order = satisfies_order(3, "order");
    value scale = "scale";

    differentiation_context<7> dctx(ctx, in, idx, ctx.uses_linear);
    auto vars = dctx.vars;

    value positive_stencil = -3 * vars[2] - 10 * vars[3] + 18 * vars[4] - 6 * vars[5] + vars[6];
    value negative_stencil = -vars[0] + 6 * vars[1] - 18 * vars[2] + 10 * vars[3] + 3 * vars[4];

    value high_order_upwind = prefix * if_v(prefix > 0, positive_stencil, negative_stencil) / (12 * scale);

    return if_v(is_high_order, high_order_upwind, prefix * diff1(ctx, in, idx));
    #else
    return prefix * diff1(ctx, in, idx);
    #endif
}
