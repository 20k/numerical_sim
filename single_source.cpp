#include "single_source.hpp"
#include <ranges>

struct codegen_override : dual_types::codegen
{
    std::optional<std::string> bracket_linear(const value_base<std::monostate>& op) const override
    {
        assert(op.args.size() == 7);

        if(op.original_type == dual_types::name_type(float()))
        {
            return type_to_string(dual_types::apply<float>("buffer_read_linear2", op.args[0],
                                                            op.args[1], op.args[2], op.args[3],
                                                            op.args[4], op.args[5], op.args[6]), *this);
        }
        else
        {
            return type_to_string(dual_types::apply<float>("buffer_read_linearh2", op.args[0],
                                                            op.args[1], op.args[2], op.args[3],
                                                            op.args[4], op.args[5], op.args[6]), *this);
        }
    }
};

std::string single_source::impl::generate_kernel_string(kernel_context& kctx, equation_context& ctx, const std::string& kernel_name, bool& any_uses_half)
{
    for(auto& in : kctx.inputs.args)
    {
        if(in.type == "half")
            any_uses_half = true;
    }

    std::string structs;
    std::map<std::string, bool> generated;
    std::vector<std::pair<std::string, type_storage>> pending_structs;

    for(input& arg : kctx.inputs.args)
    {
        if(arg.is_struct)
        {
            if(generated[arg.type])
                continue;

            generated[arg.type] = true;

            pending_structs.push_back({arg.type, arg.defines_structs.at(0)});
        }
    }

    for(auto& [type, store] : pending_structs)
    {
        structs += make_struct(type, store) + "\n";
    }

    std::string functions;

    for(auto& [name, func_kctx, func_ectx] : ctx.functions)
    {
        functions += generate_kernel_string(func_kctx, func_ectx, name, any_uses_half);
    }

    std::string base;

    if(!kctx.is_func && any_uses_half)
    {
        base += "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n\n";
    }

    base += structs;
    base += functions;

    if(!kctx.is_func)
        base += "__kernel void " + kernel_name + "(";
    else
        base += kctx.ret.args.at(0).type + " " + kernel_name + "(";

    for(int i=0; i < (int)kctx.inputs.args.size(); i++)
    {
        base += kctx.inputs.args[i].format();

        if(i != (int)kctx.inputs.args.size() - 1)
            base += ",";
    }

    base += ")\n{\n";

    ctx.strip_unused();
    ctx.substitute_aliases();

    std::vector<std::vector<value>> blocks;
    blocks.emplace_back();

    for(auto& v : ctx.sequenced)
    {
        bool is_bad = dual_types::get_description(v.type).introduces_block;

        is_bad = is_bad || dual_types::get_description(v.type).reordering_hazard;

        if(!(v.type == dual_types::ops::ASSIGN && v.args.at(0).is_memory_access()))
        {
            v.recurse_arguments([&](const value& in)
            {
                if(in.is_mutable)
                    is_bad = true;
            });
        }

        ///control flow constructs are in their own dedicated segment
        if(is_bad)
        {
            blocks.emplace_back();
            blocks.back().push_back(v);
            blocks.emplace_back();
            continue;
        }

        blocks.back().push_back(v);
    }

    #define OLD_BUT_FAST
    #ifdef OLD_BUT_FAST
    int bid = 0;

    for(const auto& block : blocks)
    {
        base += "//" + std::to_string(bid) + "\n";

        for(const auto& value : block)
        {
            if(dual_types::get_description(value.type).is_semicolon_terminated)
                base += type_to_string(value) + ";\n";
            else
                base += type_to_string(value);
        }

        bid++;
    }
    #endif

    base += "\n}\n";

    return base;
}
