#include "single_source.hpp"

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
    ctx.fix_buffers();

    std::vector<std::vector<value>> blocks;
    blocks.emplace_back();

    for(auto& value : ctx.sequenced)
    {
        ///control flow constructs are in their own dedicated segment
        if(dual_types::get_description(value.type).introduces_block)
        {
            blocks.emplace_back();
            blocks.back().push_back(value);
            blocks.emplace_back();
            continue;
        }

        ///just for simplicity
        if(dual_types::get_description(value.type).reordering_hazard)
        {
            blocks.emplace_back();
            blocks.back().push_back(value);
            blocks.emplace_back();
            continue;
        }

        blocks.back().push_back(value);
    }

    auto insert_value = [&]<typename T>(const T& in)
    {
        if(dual_types::get_description(in.type).is_semicolon_terminated)
            base += type_to_string(in) + ";\n";
        else
            base += type_to_string(in);
    };

    int gidx = 0;

    for(auto& block : blocks)
    {
        if(block.size() == 0)
            continue;

        std::vector<std::pair<value, std::string>> emitted_cache;

        for(auto& v : block)
        {
            auto checker = [&]<typename T>(value& in, T&& func)
            {
                if(in.type == dual_types::ops::IDOT)
                {
                    return;
                }

                if(in.type == dual_types::ops::RETURN || in.type == dual_types::ops::BREAK ||
                   in.type == dual_types::ops::FOR_START || in.type == dual_types::ops::IF_START ||
                   in.type == dual_types::ops::BLOCK_START || in.type == dual_types::ops::BLOCK_END)
                {
                    insert_value(in);
                    return;
                }

                if(in.type == dual_types::ops::VALUE)
                    return;

                for(int i=0; i < (int)in.args.size(); i++)
                {
                    if(in.type == dual_types::ops::DECLARE && i < 2)
                        continue;

                    if(in.type == dual_types::ops::UNKNOWN_FUNCTION && i == 0)
                        continue;

                    if(in.type == dual_types::ops::UNKNOWN_FUNCTION)
                        continue;

                    if(in.type == dual_types::ops::CONVERT && i == 1)
                        continue;

                    if(in.type == dual_types::ops::ASSIGN && i == 0)
                        continue;

                    func(in.args[i], func);
                }

                if(in.type == dual_types::ops::DECLARE || in.type == dual_types::ops::ASSIGN)
                {
                    insert_value(in);
                    return;
                }

                for(const auto& [val, name] : emitted_cache)
                {
                    if(equivalent(val, in))
                    {
                        in = name;
                        return;
                    }
                }

                std::string name = "genid" + std::to_string(gidx);

                auto [declare_op, val] = declare_raw(in, name, in.is_mutable);

                insert_value(declare_op);

                emitted_cache.push_back({in, name});

                in = val;

                gidx++;
            };

            v.recurse_lambda(checker);
        }
    }

    base += "\n}\n";

    return base;
}
