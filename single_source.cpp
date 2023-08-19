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

    std::set<std::string> emitted_variable_names;

    for(int i=0; i < (int)kctx.inputs.args.size(); i++)
    {
        emitted_variable_names.insert(kctx.inputs.args[i].name);

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
        if(in.type == dual_types::ops::DECLARE)
            emitted_variable_names.insert(type_to_string(in.args.at(1)));

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
        std::set<value*> emitted;
        std::set<std::string> emitted_decl_names;

        auto is_bottom_rung = [&](value& in)
        {
            return in.type == dual_types::ops::BREAK ||
                   in.type == dual_types::ops::FOR_START || in.type == dual_types::ops::IF_START ||
                   in.type == dual_types::ops::BLOCK_START || in.type == dual_types::ops::BLOCK_END ||
                   in.type == dual_types::ops::VALUE || in.type == dual_types::ops::IDOT || in.is_memory_access ||
                   in.type == dual_types::ops::UNKNOWN_FUNCTION || emitted.find(&in) != emitted.end();
        };

        auto all_bottom_rung = [&](value& in)
        {
            for(auto& i : in.args)
            {
                if(!is_bottom_rung(i))
                    return false;
            }

            return true;
        };

        auto can_be_assigned_to_variable = [&](value& in)
        {
            using namespace dual_types::ops;

            return  in.type != RETURN && in.type != BREAK && in.type != FOR_START && in.type != IF_START && in.type != BLOCK_START &&
                    in.type != BLOCK_END && in.type != IDOT && in.type != DECLARE && in.type != ASSIGN;
        };

        auto has_variable_deps = [&](value& in)
        {
            std::vector<std::string> variables = in.get_all_variables();

            for(auto& i : variables)
            {
                if(emitted_variable_names.find(i) == emitted_variable_names.end())
                    return false;
            }

            return true;
        };

        bool going = true;

        while(going)
        {
            std::vector<value*> all_arguments_bottom_rung;

            for(auto& v : block)
            {
                if(is_bottom_rung(v) && emitted.find(&v) == emitted.end() && has_variable_deps(v))
                {
                    all_arguments_bottom_rung.push_back(&v);
                    continue;
                }

                auto recurse = [&]<typename T>(value& in, T&& func)
                {
                    if(emitted.find(&in) != emitted.end())
                        return;

                    if(is_bottom_rung(in))
                        return;

                    if(all_bottom_rung(in) && has_variable_deps(in))
                    {
                        all_arguments_bottom_rung.push_back(&in);
                        return;
                    }

                    in.for_each_real_arg([&](value& arg)
                    {
                        func(arg, func);
                    });
                };

                v.recurse_lambda(recurse);
            }

            for(value* v : all_arguments_bottom_rung)
            {
                ///so. All our arguments are constants
                ///this means we want to ideally place ourself into a constant, then emit that new declaration
                ///if applicable
                ///otherwise just emit the statement

                if(!can_be_assigned_to_variable(*v))
                    insert_value(*v);
                else
                {
                    bool valid = true;

                    for(const auto& [val, name] : emitted_cache)
                    {
                        if(equivalent(val, *v))
                        {
                            *v = name;
                            valid = false;
                            break;
                        }
                    }

                    if(!valid)
                        continue;

                    std::string name = "genid" + std::to_string(gidx);

                    auto [declare_op, val] = declare_raw(*v, name, v->is_mutable);

                    insert_value(declare_op);

                    emitted_cache.push_back({*v, name});

                    *v = val;

                    gidx++;
                }

                emitted.insert(v);
            }

            going = all_arguments_bottom_rung.size() > 0;
        }

        #if 0
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
        #endif
    }

    base += "\n}\n";

    return base;
}
