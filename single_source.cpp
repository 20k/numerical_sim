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

    ///so, take 2
    ///we only really want to unblock by memory accesses, because that's the entire point
    ///but memory accesses aren't clear

    #if 0
    for(auto& block : blocks)
    {
        if(block.size() == 0)
            continue;

        std::set<const value*> emitted;

        std::set<const value*> block_variables;

        for(auto& v : block)
        {
            v.recurse_variables([&](const value& in)
            {
                block_variables.insert(&in);
            });
        }

        while(1)
        {
            std::map<const value*, std::vector<const value*>> depends_on;

            for(auto& v : block)
            {
                v.bottom_up_recurse([&](const value& in)
                {
                    if(emitted.find(&in) != emitted.end())
                        return;

                    depends_on[&in];

                    for(const auto& i : in.args)
                    {
                        if(emitted.find(&i) != emitted.end())
                            continue;

                        if(auto block_it = block_variables.find(&i); block_it != block_variables.end())
                        {
                            depends_on[&in].push_back(&i);
                        }

                        if(auto map_it = depends_on.find(&i); map_it != depends_on.end())
                        {
                            for(const auto& e : map_it->second)
                            {
                                depends_on[&in].push_back(e);
                            }
                        }
                    }
                });
            }

            if(depends_on.size() == 0)
                break;

            auto map_2_vec = []<typename T>(const std::map<T, int>& in)
            {
                std::vector<std::pair<T, int>> vec;

                for(auto& [a, b] : in)
                {
                    vec.push_back({a, b});
                }

                std::sort(vec.begin(), vec.end(), [](const auto& v1, const auto& v2)
                {
                    return v1.second < v2.second;
                });

                return vec;
            };

            ///needs to be by name, need dedup
            std::map<std::string, int> most_in_demand;

            for(const auto& [source, depends] : depends_on)
            {
                for(auto d : depends)
                {
                    most_in_demand[type_to_string(d)]++;
                }
            }

            std::vector<std::pair<std::string, int>> most_in_demand_vec = map_2_vec(most_in_demand);

            std::map<const value*, int> most_unblocked;

            for(const auto& [source, depends] : depends_on)
            {
                if(depends.size() == 1)
                {
                    most_unblocked[depends.back()]++;
                }
            }

            auto most_unblocked_vec = map_2_vec(most_unblocked);

            const value* next = nullptr;

            assert(most_in_demand_vec.size() > 0 || most_unblocked_vec.size() > 0);

            if(most_unblocked_vec.size() > 0)
                next = most_unblocked_vec.back().first;

            if(next == nullptr)
                next = most_in_demand_vec.back().first;

            emitted.insert(next);

            insert_value(*next);

            for(const auto& [source, on] : depends_on)
            {
                if(on.size() == 1)
                {
                    if(on[0] != next)
                        continue;
                }

                if(on.size() == 1 || on.size() == 0)
                {
                    insert_value(*source);
                }
            }
        }


        /*for(const auto& [constant, count] : most_unblocked)
        {
            std::cout << type_to_string(*constant) << " " << count << std::endl;
        }*/
    }
    #endif

    /*int block_id = 0;

    for(const auto& block : blocks)
    {
        base += "//" + std::to_string(block_id) + "\n";

        for(const auto& value : block)
        {
            if(dual_types::get_description(value.type).is_semicolon_terminated)
                base += type_to_string(value) + ";\n";
            else
                base += type_to_string(value);
        }

        block_id++;
    }*/

    base += "\n}\n";

    //std::cout << base << std::endl;

    return base;
}
