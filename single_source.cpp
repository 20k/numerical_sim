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

    for(value& v : ctx.sequenced)
    {
        auto fix_idot = [&](value& in)
        {
            if(in.type == dual_types::ops::IDOT && in.args.at(0).is_memory_access)
                in.is_memory_access = true;
        };

        v.recurse_arguments(fix_idot);
    }

    std::vector<std::vector<value>> blocks;
    blocks.emplace_back();

    for(auto& v : ctx.sequenced)
    {
        bool is_bad = dual_types::get_description(v.type).introduces_block;

        is_bad = is_bad || dual_types::get_description(v.type).reordering_hazard;

        v.recurse_arguments([&](const value& in)
        {
            if(in.is_mutable)
                is_bad = true;
        });

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

    auto insert_value = [&]<typename T>(const T& in)
    {
        if(in.type == dual_types::ops::DECLARE)
            emitted_variable_names.insert(type_to_string(in.args.at(1)));

        if(dual_types::get_description(in.type).is_semicolon_terminated)
            base += type_to_string(in) + ";\n";
        else
            base += type_to_string(in);
    };

    std::set<std::string> eventually_declared = emitted_variable_names;

    for(auto& block : blocks)
    {
        for(value& v : block)
        {
            auto substitute = [&](value& in)
            {
                if(in.type == dual_types::ops::UNKNOWN_FUNCTION)
                {
                    if(type_to_string(in.args[0]) == "buffer_index" || type_to_string(in.args[0]) == "buffer_indexh")
                    {
                        if(!in.is_memory_access)
                        {
                            std::cout << kernel_name << std::endl;
                            std::cout << type_to_string(in) << std::endl;
                        }
                        assert(in.is_memory_access);
                    }
                }
            };

            v.recurse_arguments(substitute);

            auto check = [&]<typename T>(value& in, T&& func)
            {
                if(in.type == dual_types::ops::DECLARE)
                {
                    eventually_declared.insert(type_to_string(in.args.at(1)));
                }

                for(value& i : in.args)
                {
                    i.recurse_lambda(func);
                }
            };

            v.recurse_lambda(check);
        }
    }

    auto make_unique_vec = []<typename T>(const std::vector<T>& in)
    {
        std::set<T> s(in.begin(), in.end());

        return std::vector<T>(s.begin(), s.end());
    };

    int gidx = 0;

    for(auto& block : blocks)
    {
        if(block.size() == 0)
            continue;

        std::vector<std::pair<value, std::string>> emitted_memory_requests;
        std::vector<std::pair<value, std::string>> emitted_cache;
        std::set<value*> emitted;

        auto is_bottom_rung_type = [&](value& in)
        {
            return in.type == dual_types::ops::BREAK ||
                   in.type == dual_types::ops::FOR_START || in.type == dual_types::ops::IF_START ||
                   in.type == dual_types::ops::BLOCK_START || in.type == dual_types::ops::BLOCK_END ||
                   in.type == dual_types::ops::VALUE || in.type == dual_types::ops::IDOT ||
                   in.type == dual_types::ops::SIDE_EFFECT;
        };

        auto is_bottom_rung = [&](value& in)
        {
            return is_bottom_rung_type(in) ||
                   emitted.find(&in) != emitted.end();
        };

        auto all_bottom_rung = [&](value& in)
        {
            bool valid = true;

            in.for_each_real_arg([&](value& me)
            {
                if(!is_bottom_rung(me))
                    valid = false;
            });

            return valid;
        };

        auto can_be_assigned_to_variable = [&](value& in)
        {
            using namespace dual_types::ops;

            return  in.type != RETURN && in.type != BREAK && in.type != FOR_START && in.type != IF_START && in.type != BLOCK_START &&
                    in.type != BLOCK_END && in.type != DECLARE && in.type != ASSIGN && in.type != SIDE_EFFECT;
        };

        auto has_satisfied_variable_deps = [&](value& in)
        {
            std::vector<std::string> variables = in.get_all_variables();

            for(auto& i : variables)
            {
                bool will_ever_be_declared = eventually_declared.find(i) != eventually_declared.end();

                if(!will_ever_be_declared)
                    continue;

                if(emitted_variable_names.find(i) == emitted_variable_names.end())
                    return false;
            }

            return true;
        };

        /*auto get_unsatisfied_variable_deps = [&](value& in)
        {
            std::vector<std::string> includes_dupes;

            auto getter = [&]<typename T>(value& me, T&& func)
            {
                me.for_each_real_arg([&](value& arg)
                {
                    if(arg.type == dual_types::ops::VALUE && !arg.is_constant())
                    {
                        std::string str = type_to_string(arg);

                        bool is_unsatisfied = emitted_variable_names.find(str) == emitted_variable_names.end();
                        bool will_ever_be_declared = eventually_declared.find(str) != eventually_declared.end();

                        if(will_ever_be_declared && is_unsatisfied)
                            includes_dupes.push_back(str);
                    }

                    arg.recurse_lambda(func);
                });
            };

            in.recurse_lambda(getter);

            return includes_dupes;
        };*/

        auto is_memory_request_sat = [&](value& in) -> std::optional<std::string>
        {
            assert(in.is_memory_access);

            for(auto& i : emitted_memory_requests)
            {
                if(dual_types::equivalent(i.first, in))
                    return i.second;
            }

            return std::nullopt;
        };

        auto get_unsatisfied_memory_deps = [&](value& in)
        {
            std::vector<std::string> req;

            auto getter = [&]<typename T>(value& me, T&& func)
            {
                if(me.is_memory_access && !is_memory_request_sat(me).has_value())
                {
                    req.push_back(type_to_string(me));
                    return;
                }

                me.for_each_real_arg([&](value& arg)
                {
                    arg.recurse_lambda(func);
                });
            };

            in.recurse_lambda(getter);

            return make_unique_vec(req);
        };

        auto has_unsatisfied_memory_deps = [&](value& in)
        {
            bool unsat = false;

            auto getter = [&]<typename T>(value& me, T&& func)
            {
                if(&me != &in)
                {
                    if(me.type == dual_types::ops::IDOT && me.args[0].is_memory_access)
                        return;

                    if(me.is_memory_access && !is_memory_request_sat(me).has_value())
                    {
                        unsat = true;
                        return;
                    }
                }

                me.for_each_real_arg([&](value& arg)
                {
                    arg.recurse_lambda(func);
                });
            };

            in.recurse_lambda(getter);
            return unsat;
        };

        auto get_memory_deps = [&](value& in)
        {
            std::set<std::string> req;

            auto getter = [&]<typename T>(value& me, T&& func)
            {
                me.for_each_real_arg([&](value& arg)
                {
                    if(arg.is_memory_access)
                    {
                        req.insert(type_to_string(arg));
                    }

                    arg.recurse_lambda(func);
                });
            };

            in.recurse_lambda(getter);

            return req;
        };

        bool going = true;

        #if 0
        while(going)
        {
            std::vector<std::string> unsat_memory;
            std::vector<value*> all_generatable_top_level_args;

            for(auto& v : block)
            {
                auto get_unsat = [&]<typename T>(value& in, T&& func)
                {
                    auto unsat = get_unsatisfied_memory_deps(in);

                    if(unsat.size() == 0)
                    {
                        if(has_satisfied_variable_deps(in))
                        {
                            all_generatable_top_level_args.push_back(&in);
                            return;
                        }
                    }
                    else
                    {
                        for(auto& i : unsat)
                        {
                            unsat_memory.push_back(i);
                        }
                    }

                    in.for_each_real_arg([&](value& arg)
                    {
                        func(arg, func);
                    });
                };

                v.recurse_lambda(get_unsat);
            }

            if(all_generatable_top_level_args.size() == 0)
                break;

            std::vector<value*> not_memory_access;
            std::vector<value*> memory_access;
            std::vector<value*> assigns;

            for(auto i : all_generatable_top_level_args)
            {
                if(i->is_memory_access)
                {
                    memory_access.push_back(i);
                }
                else
                {
                    if(i->type == dual_types::ops::ASSIGN)
                        assigns.push_back(i);
                    else
                        not_memory_access.push_back(i);
                }
            }

            std::vector<value*> to_emit;

            if(assigns.size() > 0)
                to_emit = assigns;
            else if(not_memory_access.size() > 0)
                to_emit = not_memory_access;
            else
                to_emit = {memory_access.at(0)};

            for(value* v : to_emit)
            {
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

                    if(valid)
                    {
                        std::string name = "genid" + std::to_string(gidx);

                        auto [declare_op, val] = declare_raw(*v, name, v->is_mutable);

                        insert_value(declare_op);

                        emitted_cache.push_back({*v, name});

                        *v = val;

                        gidx++;
                    }
                }

                emitted.insert(v);
            }

            going = all_generatable_top_level_args.size() > 0;
        }
        #endif

        #if 1
        while(1)
        {
            std::vector<value*> all_arguments_bottom_rung;
            std::map<std::string, int> could_be_satisfied_with;

            for(auto& v : block)
            {
                auto substitute = [&](value& in)
                {
                    for(auto& [val, name] : emitted_cache)
                    {
                        if(equivalent(val, in))
                        {
                            in = name;
                        }
                    }
                };

                v.recurse_arguments(substitute);

                auto mem = [&](value& in)
                {
                    for(auto& [val, name] : emitted_memory_requests)
                    {
                        if(equivalent(val, in))
                        {
                            in = name;
                        }
                    }
                };

                v.recurse_arguments(mem);

                bool already_emitted = emitted.find(&v) != emitted.end();

                if(already_emitted)
                    continue;

                if(is_bottom_rung(v) && has_satisfied_variable_deps(v))
                {
                    all_arguments_bottom_rung.push_back(&v);
                    continue;
                }

                /*if(!is_bottom_rung(v) && has_satisfied_variable_deps(v) && !has_unsatisfied_memory_deps(v))
                {
                    all_arguments_bottom_rung.push_back(&v);
                    continue;
                }*/

                if(all_bottom_rung(v) && has_satisfied_variable_deps(v))
                {
                    all_arguments_bottom_rung.push_back(&v);
                    continue;
                }

                auto recurse = [&]<typename T>(value& in, T&& func)
                {
                    if(emitted.find(&in) != emitted.end())
                        return;

                    /*if(is_bottom_rung(in))
                        return;

                    if(all_bottom_rung(in) && has_satisfied_variable_deps(in))
                    {
                        all_arguments_bottom_rung.push_back(&in);
                        return;
                    }*/

                    if(is_bottom_rung_type(in) && has_satisfied_variable_deps(in) && in.is_memory_access)
                    {
                        all_arguments_bottom_rung.push_back(&in);
                        return;
                    }

                    if(is_bottom_rung(in))
                        return;

                    ///i think the issue is we can't look through an idot
                    ///so doing int ix = points[lidx].x is always doomed to fail
                    /*if(has_satisfied_variable_deps(in) && !has_unsatisfied_memory_deps(in))
                    {
                        all_arguments_bottom_rung.push_back(&in);
                        return;
                    }*/

                    if(all_bottom_rung(in) && has_satisfied_variable_deps(in))
                    {
                        all_arguments_bottom_rung.push_back(&in);
                        return;
                    }

                    if(auto unsat_amount = get_unsatisfied_memory_deps(in); has_satisfied_variable_deps(in) && unsat_amount.size() == 1)
                    {
                        could_be_satisfied_with[unsat_amount[0]]++;
                    }

                    in.for_each_real_arg([&](value& arg)
                    {
                        func(arg, func);
                    });
                };

                v.recurse_lambda(recurse);
            }

            if(all_arguments_bottom_rung.size() == 0)
                break;

            std::vector<std::pair<std::string, int>> could_be_sat_vec;

            for(auto& i : could_be_satisfied_with)
            {
                could_be_sat_vec.push_back(i);
            }

            std::sort(could_be_sat_vec.begin(), could_be_sat_vec.end(), [](const auto& v1, const auto& v2){return v1.second < v2.second;});

            std::vector<value*> not_memory_access;
            std::vector<value*> memory_access;
            std::vector<value*> assigns;

            for(auto i : all_arguments_bottom_rung)
            {
                if(i->is_memory_access)
                {
                    memory_access.push_back(i);
                }
                else
                {
                    if(i->type == dual_types::ops::ASSIGN)
                        assigns.push_back(i);
                    else
                        not_memory_access.push_back(i);
                }
            }

            std::vector<value*> to_emit;

            if(assigns.size() > 0)
                to_emit = assigns;
            else if(not_memory_access.size() > 0)
                to_emit = not_memory_access;
            else
            {
                //assert(could_be_sat_vec.size() > 0);

                if(could_be_sat_vec.size() > 0)
                {
                    std::string last = could_be_sat_vec.back().first;

                    for(auto& i : memory_access)
                    {
                        if(type_to_string(*i) == last)
                        {
                            to_emit = {i};
                            std::cout << "Got\n";
                        }
                    }
                }
                //assert(to_emit.size() > 0);

                if(to_emit.size() == 0)
                    to_emit = {memory_access.at(0)};
            }

            for(value* v : to_emit)
            {
                ///so. All our arguments are constants
                ///this means we want to ideally place ourself into a constant, then emit that new declaration
                ///if applicable
                ///otherwise just emit the statement

                if(!can_be_assigned_to_variable(*v))
                {
                    insert_value(*v);
                }
                else
                {
                    bool valid = true;

                    for(const auto& [val, name] : emitted_cache)
                    {
                        if(equivalent(val, *v))
                        {
                            valid = false;
                            break;
                        }
                    }

                    if(valid)
                    {
                        std::string name = "genid" + std::to_string(gidx);

                        auto [declare_op, val] = declare_raw(*v, name, v->is_mutable);

                        insert_value(declare_op);

                        if(v->is_memory_access)
                        {
                            emitted_memory_requests.push_back({*v, name});
                        }

                        emitted_cache.push_back({*v, name});

                        gidx++;
                    }
                }

                emitted.insert(v);
            }
        }
        #endif

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
