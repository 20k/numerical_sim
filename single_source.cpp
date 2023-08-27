#include "single_source.hpp"
#include <ranges>

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

        if(!(v.type == dual_types::ops::ASSIGN && v.args.at(0).is_memory_access))
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

    auto insert_value = [&]<typename T>(const T& in)
    {
        if(in.type == dual_types::ops::DECLARE)
            emitted_variable_names.insert(type_to_string(in.args.at(1)));

        if(in.type == dual_types::ops::DECLARE_ARRAY)
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
                if(in.type == dual_types::ops::DECLARE || in.type == dual_types::ops::DECLARE_ARRAY)
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

    int block_id = 0;

    for(auto& block : blocks)
    {
        break;

        block_id++;

        if(block.size() == 0)
            continue;

        std::vector<std::pair<value, std::string>> emitted_memory_requests;
        std::vector<std::pair<value, std::string>> emitted_cache;
        std::set<const value*> emitted;
        std::vector<const value*> linear_emitted;
        std::set<const value*> prefetched;

        auto is_bottom_rung_type = [&](value& in)
        {
            return in.type == dual_types::ops::BREAK ||
                   in.type == dual_types::ops::FOR_START || in.type == dual_types::ops::IF_START ||
                   in.type == dual_types::ops::BLOCK_START || in.type == dual_types::ops::BLOCK_END ||
                   in.type == dual_types::ops::VALUE || in.type == dual_types::ops::IDOT ||
                   in.type == dual_types::ops::SIDE_EFFECT;
        };

        auto dont_peek = [&](const value& in)
        {
            return in.type == dual_types::ops::BREAK ||
            in.type == dual_types::ops::FOR_START || in.type == dual_types::ops::IF_START ||
            in.type == dual_types::ops::BLOCK_START || in.type == dual_types::ops::BLOCK_END ||
            in.type == dual_types::ops::VALUE ||
            in.type == dual_types::ops::SIDE_EFFECT;
        };

        /*auto is_bottom_rung = [&](value& in)
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
        };*/

        auto was_emitted = [&](const value& in)
        {
            return emitted.find(&in) != emitted.end();
        };

        auto all_args_emitted = [&](const value& in)
        {
            if(dont_peek(in))
                return true;

            bool valid = true;

            /*in.for_each_real_arg([&](const value& arg)
            {
                if(arg.type == dual_types::ops::VALUE)
                    return;

                if(emitted.find(&arg) == emitted.end())
                    valid = false;
            });*/

            in.for_each_real_arg([&](const value& arg)
            {
                if(!valid)
                    return;

                if(arg.type == dual_types::ops::VALUE)
                    return;

                if(emitted.find(&arg) != emitted.end())
                    return;

                /*if(emitted.find(&arg) == emitted.end())
                    valid = false;*/

                bool none_valid = true;

                for(auto i : linear_emitted)
                {
                    if(equivalent(arg, *i))
                    {
                        none_valid = false;
                        break;
                    }
                }

                if(none_valid)
                    valid = false;
            });

            return valid;
        };

        auto can_be_assigned_to_variable = [&](const value& in)
        {
            using namespace dual_types::ops;

            return  in.type != RETURN && in.type != BREAK && in.type != FOR_START && in.type != IF_START && in.type != BLOCK_START &&
                    in.type != BLOCK_END && in.type != DECLARE && in.type != DECLARE_ARRAY && in.type != ASSIGN && in.type != SIDE_EFFECT;
        };

        std::map<const value*, std::vector<std::string>> var_cache;

        auto has_satisfied_variable_deps = [&](const value& in)
        {
            std::vector<std::string> variables;

            if(auto it = var_cache.find(&in); it != var_cache.end())
            {
                variables = it->second;
            }
            else
            {
                variables = in.get_all_variables();
                var_cache[&in] = variables;
            }

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

        /*auto is_memory_request_sat = [&](value& in) -> std::optional<std::string>
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
        };*/

        auto get_memory_dependencies = [&](const value& in)
        {
            std::vector<std::string> req;

            auto getter = [&]<typename T>(const value& me, T&& func)
            {
                if(me.is_memory_access)
                {
                    req.push_back(type_to_string(me));
                }

                me.for_each_real_arg([&](const value& arg)
                {
                    arg.recurse_lambda(func);
                });
            };

            in.recurse_lambda(getter);

            return make_unique_vec(req);
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

        std::vector<value_base<std::monostate>> local_emit;

        auto insert_local = [&]<typename T>(const T& in)
        {
            local_emit.push_back(in.as_generic());

            if(in.type == dual_types::ops::DECLARE)
                emitted_variable_names.insert(type_to_string(in.args.at(1)));
            if(in.type == dual_types::ops::DECLARE_ARRAY)
                emitted_variable_names.insert(type_to_string(in.args.at(1)));
        };

        ///so the big problem currently is actually *using* the results we've generated in the expressions
        ///because while I can check the top level arg, the only recourse I'd have is to do a full resub which... isn't fast
        auto emit = [&](const value& v)
        {
            ///so. All our arguments are constants
            ///this means we want to ideally place ourself into a constant, then emit that new declaration
            ///if applicable
            ///otherwise just emit the statement

            value could_emit = v;

            bool is_root_relabling = false;
            bool any_sub = false;

            could_emit.recurse_arguments([&](value& arg)
            {
                for(const auto& [val, name] : emitted_cache)
                {
                    if(equivalent(arg, val))
                    {
                        if(&arg == &could_emit)
                            is_root_relabling = true;

                        any_sub = true;

                        arg = name;
                        //return;
                    }
                }
            });

            bool emitted_anything = true;

            if(!can_be_assigned_to_variable(v))
            {
                /*if(kernel_name == "evolve_1")
                {
                    std::cout << "hi " << type_to_string(could_emit) << std::endl;
                    std::cout << "? " << (v.type == dual_types::ops::DECLARE) << std::endl;

                    if(v.type == dual_types::ops::DECLARE)
                    std::cout << "Type2 " << v.args.at(2).type << " is " << type_to_string(v.args.at(2)) << " cst? " << v.args.at(2).is_constant() << std::endl;
                }*/

                insert_local(could_emit);

                if(v.type == dual_types::ops::DECLARE)
                {
                    ///lets say we've relabelled. This means we declare with a name of pv0
                    ///and a value of gen1232123
                    ///const float pv0 = genid12321
                    ///args.at(2) == genid12321
                    ///args.at(1) == pv0

                    /*if(could_emit.args.at(2).is_value())
                    {
                        emitted_cache.push_back({could_emit.args.at(1), type_to_string(could_emit.args.at(2))});

                        //std::cout << "mapping " << type_to_string(could_emit.args.at(1)) << " to " << type_to_string(could_emit.args.at(2)) << std::endl;
                    }
                    else*/
                    {

                        emitted_cache.push_back({v.args.at(2), type_to_string(v.args.at(1))});
                    }

                    //emitted_cache.push_back({type_to_string(could_emit.args.at(1)), type_to_string(could_emit.args.at(2))});
                }
            }
            else
            {
                if(!is_root_relabling)
                {
                    std::string name = "genid" + std::to_string(gidx);

                    auto [declare_op, val] = declare_raw(could_emit, name, could_emit.is_mutable);

                    insert_local(declare_op);

                    emitted_cache.push_back({v, name});
                    emitted_cache.push_back({could_emit, name});

                    gidx++;
                }
                else
                {
                    emitted_anything = false;
                }
            }

            emitted.insert(&v);

            if(!is_root_relabling)
                linear_emitted.push_back(&v);

            return emitted_anything;
        };

        ///the most deeply bracketed thing would be evaluated first. Todo, do this
        #if 0
        for(auto& v : block)
        {
            auto recurse = [&]<typename T>(const value& in, T&& func)
            {
                if(in.type == dual_types::ops::VALUE)
                    return;

                if(!dont_peek(in))
                {
                    in.for_each_real_arg([&](const value& arg)
                    {
                        arg.recurse_lambda(func);
                    });
                }

                emit(in);
            };

            v.recurse_lambda(recurse);
        }
        #endif

        #if 0
        bool any_emitted = true;

        while(any_emitted)
        {
            /*if(kernel_name == "evolve_1")
            {
                printf("hi\n");
            }*/

            any_emitted = false;

            for(auto& v : block)
            {
                auto recurse = [&]<typename T>(const value& in, T&& func)
                {
                    if(in.type == dual_types::ops::VALUE)
                        return;

                    if(was_emitted(in))
                        return;

                    if(!dont_peek(in))
                    {
                        in.for_each_real_arg([&](const value& arg)
                        {
                            arg.recurse_lambda(func);
                        });
                    }

                    if(all_args_emitted(in) && has_satisfied_variable_deps(in))
                    {
                        //if(kernel_name == "evolve_1")
                        //std::cout << "Emit " << type_to_string(in) << " " << kernel_name << std::endl;

                        emit(in);
                        any_emitted = true;
                    }
                };

                v.recurse_lambda(recurse);
            }

            //if(kernel_name == "evolve_1")
                //printf("Here\n");

            for(auto& v : block)
            {
                auto recursem = [&]<typename T>(const value& in, T&& func)
                {
                    if(any_emitted)
                        return;

                    if(!any_emitted && in.is_memory_access)
                    {
                        if(!was_emitted(in))
                        {
                            emit(in);
                            any_emitted = true;
                            return;
                        }
                    }

                    in.for_each_real_arg([&](const value& arg)
                    {
                        arg.recurse_lambda(func);
                    });
                };

                if(!any_emitted)
                    v.recurse_lambda(recursem);
            }
        }
        #endif

        #if 0
        std::map<int, int> easiest_block;

        for(int i=0; i < (int)block.size(); i++)
        {
            if(block[i].type != dual_types::ops::ASSIGN)
            {
                easiest_block[i] += 999999;
            }

            auto count_ops = [&](const value& in)
            {
                easiest_block[i]++;
            };

            block[i].recurse_arguments(count_ops);
        }

        std::vector<std::pair<int, int>> easiest_block_vec;

        for(auto& i : easiest_block)
        {
            easiest_block_vec.push_back(i);
        }

        std::sort(easiest_block_vec.begin(), easiest_block_vec.end(), [](const auto& i1,const auto& i2){return i1.second < i2.second;});

        bool any_emitted = true;

        while(any_emitted)
        {
            any_emitted = false;

            //for(auto& v : block)

            for(auto& [which_block, _] : easiest_block_vec)
            {
                auto& v = block[which_block];

                auto recurse = [&]<typename T>(const value& in, T&& func)
                {
                    if(in.type == dual_types::ops::VALUE)
                        return;

                    if(was_emitted(in))
                        return;

                    if(!dont_peek(in))
                    {
                        in.for_each_real_arg([&](const value& arg)
                        {
                            arg.recurse_lambda(func);
                        });
                    }

                    if(all_args_emitted(in) && has_satisfied_variable_deps(in))
                    {
                        emit(in);
                        any_emitted = true;
                    }
                };

                v.recurse_lambda(recurse);
            }

            for(auto& [which_block, _] : easiest_block_vec)
            {
                auto& v = block[which_block];

                auto recursem = [&]<typename T>(const value& in, T&& func)
                {
                    if(any_emitted)
                        return;

                    if(!any_emitted && in.is_memory_access)
                    {
                        if(!was_emitted(in))
                        {
                            emit(in);
                            any_emitted = true;
                            return;
                        }
                    }

                    in.for_each_real_arg([&](const value& arg)
                    {
                        arg.recurse_lambda(func);
                    });
                };

                if(!any_emitted)
                    v.recurse_lambda(recursem);
            }
        }
        #endif

        #if 0
        bool any_emitted = true;

        while(any_emitted)
        {
            any_emitted = false;

            for(auto& v : block)
            {
                auto recurse = [&]<typename T>(const value& in, T&& func)
                {
                    if(in.type == dual_types::ops::VALUE)
                        return;

                    if(was_emitted(in))
                        return;

                    if(!dont_peek(in))
                    {
                        in.for_each_real_arg([&](const value& arg)
                        {
                            arg.recurse_lambda(func);
                        });
                    }

                    if(all_args_emitted(in) && has_satisfied_variable_deps(in))
                    {
                        emit(in);
                        any_emitted = true;
                    }
                };

                v.recurse_lambda(recurse);

                auto recursem = [&]<typename T>(const value& in, T&& func)
                {
                    if(any_emitted)
                        return;

                    if(!any_emitted && in.is_memory_access)
                    {
                        if(!was_emitted(in))
                        {
                            emit(in);
                            any_emitted = true;
                            return;
                        }
                    }

                    in.for_each_real_arg([&](const value& arg)
                    {
                        arg.recurse_lambda(func);
                    });
                };

                if(!any_emitted)
                    v.recurse_lambda(recursem);
            }
        }
        #endif

        auto depends_on = []<typename T>(const T& base, const T& is_dependent_on)
        {
            bool is_dependent = false;

            auto checker = [&]<typename U>(const T& in, U&& func)
            {
                if(is_dependent)
                    return;

                if(equivalent(is_dependent_on, in))
                {
                    is_dependent = true;
                    return;
                }

                if(is_dependent_on.type == dual_types::ops::DECLARE)
                {
                    if(equivalent(is_dependent_on.args.at(1), in))
                    {
                        is_dependent = true;
                        return;
                    }
                }

                if(in.type == dual_types::ops::DECLARE)
                {
                    in.args.at(1).recurse_lambda(func);
                }

                in.for_each_real_arg([&](const T& arg)
                {
                    arg.recurse_lambda(func);
                });
            };

            base.recurse_lambda(checker);

            return is_dependent;
        };

        auto is_dep_free = [depends_on](std::vector<value_v>& in, int index)
        {
            if(in.at(index).type != dual_types::ops::DECLARE)
                return false;

            for(int j=index+1; j < (int)in.size(); j++)
            {
                if(depends_on(in.at(j), in.at(index)))
                    return false;
            }

            return true;
        };

        #if 0
        auto try_move_later = [depends_on](std::vector<value_v>& in, int index)
        {
            assert(index >= 0 && index < (int)in.size());

            for(int j=index+1; j < (int)in.size(); j++)
            {
                if(!depends_on(in.at(j), in.at(index)))
                    continue;

                if(j != index + 1)
                {
                    auto me = in[index];

                    assert(j >= 0 && j < in.size());

                    in.insert(in.begin() + j, me);

                    assert(index >= 0 && index < in.size());

                    in.erase(in.begin() + index);
                    return;
                }

                return;
            }
        };

        auto move_later = [try_move_later](std::vector<value_v>& in)
        {
            for(int i=(int)in.size() - 1; i >= 0; i--)
            {
                try_move_later(in, i);
            }
        };

        if(kernel_name== "evolve_1");
        move_later(local_emit);
        #endif

        //#define BOTTOMS_UP
        #ifdef BOTTOMS_UP
        std::vector<std::pair<const value*, int>> unevaluated_depth;
        std::vector<std::pair<const value*, int>> unevaluated_memory_depth;

        for(auto& v : block)
        {
            int level = 0;

            auto memcheck = [&]<typename T>(const value& in, T&& func)
            {
                level++;

                if(in.is_memory_access)
                {
                    unevaluated_memory_depth.push_back({&in, level});
                }

                in.for_each_real_arg([&](const value& arg)
                {
                    arg.recurse_lambda(func);
                });

                level--;
            };

            v.recurse_lambda(memcheck);

            auto recurse = [&]<typename T>(const value& in, T&& func)
            {
                level++;

                if(in.type == dual_types::ops::VALUE)
                {
                    level--;
                    return;
                }

                if(!dont_peek(in))
                {
                    in.for_each_real_arg([&](const value& arg)
                    {
                        arg.recurse_lambda(func);
                    });
                }

                if(!in.is_memory_access)
                    unevaluated_depth.push_back({&in, level});

                //emit(in);
                level--;
            };

            v.recurse_lambda(recurse);
        }

        std::sort(unevaluated_depth.begin(), unevaluated_depth.end(), [](const auto& i1, const auto& i2)
        {
            return i1.first > i2.first;
        });

        std::sort(unevaluated_memory_depth.begin(), unevaluated_memory_depth.end(), [](const auto& i1, const auto& i2)
        {
            return i1.first > i2.first;
        });

        std::vector<const value*> unevaluated_memory;
        std::vector<const value*> unevaluated;

        for(auto& [v, _] : unevaluated_memory_depth)
        {
            unevaluated_memory.push_back(v);
        }

        for(auto& [v, _] : unevaluated_depth)
        {
            unevaluated.push_back(v);
        }

        for(int i=0; i < (int)unevaluated.size(); i++)
        {
            for(int j=i+1; j < (int)unevaluated.size(); j++)
            {
                if(dual_types::equivalent(*unevaluated[i], *unevaluated[j]))
                {
                    unevaluated.erase(unevaluated.begin() + j);
                    j--;
                    continue;
                }
            }
        }

        for(int i=0; i < (int)unevaluated_memory.size(); i++)
        {
            for(int j=i+1; j < (int)unevaluated_memory.size(); j++)
            {
                if(dual_types::equivalent(*unevaluated_memory[i], *unevaluated_memory[j]))
                {
                    unevaluated_memory.erase(unevaluated_memory.begin() + j);
                    j--;
                    continue;
                }
            }
        }

        while(unevaluated.size() > 0)
        {
            bool forward_progress = false;

            for(int i=0; i < (int)unevaluated.size(); i++)
            {
                if(all_args_emitted(*unevaluated[i]) && has_satisfied_variable_deps(*unevaluated[i]))
                {
                    printf("emit %i %i\n", i, unevaluated.size());

                    emit(*unevaluated[i]);
                    unevaluated.erase(unevaluated.begin() + i);
                    i--;
                    forward_progress = true;
                    continue;
                }
            }

            if(!forward_progress)
            {
                for(int i=0; i < (int)unevaluated_memory.size(); i++)
                {
                    if(!has_satisfied_variable_deps(*unevaluated_memory[i]))
                        continue;

                    emit(*unevaluated_memory[i]);
                    unevaluated_memory.erase(unevaluated_memory.begin() + i);
                    forward_progress = true;
                    break;
                }
            }

            if(!forward_progress && unevaluated.size() > 0)
            {
                assert(false);
            }
        }
        #endif

        #if 1
        std::vector<std::pair<const value*, std::vector<std::string>>> memory_dependency_map;

        std::vector<const value*> memory_accesses;

        for(auto& v : block)
        {
            auto build_memory_accesses = [&]<typename T>(const value& in, T&& func)
            {
                if(in.is_memory_access)
                {
                    memory_accesses.push_back(&in);
                }

                in.for_each_real_arg([&](const value& arg)
                {
                    arg.recurse_lambda(func);
                });
            };

            v.recurse_lambda(build_memory_accesses);

            auto build_memory_dependencies = [&]<typename T>(const value& in, T&& func)
            {
                if(in.type == dual_types::ops::VALUE)
                    return;

                //memory_dependency_map[&in] = get_memory_dependencies(in);

                if(!in.is_memory_access)
                    memory_dependency_map.push_back({&in, get_memory_dependencies(in)});

                if(dont_peek(in))
                    return;

                in.for_each_real_arg([&](const value& arg)
                {
                    arg.recurse_lambda(func);
                });
            };

            v.recurse_lambda(build_memory_dependencies);
        }

        std::map<std::string, const value*> memory_by_name;

        for(int i=0; i < (int)memory_accesses.size(); i++)
        {
            for(int j=i+1; j < (int)memory_accesses.size(); j++)
            {
                if(dual_types::equivalent(*memory_accesses[i], *memory_accesses[j]))
                {
                    memory_accesses.erase(memory_accesses.begin() + j);
                    j--;
                    continue;
                }
            }

            memory_by_name[type_to_string(*memory_accesses[i])] = memory_accesses[i];
        }

        /*for(int i=0; i < (int)memory_dependency_map.size(); i++)
        {
            for(int j=i+1; j < (int)memory_dependency_map.size(); j++)
            {
                if(dual_types::equivalent(*memory_dependency_map[i].first, *memory_dependency_map[j].first))
                {
                    memory_dependency_map.erase(memory_dependency_map.begin() + j);
                    j--;
                    continue;
                }
            }
        }*/

        auto prefetch = [&](const value& in)
        {
            if(in.type != dual_types::ops::UNKNOWN_FUNCTION)
                return;

            if(type_to_string(in.args.at(0)) != "buffer_index" && type_to_string(in.args.at(0)) != "buffer_indexh")
                return;

            if(kernel_name != "evolve_1")
                return;

            value buffer = in.args.at(1);
            value ix = in.args.at(2);
            value iy = in.args.at(3);
            value iz = in.args.at(4);

            value_i dx = "dim.x";
            value_i dy = "dim.y";
            value_i dz = "dim.z";

            value_i index = iz.convert<int>() * dx * dy + iy.convert<int>() * dx + ix.convert<int>();

            value op = dual_types::make_op<std::monostate>(dual_types::ops::SIDE_EFFECT, "prefetch(&" + type_to_string(buffer) + "[" + type_to_string(index) + "],1);").reinterpret_as<value_base<float>>();

            emit(op);
        };

        int instr = 0;

        while(memory_dependency_map.size() > 0)
        {
            std::vector<const value*> just_emitted;

            std::map<std::string, int> best_memory_request;

            std::cout << "Mem dep map " << memory_dependency_map.size() << " kernel " << kernel_name << std::endl;

            bool any_emitted = false;

            for(auto& [v, deps] : memory_dependency_map)
            {
                const value* could_emit = v;

                if(could_emit->type != dual_types::ops::DECLARE)
                {
                    for(auto& i : deps)
                    {
                        best_memory_request[i]++;
                    }
                }

                if(deps.size() == 0 && has_satisfied_variable_deps(*could_emit) && all_args_emitted(*could_emit))
                {
                    bool real_emission = emit(*could_emit);
                    just_emitted.push_back(could_emit);

                    if(real_emission)
                    {
                        any_emitted = true;
                        instr++;
                    }

                    /*if(instr > 50)
                    {
                        for(auto i : memory_accesses)
                        {
                            if(prefetched.find(i) == prefetched.end())
                            {
                                prefetch(*i);

                                prefetched.insert(i);

                                break;
                            }
                        }

                        instr = 0;
                    }*/
                }
            }

            std::cout << "Post emit\n";

            bool any_memory = false;

            if(!any_emitted && memory_accesses.size() > 0)
            {
                //for(auto it = memory_accesses.begin(); it != memory_accesses.end(); it++)

                if(best_memory_request.size() > 0)
                {
                    std::vector<std::pair<std::string, int>> best_req;

                    for(auto& i : best_memory_request)
                    {
                        best_req.push_back(i);
                    }

                    std::sort(best_req.begin(), best_req.end(), [](const auto& v1, const auto& v2){return v1.second > v2.second;});

                    for(auto& [req, count] : best_req)
                    {
                        const value* to_emit = memory_by_name.at(req);

                        if(!has_satisfied_variable_deps(*to_emit))
                            continue;

                        std::string as_str = type_to_string(*to_emit);

                        if(as_str != req)
                            continue;

                        emit(*to_emit);
                        prefetched.insert(to_emit);
                        just_emitted.push_back(to_emit);

                        for(auto& [v, deps] : memory_dependency_map)
                        {
                            for(int idx=0; idx < (int)deps.size(); idx++)
                            {
                                if(deps[idx] == as_str)
                                {
                                    deps.erase(deps.begin() + idx);
                                    idx--;
                                    continue;
                                }
                            }
                        }

                        for(int i=0; i < (int)memory_accesses.size(); i++)
                        {
                            if(memory_accesses[i] == to_emit)
                            {
                                memory_accesses.erase(memory_accesses.begin() + i);
                                break;
                            }
                        }

                        any_memory = true;

                        break;
                    }
                }

                if(!any_memory)
                for(int midx = 0; midx < (int)memory_accesses.size(); midx++)
                {
                    const value* to_emit = memory_accesses[midx];

                    if(!has_satisfied_variable_deps(*to_emit))
                        continue;

                    std::string as_str = type_to_string(*to_emit);

                    emit(*to_emit);
                    prefetched.insert(to_emit);
                    just_emitted.push_back(to_emit);

                    for(auto& [v, deps] : memory_dependency_map)
                    {
                        for(int idx=0; idx < (int)deps.size(); idx++)
                        {
                            if(deps[idx] == as_str)
                            {
                                deps.erase(deps.begin() + idx);
                                idx--;
                                continue;
                            }
                        }
                    }

                    memory_accesses.erase(memory_accesses.begin() + midx);
                    any_memory = true;

                    break;
                }
            }

            std::cout << "Post mem\n";

            if(!any_memory && !any_emitted)
            {
                for(auto& [v, deps] : memory_dependency_map)
                {
                    std::cout << "Test\n";

                    for(auto& d : deps)
                        std::cout << kernel_name << " waiting on " << d << std::endl;
                }

                for(auto& v : memory_accesses)
                {
                    std::cout << "Check? " << type_to_string(*v) << std::endl;
                }

                assert(false);
                break;
            }

            for(auto& i : just_emitted)
            {
                for(int kk=0; kk < (int)memory_dependency_map.size(); kk++)
                {
                    if(memory_dependency_map[kk].first == i)
                    {
                        memory_dependency_map.erase(memory_dependency_map.begin() + kk);
                        kk--;
                        continue;
                    }
                }

                //if(auto it = memory_dependency_map.find(i); it != memory_dependency_map.end())
                //    memory_dependency_map.erase(it);
            }

            std::cout << "Post erase\n";
        }
        #endif

        #if 0

        int max_instr = 30;
        int current_instr = 0;

        while(1)
        {
            std::vector<value*> all_arguments_bottom_rung;
            std::map<std::string, int> could_be_satisfied_with;

            for(auto& v : block)
            {
                bool already_emitted = emitted.find(&v) != emitted.end();

                if(already_emitted)
                    continue;

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

            auto pop_best_memory_access = [&]()
            {
                if(could_be_sat_vec.size() > 0)
                {
                    std::string last = could_be_sat_vec.back().first;

                    for(auto it = memory_access.begin(); it != memory_access.end(); it++)
                    {
                        auto val = *it;

                        if(type_to_string(*val) == last)
                        {
                            could_be_sat_vec.pop_back();
                            memory_access.erase(it);

                            return val;
                        }
                    }

                    //assert(false);
                }

                auto val = memory_access.at(0);
                memory_access.erase(memory_access.begin());
                return val;
            };

            std::vector<value*> to_emit;

            if(assigns.size() > 0)
                to_emit = assigns;
            else if(not_memory_access.size() > 0)
            {
                to_emit = not_memory_access;
            }
            else
            {
                to_emit = {pop_best_memory_access()};
            }

            auto emit = [&](const value& v)
            {
                ///so. All our arguments are constants
                ///this means we want to ideally place ourself into a constant, then emit that new declaration
                ///if applicable
                ///otherwise just emit the statement

                if(!can_be_assigned_to_variable(v))
                {
                    insert_value(v);
                }
                else
                {
                    bool valid = true;

                    for(const auto& [val, name] : emitted_cache)
                    {
                        if(equivalent(val, v))
                        {
                            valid = false;
                            break;
                        }
                    }

                    if(valid)
                    {
                        std::string name = "genid" + std::to_string(gidx);

                        auto [declare_op, val] = declare_raw(v, name, v.is_mutable);

                        insert_value(declare_op);

                        if(v.is_memory_access)
                        {
                            emitted_memory_requests.push_back({v, name});
                            current_instr = 0;
                        }

                        emitted_cache.push_back({v, name});

                        gidx++;

                        current_instr++;
                    }
                }

                emitted.insert(&v);
            };

            for(value* v : to_emit)
            {
                emit(*v);

                if(current_instr > max_instr && memory_access.size() > 0)
                {
                    emit(*pop_best_memory_access());
                    current_instr = 0;
                }
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

        std::map<std::string, value_v> decl_to_value;

        for(auto& i : local_emit)
        {
            if(i.type == dual_types::ops::DECLARE)
            {
                decl_to_value[type_to_string(i.args.at(1))] = i.args.at(2);

                if(type_to_string(i.args.at(1)) == "pv100")
                    std::cout << "HI THERE\n";
            }
        }

        std::map<std::string, int> depends_on_me_count;

        for(int i=0; i < (int)local_emit.size(); i++)
        {
            for(int j=i+1; j < (int)local_emit.size(); j++)
            {
                if(depends_on(local_emit[j], local_emit[i]))
                {
                    if(local_emit[i].type == dual_types::ops::DECLARE)
                    {
                        depends_on_me_count[type_to_string(local_emit[i].args.at(1))]++;
                    }
                }
            }
        }

        auto get_arg = [&]<typename T>(const value_v& in, int which, T&& func) -> std::optional<value_v>
        {
            using namespace dual_types::ops;

            value_v arg = in.args.at(which);

            if(!arg.is_value())
                return std::nullopt;

            if(arg.is_constant())
                return std::nullopt;

            auto it = decl_to_value.find(type_to_string(arg));

            if(it == decl_to_value.end())
                return std::nullopt;

            auto last_valid_it = it;

            while(it != decl_to_value.end())
            {
                it = decl_to_value.find(type_to_string(it->second));

                if(it != decl_to_value.end())
                    last_valid_it = it;
            }

            return last_valid_it->second;
        };

        auto is_indirectly_half = [&](const value_v& in)
        {
            // bool debug = type_to_string(in) == "genid300";
            bool debug = false;

            auto check = [&]<typename T>(const value_v& v, T&& func)
            {
                if(debug)
                {
                    std::cout << "p1\n";
                }

                if(v.original_type == "half")
                    return true;

                if(debug)
                    std::cout << "p2\n";

                if(debug)
                    std::cout << "as_str " << type_to_string(v) << std::endl;

                auto found = decl_to_value.find(type_to_string(v));

                if(found != decl_to_value.end())
                {
                    return func(found->second, func);
                }

                if(debug)
                    std::cout << "p3\n";

                if(v.type == dual_types::ops::DECLARE)
                {
                    if(type_to_string(v.args.at(0)) == "half")
                        return true;

                    return func(v.args.at(2), func);
                }

                if(debug)
                    std::cout << "p4\n";

                if(v.type == dual_types::ops::CONVERT)
                    return func(v.args.at(0), func);


                if(debug)
                    std::cout << "p5\n";

                if(v.type == dual_types::ops::UMINUS)
                    return func(v.args.at(0), func);


                if(debug)
                    std::cout << "p6\n";

                return false;
            };

            return check(in, check);
        };

        if(kernel_name == "evolve_1" && block_id == blocks.size())
        {
            std::cout << "kernel " << kernel_name << std::endl;

            //assert(is_indirectly_half("genid300"));
        }

        #if 0
        ///?????????
        for(int i=(int)local_emit.size() - 1; i >= 0; i--)
        {
            ///const float my_val = a;
            value_v& me = local_emit[i];

            using namespace dual_types::ops;

            //if(me.type != DECLARE)
            //    continue;

            //value_v& test = me.args.at(2);

            //if(test.type != MULTIPLY)
            //    continue;

            me.recurse_arguments([&](value_v& decl)
            {
                if(decl.type != MULTIPLY)
                    return;

                if(decl.original_type != "float")
                    return;

                auto left_opt = get_arg(decl, 0, get_arg);
                auto right_opt = get_arg(decl, 1, get_arg);

                auto propagate_constants = [&](value_v& in, value_v& multiply_arg, int which_arg)
                {
                    value c = in.args[1-which_arg].reinterpret_as<value>();
                    value a = multiply_arg.args[0].reinterpret_as<value>();
                    value b = multiply_arg.args[1].reinterpret_as<value>();

                    if(kernel_name == "evolve_1")
                    std::cout << "M1 " << type_to_string(c) << " " << type_to_string(a) << " " << type_to_string(b) << std::endl;

                    if(c.is_constant() && a.is_constant())
                    {
                        printf("Reassoc1\n");

                        in = ((a * c) * b).as_generic();
                        return true;
                    }

                    if(c.is_constant() && b.is_constant())
                    {
                        printf("Reassoc2\n");

                        in = ((c * b) * a).as_generic();
                        return true;
                    }

                    return false;
                };

                bool left_is_multiply = left_opt && left_opt.value().type == MULTIPLY;

                if(left_is_multiply)
                {
                    if(propagate_constants(decl, left_opt.value(), 0))
                        return;
                }

                bool right_is_multiply = right_opt && right_opt.value().type == MULTIPLY;

                if(right_is_multiply)
                {
                    if(propagate_constants(decl, right_opt.value(), 1))
                        return;
                }

                if(left_is_multiply && right_is_multiply)
                {
                    value_v left = left_opt.value();
                    value_v right = right_opt.value();

                    value_v a = left.args[0];
                    value_v b = left.args[1];
                    value_v c = right.args[0];
                    value_v d = right.args[1];

                    std::optional<value_v> cst_sum;
                    std::optional<value_v> dyn_sum;

                    auto punt_cst = [&](value_v in)
                    {
                        if(cst_sum)
                            cst_sum.value() = cst_sum.value() * in;
                        else
                            cst_sum = in;
                    };
                    auto punt_dyn = [&](value_v in)
                    {
                        if(dyn_sum)
                            dyn_sum.value() = dyn_sum.value() * in;
                        else
                            dyn_sum = in;
                    };

                    auto punt = [&](value_v in)
                    {
                        if(in.is_constant())
                            punt_cst(in);
                        else
                            punt_dyn(in);
                    };

                    punt(a);
                    punt(b);
                    punt(c);
                    punt(d);

                    if(cst_sum && dyn_sum)
                    {
                        decl = cst_sum.value() * dyn_sum.value();
                    }
                }
            });
        }
        #endif

        #if 0
        for(int i=(int)local_emit.size() - 1; i >= 0; i--)
        {
            ///const float my_val = a;
            value_v& me = local_emit[i];

            using namespace dual_types::ops;

            if(me.type != DECLARE)
                continue;

            value_v& decl = me.args.at(2);

            if(decl.type != PLUS && decl.type != MINUS)
                continue;

            if(decl.original_type != "float" && decl.original_type != "half")
                continue;

            ///const float my_val = a + b

            auto left_opt = get_arg(decl, 0, get_arg);
            auto right_opt = get_arg(decl, 1, get_arg);

            auto get_check_fma = [get_arg, is_indirectly_half](value_v& in, value_v& multiply_arg, int which_arg, bool is_minus)
            {
                assert(multiply_arg.type == MULTIPLY);

                int root_arg = 1-which_arg;

                value_v c = in.args[root_arg];
                value_v a = multiply_arg.args[0];
                value_v b = multiply_arg.args[1];

                {
                    auto c_opt = get_arg(in, root_arg, get_arg);

                    if(c_opt && c_opt.value().type == MULTIPLY)
                    {
                        c = c_opt.value();
                    }
                }

                if(is_indirectly_half(a) || is_indirectly_half(b) || is_indirectly_half(c))
                    return false;

                auto hacky_fma = [](const value_v& a, const value_v& b, const value_v& c)
                {
                    if(a.is_constant() || b.is_constant() || c.is_constant())
                    {
                        return (a * b) + c;
                    }

                    return (a*b) + c;

                    //return fma(a, b, c);
                };

                ///a - (b*c)
                if(is_minus)
                {
                    if(root_arg == 0)
                        in = -hacky_fma(a, b, -c);
                    else
                        in = hacky_fma(a, b, -c);
                }
                else
                    in = hacky_fma(a, b, c);

                return true;
            };

            bool is_minus = decl.type == MINUS;

            ///if a == (c * d)
            if(left_opt.has_value() && left_opt.value().type == MULTIPLY)
            {
                value_v v = left_opt.value();

                if(get_check_fma(decl, v, 0, is_minus))
                    continue;
            }

            ///if a == (c * d)
            if(right_opt.has_value() && right_opt.value().type == MULTIPLY)
            {
                value_v v = right_opt.value();

                if(get_check_fma(decl, v, 1, is_minus))
                    continue;
            }
        }
        #endif

        #if 1
        for(int leidx=(int)local_emit.size() - 1; leidx >= 0; leidx--)
        {
            ///const float my_val = a;
            value_v& me = local_emit[leidx];

            using namespace dual_types::ops;

            if(me.type != DECLARE)
                continue;

            value_v& decl = me.args.at(2);

            if(decl.type != COMBO_PLUS)
                continue;

            if(decl.original_type != "float" && decl.original_type != "half")
                continue;

            std::vector<value_v> additions;
            std::vector<value_v> multiplications;
            std::vector<value_v> omultiplications;

            for(int i=0; i < (int)decl.args.size(); i++)
            {
                std::optional<value_v> which_opt = get_arg(decl, i, get_arg);

                if(!which_opt.has_value())
                {
                    additions.push_back(decl.args[i]);
                    continue;
                }

                if(which_opt.value().type == MULTIPLY)
                {
                    #ifdef SUPPRESS_HALF
                    if(is_indirectly_half(which_opt.value().args.at(0)) || is_indirectly_half(which_opt.value().args.at(1)))
                    {
                        additions.push_back(decl.args[i]);
                        continue;
                    }
                    #endif

                    multiplications.push_back(which_opt.value());
                    omultiplications.push_back(decl.args[i]);
                }
                else
                    additions.push_back(decl.args[i]);

                ///a + b + ... + (v1 * v2) + ... * f

                ///-> fma(v1, v2, a + b + ... + f)
            }

            if(multiplications.size() == 0)
                continue;

            std::optional<value_v> add;

            for(auto& v : additions)
            {
                if(!add)
                    add = v;
                else
                    add.value() += v;
            }

            if(add.has_value())
                add.value().group_associative_operators();

            bool is_first = true;

            std::vector<value_v> decls;

            std::optional<value_v> result = add;

            for(int kk=0; kk < multiplications.size(); kk++)
            {
                if(!result.has_value())
                {
                    result = multiplications[kk];
                    continue;
                }

                const value_v& mult = multiplications[kk];

                result = (mult.args[0] * mult.args[1]) + result.value();

                /*if(mult.args[1].is_constant())
                    result = fma(mult.args[0], mult.args[1], result.value());
                else
                    result = fma(mult.args[1], mult.args[0], result.value());*/
            }

            assert(result.has_value());

            decl = result.value();

            for(int declid=0; declid < (int)decls.size(); declid++)
            {
                local_emit.insert(local_emit.begin() + leidx, decls[declid]);
            }
        }
        #endif

        //#define SUBSTITUTE_UP
        #ifdef SUBSTITUTE_UP
        auto pull_args = [&](value_v& v)
        {
            if(v.type != dual_types::ops::MULTIPLY && v.type != dual_types::ops::PLUS && v.type != dual_types::ops::COMBO_PLUS)
                return;

            for(int i=0; i < (int)v.args.size(); i++)
            {
                auto arg_opt = get_arg(v, i, get_arg);

                if(arg_opt.has_value())
                {
                    v.args[i] = arg_opt.value();
                }
            }
        };

        ///upwards substitute everything?
        for(int leidx = (int)local_emit.size() - 1; leidx >= 0; leidx--)
        {
            value_v& me = local_emit[leidx];

            using namespace dual_types::ops;

            if(me.type != DECLARE)
                continue;

            value_v& decl = me.args.at(2);

            if(decl.type != COMBO_PLUS)
                continue;

            if(decl.original_type != "float" && decl.original_type != "half")
                continue;

            pull_args(decl);

            int depth = 0;
            int max_depth = 2;

            auto lambda = [&]<typename T>(value_v& in, T&& func)
            {
                depth++;

                if(depth == max_depth)
                {
                    depth--;
                    return;
                }

                pull_args(in);

                for(auto& i : in.args)
                {
                    func(i, func);
                }

                depth--;
            };

            decl.recurse_lambda(lambda);
        }
        #endif

        if(block_id == blocks.size())
        for(int i=(int)local_emit.size() - 1; i >= 0; i--)
        {
            if(is_dep_free(local_emit, i))
            {
                //printf("Hello\n");

                assert(i >= 0 && i < (int)local_emit.size());

                local_emit.erase(local_emit.begin() + i);
                //continue;
            }
        }

        //if(block_id == blocks.size() && kernel_name == "evolve_1")
        //move_later(local_emit);

        std::vector<value> new_local_emit;

        for(const value_v& v : local_emit)
        {
            new_local_emit.push_back(v.reinterpret_as<value>());
        }

        local_emit.clear();
        emitted.clear();
        emitted_cache.clear();
        emitted_memory_requests.clear();
        linear_emitted.clear();

        for(auto& i : new_local_emit)
        {
            emit(i);
        }

        for(const auto& v : local_emit)
        {
            insert_value(v);
        }
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
