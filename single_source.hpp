#ifndef SINGLE_SOURCE_HPP_INCLUDED
#define SINGLE_SOURCE_HPP_INCLUDED

#include <string>
#include "equation_context.hpp"
#include <geodesic/dual_value.hpp>
#include <stdfloat>
#include "cache.hpp"
#include "single_source_fw.hpp"
#include <thread>

namespace single_source
{
    namespace impl
    {
        struct input;
    }

    /*struct function_args
    {
        virtual void call(std::vector<impl::input>& result)
        {
            assert(false);
        }
    };*/

    #define TOUCH_FUNCTION_ARG(x) single_source::impl::add(x, result)

    namespace impl
    {
        template<typename T, int N>
        requires(Structy<T>)
        inline
        void add(buffer<T, N>& buf, type_storage& result);

        template<typename T, int N>
        requires(!Structy<T>)
        inline
        void add(buffer<T, N>& buf, type_storage& result);

        template<typename T, int N, size_t fN, fixed_string<fN> str>
        inline
        void add(named_buffer<T, N, str>& buf, type_storage& result);

        template<typename T>
        inline
        void add(literal<T>& lit, type_storage& result);

        template<typename T, size_t fN, fixed_string<fN> str>
        inline
        void add(named_literal<T, str>& lit, type_storage& result);

        template<typename T, int N>
        inline
        void add(tensor<T, N>& ten, type_storage& result);

        template<typename T, std::size_t N>
        inline
        void add(std::array<T, N>& buf, type_storage& result);

        template<typename T, int buffer_N, std::size_t array_N, auto str>
        inline
        void add(std::array<named_buffer<T, buffer_N, str>, array_N>& named_bufs, type_storage& result);

        template<typename T, std::size_t array_N, auto str>
        inline
        void add(std::array<named_literal<T, str>, array_N>& named_bufs, type_storage& result);

        template<typename T>
        inline
        void add(std::optional<T>& opt, type_storage& result);

        template<typename T, int N>
        requires(Structy<T>)
        inline
        void add(buffer<T, N>& buf, type_storage& result)
        {
            input in;
            in.type = T::type;
            in.pointer = true;
            in.is_struct = true;

            std::string name = buf.permanent_name ? buf.name : "s_name_" + std::to_string(result.args.size());

            in.name = name;
            buf.name = name;

            type_storage& s_info = in.defines_structs.emplace_back();

            auto do_assignment = [&](auto& v)
            {
                ::single_source::impl::add(v, s_info);
            };

            buf.storage.iterate_ext(do_assignment);

            for(auto& i : s_info.args)
            {
                buf.storage.names.push_back(i.name);
            }

            result.args.push_back(in);
        }

        template<typename T, int N>
        requires(!Structy<T>)
        inline
        void add(buffer<T, N>& buf, type_storage& result)
        {
            input in;
            in.type = dual_types::name_type(T());
            in.pointer = true;

            std::string name = buf.permanent_name ? buf.name : "buf" + std::to_string(result.args.size());

            in.name = name;
            buf.name = name;

            result.args.push_back(in);
        }

        template<typename T, int N, size_t fN, fixed_string<fN> str>
        inline
        void add(named_buffer<T, N, str>& buf, type_storage& result)
        {
            buffer<T, N>& lbuf = buf;

            add(lbuf, result);
        }

        template<typename T>
        inline
        void add(literal<T>& lit, type_storage& result)
        {
            input in;
            in.type = dual_types::name_type(T());
            in.pointer = false;

            std::string name = lit.permanent_name ? lit.name : "lit" + std::to_string(result.args.size());

            in.name = name;
            lit.name = name;

            result.args.push_back(in);
        }

        template<typename T, size_t fN, fixed_string<fN> str>
        inline
        void add(named_literal<T, str>& lit, type_storage& result)
        {
            literal<T>& llit = lit;

            add(llit, result);
        }

        template<typename T, int N>
        inline
        void add(tensor<T, N>& ten, type_storage& result)
        {
            for(auto& i : ten)
            {
                if(i.is_constant())
                    continue;

                input in;
                in.type = dual_types::name_type(T());
                in.pointer = false;

                std::string name = "ten" + std::to_string(result.args.size());

                in.name = name;
                i.name = name;

                result.args.push_back(in);
            }
        }

        template<typename T, std::size_t N>
        inline
        void add(std::array<T, N>& buf, type_storage& result)
        {
            for(int i=0; i < (int)N; i++)
            {
                add(buf[i], result);
            }
        }

        ///todo: named buffers
        /*template<typename T>
        inline
        void add(std::vector<T>& buf, type_storage& result)
        {
            for(int i=0; i < (int)buf.size(); i++)
            {
                add(buf[i], result);
            }
        }*/

        template<typename T, int buffer_N, std::size_t array_N, auto str>
        inline
        void add(std::array<named_buffer<T, buffer_N, str>, array_N>& named_bufs, type_storage& result)
        {
            for(int i=0; i < (int)array_N; i++)
            {
                input in;
                in.type = dual_types::name_type(T());
                in.pointer = true;

                std::string name = str.get() + std::to_string(i);

                in.name = name;
                named_bufs[i].name = name;

                result.args.push_back(in);
            }
        }

        template<typename T, std::size_t array_N, auto str>
        inline
        void add(std::array<named_literal<T, str>, array_N>& named_bufs, type_storage& result)
        {
            for(int i=0; i < (int)array_N; i++)
            {
                input in;
                in.type = dual_types::name_type(T());
                in.pointer = false;

                std::string name = str.get() + std::to_string(i);

                in.name = name;
                named_bufs[i].name = name;

                result.args.push_back(in);
            }
        }

        /*inline
        void add(function_args& args, type_storage& result)
        {
            args.call(result);
        }*/


        template<typename T>
        inline
        void add(std::optional<T>& opt, type_storage& result)
        {
            if(opt.has_value())
                add(opt.value(), result);
        }

        /*template<typename T>
        inline
        void add(T*& val, type_storage& result)
        {
            add(*val, result);
        }*/

        /*template<typename T>
        inline
        void add(const T&, type_storage& result)
        {
            static_assert(false);
        }*/
    }

    struct argument_generator;

    namespace impl
    {
        inline
        std::string make_struct(const std::string& type, type_storage& store)
        {
            std::string ret = "struct " + type + "{\n";

            for(input& in : store.args)
            {
                ret += "    " + in.format() + ";\n";
            }

            return ret + "\n};\n";
        }

        inline
        std::string generate_kernel_string(kernel_context& kctx, equation_context& ctx, const std::string& kernel_name, bool& any_uses_half)
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
                    /*v.recurse_variables([&](const value& in)
                    {
                        block_variables.insert(&in);
                    });*/

                    /*v.bottom_up_recurse([&](value& in)
                    {
                        if(in.type == dual_types::ops::IDOT)
                        {
                            insert_value(in);
                            return true;
                        }

                        if(in.type == dual_types::ops::RETURN || in.type == dual_types::ops::BREAK ||
                           in.type == dual_types::ops::FOR_START || in.type == dual_types::ops::IF_START ||
                           in.type == dual_types::ops::BLOCK_START || in.type == dual_types::ops::BLOCK_END)
                        {
                            insert_value(in);
                            return true;
                        }

                        if(in.type == dual_types::ops::UNKNOWN_FUNCTION)
                        {
                            //insert_value(in);
                            return true;
                        }

                        if(in.type == dual_types::ops::DECLARE || in.type == dual_types::ops::ASSIGN)
                        {
                            insert_value(in);
                            return true;
                        }

                        return false;
                    },
                    [&](value& in)
                    {
                        if(in.type == dual_types::ops::VALUE)
                            return;

                        auto [declare_op, val] = declare_raw(in, "genid" + std::to_string(gidx), in.is_mutable);

                        insert_value(declare_op);

                        in = val;

                        gidx++;
                    });
                }*/

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

                        if(in.type == dual_types::ops::UNKNOWN_FUNCTION)
                        {
                            ///treat like a constant
                            return;
                        }


                        /*if(in.type == dual_types::ops::DECLARE || in.type == dual_types::ops::ASSIGN)
                        {
                            insert_value(in);
                            return;
                        }*/

                        if(in.type == dual_types::ops::VALUE)
                            return;

                        for(int i=0; i < (int)in.args.size(); i++)
                        {
                            if(in.type == dual_types::ops::DECLARE && i < 2)
                                continue;

                            if(in.type == dual_types::ops::UNKNOWN_FUNCTION && i == 0)
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

        inline
        cl::kernel generate_kernel(const cl::context& clctx, kernel_context& kctx, equation_context& ctx, const std::string& kernel_name, const std::string& extra_args)
        {
            bool any_uses_half = false;

            std::string str = generate_kernel_string(kctx, ctx, kernel_name, any_uses_half);

            file::mkdir("generated");

            std::string name = "generated/" + std::to_string(std::hash<std::string>()(str)) + ".cl";

            file::write(name, str, file::mode::BINARY);

            cl::program prog = build_program_with_cache(clctx, name, "-cl-std=CL1.2 " + extra_args, kernel_name);

            return cl::kernel(prog, kernel_name);
        }

        template<typename R, typename... Args>
        void setup_kernel(kernel_context& kctx, equation_context& ectx, R(*func)(equation_context&, Args...))
        {
            std::tuple<equation_context&> a1 = {ectx};
            std::tuple<typename std::remove_reference<Args>::type...> a2;

            std::apply([&](auto&&... args){
                (impl::add(args, kctx.inputs), ...);
            }, a2);

            if constexpr(!std::is_same_v<R, void>)
            {
                R val = R();

                ::single_source::impl::add(val, kctx.ret);
            }

            std::tuple<Args...> a3 = a2;

            std::apply(func, std::tuple_cat(a1, a3));

            /*kctx = kernel_context();

            std::apply([&](auto&&... args){
                (kctx.add(args), ...);
            }, a3);*/
        }
    }

    struct argument_generator
    {
        impl::kernel_context kctx;

        template<typename T>
        T add()
        {
            T ret = T{};

            ::single_source::impl::add(ret, kctx.inputs);

            return ret;
        }

        template<typename T>
        void add(T& in)
        {
            ::single_source::impl::add(in, kctx.inputs);
        }

        void add(impl::input& in)
        {
            kctx.inputs.args.push_back(in);
        }

        void add(const std::string& prefix, impl::input& in)
        {
            auto v = in;
            v.name = prefix + v.name;

            add(v);
        }

        void add(std::vector<impl::input>& inputs)
        {
            for(auto& i : inputs)
            {
                add(i);
            }
        }

        void add(const std::string& prefix, std::vector<impl::input>& inputs)
        {
            for(auto& i : inputs)
            {
                add(prefix, i);
            }
        }

        template<typename... T>
        void add(T&... t)
        {
            (add(t), ...);
        }
    };

    /*struct combined_args : function_args
    {
        std::vector<function_args*> args;

        virtual void call(std::vector<impl::input>& result)
        {
            for(int i=0; i < (int)args.size(); i++)
            {
                impl::add(*args[i], result);
            }
        }
    };*/

    template<typename T, typename... U>
    inline
    cl::kernel make_dynamic_kernel_for(cl::context& clctx, equation_context& ectx, T&& func, const std::string& kernel_name, const std::string& extra_args = "", U&&... u)
    {
        argument_generator args;

        std::invoke(func, args, ectx, std::forward<U>(u)...);

        return impl::generate_kernel(clctx, args.kctx, ectx, kernel_name, extra_args);
    }

    template<typename T>
    inline
    cl::kernel make_kernel_for(cl::context& clctx, equation_context& ectx, T&& func, const std::string& kernel_name, const std::string& extra_args = "")
    {
        impl::kernel_context kctx;
        impl::setup_kernel(kctx, ectx, func);

        return impl::generate_kernel(clctx, kctx, ectx, kernel_name, extra_args);
    }

    template<typename T, typename... U>
    inline
    void make_async_dynamic_kernel_for(cl::context& clctx, T&& func, const std::string& kernel_name, const std::string& extra_args = "", U&&... u)
    {
        std::shared_ptr<cl::pending_kernel> pending = std::make_shared<cl::pending_kernel>();

        std::thread([=]() mutable
        {
            equation_context ectx;

            cl::kernel kern = single_source::make_dynamic_kernel_for(clctx, ectx, func, kernel_name, "", u...);

            pending->kernel = kern;
            pending->latch.count_down();
        }).detach();

        clctx.register_kernel(pending, kernel_name);
    }

    template<typename T>
    inline
    void make_async_kernel_for(cl::context& clctx, T&& func, const std::string& kernel_name, const std::string& extra_args = "")
    {
        std::shared_ptr<cl::pending_kernel> pending = std::make_shared<cl::pending_kernel>();

        std::thread([=]() mutable
        {
            equation_context ectx;

            cl::kernel kern = single_source::make_kernel_for(clctx, ectx, func, kernel_name, "");

            pending->kernel = kern;
            pending->latch.count_down();
        }).detach();

        clctx.register_kernel(pending, kernel_name);
    }
}

#endif // SINGLE_SOURCE_HPP_INCLUDED
