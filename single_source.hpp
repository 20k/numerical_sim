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

            std::vector<std::vector<std::pair<std::string, value>>> blocks;
            blocks.emplace_back();

            for(auto& [name, value] : ctx.sequenced)
            {
                ///control flow constructs are in their own dedicated segment
                if(dual_types::get_description(value.type).introduces_block)
                {
                    blocks.emplace_back();
                    blocks.back().push_back({name, value});
                    blocks.emplace_back();
                    continue;
                }

                ///just for simplicity
                if(dual_types::get_description(value.type).reordering_hazard)
                {
                    blocks.emplace_back();
                    blocks.back().push_back({name, value});
                    blocks.emplace_back();
                    continue;
                }

                blocks.back().push_back({name, value});
            }

            ///so. Need a getting_my_value_depends_on kind of a deal

            //std::map<value*, std::set<std::string>> depends_on;

            std::map<const value*, std::vector<std::string>> depends_on_unevaluable_constants;

            for(auto& block : blocks)
            {
                if(block.size() == 0)
                    continue;

                /*std::set<std::string> block_variables;

                for(auto& [name, v] : block)
                {
                    v.get_all_variables_impl(block_variables);
                }*/

                std::set<const value*> block_variables;

                for(auto& [name, v] : block)
                {
                    v.recurse_variables([&](const value& in)
                    {
                        block_variables.insert(&in);
                    });
                }

                std::map<const value*, std::vector<const value*>> depends_on;

                for(auto& [name, v] : block)
                {
                    v.bottom_up_recurse([&](const value& in)
                    {
                        depends_on[&in];

                        for(const auto& i : in.args)
                        {
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

                std::map<const value*, int> most_in_demand;

                for(const auto& [source, depends] : depends_on)
                {
                    for(auto d : depends)
                    {
                        most_in_demand[d]++;
                    }
                }

                /*for(const auto& [constant, count] : most_in_demand)
                {
                    std::cout << type_to_string(*constant) << " " << count << std::endl;
                }*/

                std::map<const value*, int> most_unblocked;

                for(const auto& [source, depends] : depends_on)
                {
                    if(depends.size() == 1)
                    {
                        most_unblocked[depends.back()]++;
                    }
                }

                for(const auto& [constant, count] : most_unblocked)
                {
                    std::cout << type_to_string(*constant) << " " << count << std::endl;
                }

                /*for(auto& [name, v] : block)
                {
                    v.bottom_up_recurse([&](const value& in)
                    {
                        ///so: bottom up, applied to all arguments before applied to me
                        if(in.is_value())
                            return;

                        if(in.type == dual_types::ops::IDOT)
                            return;

                        int count = 0;

                        //for(const value& arg : in.args)

                        for(int count=0; count < (int)in.args.size(); count++)
                        {
                            if(count == 0 && (in.type == dual_types::ops::UNKNOWN_FUNCTION || in.type == dual_types::ops::DECLARE))
                                continue;

                            if(count == 1 && in.type == dual_types::ops::CONVERT)
                                continue;

                            if(count == 1 && in.type == dual_types::ops::DECLARE)
                                continue;

                            const value& arg = in.args[count];

                            if(arg.is_constant())
                                continue;

                            if(!arg.is_value())
                                continue;

                            ///arg is an unevaluable constant

                            depends_on_unevaluable_constants[&in].push_back(type_to_string(arg));
                        }
                    });
                }*/
            }

            /*for(auto& [ptr, vec] : depends_on_unevaluable_constants)
            {
                for(auto& k : vec)
                {
                    std::cout << "Ptr " << type_to_string(*ptr) << " Depends on " << k << std::endl;
                }
            }*/

            int block_id = 0;

            for(const auto& block : blocks)
            {
                base += "//" + std::to_string(block_id) + "\n";

                for(const auto& [name, value] : block)
                {
                    if(name == "")
                    {
                        if(dual_types::get_description(value.type).is_semicolon_terminated)
                            base += type_to_string(value) + ";\n";
                        else
                            base += type_to_string(value);
                    }
                    else
                    {
                        std::string type = value.original_type;

                        base += "const " + type + " " + name + " = " + type_to_string(value) + ";\n";
                    }
                }

                block_id++;
            }

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
