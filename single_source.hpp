#ifndef SINGLE_SOURCE_HPP_INCLUDED
#define SINGLE_SOURCE_HPP_INCLUDED

#include <string>
#include "equation_context.hpp"
#include <vec/value.hpp>
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
        template<typename T>
        requires(Structy<T>)
        inline
        void add(buffer<T>& buf, type_storage& result);

        template<typename T>
        requires(!Structy<T>)
        inline
        void add(buffer<T>& buf, type_storage& result);

        template<typename T, size_t fN, fixed_string<fN> str>
        inline
        void add(named_buffer<T, str>& buf, type_storage& result);

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

        template<typename T, std::size_t array_N, auto str>
        inline
        void add(std::array<named_buffer<T, str>, array_N>& named_bufs, type_storage& result);

        template<typename T, std::size_t array_N, auto str>
        inline
        void add(std::array<named_literal<T, str>, array_N>& named_bufs, type_storage& result);

        template<typename T>
        inline
        void add(std::optional<T>& opt, type_storage& result);

        template<typename T>
        bool is_mutable(const T&)
        {
            return T().is_mutable;
        }

        template<typename T, int... N>
        bool is_mutable(const tensor<T, N...>&)
        {
            return is_mutable(T());
        }

        template<typename T>
        requires(Structy<T>)
        inline
        void add(buffer<T>& buf, type_storage& result)
        {
            input in;
            in.type = T().type;
            in.pointer = true;
            in.is_struct = true;
            in.is_constant = false;

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

        template<typename T>
        requires(!Structy<T>)
        inline
        void add(buffer<T>& buf, type_storage& result)
        {
            input in;
            in.type = dual_types::name_type(T());
            in.pointer = true;
            in.is_constant = !is_mutable(T());

            std::string name = buf.permanent_name ? buf.name : "buf" + std::to_string(result.args.size());

            in.name = name;
            buf.name = name;

            result.args.push_back(in);
        }

        template<typename T, size_t fN, fixed_string<fN> str>
        inline
        void add(named_buffer<T, str>& buf, type_storage& result)
        {
            buffer<T>& lbuf = buf;

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

        template<typename T, std::size_t array_N, auto str>
        inline
        void add(std::array<named_buffer<T, str>, array_N>& named_bufs, type_storage& result)
        {
            for(int i=0; i < (int)array_N; i++)
            {
                input in;
                in.type = dual_types::name_type(T());
                in.pointer = true;
                in.is_constant = !is_mutable(T());

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

        std::string generate_kernel_string(kernel_context& kctx, equation_context& ctx, const std::string& kernel_name, bool& any_uses_half);

        inline
        cl::kernel generate_kernel(const cl::context& clctx, kernel_context& kctx, equation_context& ctx, const std::string& kernel_name, const std::string& extra_args)
        {
            bool any_uses_half = false;

            std::string str = generate_kernel_string(kctx, ctx, kernel_name, any_uses_half);

            file::mkdir("generated");

            std::string name = "generated/" + kernel_name + "_" + std::to_string(std::hash<std::string>()(str)) + ".cl";

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

            std::tuple<Args...> a3 = a2;

            if constexpr(!std::is_same_v<R, void>)
            {
                R val = std::apply(func, std::tuple_cat(a1, a3));;

                ::single_source::impl::add(val, kctx.ret);
            }
            else
            {
                std::apply(func, std::tuple_cat(a1, a3));
            }

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

            cl::kernel kern = single_source::make_dynamic_kernel_for(clctx, ectx, func, kernel_name, extra_args, u...);

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
