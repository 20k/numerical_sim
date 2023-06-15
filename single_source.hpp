#ifndef SINGLE_SOURCE_HPP_INCLUDED
#define SINGLE_SOURCE_HPP_INCLUDED

#include <string>
#include "equation_context.hpp"
#include <geodesic/dual_value.hpp>
#include <stdfloat>
#include "cache.hpp"

namespace single_source
{
    namespace impl
    {
        ///https://vector-of-bool.github.io/2021/10/22/string-templates.html
        template<size_t N>
        struct fixed_string
        {
            char _chars[N+1] = {};

            //std::array<char, N+1> _chars = {};

            std::string get() const
            {
                return std::string(_chars, _chars + N);
            }

            constexpr fixed_string(){}

            constexpr fixed_string(const char (&arr)[N+1])
            {
                std::copy(arr, arr + N, _chars);
            }

            template<size_t M>
            constexpr fixed_string<N + M> operator+(const fixed_string<M>& other) const
            {
                fixed_string<N + M> result;

                for(size_t i=0; i < N; i++)
                {
                    result._chars[i] = _chars[i];
                }

                for(size_t i=0; i < M; i++)
                {
                    result._chars[i + N] = other._chars[i];
                }

                return result;
            }

            template<size_t M>
            constexpr fixed_string<N + M - 1> operator+(const char (&arr)[M]) const
            {
                fixed_string<N + M - 1> result;

                for(size_t i=0; i < N; i++)
                {
                    result._chars[i] = _chars[i];
                }

                for(size_t i=0; i < M-1; i++)
                {
                    result._chars[i + N] = arr[i];
                }

                return result;
            }
        };

        template<size_t N>
        fixed_string(const char (&arr)[N]) -> fixed_string<N-1>;  // Drop the null terminator

        struct input;
    }

    template<typename T, int dimensions, impl::fixed_string _name>
    struct named_buffer : buffer<T, dimensions>
    {
        named_buffer()
        {
            buffer<T, dimensions>::name = _name.get();
            buffer<T, dimensions>::permanent_name = true;
        }
    };

    template<typename T, impl::fixed_string _name>
    struct named_literal : literal<T>
    {
        named_literal()
        {
            literal<T>::name = _name.get();
            literal<T>::permanent_name = true;
        }
    };

    struct function_args
    {
        virtual void call(std::vector<impl::input>& result)
        {
            assert(false);
        }
    };

    #define TOUCH_FUNCTION_ARG(x) single_source::impl::add(x, result)

    namespace impl
    {
        struct input
        {
            std::string type;
            bool pointer = false;
            std::string name;

            std::string format()
            {
                if(pointer)
                {
                    return "__global " + type + "* " + name;
                }
                else
                {
                    return type + " " + name;
                }
            }
        };

        /*template<typename T, int N>
        inline
        void add(const buffer<T, N>& buf, std::vector<input>& result);

        template<typename T>
        inline
        void add(const literal<T>& lit, std::vector<input>& result);

        template<typename T, int N>
        inline
        void add(const tensor<T, N>& ten, std::vector<input>& result);*/

        template<typename T, int N>
        inline
        void add(buffer<T, N>& buf, std::vector<input>& result)
        {
            input in;
            in.type = dual_types::name_type(T());
            in.pointer = true;

            std::string name = buf.permanent_name ? buf.name : "buf" + std::to_string(result.size());

            in.name = name;
            buf.name = name;

            result.push_back(in);
        }

        template<typename T, int N, size_t fN, fixed_string<fN> str>
        inline
        void add(named_buffer<T, N, str>& buf, std::vector<input>& result)
        {
            buffer<T, N>& lbuf = buf;

            add(lbuf, result);
        }

        template<typename T>
        inline
        void add(literal<T>& lit, std::vector<input>& result)
        {
            input in;
            in.type = dual_types::name_type(T());
            in.pointer = false;

            std::string name = lit.permanent_name ? lit.name : "lit" + std::to_string(result.size());

            in.name = name;
            lit.name = name;

            result.push_back(in);
        }


        template<typename T, size_t fN, fixed_string<fN> str>
        inline
        void add(named_literal<T, str>& lit, std::vector<input>& result)
        {
            literal<T>& llit = lit;

            add(llit, result);
        }

        template<typename T, int N>
        inline
        void add(tensor<T, N>& ten, std::vector<input>& result)
        {
            for(auto& i : ten)
            {
                if(i.is_constant())
                    continue;

                input in;
                in.type = dual_types::name_type(T());
                in.pointer = false;

                std::string name = "ten" + std::to_string(result.size());

                in.name = name;
                i.name = name;

                result.push_back(in);
            }
        }

        template<typename T, std::size_t N>
        inline
        void add(std::array<T, N>& buf, std::vector<input>& result)
        {
            for(int i=0; i < (int)N; i++)
            {
                add(buf[i], result);
            }
        }

        ///todo: named buffers
        /*template<typename T>
        inline
        void add(std::vector<T>& buf, std::vector<input>& result)
        {
            for(int i=0; i < (int)buf.size(); i++)
            {
                add(buf[i], result);
            }
        }*/

        template<typename T, int buffer_N, std::size_t array_N, auto str>
        inline
        void add(std::array<named_buffer<T, buffer_N, str>, array_N>& named_bufs, std::vector<input>& result)
        {
            for(int i=0; i < (int)array_N; i++)
            {
                input in;
                in.type = dual_types::name_type(T());
                in.pointer = true;

                std::string name = str.get() + std::to_string(i);

                in.name = name;
                named_bufs[i].name = name;

                result.push_back(in);
            }
        }

        template<typename T, std::size_t array_N, auto str>
        inline
        void add(std::array<named_literal<T, str>, array_N>& named_bufs, std::vector<input>& result)
        {
            for(int i=0; i < (int)array_N; i++)
            {
                input in;
                in.type = dual_types::name_type(T());
                in.pointer = false;

                std::string name = str.get() + std::to_string(i);

                in.name = name;
                named_bufs[i].name = name;

                result.push_back(in);
            }
        }

        inline
        void add(function_args& args, std::vector<input>& result)
        {
            args.call(result);
        }

        template<typename T>
        inline
        void add(std::optional<T>& opt, std::vector<input>& result)
        {
            if(opt.has_value())
                add(opt.value(), result);
        }

        /*template<typename T>
        inline
        void add(T*& val, std::vector<input>& result)
        {
            add(*val, result);
        }*/

        /*template<typename T>
        inline
        void add(const T&, std::vector<input>& result)
        {
            static_assert(false);
        }*/

        struct kernel_context
        {
            std::vector<input> inputs;

            template<typename T>
            void add(T& t)
            {
                impl::add(t, inputs);
            }
        };

        inline
        std::string generate_kernel_string(kernel_context& kctx, equation_context& ctx, const std::string& kernel_name)
        {
            std::string base = "__kernel void " + kernel_name + "(";

            for(int i=0; i < (int)kctx.inputs.size(); i++)
            {
                base += kctx.inputs[i].format();

                if(i != (int)kctx.inputs.size() - 1)
                    base += ",";
            }

            base += ")\n{\n";

            ctx.strip_unused();
            ctx.substitute_aliases();

            for(auto& [name, value] : ctx.sequenced)
            {
                if(name == "")
                {
                    base += type_to_string(value) + ";\n";
                }
                else
                {
                    std::string type = dual_types::name_type(decltype(value)());

                    base += type + " " + name + " = " + type_to_string(value) + ";\n";
                }
            }

            base += "\n}";

            std::cout << base << std::endl;

            return base;
        }

        inline
        cl::kernel generate_kernel(cl::context& clctx, kernel_context& kctx, equation_context& ctx, const std::string& kernel_name, const std::string& extra_args)
        {
            std::string str = generate_kernel_string(kctx, ctx, kernel_name);

            file::mkdir("generated");

            std::string name = "generated/" + std::to_string(std::hash<std::string>()(str)) + ".cl";

            file::write(name, str, file::mode::BINARY);

            cl::program prog = build_program_with_cache(clctx, name, "-cl-std=CL1.2 " + extra_args);

            return cl::kernel(prog, kernel_name);
        }

        template<typename R, typename... Args>
        void setup_kernel(kernel_context& kctx, equation_context& ectx, R(*func)(equation_context&, Args...))
        {
            std::tuple<equation_context&> a1 = {ectx};
            std::tuple<typename std::remove_reference<Args>::type...> a2;

            std::apply([&](auto&&... args){
                (kctx.add(args), ...);
            }, a2);

            std::tuple<Args...> a3 = a2;

            std::apply(func, std::tuple_cat(a1, a3));

            kctx = kernel_context();

            std::apply([&](auto&&... args){
                (kctx.add(args), ...);
            }, a3);
        }
    }

    struct combined_args : function_args
    {
        std::vector<function_args*> args;

        virtual void call(std::vector<impl::input>& result)
        {
            for(int i=0; i < (int)args.size(); i++)
            {
                impl::add(*args[i], result);
            }
        }
    };

    template<typename T>
    inline
    cl::kernel make_kernel_for(cl::context& clctx, equation_context& ectx, T&& func, const std::string& kernel_name = "kernel_name", const std::string& extra_args = "")
    {
        impl::kernel_context kctx;
        impl::setup_kernel(kctx, ectx, func);

        return impl::generate_kernel(clctx, kctx, ectx, kernel_name, extra_args);
    }

}

#endif // SINGLE_SOURCE_HPP_INCLUDED
