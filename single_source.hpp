#ifndef SINGLE_SOURCE_HPP_INCLUDED
#define SINGLE_SOURCE_HPP_INCLUDED

#include <string>
#include "equation_context.hpp"
#include <geodesic/dual_value.hpp>

namespace single_source
{
    namespace impl
    {
        template<typename T>
        inline
        std::string name_type()
        {
            if constexpr(std::is_same_v<T, float>)
            {
                return "float";
            }

            else if constexpr(std::is_same_v<T, double>)
            {
                return "double";
            }

            else if constexpr(std::is_same_v<T, std::float16_t>)
            {
                return "half";
            }

            else if constexpr(std::is_same_v<T, int>)
            {
                return "int";
            }

            else if constexpr(std::is_same_v<T, short>)
            {
                return "short";
            }
        }


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

        template<typename T, int N>
        inline
        void add(const buffer<T, N>& buf, std::vector<input>& result);

        template<typename T>
        inline
        void add(const literal<T>& lit, std::vector<input>& result);

        template<typename T, int N>
        inline
        void add(const tensor<T, N>& ten, std::vector<input>& result);

        template<typename T, int N>
        inline
        void add(buffer<T, N>& buf, std::vector<input>& result)
        {
            input in;
            in.type = name_type<typename T::value_type>();
            in.pointer = true;

            std::string name = "buf" + std::to_string(result.size());

            in.name = name;
            buf.name = name;

            result.push_back(in);
        }

        template<typename T>
        inline
        void add(literal<T>& lit, std::vector<input>& result)
        {
            input in;
            in.type = name_type<typename T::value_type>();
            in.pointer = false;

            std::string name = "lit" + std::to_string(result.size());

            in.name = name;
            lit.name = name;

            result.push_back(in);
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
                in.type = name_type<typename T::value_type>();
                in.pointer = false;

                std::string name = "ten" + std::to_string(result.size());

                in.name = name;
                i.name = name;

                result.push_back(in);
            }
        }

        template<typename T>
        inline
        void add(const T&, std::vector<input>& result)
        {

        }

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
        std::string generate_kernel_string(kernel_context& kctx, equation_context& ctx)
        {
            std::string base = "__kernel void kernel_name(";

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
                    std::string type = name_type<decltype(value)::value_type>();

                    base += type + " " + name + " = " + type_to_string(value) + ";\n";
                }
            }

            base += "\n}";

            return base;
        }

        inline
        cl::kernel generate_kernel(cl::context& clctx, kernel_context& kctx, equation_context& ctx)
        {
            std::string str = generate_kernel_string(kctx, ctx);

            file::mkdir("generated");

            std::string name = "generated/" + std::to_string(std::hash<std::string>()(str)) + ".cl";

            file::write(name, str, file::mode::BINARY);

            cl::program prog = build_program_with_cache(clctx, name, "-cl-std=CL1.2");

            return cl::kernel(prog, "kernel_name");
        }

        template<typename R, typename... Args>
        void setup_kernel(kernel_context& kctx, equation_context& ectx, R(*func)(equation_context&, Args...))
        {
            std::tuple<equation_context&> a1 = {ectx};
            std::tuple<Args...> a2;

            std::apply([&](auto&&... args){
                (kctx.add(args), ...);
            }, a2);

            std::apply(func, std::tuple_cat(a1, a2));
        }
    }


    template<typename T>
    inline
    cl::kernel make_kernel_for(cl::context& clctx, equation_context& ectx, T&& func)
    {
        impl::kernel_context kctx;
        impl::setup_kernel(kctx, ectx, func);

        return impl::generate_kernel(clctx, kctx, ectx);
    }

}

#endif // SINGLE_SOURCE_HPP_INCLUDED
