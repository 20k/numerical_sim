#ifndef SINGLE_SOURCE_KERNEL_CONTEXT_HPP_INCLUDED
#define SINGLE_SOURCE_KERNEL_CONTEXT_HPP_INCLUDED

#include <vector>
#include <vec/value.hpp>
#include <vec/tensor.hpp>

struct equation_context;

namespace single_source
{
    namespace impl
    {
        ///https://vector-of-bool.github.io/2021/10/22/string-templates.html
        template<size_t N>
        struct fixed_string
        {
            char _chars[N+1] = {};

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

        struct type_storage
        {
            std::vector<input> args;
        };

        struct kernel_context
        {
            bool is_func = false;

            type_storage inputs;
            type_storage ret;
        };

        struct input
        {
            std::vector<type_storage> defines_structs;

            std::string type;
            bool pointer = false;
            bool is_struct = false;
            bool is_constant = false;
            std::string name;

            std::string format()
            {
                std::string str = "";

                if(is_struct)
                    str = "struct ";

                if(pointer)
                {
                    std::string cst = is_constant ? " const " : "";

                    return "__global " + cst + str + type + "* __restrict__ " + name;
                }
                else
                {
                    return type + " " + str + name;
                }
            }
        };

        template<typename R, typename... Args>
        void setup_kernel(kernel_context& kctx, equation_context& ectx, R(*func)(equation_context&, Args...));
    }

    template<typename T>
    struct struct_base
    {
        using is_struct = std::true_type;

        std::vector<std::string> names;

        template<typename U>
        void iterate_ext(U&& u)
        {
            auto tup = static_cast<T*>(this)->as_tuple();

            std::apply([&](auto& ...x){(..., u(x));}, tup);
        }
    };

    template<typename T>
    concept Structy = T::is_struct::value;

    ///all of this parse_tensor stuff needs to be fixed well in advanced of being a compiler
    template<typename U>
    requires(!Structy<U>)
    inline U parse_tensor(U& tag, value_i op, bool is_memory_access)
    {
        auto result = op.reinterpret_as<U>();
        result.is_mutable = tag.is_mutable;
        result.original_type = tag.original_type;
        result.is_memory_access = is_memory_access;
        return result;
    }

    template<typename U, int N>
    inline tensor<U, N> parse_tensor(tensor<U, N>& tag, value_i op, bool is_memory_access)
    {
        tensor<U, N> ret;

        for(int i=0; i < N; i++)
        {
            ret[i] = op.index(i).reinterpret_as<U>();
            ret[i].is_mutable = tag[i].is_mutable;
            ret[i].original_type = tag[i].original_type;
            ret[i].is_memory_access = is_memory_access;
        }

        return ret;
    }

    template<typename T>
    requires(Structy<T>)
    inline T parse_tensor(T& val, value_i op, bool unused)
    {
        T result;

        ///so op either contains "name", or name[index]

        int name_offset = 0;

        auto assign_names = [&](auto& in)
        {
            std::string name = val.names[name_offset];

            ///the double value_type is because a struct is built of literal<>s
            //in = op.property(name).as<decltype(in)::value_type::value_type>();

            in.name = type_to_string(op.property(name));

            name_offset++;
        };

        result.iterate_ext(assign_names);

        return result;
    }

    template<typename T>
    struct literal
    {
        using value_type = T;
        T storage = T{};

        bool permanent_name = false;
        std::string name;

        literal(){}
        literal(const std::string& str) : name(str){}
        literal(const char* str) : name(str){}

        T get()
        {
            value_i op(name);

            return parse_tensor(storage, op, false);
        }

        operator T()
        {
            return get();
        }
    };

    template<typename T>
    struct buffer
    {
        using value_type = T;
        T storage = T{};

        bool permanent_name = false;
        std::string name;

        buffer(){}
        buffer(const std::string& str) : name(str){}
        buffer(const char* str) : name(str){}

        ///detect accidental usage of operator,
        template<typename... U, typename V, typename W, typename X>
        T operator[](U&&... in, V&& in2, W&& in3, X&& in4)
        {
            static_assert(false);
            return T();
        }

        T operator[](const value_i& in)
        {
            value_i op = make_op<int>(dual_types::ops::BRACKET2, value_i(name), in);

            return parse_tensor(storage, op, true);
        }

        template<int N>
        T operator[](const tensor<value_i, N>& pos, const tensor<value_i, N>& dim)
        {
            static_assert(N == 3 || N == 4);

            if constexpr(N == 3)
            {
                value_i op = make_op<int>(dual_types::ops::BRACKET2, value_i(name), pos.x(), pos.y(), pos.z(), dim.x(), dim.y(), dim.z());

                return parse_tensor(storage, op, true);
            }

            else if constexpr(N == 4)
            {
                value_i op = make_op<int>(dual_types::ops::BRACKET2, value_i(name), pos.x(), pos.y(), pos.z(), pos.w(), dim.x(), dim.y(), dim.z(), dim.w());

                return parse_tensor(storage, op, true);
            }
        }

        /*T assign(const T& location, const T& what)
        {
            return make_op<typename T::value_type>(dual_types::ops::ASSIGN, location, what);
        }*/
    };

    template<typename T, impl::fixed_string _name>
    struct named_buffer : buffer<T>
    {
        named_buffer()
        {
            buffer<T>::name = _name.get();
            buffer<T>::permanent_name = true;
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
}

template<typename T>
using buffer = single_source::buffer<T>;
template<typename T>
using literal = single_source::literal<T>;

#endif // SINGLE_SOURCE_KERNEL_CONTEXT_HPP_INCLUDED
