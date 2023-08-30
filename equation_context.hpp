#ifndef EQUATION_CONTEXT_HPP_INCLUDED
#define EQUATION_CONTEXT_HPP_INCLUDED

#include <vector>
#include <utility>
#include <string>
#include <geodesic/dual_value.hpp>
#include <vec/tensor.hpp>
#include "differentiator.hpp"
#include "single_source_fw.hpp"
#include "util.hpp"

struct equation_context;

value diff1(equation_context& ctx, const value& in, int idx);
//value diff1(equation_context& ctx, const buffer<value, 3>& in, int idx, const v3i& where, const value& scale);
value diff2(equation_context& ctx, const value& in, int idx, int idy, const value& first_x, const value& first_y);
value upwind(equation_context& ctx, const value& prefix, const value& in, int idx);

struct equation_context : differentiator
{
    std::vector<std::tuple<std::string, single_source::impl::kernel_context, equation_context>> functions;

    std::vector<std::pair<std::string, value>> values;
    std::vector<std::tuple<std::string, value, int>> temporaries;
    std::vector<int> can_cache_status;
    std::vector<value> sequenced;
    std::vector<std::pair<value, value>> aliases;
    bool uses_linear = false;
    bool debug = false;
    bool use_precise_differentiation = true;
    bool always_directional_derivatives = false;
    bool is_derivative_free = false;
    bool dynamic_drop = false;
    bool better_buffer_index = false;
    std::optional<vec3i> fixed_dim;

    int order = 2;

    int current_block_level = 0;

    virtual value diff1(const value& in, int idx) override {return ::diff1(*this, in, idx);};
    //virtual value diff1(const buffer<value, 3>& in, int idx, const v3i& where, const value& scale) override {return ::diff1(*this, in, idx, where, scale);};
    virtual value diff2(const value& in, int idx, int idy, const value& dx, const value& dy) override {return ::diff2(*this, in, idx, idy, dx, dy);};

    virtual value upwind(const value& prefix, const value& in, int idx) override {return ::upwind(*this, prefix, in, idx);}

    template<typename T>
    void add_function(const std::string& name, const T& func)
    {
        equation_context ectx;

        single_source::impl::kernel_context kctx;
        kctx.is_func = true;
        single_source::impl::setup_kernel(kctx, ectx, func);

        functions.push_back({name, kctx, ectx});
    }

    void exec(const value& v)
    {
        if(v.type == dual_types::ops::BLOCK_START)
            current_block_level++;

        if(v.type == dual_types::ops::BLOCK_END)
            current_block_level--;

        sequenced.push_back(v);
    }

    ////errrr this is starting to become a real type mess
    template<typename T>
    void exec(const value_base<T>& v)
    {
        return exec(v.template reinterpret_as<value_base<float>>());
    }

    template<typename T>
    void exec(const value_base_mut<T>& v)
    {
        return exec(v.template reinterpret_as<value_base<float>>());
    }

    template<typename T, int N>
    void exec(const tensor<T, N>& v)
    {
        for(int i=0; i < N; i++)
        {
            exec(v[i]);
        }
    }

    void pin(value& v)
    {
        ///so: case 1
        ///expr1 = ix + 1
        ///ix = ix + 1
        ///expr2 = ix + 1
        ///expr2 should not equal expr1
        ///
        ///expr1 = ix + 1
        ///for(idx) {
        ///ix = ix + 1
        ///}
        ///expr2 = ix + 1
        ///expr2 != expr1
        ///expr3 = ix + 1
        ///expr2 == expr3
        ///ok so. Types must be declared as mutable or not mutable
        ///only mutable types can be mutated, obviously
        ///if an expression contains a mutable type, it can't be cached

        bool can_cache = true;

        v.recurse_arguments([&](const value& in)
        {
            if(in.is_mutable)
                can_cache = false;
        });

        if(can_cache)
        {
            for(int i=0; i < (int)temporaries.size(); i++)
            {
                auto& [name, val, level] = temporaries[i];

                if(!can_cache_status[i])
                    continue;

                if(dual_types::equivalent<float>(v, name) || dual_types::equivalent<float>(v, val))
                {
                    value facade;
                    facade.make_value(name);

                    v = facade;
                    return;
                }
            }
        }


        std::string name = "pv" + std::to_string(temporaries.size());
        //std::string name = "pv[" + std::to_string(temporaries.size()) + "]";

        value old = v;

        temporaries.push_back({name, old, current_block_level});
        can_cache_status.push_back(can_cache);

        declare(*this, old, name);

        value facade;
        facade.make_value(name);

        v = facade;
    }

    void pin(dual& d)
    {
        pin(d.real);
        pin(d.dual);
    }

    template<typename T, int N>
    void pin(tensor<T, N>& mT)
    {
        for(int i=0; i < N; i++)
        {
            pin(mT.idx(i));
        }
    }

    template<typename T, int N>
    void pin(vec<N, T>& mT)
    {
        for(int i=0; i < N; i++)
        {
            pin(mT[i]);
        }
    }

    template<typename T, int N>
    void pin(tensor<T, N, N>& mT)
    {
        for(int i=0; i < N; i++)
        {
            for(int j=0; j < N; j++)
            {
                pin(mT.idx(i, j));
            }
        }
    }

    template<typename T, int N>
    void pin(tensor<T, N, N, N>& mT)
    {
        for(int i=0; i < N; i++)
        {
            for(int j=0; j < N; j++)
            {
                for(int k=0; k < N; k++)
                {
                    pin(mT.idx(i, j, k));
                }
            }
        }
    }

    template<typename T, int N>
    void pin(inverse_metric<T, N, N>& mT)
    {
        for(int i=0; i < N; i++)
        {
            for(int j=0; j < N; j++)
            {
                pin(mT.idx(i, j));
            }
        }
    }

    template<typename T, int N>
    void pin(metric<T, N, N>& mT)
    {
        for(int i=0; i < N; i++)
        {
            for(int j=0; j < N; j++)
            {
                pin(mT.idx(i, j));
            }
        }
    }

    template<typename T, std::size_t N>
    void pin(std::array<T, N>& arr)
    {
        for(int i=0; i < (int)N; i++)
        {
            pin(arr[i]);
        }
    }

    /*template<typename T>
    void pin(T& mT)
    {
        pin(mT.to_concrete());
    }*/

    void alias(const value& concrete, const value& alias)
    {
        for(auto& i : aliases)
        {
            if(dual_types::equivalent(concrete, i.first))
            {
                //std::cout << "CONC " << type_to_string(concrete)  << " ALIAS " << type_to_string(alias) << std::endl;
                //std::cout << "ALIAS " << type_to_string(alias) << " WITH " << type_to_string(i.second) << std::endl;

                assert(dual_types::equivalent(alias, i.second));
                return;
            }
        }

        aliases.push_back({concrete, alias});
    }

    void add(const std::string& name, const value& v)
    {
        values.push_back({name, v});
        declare(*this, v, name);
    }

    void strip_unused()
    {
        std::set<std::string> used_names;

        for(auto& i : values)
        {
            auto& name = i.first;
            value& v = i.second;

            used_names.insert(name);

            std::vector<std::string> all_used = v.get_all_variables();

            for(auto& k : all_used)
            {
                used_names.insert(k);
            }
        }

        std::vector<std::tuple<std::string, value, int>> unprocessed_temporaries = temporaries;

        bool any_change = true;

        while(any_change && unprocessed_temporaries.size() > 0)
        {
            any_change = false;

            for(int i=0; i < (int)unprocessed_temporaries.size(); i++)
            {
                std::tuple<std::string, value, int>& next = unprocessed_temporaries[i];

                //unprocessed_temporaries.erase(unprocessed_temporaries.begin());

                if(used_names.find(std::get<0>(next)) != used_names.end())
                {
                    std::vector<std::string> all_used = std::get<1>(next).get_all_variables();

                    used_names.insert(std::get<0>(next));

                    for(auto& kk : all_used)
                    {
                        used_names.insert(kk);
                    }

                    any_change = true;

                    unprocessed_temporaries.erase(unprocessed_temporaries.begin() + i);
                    i--;
                    continue;
                }
            }
        }

        std::set<std::string> to_erase;

        for(auto& i : unprocessed_temporaries)
        {
            to_erase.insert(std::get<0>(i));
        }

        for(int i=0; i < (int)temporaries.size(); i++)
        {
            if(to_erase.find(std::get<0>(temporaries[i])) != to_erase.end())
            {
                temporaries.erase(temporaries.begin() + i);
                i--;
                continue;
            }
        }
    }

    void substitute_aliases()
    {
        for(auto& [name, v] : values)
        {
            v.substitute(aliases);
        }

        for(auto& [name, v, level] : temporaries)
        {
            v.substitute(aliases);
        }

        for(auto& v : sequenced)
        {
            v.substitute(aliases);
        }
    }

    void build_impl(std::string& argument_string, const std::string& str)
    {
        assert(current_block_level == 0);

        strip_unused();
        substitute_aliases();

        for(auto& i : values)
        {
            std::string str = "-D" + i.first + "=" + type_to_string(i.second) + " ";

            argument_string += str;
        }

        if(temporaries.size() == 0)
        {
            argument_string += "-DTEMPORARIES" + str + "=DUMMY ";
            return;
        }

        std::string temporary_string;

        for(auto& [current_name, value, level] : temporaries)
        {
            temporary_string += current_name + "=" + type_to_string(value) + ",";
        }

        ///remove trailing comma
        if(temporary_string.size() > 0)
            temporary_string.pop_back();

        argument_string += "-DTEMPORARIES" + str + "=" + temporary_string + " ";
    }

    void build(std::string& argument_string, const std::string& str)
    {
        int old_length = argument_string.size();

        build_impl(argument_string, str);

        int new_length = argument_string.size();

        std::cout << "EXTRA LENGTH " << (new_length - old_length) << " " << str << std::endl;
    }

    void build(std::string& argument_string, int idx)
    {
        build(argument_string, std::to_string(idx));
    }

    void fix_buffers()
    {
        //#define SUB_IN_INDEXING
        #ifdef SUB_IN_INDEXING
        auto substitute = [](value& v)
        {
            if(v.type != dual_types::ops::UNKNOWN_FUNCTION)
                return;

            std::string function_name = type_to_string(v.args[0]);

            if(function_name == "buffer_index" || function_name == "buffer_indexh")
            {
                buffer<value, 3> buf;
                buf.name = type_to_string(v.args[1]);

                value_i x = v.args[2].convert<int>();
                value_i y = v.args[3].convert<int>();
                value_i z = v.args[4].convert<int>();
                value_i dim = v.args[5].reinterpret_as<value_i>();

                ///buffer[z * dim.x * dim.y + y * dim.x + x];
                value indexed = buf[z * dim.x() * dim.y() + y * dim.x() + x];
                indexed.is_memory_access = true;

                v = indexed;
            }

            if(function_name == "buffer_index_2" || function_name == "buffer_indexh_2")
            {
                buffer<value, 3> buf;
                buf.name = type_to_string(v.args[1]);

                value_i index = v.args[2].reinterpret_as<value_i>();

                value indexed = buf[index];
                indexed.is_memory_access = true;

                v = indexed;
            }
        };

        for(auto& v : sequenced)
        {
            v.recurse_arguments(substitute);
        }

        for(auto& [_, v] : aliases)
        {
            v.recurse_arguments(substitute);
        }

        for(auto& [_, v, _2] : temporaries)
        {
            v.recurse_arguments(substitute);
        }
        #endif
    }
};


#endif // EQUATION_CONTEXT_HPP_INCLUDED
