#ifndef EQUATION_CONTEXT_HPP_INCLUDED
#define EQUATION_CONTEXT_HPP_INCLUDED

#include <vector>
#include <utility>
#include <string>
#include <geodesic/dual_value.hpp>
#include <vec/tensor.hpp>
#include "differentiator.hpp"

template<typename T, int N>
inline
T buffer_read_linear(const buffer<T, N>& buf, const v3f& position, const v3i& dim);

template<typename T, int N>
inline
T buffer_index(const buffer<T, N>& buf, const tensor<value_i, 3>& pos, const tensor<value_i, 3>& dim);

struct equation_context;

value diff1(equation_context& ctx, const value& in, int idx);
value diff1(equation_context& ctx, const buffer<value, 3>& in, int idx, const v3i& where, const value& scale);
value diff2(equation_context& ctx, const value& in, int idx, int idy, const value& first_x, const value& first_y);

struct equation_context : differentiator
{
    std::vector<std::pair<std::string, value>> values;
    std::vector<std::pair<std::string, value>> temporaries;
    std::vector<std::pair<std::string, value>> sequenced;
    std::vector<std::pair<value, value>> aliases;
    bool uses_linear = false;
    bool debug = false;
    bool use_precise_differentiation = true;
    bool always_directional_derivatives = false;
    bool is_derivative_free = false;

    int order = 2;

    virtual value diff1(const value& in, int idx) override {return ::diff1(*this, in, idx);};
    virtual value diff1(const buffer<value, 3>& in, int idx, const v3i& where, const value& scale) override {return ::diff1(*this, in, idx, where, scale);};
    virtual value diff2(const value& in, int idx, int idy, const value& dx, const value& dy) override {return ::diff2(*this, in, idx, idy, dx, dy);};

    void exec(const value& v)
    {
        sequenced.push_back({"", v});
    }

    ////errrr this is starting to become a real type mess
    template<typename T>
    void exec(const value_base<T>& v)
    {
        return exec(v.template as<float>());
    }

    void pin(value& v)
    {
        for(auto& i : temporaries)
        {
            if(dual_types::equivalent<float>(v, i.first) || dual_types::equivalent<float>(v, i.second))
            {
                value facade;
                facade.make_value(i.first);

                v = facade;
                return;
            }
        }

        std::string name = "pv" + std::to_string(temporaries.size());
        //std::string name = "pv[" + std::to_string(temporaries.size()) + "]";

        value old = v;

        temporaries.push_back({name, old});
        sequenced.push_back({name, old});

        value facade;
        facade.make_value(name);

        v = facade;
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
        sequenced.push_back({name, v});
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

        std::vector<std::pair<std::string, value>> unprocessed_temporaries = temporaries;

        bool any_change = true;

        while(any_change && unprocessed_temporaries.size() > 0)
        {
            any_change = false;

            for(int i=0; i < (int)unprocessed_temporaries.size(); i++)
            {
                std::pair<std::string, value>& next = unprocessed_temporaries[i];

                //unprocessed_temporaries.erase(unprocessed_temporaries.begin());

                if(used_names.find(next.first) != used_names.end())
                {
                    std::vector<std::string> all_used = next.second.get_all_variables();

                    used_names.insert(next.first);

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
            to_erase.insert(i.first);
        }

        for(int i=0; i < (int)temporaries.size(); i++)
        {
            if(to_erase.find(temporaries[i].first) != to_erase.end())
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

        for(auto& [name, v] : temporaries)
        {
            v.substitute(aliases);
        }

        for(auto& [name, v] : sequenced)
        {
            v.substitute(aliases);
        }
    }

    void build_impl(std::string& argument_string, const std::string& str)
    {
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

        for(auto& [current_name, value] : temporaries)
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
                value_i dim = v.args[5].as<int>();

                ///buffer[z * dim.x * dim.y + y * dim.x + x];
                value indexed = buf[z * dim.x() * dim.y() + y * dim.x() + x];

                v = indexed;
            }

            if(function_name == "buffer_read_linear")
            {
                buffer<value, 3> buf;
                buf.name = type_to_string(v.args[1]);

                ///to_sub = apply(value(function_name), variables.args[1], as_float3(xs[kk], ys[kk], zs[kk]), "dim");

                value index_vars = v.args[2];
                value dim = v.args[3];

                assert(type_to_string(index_vars.args.at(0)) == "(float3)");

                value x = index_vars.args.at(1);
                value y = index_vars.args.at(2);
                value z = index_vars.args.at(3);

                value indexed = buffer_read_linear(buf, (v3f){x, y, z}, (v3i){dim.x().as<int>(), dim.y().as<int>(), dim.z().as<int>()});

                v = indexed;
            }
        };

        for(auto& [_, v] : sequenced)
        {
            v.recurse_arguments(substitute);
        }

        for(auto& [_, v] : aliases)
        {
            v.recurse_arguments(substitute);
        }

        for(auto& [_, v] : temporaries)
        {
            v.recurse_arguments(substitute);
        }
    }
};


#endif // EQUATION_CONTEXT_HPP_INCLUDED
