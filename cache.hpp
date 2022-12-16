#ifndef CACHE_HPP_INCLUDED
#define CACHE_HPP_INCLUDED

#include <toolkit/opencl.hpp>
#include <string>
#include <toolkit/fs_helpers.hpp>

template<typename T>
inline void hash_combine(std::size_t& seed, const T& v)
{
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}

inline
cl::program build_program_with_cache(cl::context& clctx, const std::string& filename, const std::string& options)
{
    std::string file_data = file::read(filename, file::mode::BINARY);

    std::optional<cl::program> prog_opt;

    bool needs_cache = false;

    std::size_t hsh = 0;

    hash_combine(hsh, options);
    hash_combine(hsh, file_data);

    file::mkdir("cache");

    std::string name = filename + "_" + std::to_string(hsh);

    if(file::exists("cache/" + name))
    {
        std::string bin = file::read("cache/" + name, file::mode::BINARY);

        prog_opt.emplace(clctx, bin, cl::program::binary_tag{});
    }
    else
    {
        prog_opt.emplace(clctx, file_data, false);

        needs_cache = true;
    }

    cl::program& t_program = prog_opt.value();

    t_program.build(clctx, options);

    if(needs_cache)
    {
        t_program.ensure_built();

        file::write("cache/" + name, prog_opt.value().get_binary(), file::mode::BINARY);
    }

    return t_program;
}

#endif // CACHE_HPP_INCLUDED
