#ifndef UTIL_HPP_INCLUDED
#define UTIL_HPP_INCLUDED

#include <geodesic/dual_value.hpp>
#include "single_source_fw.hpp"

template<typename T, int N>
inline
T buffer_index(buffer<T, N> buf, const tensor<value_i, 3>& pos, const tensor<value_i, 3>& dim)
{
    return buf[pos.z() * dim.x() * dim.y() + pos.y() * dim.x() + pos.x()];
}

template<typename T, int N>
inline
T buffer_index(buffer<T, N> buf, const tensor<value_i, 4>& pos, const tensor<value_i, 4>& dim)
{
    return buf[pos.w() * dim.x() * dim.y() * dim.z() + pos.z() * dim.x() * dim.y() + pos.y() * dim.x() + pos.x()];
}

template<typename T, int N>
inline
value buffer_read_linear(buffer<T, N> buf, const v3f& position, const v3i& dim)
{
    auto clamped_read = [&](v3i in)
    {
        in = clamp(in, (v3i){0,0,0}, dim - 1);

        return (value)buffer_index(buf, in, dim);
    };

    v3f floored = floor(position);

    v3i ipos = (v3i)(floored);

    value c000 = clamped_read(ipos + (v3i){0,0,0});
    value c100 = clamped_read(ipos + (v3i){1,0,0});

    value c010 = clamped_read(ipos + (v3i){0,1,0});
    value c110 = clamped_read(ipos + (v3i){1,1,0});

    value c001 = clamped_read(ipos + (v3i){0,0,1});
    value c101 = clamped_read(ipos + (v3i){1,0,1});

    value c011 = clamped_read(ipos + (v3i){0,1,1});
    value c111 = clamped_read(ipos + (v3i){1,1,1});

    v3f frac = position - floored;

    value c00 = c000 - (value)frac.x() * (c000 - c100);
    value c01 = c001 - (value)frac.x() * (c001 - c101);

    value c10 = c010 - (value)frac.x() * (c010 - c110);
    value c11 = c011 - (value)frac.x() * (c011 - c111);

    value c0 = c00 - (value)frac.y() * (c00 - c10);
    value c1 = c01 - (value)frac.y() * (c01 - c11);

    return c0 - (value)frac.z() * (c0 - c1);
}

template<typename T, int N>
inline
value buffer_read_linear(buffer<T, N> buf, const v4f& position, const v4i& dim)
{
    auto clamped_read = [&](v4i in)
    {
        in = clamp(in, (v4i){0,0,0,0}, dim - 1);

        return (value)buffer_index(buf, in, dim);
    };

    v4f floored = floor(position);
    v4i ipos = (v4i)floored;

    v4f frac = position - floored;

    value linear_1 = 0;

    {
        value c000 = clamped_read(ipos + (v4i){0,0,0,0});
        value c100 = clamped_read(ipos + (v4i){1,0,0,0});

        value c010 = clamped_read(ipos + (v4i){0,1,0,0});
        value c110 = clamped_read(ipos + (v4i){1,1,0,0});

        value c001 = clamped_read(ipos + (v4i){0,0,1,0});
        value c101 = clamped_read(ipos + (v4i){1,0,1,0});

        value c011 = clamped_read(ipos + (v4i){0,1,1,0});
        value c111 = clamped_read(ipos + (v4i){1,1,1,0});

        value c00 = c000 - (value)frac.x() * (c000 - c100);
        value c01 = c001 - (value)frac.x() * (c001 - c101);

        value c10 = c010 - (value)frac.x() * (c010 - c110);
        value c11 = c011 - (value)frac.x() * (c011 - c111);

        value c0 = c00 - (value)frac.y() * (c00 - c10);
        value c1 = c01 - (value)frac.y() * (c01 - c11);

        linear_1 = c0 - (value)frac.z() * (c0 - c1);
    }

    value linear_2 = 0;

    {
        value c000 = clamped_read(ipos + (v4i){0,0,0,1});
        value c100 = clamped_read(ipos + (v4i){1,0,0,1});

        value c010 = clamped_read(ipos + (v4i){0,1,0,1});
        value c110 = clamped_read(ipos + (v4i){1,1,0,1});

        value c001 = clamped_read(ipos + (v4i){0,0,1,1});
        value c101 = clamped_read(ipos + (v4i){1,0,1,1});

        value c011 = clamped_read(ipos + (v4i){0,1,1,1});
        value c111 = clamped_read(ipos + (v4i){1,1,1,1});

        value c00 = c000 - (value)frac.x() * (c000 - c100);
        value c01 = c001 - (value)frac.x() * (c001 - c101);

        value c10 = c010 - (value)frac.x() * (c010 - c110);
        value c11 = c011 - (value)frac.x() * (c011 - c111);

        value c0 = c00 - (value)frac.y() * (c00 - c10);
        value c1 = c01 - (value)frac.y() * (c01 - c11);

        linear_2 = c0 - (value)frac.z() * (c0 - c1);
    }

    return linear_1 - (value)frac.w() * (linear_1 - linear_2);
}


template<typename T, int N>
inline
literal<value> buffer_read_linear_f(equation_context& ctx, buffer<T, N> buf, literal<v3f> position, literal<v4i> dim)
{
    ctx.order = 1;
    ctx.uses_linear = true;

    v3i idim = dim.get().xyz();
    v3f ipos = position.get();

    ctx.exec(return_v(buffer_read_linear(buf, ipos, idim)));

    literal<value> result;
    result.storage.is_memory_access = true;
    return result;
}

template<typename T, int N>
inline
literal<value> buffer_read_linear_f4(equation_context& ctx, buffer<T, N> buf, literal<v4f> position, literal<v4i> dim)
{
    ctx.order = 1;
    ctx.uses_linear = true;

    v4i idim = dim.get();
    v4f ipos = position.get();

    ctx.exec(return_v(buffer_read_linear(buf, ipos, idim)));

    literal<value> result;
    result.storage.is_memory_access = true;
    return result;
}

template<typename T, int N>
inline
literal<value> buffer_index_f(equation_context& ctx, buffer<T, N>& buf, literal<value_i> x, literal<value_i> y, literal<value_i> z, literal<v4i> dim)
{
    value v = (value)buf[z.get() * dim.get().x() * dim.get().y() + y.get() * dim.get().x() + x.get()];

    ctx.exec(return_v(v));

    literal<value> result;
    result.storage.is_memory_access = true;
    return result;
}

template<typename T, int N>
inline
literal<T> buffer_index_f2(equation_context& ctx, buffer<T, N>& buf, literal<value_i> index)
{
    T v = buf[index.get()];

    ctx.exec(return_v(v));

    literal<T> result;
    result.storage.is_memory_access = true;
    return result;
}

inline
value as_float3(const value& x, const value& y, const value& z)
{
    return "(float3)(" + type_to_string(x) + "," + type_to_string(y) + "," + type_to_string(z) + ")";
    //return dual_types::apply(value("(float3)"), x, y, z);
}

#endif // UTIL_HPP_INCLUDED
