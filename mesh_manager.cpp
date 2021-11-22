#include "mesh_manager.hpp"
#include <toolkit/opencl.hpp>
#include <execution>
#include <optional>

buffer_set::buffer_set(cl::context& ctx, vec3i size)
{
    for(int kk=0; kk < buffer_count; kk++)
    {
        buffers.emplace_back(ctx);
        buffers.back().alloc(size.x() * size.y() * size.z() * sizeof(cl_float));
    }
}

inline
std::pair<cl::buffer, int> generate_sponge_points(cl::context& ctx, cl::command_queue& cqueue, float scale, vec3i size)
{
    cl::buffer points(ctx);
    cl::buffer real_count(ctx);

    points.alloc(size.x() * size.y() * size.z() * sizeof(cl_ushort4));
    real_count.alloc(sizeof(cl_int));
    real_count.set_to_zero(cqueue);

    vec<4, cl_int> clsize = {size.x(), size.y(), size.z(), 0};

    cl::args args;
    args.push_back(points);
    args.push_back(real_count);
    args.push_back(scale);
    args.push_back(clsize);

    cqueue.exec("generate_sponge_points", args, {size.x(),  size.y(),  size.z()}, {8, 8, 1});

    std::vector<cl_ushort4> cpu_points = points.read<cl_ushort4>(cqueue);

    printf("Original sponge points %i\n", cpu_points.size());

    cl_int count = real_count.read<cl_int>(cqueue).at(0);

    assert(count > 0);

    cpu_points.resize(count);

    std::sort(std::execution::par_unseq, cpu_points.begin(), cpu_points.end(), [](const cl_ushort4& p1, const cl_ushort4& p2)
    {
        return std::tie(p1.s[2], p1.s[1], p1.s[0]) < std::tie(p2.s[2], p2.s[1], p2.s[0]);
    });

    cl::buffer real(ctx);
    real.alloc(cpu_points.size() * sizeof(cl_ushort4));
    real.write(cqueue, cpu_points);

    printf("Sponge point reduction %i\n", count);

    return {real, count};
}


inline
evolution_points generate_evolution_points(cl::context& ctx, cl::command_queue& cqueue, float scale, vec3i size)
{
    cl::buffer points_1(ctx);
    cl::buffer count_1(ctx);

    cl::buffer points_2(ctx);
    cl::buffer count_2(ctx);

    points_1.alloc(size.x() * size.y() * size.z() * sizeof(cl_ushort4));
    points_2.alloc(size.x() * size.y() * size.z() * sizeof(cl_ushort4));

    count_1.alloc(sizeof(cl_int));
    count_2.alloc(sizeof(cl_int));

    count_1.set_to_zero(cqueue);
    count_2.set_to_zero(cqueue);

    vec<4, cl_int> clsize = {size.x(), size.y(), size.z(), 0};

    cl::args args;
    args.push_back(points_1);
    args.push_back(count_1);
    args.push_back(points_2);
    args.push_back(count_2);
    args.push_back(scale);
    args.push_back(clsize);

    cqueue.exec("generate_evolution_points", args, {size.x(),  size.y(),  size.z()}, {8, 8, 1});

    std::vector<cl_ushort4> cpu_points_1 = points_1.read<cl_ushort4>(cqueue);
    std::vector<cl_ushort4> cpu_points_2 = points_2.read<cl_ushort4>(cqueue);

    printf("Original evolve points %i\n", cpu_points_1.size());

    cl_int cpu_count_1 = count_1.read<cl_int>(cqueue).at(0);
    cl_int cpu_count_2 = count_2.read<cl_int>(cqueue).at(0);

    assert(cpu_count_1 > 0);
    assert(cpu_count_2 > 0);

    cpu_points_1.resize(cpu_count_1);
    cpu_points_2.resize(cpu_count_2);

    std::sort(std::execution::par_unseq, cpu_points_1.begin(), cpu_points_1.end(), [](const cl_ushort4& p1, const cl_ushort4& p2)
    {
        return std::tie(p1.s[2], p1.s[1], p1.s[0]) < std::tie(p2.s[2], p2.s[1], p2.s[0]);
    });

    std::sort(std::execution::par_unseq, cpu_points_2.begin(), cpu_points_2.end(), [](const cl_ushort4& p1, const cl_ushort4& p2)
    {
        return std::tie(p1.s[2], p1.s[1], p1.s[0]) < std::tie(p2.s[2], p2.s[1], p2.s[0]);
    });

    cl::buffer shrunk_points_1(ctx);
    shrunk_points_1.alloc(cpu_points_1.size() * sizeof(cl_ushort4));
    shrunk_points_1.write(cqueue, cpu_points_1);

    cl::buffer shrunk_points_2(ctx);
    shrunk_points_2.alloc(cpu_points_2.size() * sizeof(cl_ushort4));
    shrunk_points_2.write(cqueue, cpu_points_2);

    evolution_points ret(ctx);
    ret.first_count = cpu_count_1;
    ret.second_count = cpu_count_2;

    ret.first_derivative_points = shrunk_points_1;
    ret.second_derivative_points = shrunk_points_2;

    printf("Evolve point reduction %i\n", cpu_count_1);

    return ret;
}


cl::buffer thin_intermediates_pool::request(cl::context& ctx, cl::command_queue& cqueue, int id, vec3i size, int element_size)
{
    for(const buffer_descriptor& desc : pool)
    {
        int my_size = size.x() * size.y() * size.z() * element_size;
        int desc_size = desc.size.x() * desc.size.y() * desc.size.z() * desc.element_size;

        if(desc.id == id && desc_size >= my_size)
        {
            return desc.buf;
        }
    }

    cl::buffer next(ctx);
    next.alloc(size.x() * size.y() * size.z() * element_size);

    #ifdef NANFILL
    cl_float nan = std::nanf("");
    next.fill(cqueue, nan);
    #else
    next.set_to_zero(cqueue);
    #endif // NANFILL

    buffer_descriptor& buf = pool.emplace_back(ctx);
    buf.id = id;
    buf.size = size;
    buf.element_size = element_size;
    buf.buf = next;

    return next;
}

cpu_mesh::cpu_mesh(cl::context& ctx, cl::command_queue& cqueue, vec3f _centre, vec3i _dim, cpu_mesh_settings _sett) :
        data{buffer_set(ctx, _dim), buffer_set(ctx, _dim)}, scratch{ctx, _dim}, points_set{ctx}, sponge_positions{ctx},
        momentum_constraint{ctx, ctx, ctx}, u_arg{ctx}
{
    centre = _centre;
    dim = _dim;
    sett = _sett;

    scale = calculate_scale(get_c_at_max(), dim);

    points_set = generate_evolution_points(ctx, cqueue, scale, dim);
    std::tie(sponge_positions, sponge_positions_count) = generate_sponge_points(ctx, cqueue, scale, dim);

    for(auto& i : momentum_constraint)
    {
        if(sett.calculate_momentum_constraint)
        {
            i.alloc(dim.x() * dim.y() * dim.z() * sizeof(cl_float));
            i.set_to_zero(cqueue);
        }
        else
        {
            i.alloc(sizeof(cl_int));
        }
    }
}

void cpu_mesh::flip()
{
    which_data = (which_data + 1) % 2;
}

buffer_set& cpu_mesh::get_input()
{
    return data[which_data];
}

buffer_set& cpu_mesh::get_output()
{
    return data[(which_data + 1) % 2];
}

buffer_set& cpu_mesh::get_scratch(int which)
{
    assert(which == 0);
    return scratch;
}

void cpu_mesh::init(cl::command_queue& cqueue, cl::buffer& in_u_arg)
{
    u_arg = in_u_arg;

    cl_int4 clsize = {dim.x(), dim.y(), dim.z(), 0};
    cl_float4 clmeshpos = {centre.x(), centre.y(), centre.z()};

    {
        cl::args init;

        for(auto& i : data[0].buffers)
        {
            init.push_back(i);
        }

        init.push_back(u_arg);
        init.push_back(scale);
        init.push_back(clsize);
        init.push_back(clmeshpos);

        cqueue.exec("calculate_initial_conditions", init, {dim.x(), dim.y(), dim.z()}, {8, 8, 1});
    }

    for(int i=0; i < (int)data[0].buffers.size(); i++)
    {
        cl::copy(cqueue, data[0].buffers[i], data[1].buffers[i]);
        cl::copy(cqueue, data[0].buffers[i], scratch.buffers[i]);
    }
}

cl::buffer cpu_mesh::get_thin_buffer(cl::context& ctx, cl::command_queue& cqueue, thin_intermediates_pool& pool, int id)
{
    if(sett.use_half_intermediates)
        return pool.request(ctx, cqueue, id, dim, sizeof(cl_half));
    else
        return pool.request(ctx, cqueue, id, dim, sizeof(cl_float));
}

///returns buffers and intermediates
std::pair<std::vector<cl::buffer>, std::vector<cl::buffer>> cpu_mesh::full_step(cl::context& ctx, cl::command_queue& cqueue, float timestep, thin_intermediates_pool& pool)
{
    auto buffer_to_index = [&](const std::string& name)
    {
        for(int idx = 0; idx < buffer_set::buffer_count; idx++)
        {
            if(buffer_names[idx] == name)
                return idx;
        }

        assert(false);
    };

    cl_int4 clsize = {dim.x(), dim.y(), dim.z(), 0};

    std::vector<cl::buffer>* last_valid_thin_buffer = &get_input().buffers;

    auto& base_yn = get_input().buffers;

    std::vector<cl::buffer> intermediates;

    auto step = [&](auto& generic_in, auto& generic_out, float current_timestep)
    {
        intermediates.clear();

        last_valid_thin_buffer = &generic_in;

        {
            auto differentiate = [&](const std::string& name, cl::buffer& out1, cl::buffer& out2, cl::buffer& out3)
            {
                int idx = buffer_to_index(name);

                cl::args thin;
                thin.push_back(points_set.first_derivative_points);
                thin.push_back(points_set.first_count);
                thin.push_back(generic_in[idx]);
                thin.push_back(out1);
                thin.push_back(out2);
                thin.push_back(out3);
                thin.push_back(scale);
                thin.push_back(clsize);

                cqueue.exec("calculate_intermediate_data_thin", thin, {points_set.first_count}, {128});
            };

            std::array buffers = {"cY0", "cY1", "cY2", "cY3", "cY4", "cY5",
                                  "gA", "gB0", "gB1", "gB2", "X"};

            for(int idx = 0; idx < (int)buffers.size(); idx++)
            {
                int i1 = idx * 3 + 0;
                int i2 = idx * 3 + 1;
                int i3 = idx * 3 + 2;

                cl::buffer b1 = get_thin_buffer(ctx, cqueue, pool, i1);
                cl::buffer b2 = get_thin_buffer(ctx, cqueue, pool, i2);
                cl::buffer b3 = get_thin_buffer(ctx, cqueue, pool, i3);

                differentiate(buffers[idx], b1, b2, b3);

                intermediates.push_back(b1);
                intermediates.push_back(b2);
                intermediates.push_back(b3);
            }
        }

        if(sett.calculate_momentum_constraint)
        {
            cl::args momentum_args;

            momentum_args.push_back(points_set.first_derivative_points);
            momentum_args.push_back(points_set.first_count);

            for(auto& i : generic_in)
            {
                momentum_args.push_back(i);
            }

            for(auto& i : momentum_constraint)
            {
                momentum_args.push_back(i);
            }

            momentum_args.push_back(scale);
            momentum_args.push_back(clsize);

            cqueue.exec("calculate_momentum_constraint", momentum_args, {points_set.first_count}, {128});
        }

        auto step_kernel = [&](const std::string& name)
        {
            cl::args a1;

            a1.push_back(points_set.second_derivative_points);
            a1.push_back(points_set.second_count);

            for(auto& i : generic_in)
            {
                a1.push_back(i);
            }

            for(auto& i : generic_out)
            {
                a1.push_back(i);
            }

            for(auto& i : base_yn)
            {
                a1.push_back(i);
            }

            for(auto& i : momentum_constraint)
            {
                a1.push_back(i);
            }

            for(auto& i : intermediates)
            {
                a1.push_back(i);
            }

            a1.push_back(scale);
            a1.push_back(clsize);
            a1.push_back(current_timestep);

            cqueue.exec(name, a1, {points_set.second_count}, {128});
        };

        step_kernel("evolve_cY");
        step_kernel("evolve_cA");
        step_kernel("evolve_cGi");
        step_kernel("evolve_K");
        step_kernel("evolve_X");
        step_kernel("evolve_gA");
        step_kernel("evolve_gB");
    };

    auto enforce_constraints = [&](auto& generic_out)
    {
        cl::args constraints;

        ///technically this function could work anywhere as it does not need derivatives
        ///but only the valid second derivative points are used
        constraints.push_back(points_set.second_derivative_points);
        constraints.push_back(points_set.second_count);

        for(auto& i : generic_out)
        {
            constraints.push_back(i);
        }

        constraints.push_back(scale);
        constraints.push_back(clsize);

        cqueue.exec("enforce_algebraic_constraints", constraints, {points_set.second_count}, {128});
    };

    auto diff_to_input = [&](auto& buffer_in, cl_float factor)
    {
        for(int i=0; i < (int)buffer_in.size(); i++)
        {
            cl::args accum;
            accum.push_back(points_set.second_derivative_points);
            accum.push_back(points_set.second_count);
            accum.push_back(clsize);
            accum.push_back(buffer_in[i]);
            accum.push_back(base_yn[i]);
            accum.push_back(factor);

            cqueue.exec("calculate_rk4_val", accum, {points_set.second_count}, {128});
        }
    };

    auto copy_valid = [&](auto& in, auto& out)
    {
        for(int i=0; i < (int)in.size(); i++)
        {
            cl::args copy;
            copy.push_back(points_set.second_derivative_points);
            copy.push_back(points_set.second_count);
            copy.push_back(in[i]);
            copy.push_back(out[i]);
            copy.push_back(clsize);

            cqueue.exec("copy_valid", copy, {points_set.second_count}, {128});
        }
    };

    auto dissipate = [&](auto& base_reference, auto& inout)
    {
        for(int i=0; i < buffer_set::buffer_count; i++)
        {
            cl::args diss;

            diss.push_back(points_set.second_derivative_points);
            diss.push_back(points_set.second_count);

            diss.push_back(base_reference[i]);
            diss.push_back(inout[i]);

            float coeff = dissipation_coefficients[i];

            diss.push_back(coeff);
            diss.push_back(scale);
            diss.push_back(clsize);
            diss.push_back(timestep);

            if(coeff == 0)
                continue;

            cqueue.exec("dissipate_single", diss, {points_set.second_count}, {128});
        }
    };

    ///https://mathworld.wolfram.com/Runge-KuttaMethod.html
    //#define RK4
    #ifdef RK4
    auto& b1 = generic_data[which_data];
    auto& b2 = generic_data[(which_data + 1) % 2];

    cl_int size_1d = size.x() * size.y() * size.z();

    auto copy_all = [&](auto& in, auto& out)
    {
        for(int i=0; i < (int)in.size(); i++)
        {
            cl::args copy;
            copy.push_back(in[i]);
            copy.push_back(out[i]);
            copy.push_back(size_1d);

            clctx.cqueue.exec("copy_buffer", copy, {size_1d}, {128});
        }
    };

    copy_all(b1.buffers, rk4_intermediate.buffers);

    //copy_all(b1.buffers, rk4_xn.buffers);

    auto accumulate_rk4 = [&](auto& buffers, cl_float factor)
    {
        for(int i=0; i < (int)buffers.size(); i++)
        {
            cl::args accum;
            accum.push_back(evolution_positions);
            accum.push_back(evolution_positions_count);
            accum.push_back(clsize);
            accum.push_back(rk4_intermediate.buffers[i]);
            accum.push_back(buffers[i]);
            accum.push_back(factor);

            clctx.cqueue.exec("accumulate_rk4", accum, {size_1d}, {128});
        }
    };

    ///the issue is scratch buffers not being populatd with initial conditions

    auto& scratch_2 = generic_data[(which_data + 1) % 2];

    ///gives an
    step(base_yn, rk4_scratch.buffers, 0.f);
    ///accumulate an
    accumulate_rk4(rk4_scratch.buffers, timestep/6.f);

    ///gives xn + h/2 an
    diff_to_input(rk4_scratch.buffers, timestep/2);

    enforce_constraints(rk4_scratch.buffers);

    ///gives bn
    step(rk4_scratch.buffers, scratch_2.buffers, 0.f);

    ///accumulate bn
    accumulate_rk4(scratch_2.buffers, timestep * 2.f / 6.f);

    ///gives xn + h/2 bn
    diff_to_input(scratch_2.buffers, timestep/2);

    enforce_constraints(scratch_2.buffers);

    ///gives cn
    step(scratch_2.buffers, rk4_scratch.buffers, 0.f);

    ///accumulate cn
    accumulate_rk4(rk4_scratch.buffers, timestep * 2.f / 6.f);

    ///gives xn + h * cn
    diff_to_input(rk4_scratch.buffers, timestep);

    enforce_constraints(rk4_scratch.buffers);

    ///gives dn
    step(rk4_scratch.buffers, scratch_2.buffers, 0.f);

    ///accumulate dn
    accumulate_rk4(scratch_2.buffers, timestep/6.f);

    //copy_all(base_yn.buffers, generic_data[which_data].buffers);
    copy_valid(rk4_intermediate.buffers, generic_data[(which_data + 1) % 2].buffers);
    //copy_all(rk4_intermediate.buffers, generic_data[(which_data + 1) % 2].buffers);

    #endif // RK4

    //#define FORWARD_EULER
    #ifdef FORWARD_EULER
    step(generic_data[which_data].buffers, generic_data[(which_data + 1) % 2].buffers, timestep);

    diff_to_input(generic_data[(which_data + 1) % 2].buffers, timestep);
    #endif

    #define BACKWARD_EULER
    #ifdef BACKWARD_EULER
    auto& b1 = get_input();
    auto& b2 = get_output();

    int iterations = 2;

    for(int i=0; i < iterations; i++)
    {
        if(i != 0)
            step(scratch.buffers, b2.buffers, timestep);
        else
            step(b1.buffers, b2.buffers, timestep);

        if(i != iterations - 1)
        {
            #ifdef INTERMEDIATE_DISSIPATE
            dissipate(base_yn, b2.buffers);
            #endif

            enforce_constraints(b2.buffers);
            std::swap(b2, scratch);
        }
    }
    #endif

    //#define TRAPEZOIDAL
    #ifdef TRAPEZOIDAL
    auto& b1 = generic_data[which_data];
    auto& b2 = generic_data[(which_data + 1) % 2];

    auto& f_y1 = rk4_intermediate;
    auto& f_y2 = rk4_scratch;

    //if(!trapezoidal_init)

    //step(b1.buffers, f_y1.buffers, timestep);

    step(b1.buffers, f_y1.buffers, timestep);
    diff_to_input(f_y1.buffers, timestep);
    enforce_constraints(f_y1.buffers);

    step(f_y1.buffers, f_y2.buffers, timestep);

    step(b1.buffers, f_y1.buffers, timestep);

    int iterations = 4;

    for(int i=0; i < iterations; i++)
    {
        for(int bidx = 0; bidx < f_y1.buffers.size(); bidx++)
        {
            cl::args trapezoidal;
            trapezoidal.push_back(evolution_positions);
            trapezoidal.push_back(evolution_positions_count);
            trapezoidal.push_back(clsize);
            trapezoidal.push_back(b1.buffers[bidx]); ///yn
            trapezoidal.push_back(f_y1.buffers[bidx]); ///f(Yn)
            trapezoidal.push_back(f_y2.buffers[bidx]); ///f(Yn+1) INPUT OUTPUT ARG, CONTAINS Yn+1
            trapezoidal.push_back(timestep);

            clctx.cqueue.exec("trapezoidal_accumulate", trapezoidal, {evolution_positions_count}, {128});
        }


        //diff_to_input(f_y2.buffers, timestep);
        std::swap(f_y2, b2);

        if(i != iterations - 1)
        {
            enforce_constraints(b2.buffers);
            step(b2.buffers, f_y2.buffers, timestep);
        }
    }
    #endif // TRAPEZOIDAL

    #ifdef DOUBLE_ENFORCEMENT
    enforce_constraints(generic_data[(which_data + 1) % 2].buffers);
    #endif // DOUBLE_ENFORCEMENT

    //#define DISSIPATE_SELF
    #ifdef DISSIPATE_SELF
    copy_valid(generic_data[(which_data + 1) % 2].buffers, generic_data[which_data].buffers);
    #endif // DISSIPATE_SELF

    dissipate(get_input().buffers, get_output().buffers);

    cl_float4 clmeshpos = {centre.x(), centre.y(), centre.z()};

    {
        cl::args cleaner;
        cleaner.push_back(sponge_positions);
        cleaner.push_back(sponge_positions_count);

        for(auto& i : get_output().buffers)
        {
            cleaner.push_back(i);
        }

        //cleaner.push_back(bssnok_datas[which_data]);
        cleaner.push_back(u_arg);
        cleaner.push_back(scale);
        cleaner.push_back(clsize);
        cleaner.push_back(clmeshpos);
        cleaner.push_back(timestep);

        cqueue.exec("clean_data", cleaner, {sponge_positions_count}, {256});
    }

    enforce_constraints(get_output().buffers);

    flip();

    return {*last_valid_thin_buffer, intermediates};
}

cl::buffer solve_for_u(cl::context& ctx, cl::command_queue& cqueue, vec<4, cl_int> base_size, vec3f mesh_position, float c_at_max, int scale_factor, std::optional<cl::buffer> base)
{
    vec<4, cl_int> reduced_clsize = ((base_size - 1) / scale_factor) + 1;

    std::array<cl::buffer, 2> reduced_u_args{ctx, ctx};

    if(base.has_value())
        reduced_u_args[0] = base.value();
    else
        reduced_u_args[0].alloc(reduced_clsize.x() * reduced_clsize.y() * reduced_clsize.z() * sizeof(cl_float));

    reduced_u_args[1].alloc(reduced_clsize.x() * reduced_clsize.y() * reduced_clsize.z() * sizeof(cl_float));

    int which_reduced = 0;

    {
        cl::args initial_u_args;
        initial_u_args.push_back(reduced_u_args[1]);
        initial_u_args.push_back(reduced_clsize);

        cqueue.exec("setup_u_offset", initial_u_args, {reduced_clsize.x(), reduced_clsize.y(), reduced_clsize.z()}, {8, 8, 1});

        if(!base.has_value())
            cl::copy(cqueue, reduced_u_args[1], reduced_u_args[0]);
    }

    int N = 8000;

    #ifdef GPU_PROFILE
    N = 1000;
    #endif // GPU_PROFILE

    #ifdef QUICKSTART
    N = 200;
    #endif // QUICKSTART

    vec<4, cl_float> clmeshpos = {mesh_position.x(), mesh_position.y(), mesh_position.z(), 0.f};

    for(int i=0; i < N; i++)
    {
        float local_scale = calculate_scale(c_at_max, reduced_clsize);

        cl::args iterate_u_args;
        iterate_u_args.push_back(reduced_u_args[which_reduced]);
        iterate_u_args.push_back(reduced_u_args[(which_reduced + 1) % 2]);
        iterate_u_args.push_back(local_scale);
        iterate_u_args.push_back(reduced_clsize);
        iterate_u_args.push_back(clmeshpos);

        cqueue.exec("iterative_u_solve", iterate_u_args, {reduced_clsize.x(), reduced_clsize.y(), reduced_clsize.z()}, {8, 8, 1});

        which_reduced = (which_reduced + 1) % 2;
    }

    return reduced_u_args[which_reduced];
}

cl::buffer upscale_u(cl::context& ctx, cl::command_queue& cqueue, cl::buffer& source_buffer, vec<4, cl_int> base_size, int upscale_scale, int source_scale)
{
    vec<4, cl_int> reduced_clsize = ((base_size - 1) / source_scale) + 1;
    vec<4, cl_int> upper_clsize = ((base_size - 1) / upscale_scale) + 1;

    cl::buffer u_arg(ctx);
    u_arg.alloc(upper_clsize.x() * upper_clsize.y() * upper_clsize.z() * sizeof(cl_float));

    cl::args upscale_args;
    upscale_args.push_back(source_buffer);
    upscale_args.push_back(u_arg);
    upscale_args.push_back(reduced_clsize);
    upscale_args.push_back(upper_clsize);

    cqueue.exec("upscale_u", upscale_args, {upper_clsize.x(), upper_clsize.y(), upper_clsize.z()}, {8, 8, 1});

    return u_arg;
}

cl::buffer iterate_u(cl::context& ctx, cl::command_queue& cqueue, vec3i size, vec3f mesh_position, float c_at_max)
{
    vec<4, cl_int> clsize = {size.x(), size.y(), size.z(), 0};

    std::optional<cl::buffer> last;

    for(int i=2; i >= 0; i--)
    {
        int up_size = pow(2, i+1);
        int current_size = pow(2, i);

        cl::buffer reduced = solve_for_u(ctx, cqueue, clsize, mesh_position, c_at_max, up_size, last);

        cl::buffer upscaled = upscale_u(ctx, cqueue, reduced, clsize, current_size, up_size);

        last = upscaled;
    }

    return solve_for_u(ctx, cqueue, clsize, mesh_position, c_at_max, 1, last);
}