#include "mesh_manager.hpp"
#include <toolkit/opencl.hpp>
#include <execution>
#include <iostream>

buffer_set::buffer_set(cl::context& ctx, vec3i size, bool use_matter)
{
    std::vector<std::tuple<std::string, std::string, float, bool>> values =
    {
        {"cY0", "evolve_cY", cpu_mesh::dissipate_low, false},
        {"cY1", "evolve_cY", cpu_mesh::dissipate_low, false},
        {"cY2", "evolve_cY", cpu_mesh::dissipate_low, false},
        {"cY3", "evolve_cY", cpu_mesh::dissipate_low, false},
        {"cY4", "evolve_cY", cpu_mesh::dissipate_low, false},
        {"cY5", "evolve_cY", cpu_mesh::dissipate_low, false},

        {"cA0", "evolve_cA", cpu_mesh::dissipate_high, false},
        {"cA1", "evolve_cA", cpu_mesh::dissipate_high, false},
        {"cA2", "evolve_cA", cpu_mesh::dissipate_high, false},
        {"cA3", "evolve_cA", cpu_mesh::dissipate_high, false},
        {"cA4", "evolve_cA", cpu_mesh::dissipate_high, false},
        {"cA5", "evolve_cA", cpu_mesh::dissipate_high, false},

        {"cGi0", "evolve_cGi", cpu_mesh::dissipate_low, false},
        {"cGi1", "evolve_cGi", cpu_mesh::dissipate_low, false},
        {"cGi2", "evolve_cGi", cpu_mesh::dissipate_low, false},

        {"K", "evolve_K", cpu_mesh::dissipate_high, false},
        {"X", "evolve_X", cpu_mesh::dissipate_low, false},

        {"gA", "evolve_gA", cpu_mesh::dissipate_gauge, false},
        {"gB0", "evolve_gB", cpu_mesh::dissipate_gauge, false},
        {"gB1", "evolve_gB", cpu_mesh::dissipate_gauge, false},
        {"gB2", "evolve_gB", cpu_mesh::dissipate_gauge, false},

        {"Dp_star", "evolve_hydro_all", 0.01f, true},
        {"De_star", "evolve_hydro_all", 0.025f, true},
        {"DcS0", "evolve_hydro_all", 0.15f, true},
        {"DcS1", "evolve_hydro_all", 0.15f, true},
        {"DcS2", "evolve_hydro_all", 0.15f, true},
        {"DW_stashed", "calculate_hydro_W", 0.1f, true},
    };

    for(int kk=0; kk < (int)values.size(); kk++)
    {
        named_buffer& buf = buffers.emplace_back(ctx);

        if(std::get<3>(values[kk]))
        {
            if(use_matter)
                buf.buf.alloc(size.x() * size.y() * size.z() * sizeof(cl_float));
            else
                buf.buf.alloc(sizeof(cl_int));
        }
        else
        {
            buf.buf.alloc(size.x() * size.y() * size.z() * sizeof(cl_float));
        }

        buf.name = std::get<0>(values[kk]);
        buf.modified_by = std::get<1>(values[kk]);
        buf.dissipation_coeff = std::get<2>(values[kk]);
        buf.matter_term = std::get<3>(values[kk]);
    }
}

named_buffer& buffer_set::lookup(const std::string& name)
{
    for(named_buffer& buf : buffers)
    {
        if(buf.name == name)
            return buf;
    }

    assert(false);
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

    return {real.as_device_read_only(), count};
}

inline
std::pair<cl::buffer, int> extract_buffer(cl::context& ctx, cl::command_queue& cqueue, cl::buffer& buf, cl::buffer& count)
{
    std::vector<cl_ushort4> cpu_buf = buf.read<cl_ushort4>(cqueue);
    cl_int cpu_count_1 = count.read<cl_int>(cqueue).at(0);

    assert(cpu_count_1 > 0);

    cpu_buf.resize(cpu_count_1);

    std::sort(std::execution::par_unseq, cpu_buf.begin(), cpu_buf.end(), [](const cl_ushort4& p1, const cl_ushort4& p2)
    {
        return std::tie(p1.s[2], p1.s[1], p1.s[0]) < std::tie(p2.s[2], p2.s[1], p2.s[0]);
    });

    cl::buffer shrunk_points(ctx);
    shrunk_points.alloc(cpu_buf.size() * sizeof(cl_ushort4));
    shrunk_points.write(cqueue, cpu_buf);

    printf("COUNT %i\n", cpu_count_1);

    return {shrunk_points.as_device_read_only(), cpu_count_1};
}

evolution_points generate_evolution_points(cl::context& ctx, cl::command_queue& cqueue, float scale, vec3i size)
{
    cl::buffer points_1(ctx);
    cl::buffer count_1(ctx);

    cl::buffer points_2(ctx);
    cl::buffer count_2(ctx);

    cl::buffer border_points(ctx);
    cl::buffer border_count(ctx);

    cl::buffer all_points(ctx);
    cl::buffer all_count(ctx);

    cl::buffer order(ctx);

    points_1.alloc(size.x() * size.y() * size.z() * sizeof(cl_ushort4));
    points_2.alloc(size.x() * size.y() * size.z() * sizeof(cl_ushort4));
    border_points.alloc(size.x() * size.y() * size.z() * sizeof(cl_ushort4));
    all_points.alloc(size.x() * size.y() * size.z() * sizeof(cl_ushort4));
    order.alloc(size.x() * size.y() * size.z() * sizeof(cl_ushort));

    count_1.alloc(sizeof(cl_int));
    count_2.alloc(sizeof(cl_int));
    border_count.alloc(sizeof(cl_int));
    all_count.alloc(sizeof(cl_int));

    count_1.set_to_zero(cqueue);
    count_2.set_to_zero(cqueue);
    border_count.set_to_zero(cqueue);
    all_count.set_to_zero(cqueue);
    order.set_to_zero(cqueue);

    vec<4, cl_int> clsize = {size.x(), size.y(), size.z(), 0};

    cl::args args;
    args.push_back(points_1);
    args.push_back(count_1);
    args.push_back(points_2);
    args.push_back(count_2);
    args.push_back(border_points);
    args.push_back(border_count);
    args.push_back(all_points);
    args.push_back(all_count);
    args.push_back(order);
    args.push_back(scale);
    args.push_back(clsize);

    cqueue.exec("generate_evolution_points", args, {size.x(),  size.y(),  size.z()}, {8, 8, 1});

    //auto [shrunk_points_1, cpu_count_1] = extract_buffer(ctx, cqueue, points_1, count_1);
    //auto [shrunk_points_2, cpu_count_2] = extract_buffer(ctx, cqueue, points_2, count_2);
    auto [shrunk_border, cpu_border_count] = extract_buffer(ctx, cqueue, border_points, border_count);
    auto [shrunk_all, cpu_all_count] = extract_buffer(ctx, cqueue, all_points, all_count);

    evolution_points ret(ctx);
    //ret.first_count = cpu_count_1;
    //ret.second_count = cpu_count_2;
    ret.border_count = cpu_border_count;
    ret.all_count = cpu_all_count;

    //ret.first_derivative_points = shrunk_points_1;
    //ret.second_derivative_points = shrunk_points_2;
    ret.border_points = shrunk_border;
    ret.all_points = shrunk_all;
    ret.order = order.as_device_read_only();

    //printf("Evolve point reduction %i\n", cpu_count_1);

    return ret;
}

ref_counted_buffer thin_intermediates_pool::request(cl::context& ctx, cl::managed_command_queue& cqueue, vec3i size, int element_size)
{
    for(ref_counted_buffer& desc : pool)
    {
        int my_size = size.x() * size.y() * size.x() * element_size;
        int desc_size = desc.alloc_size;

        int rc = desc.ref_count();

        if(rc == 1 && desc_size >= my_size)
        {
            return desc;
        }
    }

    ref_counted_buffer next(ctx);
    next.alloc(size.x() * size.y() * size.z() * element_size);

    #ifdef NANFILL
    cl_float nan = std::nanf("");
    cl::event evt = next.fill(cqueue, nan);
    cqueue.getting_value_depends_on(next, evt);
    #else
    cl::event evt = next.set_to_zero(cqueue.mqueue.next());
    cqueue.getting_value_depends_on(next, evt);
    #endif // NANFILL

    pool.push_back(next);

    return next;
}

cpu_mesh::cpu_mesh(cl::context& ctx, cl::command_queue& cqueue, vec3i _centre, vec3i _dim, cpu_mesh_settings _sett, evolution_points& points) :
        data{buffer_set(ctx, _dim, _sett.use_matter), buffer_set(ctx, _dim, _sett.use_matter)}, scratch{ctx, _dim, _sett.use_matter}, points_set{ctx}, sponge_positions{ctx},
        momentum_constraint{ctx, ctx, ctx}
{
    centre = _centre;
    dim = _dim;
    sett = _sett;

    scale = calculate_scale(get_c_at_max(), dim);

    points_set = points;
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

void cpu_mesh::init(cl::command_queue& cqueue, cl::buffer& u_arg)
{
    cl_int4 clsize = {dim.x(), dim.y(), dim.z(), 0};

    {
        cl::args init;

        for(auto& i : data[0].buffers)
        {
            init.push_back(i.buf);
        }

        init.push_back(u_arg);
        init.push_back(scale);
        init.push_back(clsize);

        cqueue.exec("calculate_initial_conditions", init, {dim.x(), dim.y(), dim.z()}, {8, 8, 1});
    }

    if(sett.use_matter)
    {
        cl::args hydro_init;

        for(auto& i : data[0].buffers)
        {
            hydro_init.push_back(i.buf);
        }

        hydro_init.push_back(u_arg);
        hydro_init.push_back(scale);
        hydro_init.push_back(clsize);

        cqueue.exec("calculate_hydrodynamic_initial_conditions", hydro_init, {dim.x(), dim.y(), dim.z()}, {8, 8, 1});
    }

    for(int i=0; i < (int)data[0].buffers.size(); i++)
    {
        cl::copy(cqueue, data[0].buffers[i].buf, data[1].buffers[i].buf);
        cl::copy(cqueue, data[0].buffers[i].buf, scratch.buffers[i].buf);
    }
}

void cpu_mesh::calculate_hydro_w(cl::managed_command_queue& cqueue, buffer_set& in)
{
    if(!sett.use_matter)
        return;

    cl::args hyd;
    hyd.push_back(points_set.all_points);
    hyd.push_back(points_set.all_count);

    for(auto& i : in.buffers)
    {
        hyd.push_back(i.buf);
    }

    vec<4, cl_int> clsize = {dim.x(), dim.y(), dim.z(), 0};

    hyd.push_back(scale);
    hyd.push_back(clsize);
    hyd.push_back(points_set.order);

    cqueue.exec("calculate_hydro_W", hyd, {points_set.all_count}, {128});
}

void cpu_mesh::step_hydro(cl::context& ctx, cl::managed_command_queue& cqueue, thin_intermediates_pool& pool, buffer_set& in, buffer_set& out, buffer_set& base, float timestep)
{
    if(!sett.use_matter)
        return;

    cl_int4 clsize = {dim.x(), dim.y(), dim.z(), 0};

    calculate_hydro_w(cqueue, in);

    int intermediate_count = 13;

    std::vector<ref_counted_buffer> intermediates;

    for(int i=0; i < intermediate_count; i++)
    {
        intermediates.push_back(pool.request(ctx, cqueue, dim, sizeof(cl_float)));

        intermediates.back().set_to_zero(cqueue);
    }

    {
        cl::args calc_intermediates;
        calc_intermediates.push_back(points_set.all_points);
        calc_intermediates.push_back(points_set.all_count);

        for(auto& buf : in.buffers)
        {
            calc_intermediates.push_back(buf.buf.as_device_read_only());
        }

        for(auto& i : intermediates)
        {
            calc_intermediates.push_back(i);
        }

        calc_intermediates.push_back(scale);
        calc_intermediates.push_back(clsize);
        calc_intermediates.push_back(points_set.order);

        cqueue.exec("calculate_hydro_intermediates", calc_intermediates, {points_set.all_count}, {128});
    }

    {
        cl::args evolve;
        evolve.push_back(points_set.all_points);
        evolve.push_back(points_set.all_count);

        for(auto& buf : in.buffers)
        {
            evolve.push_back(buf.buf.as_device_read_only());
        }

        for(auto& buf : out.buffers)
        {
            evolve.push_back(buf.buf.as_device_write_only());
        }

        for(auto& buf : base.buffers)
        {
            evolve.push_back(buf.buf.as_device_read_only());
        }

        for(auto& buf : intermediates)
        {
            evolve.push_back(buf.as_device_read_only());
        }

        evolve.push_back(scale);
        evolve.push_back(clsize);
        evolve.push_back(points_set.order);
        evolve.push_back(timestep);

        cqueue.exec("evolve_hydro_all", evolve, {points_set.all_count}, {128});
    }

    ///temporary
    //calculate_hydro_w(cqueue);
}

ref_counted_buffer cpu_mesh::get_thin_buffer(cl::context& ctx, cl::managed_command_queue& cqueue, thin_intermediates_pool& pool)
{
    if(sett.use_half_intermediates)
        return pool.request(ctx, cqueue, dim, sizeof(cl_half));
    else
        return pool.request(ctx, cqueue, dim, sizeof(cl_float));
}

std::vector<ref_counted_buffer> cpu_mesh::get_derivatives_of(cl::context& ctx, buffer_set& generic_in, cl::managed_command_queue& mqueue, thin_intermediates_pool& pool)
{
    cl_int4 clsize = {dim.x(), dim.y(), dim.z(), 0};

    std::vector<ref_counted_buffer> intermediates;

    auto differentiate = [&](cl::managed_command_queue& cqueue, cl::buffer in_buffer, cl::buffer& out1, cl::buffer& out2, cl::buffer& out3)
    {
        cl::args thin;
        thin.push_back(points_set.all_points);
        thin.push_back(points_set.all_count);
        thin.push_back(in_buffer.as_device_read_only());
        thin.push_back(out1);
        thin.push_back(out2);
        thin.push_back(out3);
        thin.push_back(scale);
        thin.push_back(clsize);
        thin.push_back(points_set.order);

        cqueue.exec("calculate_intermediate_data_thin", thin, {points_set.all_count}, {128});
    };

    std::array buffers = {"cY0", "cY1", "cY2", "cY3", "cY4", "cY5",
                          "gA", "gB0", "gB1", "gB2", "X"};

    for(int idx = 0; idx < (int)buffers.size(); idx++)
    {
        ref_counted_buffer b1 = get_thin_buffer(ctx, mqueue, pool);
        ref_counted_buffer b2 = get_thin_buffer(ctx, mqueue, pool);
        ref_counted_buffer b3 = get_thin_buffer(ctx, mqueue, pool);

        cl::buffer found = generic_in.lookup(buffers[idx]).buf;

        differentiate(mqueue, found, b1, b2, b3);

        intermediates.push_back(b1);
        intermediates.push_back(b2);
        intermediates.push_back(b3);
    }

    return intermediates;
}

///returns buffers and intermediates
std::pair<std::vector<cl::buffer>, std::vector<ref_counted_buffer>> cpu_mesh::full_step(cl::context& ctx, cl::command_queue& main_queue, cl::managed_command_queue& mqueue, float timestep, thin_intermediates_pool& pool, cl::buffer& u_arg)
{
    cl_int4 clsize = {dim.x(), dim.y(), dim.z(), 0};

    std::vector<cl::buffer> last_valid_thin_buffer;

    for(auto& i : get_input().buffers)
    {
        last_valid_thin_buffer.push_back(i.buf);
    }

    auto& base_yn = get_input();

    std::vector<ref_counted_buffer> intermediates;

    mqueue.begin_splice(main_queue);

    auto check_for_nans = [&](const std::string& name, cl::buffer& buf)
    {
        return;

        mqueue.block();

        std::cout << "checking " << name << std::endl;

        cl::args nan_buf;
        nan_buf.push_back(points_set.border_points);
        nan_buf.push_back(points_set.border_count);
        nan_buf.push_back(buf);
        nan_buf.push_back(scale);
        nan_buf.push_back(clsize);

        mqueue.exec("nan_checker", nan_buf, {points_set.border_count}, {128});

        mqueue.block();
    };

    #if 0
    auto copy_border = [&](auto& in, auto& out)
    {
        for(int i=0; i < (int)in.size(); i++)
        {
            cl::args copy;
            copy.push_back(points_set.border_points);
            copy.push_back(points_set.border_count);
            copy.push_back(in[i]);
            copy.push_back(out[i]);
            copy.push_back(clsize);

            mqueue.exec("copy_valid", copy, {points_set.border_count}, {128});
        }
    };
    #endif // 0

    auto clean = [&](auto& in, auto& inout)
    {
        cl::args cleaner;
        cleaner.push_back(points_set.border_points);
        cleaner.push_back(points_set.border_count);

        for(auto& i : in.buffers)
        {
            cleaner.push_back(i.buf.as_device_read_only());
        }

        for(auto& i : base_yn.buffers)
        {
            cleaner.push_back(i.buf.as_device_read_only());
        }

        for(auto& i : inout.buffers)
        {
            cleaner.push_back(i.buf);
        }

        //cleaner.push_back(bssnok_datas[which_data]);
        cleaner.push_back(u_arg);
        cleaner.push_back(points_set.order);
        cleaner.push_back(scale);
        cleaner.push_back(clsize);
        cleaner.push_back(timestep);

        mqueue.exec("clean_data", cleaner, {points_set.border_count}, {256});

        for(auto& i : inout.buffers)
        {
            check_for_nans(i.name + "_clean", i.buf);
        }
    };

    auto step = [&](auto& generic_in, auto& generic_out, float current_timestep)
    {
        intermediates.clear();

        last_valid_thin_buffer.clear();

        step_hydro(ctx, mqueue, pool, generic_in, generic_out, base_yn, current_timestep);

        for(auto& i : generic_in.buffers)
        {
            last_valid_thin_buffer.push_back(i.buf);
        }

        intermediates = get_derivatives_of(ctx, generic_in, mqueue, pool);

        ///end all the differentiation work before we move on
        if(sett.calculate_momentum_constraint)
        {
            assert(false);

            /*cl::args momentum_args;

            momentum_args.push_back(points_set.first_derivative_points);
            momentum_args.push_back(points_set.first_count);

            for(auto& i : generic_in)
            {
                momentum_args.push_back(i.as_device_read_only());
            }

            for(auto& i : momentum_constraint)
            {
                momentum_args.push_back(i);
            }

            momentum_args.push_back(scale);
            momentum_args.push_back(clsize);

            mqueue.exec("calculate_momentum_constraint", momentum_args, {points_set.first_count}, {128});*/
        }

        auto step_kernel = [&](const std::string& name)
        {
            cl::args a1;

            a1.push_back(points_set.all_points);
            a1.push_back(points_set.all_count);

            for(auto& i : generic_in.buffers)
            {
                a1.push_back(i.buf.as_device_read_only());
            }

            for(named_buffer& i : generic_out.buffers)
            {
                if(i.modified_by == name)
                    a1.push_back(i.buf);
                else
                    a1.push_back(i.buf.as_device_inaccessible());
            }

            for(auto& i : base_yn.buffers)
            {
                a1.push_back(i.buf.as_device_read_only());
            }

            for(auto& i : momentum_constraint)
            {
                a1.push_back(i.as_device_read_only());
            }

            for(auto& i : intermediates)
            {
                a1.push_back(i.as_device_read_only());
            }

            a1.push_back(scale);
            a1.push_back(clsize);
            a1.push_back(current_timestep);
            a1.push_back(points_set.order);

            mqueue.exec(name, a1, {points_set.all_count}, {128});
            //mqueue.flush();

            for(auto& i : generic_out.buffers)
            {
                if(i.modified_by != name)
                    continue;

                check_for_nans(i.name + "_step", i.buf);
            }
        };

        step_kernel("evolve_cY");
        step_kernel("evolve_cA");
        step_kernel("evolve_cGi");
        step_kernel("evolve_K");
        step_kernel("evolve_X");
        step_kernel("evolve_gA");
        step_kernel("evolve_gB");

        clean(generic_in, generic_out);

        //copy_border(generic_in, generic_out);
    };

    auto enforce_constraints = [&](auto& generic_out)
    {
        cl::args constraints;

        ///technically this function could work anywhere as it does not need derivatives
        ///but only the valid second derivative points are used
        constraints.push_back(points_set.all_points);
        constraints.push_back(points_set.all_count);

        for(auto& i : generic_out.buffers)
        {
            constraints.push_back(i.buf);
        }

        constraints.push_back(scale);
        constraints.push_back(clsize);

        mqueue.exec("enforce_algebraic_constraints", constraints, {points_set.all_count}, {128});

        for(auto& i : generic_out.buffers)
        {
            check_for_nans(i.name + "_constrain", i.buf);
        }
    };

    #if 0
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

            mqueue.exec("calculate_rk4_val", accum, {points_set.second_count}, {128});
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

            mqueue.exec("copy_valid", copy, {points_set.second_count}, {128});
        }
    };

    auto dissipate = [&](auto& base_reference, auto& inout)
    {
        for(int i=0; i < buffer_set::buffer_count; i++)
        {
            cl::args diss;

            diss.push_back(points_set.second_derivative_points);
            diss.push_back(points_set.second_count);

            diss.push_back(base_reference[i].as_device_read_only());
            diss.push_back(inout[i]);

            float coeff = dissipation_coefficients[i];

            diss.push_back(coeff);
            diss.push_back(scale);
            diss.push_back(clsize);
            diss.push_back(timestep);

            if(coeff == 0)
                continue;

            mqueue.exec("dissipate_single", diss, {points_set.second_count}, {128});
            //mqueue.flush();
        }
    };
    #endif // 0

    auto dissipate_unidir = [&](auto& in, auto& out)
    {
        assert(in.buffers.size() == out.buffers.size());

        for(int i=0; i < (int)in.buffers.size(); i++)
        {
            if(in.buffers[i].buf.alloc_size != sizeof(cl_float) * dim.x() * dim.y() * dim.z() || in.buffers[i].dissipation_coeff == 0.f)
            {
                //assert(false);
                //printf("hi\n");

                std::swap(in.buffers[i], out.buffers[i]);
                continue;
            }

            cl::args diss;

            diss.push_back(points_set.all_points);
            diss.push_back(points_set.all_count);

            diss.push_back(in.buffers[i].buf.as_device_read_only());
            diss.push_back(out.buffers[i].buf);

            float coeff = in.buffers[i].dissipation_coeff;

            diss.push_back(coeff);
            diss.push_back(scale);
            diss.push_back(clsize);
            diss.push_back(timestep);
            diss.push_back(points_set.order);

            //if(coeff == 0)
            //    continue;

            mqueue.exec("dissipate_single_unidir", diss, {points_set.all_count}, {128});

            check_for_nans(in.buffers[i].name + "_diss", out.buffers[i].buf);
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

    int iterations = 3;

    if(iterations == 1)
    {
        printf("You're going to forget every single time when you change this for debugging reasons, this will cause everything to break\n");
    }

    for(int i=0; i < iterations; i++)
    {
        if(i != 0)
            step(scratch, b2, timestep);
        else
            step(b1, b2, timestep);

        if(i != iterations - 1)
        {
            //#define INTERMEDIATE_DISSIPATE
            #ifdef INTERMEDIATE_DISSIPATE
            dissipate(base_yn, b2.buffers);
            #endif

            dissipate_unidir(b2, scratch);

            enforce_constraints(scratch);
            //std::swap(b2, scratch);
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

    dissipate_unidir(b2, scratch);

    std::swap(b2, scratch);;

    //dissipate(get_input().buffers, get_output().buffers);

    //clean(scratch.buffers, b2.buffers);

    enforce_constraints(get_output());

    mqueue.end_splice(main_queue);

    flip();

    return {last_valid_thin_buffer, intermediates};
}
