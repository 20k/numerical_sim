#include "mesh_manager.hpp"
#include <toolkit/opencl.hpp>
#include <execution>
#include <iostream>

buffer_set::buffer_set(cl::context& ctx, vec3i size, const std::vector<buffer_descriptor>& in_buffers)
{
    uint64_t buf_size = size.x() * size.y() * size.z() * sizeof(cl_float);

    for(const buffer_descriptor& desc : in_buffers)
    {
        named_buffer& buf = buffers.emplace_back(ctx);

        buf.buf.alloc(buf_size);
        buf.desc = desc;
    }
}

named_buffer& buffer_set::lookup(const std::string& name)
{
    for(named_buffer& buf : buffers)
    {
        if(buf.desc.name == name)
            return buf;
    }

    assert(false);
}

///:O ok so, SO the factor that we add to these buffers is constant
///might be able to add it to base just once
template<typename T>
void dissipate_set(cl::managed_command_queue& mqueue, T& base_reference, T& inout, evolution_points& points_set, float timestep, vec3i dim, float scale)
{
    cl_int4 clsize = {dim.x(), dim.y(), dim.z(), 0};

    for(int i=0; i < base_reference.buffers.size(); i++)
    {
        if(base_reference.buffers[i].buf.alloc_size != sizeof(cl_float) * dim.x() * dim.y() * dim.z())
            continue;

        if(inout.buffers[i].desc.dissipation_coeff == 0)
            continue;

        cl::args diss;

        diss.push_back(points_set.all_points);
        diss.push_back(points_set.all_count);

        diss.push_back(base_reference.buffers[i].buf.as_device_read_only());
        diss.push_back(inout.buffers[i].buf);

        float coeff = inout.buffers[i].desc.dissipation_coeff;

        diss.push_back(coeff);
        diss.push_back(scale);
        diss.push_back(clsize);
        diss.push_back(timestep);
        diss.push_back(points_set.order);

        mqueue.exec("dissipate_single", diss, {points_set.all_count}, {128});

        //check_for_nans(inout.buffers[i].name + "_diss_single", inout.buffers[i].buf);
    }
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

cpu_mesh::cpu_mesh(cl::context& ctx, cl::command_queue& cqueue, vec3i _centre, vec3i _dim, cpu_mesh_settings _sett, evolution_points& points, const std::vector<buffer_descriptor>& buffers, const std::vector<buffer_descriptor>& utility_buffers, std::vector<plugin*> _plugins) :
        data{buffer_set(ctx, _dim, buffers), buffer_set(ctx, _dim, buffers), buffer_set(ctx, _dim, buffers), buffer_set(ctx, _dim, buffers)},
        utility_data{buffer_set(ctx, _dim, utility_buffers)},
        points_set{ctx},
        momentum_constraint{ctx, ctx, ctx},
        plugins(_plugins)
{
    centre = _centre;
    dim = _dim;
    sett = _sett;

    scale = calculate_scale(get_c_at_max(), dim);

    points_set = points;

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

void cpu_mesh::init(cl::context& ctx, cl::command_queue& cqueue, thin_intermediates_pool& pool, cl::buffer& u_arg, std::array<cl::buffer, 6>& bcAij)
{
    cl_int4 clsize = {dim.x(), dim.y(), dim.z(), 0};

    {
        cl::args init;

        for(auto& i : data[0].buffers)
        {
            init.push_back(i.buf);
        }

        init.push_back(u_arg);

        for(int i = 0; i < (int)bcAij.size(); i++)
        {
            init.push_back(bcAij[i]);
        }

        init.push_back(scale);
        init.push_back(clsize);

        cqueue.exec("calculate_initial_conditions", init, {dim.x(), dim.y(), dim.z()}, {8, 8, 1});
    }

    for(plugin* p : plugins)
    {
        p->init(*this, ctx, cqueue, pool, data[0]);
    }

    for(int i=0; i < (int)data[0].buffers.size(); i++)
    {
        cl::copy(cqueue, data[0].buffers[i].buf, data[1].buffers[i].buf);
        cl::copy(cqueue, data[0].buffers[i].buf, data[2].buffers[i].buf);
        cl::copy(cqueue, data[0].buffers[i].buf, data[3].buffers[i].buf);
    }
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

void cpu_mesh::clean_buffer(cl::managed_command_queue& mqueue, cl::buffer& in, cl::buffer& out, cl::buffer& base, float asym, float speed, float timestep)
{
    if(in.alloc_size != sizeof(cl_float) * dim.x() * dim.y() * dim.z())
        return;

    cl_int4 clsize = {dim.x(), dim.y(), dim.z(), 0};

    cl::args cleaner;
    cleaner.push_back(points_set.border_points);
    cleaner.push_back(points_set.border_count);

    cleaner.push_back(in.as_device_read_only());
    cleaner.push_back(base.as_device_read_only());
    cleaner.push_back(out);

    cleaner.push_back(points_set.order);
    cleaner.push_back(scale);
    cleaner.push_back(clsize);
    cleaner.push_back(timestep);
    cleaner.push_back(asym);
    cleaner.push_back(speed);

    mqueue.exec("clean_data_thin", cleaner, {points_set.border_count}, {256});
}

///returns buffers and intermediates
void cpu_mesh::full_step(cl::context& ctx, cl::command_queue& main_queue, cl::managed_command_queue& mqueue, float timestep, thin_intermediates_pool& pool, step_callback callback)
{
    cl_int4 clsize = {dim.x(), dim.y(), dim.z(), 0};

    mqueue.begin_splice(main_queue);

    ///need to size check the buffers
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

    auto clean_thin = [&](auto& in_buf, auto& out_buf, auto& base_buf, float current_timestep)
    {
        clean_buffer(mqueue, in_buf.buf, out_buf.buf, base_buf.buf, in_buf.desc.asymptotic_value, in_buf.desc.wave_speed, current_timestep);
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
            check_for_nans(i.desc.name + "_constrain", i.buf);
        }
    };

    auto step = [&](int root_index, int generic_in_index, int generic_out_index, float current_timestep, bool trigger_callbacks, int iteration, int max_iteration)
    {
        auto& generic_in = data[generic_in_index];
        auto& generic_out = data[generic_out_index];
        auto& generic_base = data[root_index];

        if(!generic_in.currently_physical)
        {
            enforce_constraints(generic_in);
            generic_in.currently_physical = true;
        }

        buffer_pack pack(generic_in, generic_out, generic_base, generic_in_index, generic_out_index, root_index);

        for(plugin* p : plugins)
        {
            p->step(*this, ctx, mqueue, pool, pack, current_timestep, iteration, max_iteration);
        }

        std::vector<ref_counted_buffer> intermediates = get_derivatives_of(ctx, generic_in, mqueue, pool);

        if(trigger_callbacks)
        {
            std::vector<cl::buffer> linear_bufs;

            for(auto& i : data[0].buffers)
            {
                linear_bufs.push_back(i.buf);
            }

            callback(mqueue, linear_bufs, intermediates);
        }

        ///end all the differentiation work before we move on
        if(sett.calculate_momentum_constraint)
        {
            cl::args momentum_args;

            momentum_args.push_back(points_set.all_points);
            momentum_args.push_back(points_set.all_count);

            for(auto& i : generic_in.buffers)
            {
                momentum_args.push_back(i.buf.as_device_read_only());
            }

            for(auto& i : momentum_constraint)
            {
                momentum_args.push_back(i);
            }

            append_utility_buffers("calculate_momentum_constraint", momentum_args);

            momentum_args.push_back(scale);
            momentum_args.push_back(clsize);
            momentum_args.push_back(points_set.order);

            mqueue.exec("calculate_momentum_constraint", momentum_args, {points_set.all_count}, {128});
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
                if(i.desc.modified_by == name)
                    a1.push_back(i.buf);
                else
                    a1.push_back(i.buf.as_device_inaccessible());
            }

            for(auto& i : generic_base.buffers)
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

            append_utility_buffers(name, a1);

            a1.push_back(scale);
            a1.push_back(clsize);
            a1.push_back(current_timestep);
            a1.push_back(points_set.order);

            mqueue.exec(name, a1, {points_set.all_count}, {128});
            //mqueue.flush();

            for(auto& i : generic_out.buffers)
            {
                if(i.desc.modified_by != name)
                    continue;

                check_for_nans(i.desc.name + "_step", i.buf);
            }

            ///clean
            for(int i=0; i < (int)generic_in.buffers.size(); i++)
            {
                named_buffer& buf_in = generic_in.buffers[i];
                named_buffer& buf_base = generic_base.buffers[i];
                named_buffer& buf_out = generic_out.buffers[i];

                if(buf_in.desc.modified_by != name)
                    continue;

                clean_thin(buf_in, buf_out, buf_base, current_timestep);
            }
        };

        step_kernel("evolve_cY");
        step_kernel("evolve_cA");
        step_kernel("evolve_cGi");
        step_kernel("evolve_K");
        step_kernel("evolve_X");
        step_kernel("evolve_gA");
        step_kernel("evolve_gB");

        generic_out.currently_physical = false;

        //enforce_constraints(generic_out);

        //copy_border(generic_in, generic_out);
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
            if(in.buffers[i].buf.alloc_size != sizeof(cl_float) * dim.x() * dim.y() * dim.z() || in.buffers[i].desc.dissipation_coeff == 0.f)
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
            diss.push_back(out.buffers[i].buf.as_device_write_only());

            float coeff = in.buffers[i].desc.dissipation_coeff;

            diss.push_back(coeff);
            diss.push_back(scale);
            diss.push_back(clsize);
            diss.push_back(timestep);
            diss.push_back(points_set.order);

            //if(coeff == 0)
            //    continue;

            mqueue.exec("dissipate_single_unidir", diss, {points_set.all_count}, {128});

            //check_for_nans(in.buffers[i].name + "_diss", out.buffers[i].buf);
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

    ///so. data[0] is current data, data[1] is old data

    {
        dissipate_unidir(data[0], data[2]);
        std::swap(data[0], data[2]);
        data[0].currently_physical = false;
    }

    auto copy_valid = [&](auto& in, auto& out)
    {
        out.currently_physical = in.currently_physical;

        for(int i=0; i < (int)in.buffers.size(); i++)
        {
            cl::args copy;
            copy.push_back(points_set.all_points);
            copy.push_back(points_set.all_count);
            copy.push_back(in.buffers[i].buf);
            copy.push_back(out.buffers[i].buf);
            copy.push_back(clsize);

            mqueue.exec("copy_valid", copy, {points_set.all_count}, {128});
        }
    };

    auto finish_midpoint = [&](auto& summed, auto& znm1, auto& out)
    {
        summed.currently_physical = false;

        for(int i=0; i < (int)summed.buffers.size(); i++)
        {
            cl::args args;

            args.push_back(points_set.all_points);
            args.push_back(points_set.all_count);
            args.push_back(summed.buffers[i].buf);
            args.push_back(znm1.buffers[i].buf);
            args.push_back(out.buffers[i].buf);
            args.push_back(clsize);

            mqueue.exec("finish_midpoint_impl", args, {points_set.all_count}, {128});
        }
    };

    auto finish_bs = [&](int high, int low)
    {
        data[high].currently_physical = false;

        for(int i=0; i < (int)data[high].buffers.size(); i++)
        {
            cl::args args;

            args.push_back(points_set.all_points);
            args.push_back(points_set.all_count);
            args.push_back(data[high].buffers[i].buf);
            args.push_back(data[low].buffers[i].buf);
            args.push_back(clsize);

            mqueue.exec("finish_bs_impl", args, {points_set.all_count}, {128});
        }
    };

    //#define IMPLICIT_SECOND
    #ifdef IMPLICIT_SECOND
    ///https://assets.researchsquare.com/files/rs-1517205/v1_covered.pdf?c=1649867432 (3)
    ///K1 = yi+1 + -0.5 * h f(ti+1, yi+1)
    ///yi+1 = yi + h f (ti + h/2, k1)

    step(0, 0, 1, -timestep * 0.5f, true, 0, 1);
    ///data[1] contains k1

    step(1, 0, 2, timestep, false, 0, 1);

    ///data[2] contains yi+1
    ///data[0] contains yi
    step(2, 2, 1, -timestep * 0.5f, false, 0, 1);

    ///data[1] contains K1
    step(0, 1, 2, timestep, false, 0, 1);;

    std::swap(data[2], data[1]);

    #endif // IMPLICIT_SECOND

    ///https://www.physics.unlv.edu/~jeffery/astro/computer/numrec/f16-3.pdf
    //#define MODIFIED_MIDPOINT
    #ifdef MODIFIED_MIDPOINT
    auto do_midpoint = [&](int in, int intermediate, int out, int N)
    {
        float littleh = timestep / N;

        step(in, in, intermediate, littleh, true, 0, 1); ///produces z1

        int zmm1 = in;
        int zm = intermediate;
        int zmp1 = out;

        for(int m=1; m <= N - 1; m++)
        {
            step(zmm1, zm, zmp1, 2 * littleh, false, 0, 1);

            //zm = zmp1
            //zmp1 = zmm1
            //zmm1 = zm

            int old_zm = zm;
            zm = zmp1;
            zmp1 = zmm1;
            zmm1 = old_zm;
        }

        step(zm, zm, zmp1, littleh, false, 0, 1);

        finish_midpoint(data[zmp1], data[zmm1], data[zm]);

        std::swap(data[zm], data[out]);
    };

    //#define RICHARDSON
    #ifndef RICHARDSON
    int N = 3;

    do_midpoint(0, 1, 2, N);

    std::swap(data[2], data[1]);
    #else
    copy_valid(data[0], data[1]);

    int N = 6;

    do_midpoint(1, 2, 3, N);

    do_midpoint(0, 1, 2, N/2);

    finish_bs(3, 2);

    std::swap(data[2], data[1]);
    #endif
    #endif

    auto construct_guess = [&](auto& a, auto& b)
    {
        a.currently_physical = false;

        for(int i=0; i < (int)a.buffers.size(); i++)
        {
            cl::args args;

            args.push_back(points_set.all_points);
            args.push_back(points_set.all_count);
            args.push_back(a.buffers[i].buf);
            args.push_back(b.buffers[i].buf);
            args.push_back(clsize);

            mqueue.exec("construct_guess_impl", args, {points_set.all_count}, {128});
        }
    };


    auto midpoint_guess = [&](auto& a, auto& b)
    {
        a.currently_physical = false;

        for(int i=0; i < (int)a.buffers.size(); i++)
        {
            cl::args args;

            args.push_back(points_set.all_points);
            args.push_back(points_set.all_count);
            args.push_back(a.buffers[i].buf);
            args.push_back(b.buffers[i].buf);
            args.push_back(clsize);

            mqueue.exec("midpoint_guess_impl", args, {points_set.all_count}, {128});
        }
    };

    #ifdef EULER
    step(0, 0, 1, timestep, true, 0, 1);
    #endif

    auto finish_heun = [&](auto& yip1, auto& yip2)
    {
        yip2.currently_physical = false;

        for(int i=0; i < (int)yip1.buffers.size(); i++)
        {
            cl::args args;

            args.push_back(points_set.all_points);
            args.push_back(points_set.all_count);
            args.push_back(yip1.buffers[i].buf);
            args.push_back(yip2.buffers[i].buf);
            args.push_back(clsize);

            mqueue.exec("finish_heun_impl", args, {points_set.all_count}, {128});
        }
    };

    auto bdf_sum = [&](auto& yp1, auto& yn)
    {
        yn.currently_physical = false;

        for(int i=0; i < (int)yp1.buffers.size(); i++)
        {
            cl::args args;

            args.push_back(points_set.all_points);
            args.push_back(points_set.all_count);
            args.push_back(yp1.buffers[i].buf);
            args.push_back(yn.buffers[i].buf);
            args.push_back(clsize);

            mqueue.exec("bdf_sum_impl", args, {points_set.all_count}, {128});
        }
    };

    //#define BACKWARDS2
    #ifdef BACKWARDS2

    if(first_step)
    {
        step(0, 0, 1, timestep * 0.5f, true, 0, 1);
        step(0, 1, 2, timestep * 0.5f, false, 0, 1);
        step(0, 2, 1, timestep * 0.5f, false, 0, 1);
        //step(0, 1, 2, timestep * 0.5f, false, 0, 1);

        std::swap(data[2], data[1]);

        ///data[2] == yn+0.5

        midpoint_guess(data[2], data[0]);
        std::swap(data[2], data[1]);
        first_step = false;
    }
    else
    {
        ///data[1] contains yn
        ///data[0] contains yn+1

        bdf_sum(data[0], data[1]);

        ///data[1] contains bdf sum

        int iterations = 3;

        for(int i=0; i < iterations; i++)
        {
            if(i == 0)
                step(1, 0, 2, timestep * (2.f/3.f), true, 0, 1);
            else
                step(1, 2, 3, timestep * (2.f/3.f), false, 0, 1);

            if(i != 0)
            {
                std::swap(data[3], data[2]);
            }
        }
    }

    std::swap(data[2], data[1]);

    #endif // BACKWARDS2

    //#define HEUNS
    #ifdef HEUNS
    step(0, 0, 1, timestep, true, 0, 1);

    ///yi + h f(yi+1)
    step(0, 1, 2, timestep, false, 0, 1);

    ///calculates 2 * yi + h f(yi) + h f(yi+1), then divides by 2
    ///stores in data[1]

    finish_heun(data[0], data[1]);

    #endif // HEUNS

    //#define IMPLICIT_MIDPOINT
    #ifdef IMPLICIT_MIDPOINT
    ///ynp1 = yn + h f (0.5 yn + 0.5 yn+1)
    ///ynp1 == yn

    ///first step == yn + h f 0.5 yn + 0.5 yn == yn + h f yn == euler

    step(0, 0, 1, timestep, true, 0, 1);

    ///data[1] contains guess
    construct_guess(data[1], data[0]);

    step(0, 1, 2, timestep, false, 0, 1);
    ///yn+1 is in data[2]

    construct_guess(data[2], data[0]);

    step(0, 2, 1, timestep, false, 0, 1);

    //construct_guess(data[1], data[0]);

    //step(0, 1, 2, timestep, false, 0, 1);

    //construct_guess(data[2], data[0]);

    //step(0, 2, 1, timestep, false, 0, 1);

    #endif // IMPLICIT_MIDPOINT

    //#define IMPLICIT_MIDPOINT2
    #ifdef IMPLICIT_MIDPOINT2

    step(0, 0, 1, timestep * 0.5f, true, 0, 1);
    step(0, 1, 2, timestep * 0.5f, false, 0, 1);
    step(0, 2, 1, timestep * 0.5f, false, 0, 1);
    //step(0, 1, 2, timestep * 0.5f, false, 0, 1);

    std::swap(data[2], data[1]);

    ///data[2] == yn+0.5

    midpoint_guess(data[2], data[0]);
    std::swap(data[2], data[1]);

    #endif // IMPLICIT_MIDPOINT2

    //#define MIDPOINT
    #ifdef MIDPOINT
    {
        step(0, 0, 1, timestep * 0.5f, true, 0, 2);
        step(0, 1, 2, timestep, false, 1, 2);

        std::swap(data[2], data[1]);
    }
    #endif

    ///so
    ///each tick we do buffer -> base + dt * dx. Then we dissipate result. That dissipated result is used to calculate derivatives
    ///for the next tick, and is also used as the values of the inputs et
    ///buffer has kreiss oliger applied, which is of the form buffer -> buffer + f(base)
    ///so. Buffer -> base + dt * dx + f(base)
    ///this implies that I can redefine base to be base + f(base) and get the same effect
    //#define BACKWARD_EULER
    #ifdef BACKWARD_EULER
    int iterations = 2;

    if(iterations == 1)
    {
        printf("You're going to forget every single time when you change this for debugging reasons, this will cause everything to break\n");
    }

    ///so
    ///ynp1 = yn + dt f(yn+1)
    ///F(x) = yn - x + dt f(x) = 0
    for(int i=0; i < iterations; i++)
    {
        if(i != 0)
            step(0, 2, 1, timestep, false, i, iterations);
        else
            step(0, 0, 1, timestep, true, i, iterations);

        if(i != iterations - 1)
        {
            //#define INTERMEDIATE_DISSIPATE
            #ifdef INTERMEDIATE_DISSIPATE
            dissipate(base_yn, b2.buffers);
            #endif

            ///this is actually fundamentally different from below hmm
            #ifdef DISS_UNIDIR
            dissipate_unidir(b2, scratch);
            enforce_constraints(scratch);
            #else
            //dissipate_set(mqueue, data[0], data[1], points_set, timestep, dim, scale);

            std::swap(data[1], data[2]);
            #endif // DISS_UNIDIR
        }
    }
    #endif

    ///so ok: at the end of these iterations, data[0] is the original data, data[1] is the output data

    #ifdef RK4_2
    auto post_step = [&](auto& buf, float step)
    {
        dissipate_set(mqueue, data[0], buf, points_set, step, dim, scale);
        enforce_constraints(buf);
    };

    auto copy_points = [&](auto& in, auto& out)
    {
        assert(in.buffers.size() == out.buffers.size());

        for(int i=0; i < (int)in.buffers.size(); i++)
        {
            if(in.buffers[i].buf.alloc_size != sizeof(cl_float) * dim.x() * dim.y() * dim.z())
                continue;

            assert(in.buffers[i].buf.alloc_size == out.buffers[i].buf.alloc_size);

            cl::args copy;
            copy.push_back(points_set.all_points);
            copy.push_back(points_set.all_count);
            copy.push_back(in.buffers[i].buf.as_device_read_only());
            copy.push_back(out.buffers[i].buf.as_device_write_only());
            copy.push_back(clsize);

            mqueue.exec("copy_valid", copy, {points_set.all_count}, {128});
        }
    };

    ///performs accum += (q - base) * factor
    auto accumulator = [&](auto& q_val, auto& accum, float factor)
    {
        for(int i=0; i < (int)q_val.buffers.size(); i++)
        {
            if(q_val.buffers[i].buf.alloc_size != sizeof(cl_float) * dim.x() * dim.y() * dim.z())
                continue;

            cl::args acc;
            acc.push_back(points_set.all_points);
            acc.push_back(points_set.all_count);
            acc.push_back(clsize);
            acc.push_back(accum.buffers[i].buf);
            acc.push_back(base_yn.buffers[i].buf.as_device_read_only());
            acc.push_back(q_val.buffers[i].buf.as_device_read_only());
            acc.push_back(factor);

            mqueue.exec("do_rk4_accumulate", acc, {points_set.all_count}, {128});
        }
    };

    auto& accum = data[1];

    copy_points(data[0], accum);

    auto& temp_1 = data[2];

    auto data_get = [&]()
    {
        return buffer_set(ctx, dim, get_buffer_cfg(sett));
    };

    auto& temp_2 = free_data.get_named(data_get, "temp2");

    ///temp_1 == q1
    step(data[0], temp_1, timestep * 0.5f, true);

    accumulator(temp_1, accum, 2.f/6.f);

    post_step(temp_1, timestep * 0.5f);

    ///temp_2 == q2
    step(temp_1, temp_2, timestep * 0.5f, false);

    accumulator(temp_2, accum, 4.f/6.f);

    post_step(temp_2, timestep * 0.5f);

    ///temp_1 now == q3
    step(temp_2, temp_1, timestep, false);

    accumulator(temp_1, accum, 2.f/6.f);

    post_step(temp_1, timestep);

    step(temp_1, temp_2, timestep, false);

    accumulator(temp_2, accum, 1.f/6.f);

    //post_step(temp_2);

    #endif

    //#define RK4_3
    #ifdef RK4_3
    ///substantial rearrangement of the rk4 terms to make it implement better
    /*yn+1 = yn + (1/6) (k1 + 2k2 + 2 k3 + k4) h

    k1 = f(yn)
    k2 = f(yn + hk1/2)
    k3 = f(yn + hk2/2)
    k4 = f(yn + hk3)
    */

    /*
    b1 = yn + 0.5 h f(yn)
    b2 = yn + 0.5 h f(b1)
    b3 = yn + h f(b2)
    b4 = yn + 0.5 h f(b3)

    yn+1 = -2/3 yn + 1/3 (b1 + 2b2 + b3 + b4)

    ///b4s left could be any buffer

    bx = G + (1/a) h f(b3)

    yn+1 = -1/3 yn - 1/6 G a + 1/3 (b1 + 2b2 + b3 + 0.5 a bx)

    a = 2, G = b1 + 2b2 + b3

    -1/3 yn + 1/3 bx
    */

    auto multiply_add = [&](auto& inout, auto& right, float left_cst, float right_cst)
    {
        inout.currently_physical = false;

        for(int i=0; i < (int)inout.buffers.size(); i++)
        {
            cl::args args;

            args.push_back(points_set.all_points);
            args.push_back(points_set.all_count);
            args.push_back(inout.buffers[i].buf);
            args.push_back(right.buffers[i].buf.as_device_read_only());
            args.push_back(left_cst);
            args.push_back(right_cst);
            args.push_back(i);
            args.push_back(clsize);

            mqueue.exec("multiply_add_impl", args, {points_set.all_count}, {128});
        }
    };

    step(0, 0, 1, 0.5f * timestep, true, 0, 1);
    ///data[1] contains b1, and will be the starting point for the rk4 additions
    step(0, 1, 2, 0.5f * timestep, false, 0, 1);
    ///data[1] contains b1. data[2] contains b2

    multiply_add(data[1], data[2], 1.f, 2.f);

    step(0, 2, 3, timestep, false, 0, 1);
    ///data[3] now contains b3

    multiply_add(data[1], data[3], 1.f, 1.f);

    ///data[1] == b1 + 2b2 + b3

    ///bx = G + (1/a) h f(b3)
    ///G = b1 + 2b2 + b3
    ///a = 2

    step(1, 3, 2, 0.5f * timestep, false, 0, 1);
    ///data[2] == bx

    multiply_add(data[0], data[2], -(1.f/3.f), 1.f/3.f);
    std::swap(data[0], data[1]);

    #endif // RK4_3

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

    ///output is in b2
    #ifdef DISS_UNIDIR
    dissipate_unidir(b2, scratch);

    std::swap(b2, scratch);
    #else
    //dissipate_set(mqueue, data[0], data[1], points_set, timestep, dim, scale);
    #endif

    //dissipate(get_input().buffers, get_output().buffers);

    //clean(scratch.buffers, b2.buffers);

    for(plugin* p : plugins)
    {
        p->finalise(*this, ctx, mqueue, pool, timestep);
    }

    mqueue.end_splice(main_queue);

    std::swap(data[1], data[0]);
    ///data[0] is now the new output data, data[1] is the old data, data[2] is the old intermediate data
}

void cpu_mesh::append_utility_buffers(const std::string& kernel_name, cl::args& args)
{
    if(utility_data.buffers.size() == 0)
    {
        args.push_back(nullptr);
    }
    else
    {
        for(named_buffer& buf : utility_data.buffers)
        {
            if(buf.desc.modified_by == kernel_name)
                args.push_back(buf.buf);
            else
                args.push_back(buf.buf.as_device_read_only());
        }
    }
}
