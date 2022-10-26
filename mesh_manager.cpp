#include "mesh_manager.hpp"
#include <toolkit/opencl.hpp>
#include <execution>
#include <iostream>

buffer_set::buffer_set(cl::context& ctx, vec3i size, buffer_set_cfg cfg)
{
    ///often 1 is used here as well. Seems to make a noticable difference to reflections
    float gauge_wave_speed = sqrt(2);

    std::vector<std::tuple<std::string, std::string, float, float, float, int>> values =
    {
        {"cY0", "evolve_cY", cpu_mesh::dissipate_low, 1, 1, 0},
        {"cY1", "evolve_cY", cpu_mesh::dissipate_low, 0, 1, 0},
        {"cY2", "evolve_cY", cpu_mesh::dissipate_low, 0, 1, 0},
        {"cY3", "evolve_cY", cpu_mesh::dissipate_low, 1, 1, 0},
        {"cY4", "evolve_cY", cpu_mesh::dissipate_low, 0, 1, 0},
        {"cY5", "evolve_cY", cpu_mesh::dissipate_low, 1, 1, 0},

        {"cA0", "evolve_cA", cpu_mesh::dissipate_high, 0, 1, 0},
        {"cA1", "evolve_cA", cpu_mesh::dissipate_high, 0, 1, 0},
        {"cA2", "evolve_cA", cpu_mesh::dissipate_high, 0, 1, 0},
        {"cA3", "evolve_cA", cpu_mesh::dissipate_high, 0, 1, 0},
        {"cA4", "evolve_cA", cpu_mesh::dissipate_high, 0, 1, 0},
        {"cA5", "evolve_cA", cpu_mesh::dissipate_high, 0, 1, 0},

        {"cGi0", "evolve_cGi", cpu_mesh::dissipate_low, 0, 1, 0},
        {"cGi1", "evolve_cGi", cpu_mesh::dissipate_low, 0, 1, 0},
        {"cGi2", "evolve_cGi", cpu_mesh::dissipate_low, 0, 1, 0},

        {"K", "evolve_K", cpu_mesh::dissipate_high, 0, 1, 0},
        {"X", "evolve_X", cpu_mesh::dissipate_low, 1, 1, 0},

        {"gA", "evolve_gA", cpu_mesh::dissipate_gauge, 1, gauge_wave_speed, 0},
        {"gB0", "evolve_gB", cpu_mesh::dissipate_gauge, 0, gauge_wave_speed, 0},
        {"gB1", "evolve_gB", cpu_mesh::dissipate_gauge, 0, gauge_wave_speed, 0},
        {"gB2", "evolve_gB", cpu_mesh::dissipate_gauge, 0, gauge_wave_speed, 0},

        {"gBB0", "evolve_cGi", cpu_mesh::dissipate_gauge, 0, gauge_wave_speed, 2},
        {"gBB1", "evolve_cGi", cpu_mesh::dissipate_gauge, 0, gauge_wave_speed, 2},
        {"gBB2", "evolve_cGi", cpu_mesh::dissipate_gauge, 0, gauge_wave_speed, 2},

        {"Dp_star", "evolve_hydro_all", 0.25f, 0, 1, 1},
        {"De_star", "evolve_hydro_all", 0.25f, 0, 1, 1},
        {"DcS0", "evolve_hydro_all", 0.25f, 0, 1, 1},
        {"DcS1", "evolve_hydro_all", 0.25f, 0, 1, 1},
        {"DcS2", "evolve_hydro_all", 0.25f, 0, 1, 1},

        {"dRed", "evolve_advect", 0.25f, 0, 1, 3},
        {"dGreen", "evolve_advect", 0.25f, 0, 1, 3},
        {"dBlue", "evolve_advect", 0.25f, 0, 1, 3},
    };

    for(int kk=0; kk < (int)values.size(); kk++)
    {
        uint64_t buf_size = size.x() * size.y() * size.z() * sizeof(cl_float);

        named_buffer& buf = buffers.emplace_back(ctx);

        int type = std::get<5>(values[kk]);

        if(type == 0)
        {
            buf.buf.alloc(buf_size);
        }
        else if(type == 1 && cfg.use_matter)
        {
            buf.buf.alloc(buf_size);
        }
        else if(type == 2 && cfg.use_gBB)
        {
            buf.buf.alloc(buf_size);
        }
        else if(type == 3 && cfg.use_matter_colour)
        {
            buf.buf.alloc(buf_size);
        }
        else
        {
            buf.buf.alloc(sizeof(cl_int));
        }

        buf.name = std::get<0>(values[kk]);
        buf.modified_by = std::get<1>(values[kk]);
        buf.dissipation_coeff = std::get<2>(values[kk]);
        buf.asymptotic_value = std::get<3>(values[kk]);
        buf.wave_speed = std::get<4>(values[kk]);
        buf.matter_term = std::get<5>(values[kk]) == 1 || std::get<5>(values[kk]) == 3;
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

template<typename T>
void dissipate_set(cl::managed_command_queue& mqueue, T& base_reference, T& inout, evolution_points& points_set, float timestep, vec3i dim, float scale)
{
    cl_int4 clsize = {dim.x(), dim.y(), dim.z(), 0};

    for(int i=0; i < base_reference.buffers.size(); i++)
    {
        if(base_reference.buffers[i].buf.alloc_size != sizeof(cl_float) * dim.x() * dim.y() * dim.z())
            continue;

        cl::args diss;

        diss.push_back(points_set.all_points);
        diss.push_back(points_set.all_count);

        diss.push_back(base_reference.buffers[i].buf.as_device_read_only());
        diss.push_back(inout.buffers[i].buf);

        float coeff = inout.buffers[i].dissipation_coeff;

        diss.push_back(coeff);
        diss.push_back(scale);
        diss.push_back(clsize);
        diss.push_back(timestep);
        diss.push_back(points_set.order);

        if(coeff == 0)
            continue;

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

buffer_set_cfg get_buffer_cfg(cpu_mesh_settings sett)
{
    buffer_set_cfg cfg;
    cfg.use_matter = sett.use_matter;
    cfg.use_matter_colour = sett.use_matter_colour;
    cfg.use_gBB = sett.use_gBB;

    return cfg;
}

cpu_mesh::cpu_mesh(cl::context& ctx, cl::command_queue& cqueue, vec3i _centre, vec3i _dim, cpu_mesh_settings _sett, evolution_points& points) :
        data{buffer_set(ctx, _dim, get_buffer_cfg(_sett)), buffer_set(ctx, _dim, get_buffer_cfg(_sett)), buffer_set(ctx, _dim, get_buffer_cfg(_sett))},
        points_set{ctx},
        momentum_constraint{ctx, ctx, ctx}, hydro_st(ctx)
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

void cpu_mesh::init(cl::command_queue& cqueue, cl::buffer& u_arg, matter_initial_vars& vars)
{
    cl_int4 clsize = {dim.x(), dim.y(), dim.z(), 0};

    {
        cl::args init;

        for(auto& i : data[0].buffers)
        {
            init.push_back(i.buf);
        }

        init.push_back(u_arg);

        for(int i = 0; i < (int)vars.bcAij.size(); i++)
        {
            init.push_back(vars.bcAij[i]);
        }

        init.push_back(scale);
        init.push_back(clsize);

        cqueue.exec("calculate_initial_conditions", init, {dim.x(), dim.y(), dim.z()}, {8, 8, 1});
    }

    if(sett.use_matter)
    {
        hydro_st.should_evolve.alloc(dim.x() * dim.y() * dim.z() * sizeof(cl_char));
        hydro_st.should_evolve.fill(cqueue, cl_char{1});

        cl::args hydro_init;

        for(auto& i : data[0].buffers)
        {
            hydro_init.push_back(i.buf);
        }

        hydro_init.push_back(vars.pressure_buf);
        hydro_init.push_back(vars.rho_buf);
        hydro_init.push_back(vars.rhoH_buf);
        hydro_init.push_back(vars.p0_buf);
        hydro_init.push_back(vars.Si_buf[0]);
        hydro_init.push_back(vars.Si_buf[1]);
        hydro_init.push_back(vars.Si_buf[2]);
        hydro_init.push_back(vars.colour_buf[0]);
        hydro_init.push_back(vars.colour_buf[1]);
        hydro_init.push_back(vars.colour_buf[2]);

        hydro_init.push_back(u_arg);
        hydro_init.push_back(vars.superimposed_tov_phi);
        hydro_init.push_back(scale);
        hydro_init.push_back(clsize);

        cl_int use_colour = sett.use_matter_colour;

        hydro_init.push_back(use_colour);

        cqueue.exec("calculate_hydrodynamic_initial_conditions", hydro_init, {dim.x(), dim.y(), dim.z()}, {8, 8, 1});
    }

    for(int i=0; i < (int)data[0].buffers.size(); i++)
    {
        cl::copy(cqueue, data[0].buffers[i].buf, data[1].buffers[i].buf);
        cl::copy(cqueue, data[0].buffers[i].buf, data[2].buffers[i].buf);
    }
}

void cpu_mesh::step_hydro(cl::context& ctx, cl::managed_command_queue& cqueue, thin_intermediates_pool& pool, buffer_set& in, buffer_set& out, buffer_set& base, float timestep)
{
    if(!sett.use_matter)
        return;

    cl_int4 clsize = {dim.x(), dim.y(), dim.z(), 0};

    int intermediate_count = 1;

    std::vector<ref_counted_buffer> intermediates;

    for(int i=0; i < intermediate_count; i++)
    {
        intermediates.push_back(pool.request(ctx, cqueue, dim, sizeof(cl_float)));

        //intermediates.back().fill(cqueue, std::numeric_limits<float>::quiet_NaN());
    }

    ///only need this in the case of quadratic viscosity
    ref_counted_buffer w_buf = pool.request(ctx, cqueue, dim, sizeof(cl_float));

    {
        cl::args build;
        build.push_back(points_set.all_points);
        build.push_back(points_set.all_count);

        for(auto& buf : in.buffers)
        {
            build.push_back(buf.buf.as_device_read_only());
        }

        build.push_back(scale);
        build.push_back(clsize);
        build.push_back(points_set.order);
        build.push_back(hydro_st.should_evolve);

        cqueue.exec("calculate_hydro_evolved", build, {points_set.all_count}, {128});
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

        calc_intermediates.push_back(w_buf);

        calc_intermediates.push_back(scale);
        calc_intermediates.push_back(clsize);
        calc_intermediates.push_back(points_set.order);
        calc_intermediates.push_back(hydro_st.should_evolve);

        cqueue.exec("calculate_hydro_intermediates", calc_intermediates, {points_set.all_count}, {128});
    }

    {
        cl::args visco;
        visco.push_back(points_set.all_points);
        visco.push_back(points_set.all_count);

        for(auto& buf : in.buffers)
        {
            visco.push_back(buf.buf.as_device_read_only());
        }

        for(auto& i : intermediates)
        {
            visco.push_back(i);
        }

        visco.push_back(w_buf);

        visco.push_back(scale);
        visco.push_back(clsize);
        visco.push_back(points_set.order);
        visco.push_back(hydro_st.should_evolve);

        cqueue.exec("add_hydro_artificial_viscosity", visco, {points_set.all_count}, {128});
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

        evolve.push_back(w_buf.as_device_read_only());

        evolve.push_back(scale);
        evolve.push_back(clsize);
        evolve.push_back(points_set.order);
        evolve.push_back(hydro_st.should_evolve);
        evolve.push_back(timestep);

        cqueue.exec("evolve_hydro_all", evolve, {points_set.all_count}, {128});
    }

    auto clean_by_name = [&](const std::string& name)
    {
        clean_buffer(cqueue, in.lookup(name).buf, out.lookup(name).buf, base.lookup(name).buf, in.lookup(name).asymptotic_value, in.lookup(name).wave_speed, timestep);
    };

    if(sett.use_matter_colour)
    {
        std::vector<std::string> cols = {"dRed", "dGreen", "dBlue"};

        for(const std::string& buf_name : cols)
        {
            cl::buffer buf_in = in.lookup(buf_name).buf;
            cl::buffer buf_out = out.lookup(buf_name).buf;
            cl::buffer buf_base = base.lookup(buf_name).buf;

            cl::args advect;
            advect.push_back(points_set.all_points);
            advect.push_back(points_set.all_count);

            for(auto& buf : in.buffers)
            {
                advect.push_back(buf.buf.as_device_read_only());
            }

            advect.push_back(w_buf.as_device_read_only());

            advect.push_back(buf_base.as_device_read_only());
            advect.push_back(buf_in.as_device_read_only());
            advect.push_back(buf_out.as_device_write_only());

            advect.push_back(scale);
            advect.push_back(clsize);
            advect.push_back(points_set.order);
            advect.push_back(hydro_st.should_evolve);
            advect.push_back(timestep);

            cqueue.exec("hydro_advect", advect, {points_set.all_count}, {128});
        }

        clean_by_name("dRed");
        clean_by_name("dGreen");
        clean_by_name("dBlue");
    }

    clean_by_name("Dp_star");
    clean_by_name("De_star");
    clean_by_name("DcS0");
    clean_by_name("DcS1");
    clean_by_name("DcS2");
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

    auto& base_yn = data[0];

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
        clean_buffer(mqueue, in_buf.buf, out_buf.buf, base_buf.buf, in_buf.asymptotic_value, in_buf.wave_speed, current_timestep);
    };

    auto step = [&](auto& generic_in, auto& generic_out, float current_timestep, bool first)
    {
        step_hydro(ctx, mqueue, pool, generic_in, generic_out, base_yn, current_timestep);

        std::vector<ref_counted_buffer> intermediates = get_derivatives_of(ctx, generic_in, mqueue, pool);

        if(first)
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

            ///clean
            for(int i=0; i < (int)generic_in.buffers.size(); i++)
            {
                named_buffer& buf_in = generic_in.buffers[i];
                named_buffer& buf_base = base_yn.buffers[i];
                named_buffer& buf_out = generic_out.buffers[i];

                if(buf_in.modified_by != name)
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

    //#define BACKWARD_EULER
    #ifdef BACKWARD_EULER
    int iterations = 2;

    if(iterations == 1)
    {
        printf("You're going to forget every single time when you change this for debugging reasons, this will cause everything to break\n");
    }

    for(int i=0; i < iterations; i++)
    {
        if(i != 0)
            step(data[2], data[1], timestep, false);
        else
            step(data[0], data[1], timestep, true);

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
            dissipate_set(mqueue, data[0], data[1], points_set, timestep, dim, scale);
            enforce_constraints(data[1]);

            std::swap(data[1], data[2]);
            #endif // DISS_UNIDIR
        }
    }
    #endif

    auto post_step = [&](auto& buf)
    {
        dissipate_set(mqueue, data[0], buf, points_set, timestep, dim, scale);
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
            acc.push_back(base_yn.buffers[i].buf);
            acc.push_back(q_val.buffers[i].buf);
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

    auto temp_2 = free_data.get(data_get);

    ///temp_1 == q1
    step(data[0], temp_1, timestep * 0.5f, true);

    accumulator(temp_1, accum, 2.f/6.f);

    post_step(temp_1);

    ///temp_2 == q2
    step(temp_1, temp_2, timestep * 0.5f, false);

    accumulator(temp_2, accum, 4.f/6.f);

    post_step(temp_2);

    ///temp_1 now == q3
    step(temp_2, temp_1, timestep, false);

    accumulator(temp_1, accum, 2.f/6.f);

    post_step(temp_1);

    step(temp_1, temp_2, timestep, false);

    accumulator(temp_2, accum, 1.f/6.f);

    //post_step(temp_2);

    free_data.give_back(std::move(temp_2));

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
    dissipate_set(mqueue, data[0], data[1], points_set, timestep, dim, scale);
    #endif

    //dissipate(get_input().buffers, get_output().buffers);

    //clean(scratch.buffers, b2.buffers);

    enforce_constraints(data[1]);

    mqueue.end_splice(main_queue);

    std::swap(data[1], data[0]);
}
