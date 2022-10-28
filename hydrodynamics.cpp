#include "hydrodynamics.hpp"

eularian_hydrodynamics::eularian_hydrodynamics(cl::context& ctx, matter_initial_vars _vars, cl::buffer _u_arg) : hydro_st(ctx), vars(_vars), u_arg(_u_arg){}

void eularian_hydrodynamics::init(cpu_mesh& mesh, cl::context& ctx, cl::command_queue& cqueue, thin_intermediates_pool& pool, buffer_set& to_init)
{
    vec3i dim = mesh.dim;
    cl_float scale = mesh.scale;

    cl_int4 clsize = {dim.x(), dim.y(), dim.z(), 0};

    hydro_st.should_evolve.alloc(dim.x() * dim.y() * dim.z() * sizeof(cl_char));
    hydro_st.should_evolve.fill(cqueue, cl_char{1});

    cl::args hydro_init;

    for(auto& i : to_init.buffers)
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

    cl_int use_colour = mesh.sett.use_matter_colour;

    hydro_init.push_back(use_colour);

    cqueue.exec("calculate_hydrodynamic_initial_conditions", hydro_init, {dim.x(), dim.y(), dim.z()}, {8, 8, 1});

    vars = matter_initial_vars(ctx);
    u_arg = cl::buffer(ctx);
}

void eularian_hydrodynamics::step(cpu_mesh& mesh, cl::context& ctx, cl::managed_command_queue& cqueue, thin_intermediates_pool& pool, buffer_set& in, buffer_set& out, buffer_set& base, float timestep)
{
    vec3i dim = mesh.dim;
    cl_int4 clsize = {dim.x(), dim.y(), dim.z(), 0};
    cl_float scale = mesh.scale;
    auto& points_set = mesh.points_set;

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
        mesh.clean_buffer(cqueue, in.lookup(name).buf, out.lookup(name).buf, base.lookup(name).buf, in.lookup(name).desc.asymptotic_value, in.lookup(name).desc.wave_speed, timestep);
    };

    if(mesh.sett.use_matter_colour)
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
