#include "raytracing.hpp"
#include "bssn.hpp"
#include "single_source.hpp"

using single_source::named_buffer;
using single_source::named_literal;

void get_raytraced_quantities(single_source::argument_generator& arg_gen, equation_context& ctx, base_bssn_args& bssn_args)
{
    ctx.add_function("buffer_index", buffer_index_f<value, 3>);
    ctx.add_function("buffer_indexh", buffer_index_f<value_h, 3>);
    ctx.add_function("buffer_read_linear", buffer_read_linear_f<value, 3>);

    arg_gen.add(bssn_args.buffers);
    arg_gen.add<named_literal<v4i, "dim">>();
    arg_gen.add<named_literal<v4i, "out_dim">>();

    v3i in_dim = {"dim.x", "dim.y", "dim.z"};
    v3i out_dim = {"out_dim.x", "out_dim.y", "out_dim.z"};

    auto Yij_out = arg_gen.add<std::array<buffer<value>, 6>>();
    auto Kij_out = arg_gen.add<std::array<buffer<value>, 6>>();
    auto gA_out = arg_gen.add<buffer<value>>();
    auto gB_out = arg_gen.add<std::array<buffer<value>, 3>>();
    //auto slice = arg_gen.add<literal<value_i>>();

    ctx.exec("int ix = get_global_id(0)");
    ctx.exec("int iy = get_global_id(1)");
    ctx.exec("int iz = get_global_id(2)");

    v3i pos = {"ix", "iy", "iz"};

    ctx.exec(if_s(pos.x() >= out_dim.x() || pos.y() >= out_dim.y() || pos.z() >= out_dim.z(), return_s));

    v3f in_dimf = (v3f)in_dim;
    v3f out_dimf = (v3f)out_dim;

    v3f in_ratio = in_dimf / out_dimf;

    v3f upper_pos = (v3f)pos * in_ratio;

    ctx.uses_linear = true;

    ctx.exec("float fx = " + type_to_string(upper_pos.x()));
    ctx.exec("float fy = " + type_to_string(upper_pos.y()));
    ctx.exec("float fz = " + type_to_string(upper_pos.z()));

    standard_arguments args(ctx);

    ///don't need to do the slice thing, because all rays share coordinate time
    value_i idx = pos.z() * out_dim.y() * out_dim.x() + pos.y() * out_dim.x() + pos.x();
    //value_i idx = slice * out_dim.z() * out_dim.y() * out_dim.x() + pos.z() * out_dim.y() * out_dim.x() + pos.y() * out_dim.x() + pos.x();

    for(int i=0; i < 6; i++)
    {
        vec2i vidx = args.linear_indices[i];

        ctx.exec(assign(Yij_out[i][idx], args.Yij[vidx.x(), vidx.y()]));
        ctx.exec(assign(Kij_out[i][idx], args.Kij[vidx.x(), vidx.y()]));
    }

    for(int i=0; i < 3; i++)
    {
        ctx.exec(assign(gB_out[i][idx], args.gB[i]));
    }

    ctx.exec(assign(gA_out[idx], args.gA));
}

lightray make_lightray(equation_context& ctx,
                       const tensor<value, 3>& world_position, const tensor<value, 4>& camera_quat, v2i screen_size, v2i xy,
                       const metric<value, 3, 3>& Yij, const value& gA, const tensor<value, 3>& gB)
{
    value cx = (value)xy.x();
    value cy = (value)xy.y();

    float FOV = 90;

    float fov_rad = (FOV / 360.f) * 2 * M_PI;

    value nonphysical_plane_half_width = (value)screen_size.x()/2;
    value nonphysical_f_stop = nonphysical_plane_half_width / tan(fov_rad/2);

    tensor<value, 3> pixel_direction = {cx - (value)screen_size.x()/2, cy - (value)screen_size.y()/2, nonphysical_f_stop};

    pixel_direction = rot_quat(pixel_direction, camera_quat);

    ctx.pin(pixel_direction);

    pixel_direction = pixel_direction.norm();

    metric<value, 4, 4> real_metric = calculate_real_metric(Yij, gA, gB);

    ctx.pin(real_metric);

    frame_basis basis = calculate_frame_basis(ctx, real_metric);

    vec<4, value> e0 = basis.v1;
    vec<4, value> e1 = basis.v2;
    vec<4, value> e2 = basis.v3;
    vec<4, value> e3 = basis.v4;

    ctx.pin(e0);
    ctx.pin(e1);
    ctx.pin(e2);
    ctx.pin(e3);

    vec<4, value> basis_x = e2;
    vec<4, value> basis_y = e3;
    vec<4, value> basis_z = e1;

    bool should_orient = true;

    if(should_orient)
    {
        tetrad tet = {e0, e1, e2, e3};
        inverse_tetrad itet = get_tetrad_inverse(tet);

        ctx.pin(itet.e[0]);
        ctx.pin(itet.e[1]);
        ctx.pin(itet.e[2]);
        ctx.pin(itet.e[3]);

        vec<4, value> cartesian_basis_x = {0, 1, 0, 0};
        vec<4, value> cartesian_basis_y = {0, 0, 1, 0};
        vec<4, value> cartesian_basis_z = {0, 0, 0, 1};

        vec<4, value> tE1 = coordinate_to_tetrad_basis(cartesian_basis_y, itet);
        vec<4, value> tE2 = coordinate_to_tetrad_basis(cartesian_basis_x, itet);
        vec<4, value> tE3 = coordinate_to_tetrad_basis(cartesian_basis_z, itet);

        ctx.pin(tE1);
        ctx.pin(tE2);
        ctx.pin(tE3);

        ortho_result result = orthonormalise(tE1.yzw(), tE2.yzw(), tE3.yzw());

        basis_x = {0, result.v2.x(), result.v2.y(), result.v2.z()};
        basis_y = {0, result.v1.x(), result.v1.y(), result.v1.z()};
        basis_z = {0, result.v3.x(), result.v3.y(), result.v3.z()};
        ///basis_t == e0
    }

    tetrad oriented = {e0, basis_x, basis_y, basis_z};

    tensor<value, 4> observer_velocity = {oriented.e[0][0], oriented.e[0][1], oriented.e[0][2], oriented.e[0][3]};

    vec<4, value> pixel_x = pixel_direction.x() * oriented.e[1];
    vec<4, value> pixel_y = pixel_direction.y() * oriented.e[2];
    vec<4, value> pixel_z = pixel_direction.z() * oriented.e[3];
    vec<4, value> pixel_t = -oriented.e[0];

    #define INVERT_TIME
    #ifdef INVERT_TIME
    pixel_t = -pixel_t;
    #endif // INVERT_TIME

    vec<4, value> lightray_velocity = pixel_x + pixel_y + pixel_z + pixel_t;
    tensor<value, 4> lightray_position = {0, world_position.x(), world_position.y(), world_position.z()};

    tensor<value, 4> tensor_velocity = {lightray_velocity.x(), lightray_velocity.y(), lightray_velocity.z(), lightray_velocity.w()};

    tensor<value, 4> tensor_velocity_lowered = lower_index(tensor_velocity, real_metric, 0);

    value ku_uobsu = sum_multiply(tensor_velocity_lowered, observer_velocity);

    //ctx.add("GET_KU_UOBSU", ku_uobsu);

    tensor<value, 4> N = get_adm_hypersurface_normal_raised(gA, gB);

    value E = -sum_multiply(tensor_velocity_lowered, N);

    tensor<value, 4> adm_velocity = (tensor_velocity / E) - N;

    lightray ret;
    ret.adm_pos = world_position;
    ret.adm_vel = {adm_velocity[1], adm_velocity[2], adm_velocity[3]};

    ret.ku_uobsu = ku_uobsu;

    ret.pos4 = lightray_position;
    ret.vel4 = tensor_velocity;

    return ret;
}

void init_slice_rays(equation_context& ctx, literal<v3f> camera_pos, literal<v4f> camera_quat, literal<v2i> screen_size,
                     std::array<buffer<value, 3>, 6> linear_Yij_1, std::array<buffer<value, 3>, 6> linear_Kij_1, buffer<value, 3> linear_gA_1, std::array<buffer<value, 3>, 3> linear_gB_1,
                     named_literal<value, "scale"> scale, named_literal<v4i, "dim"> dim,
                     std::array<buffer<value>, 3> positions_out, std::array<buffer<value>, 3> velocities_out
                     )
{
    ctx.add_function("buffer_index", buffer_index_f<value, 3>);
    ctx.add_function("buffer_indexh", buffer_index_f<value_h, 3>);
    ctx.add_function("buffer_read_linear", buffer_read_linear_f<value, 3>);

    ctx.order = 1;
    ctx.uses_linear = true;

    metric<value, 3, 3> Yij;
    tensor<value, 3, 3> Kij;
    tensor<value, 3> gB;
    value gA;

    int index_table[3][3] = {{0, 1, 2},
                             {1, 3, 4},
                             {2, 4, 5}};

    v3f pos = camera_pos.get();

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            int tidx = index_table[i][j];

            Yij[i, j] = buffer_index_generic(linear_Yij_1[tidx], pos, dim.name);
            Kij[i, j] = buffer_index_generic(linear_Kij_1[tidx], pos, dim.name);
        }

        gB[i] = buffer_index_generic(linear_gB_1[i], pos, dim.name);
    }

    gA = buffer_index_generic(linear_gA_1, pos, dim.name);

    v2i in_xy = {"get_global_id(0)", "get_global_id(1)"};

    v2i xy = declare(ctx, in_xy);

    ctx.exec(if_s(xy.x() >= screen_size.get().x() || xy.y() >= screen_size.get().y(), return_s));

    lightray ray = make_lightray(ctx, camera_pos.get(), camera_quat.get(), screen_size.get(), xy, Yij, gA, gB);

    value_i out_idx = xy.y() * screen_size.get().x() + xy.x();

    positions_out[0][out_idx] = ray.adm_pos.x();
    positions_out[1][out_idx] = ray.adm_pos.y();
    positions_out[2][out_idx] = ray.adm_pos.z();

    velocities_out[0][out_idx] = ray.adm_vel.x();
    velocities_out[1][out_idx] = ray.adm_vel.y();
    velocities_out[2][out_idx] = ray.adm_vel.z();

    ctx.fix_buffers();
}

struct render_ray_info : single_source::struct_base<render_ray_info>
{
    static constexpr std::string type = "ray_render_info";

    literal<value> X, Y, Z;
    literal<value> dX, dY, dZ;

    literal<value_i> hit_type;

    literal<value> R, G, B;
    literal<value> background_power;

    literal<value_i> x, y;
    literal<value> zp1;

    auto as_tuple()
    {
        return std::tie(X, Y, Z, dX, dY, dZ, hit_type, R, G, B, background_power, x, y, zp1);
    }
};

void trace_slice(equation_context& ctx,
                 std::array<buffer<value, 3>, 6> linear_Yij_1, std::array<buffer<value, 3>, 6> linear_Kij_1, buffer<value, 3> linear_gA_1, std::array<buffer<value, 3>, 3> linear_gB_1,
                 std::array<buffer<value, 3>, 6> linear_Yij_2, std::array<buffer<value, 3>, 6> linear_Kij_2, buffer<value, 3> linear_gA_2, std::array<buffer<value, 3>, 3> linear_gB_2,
                 named_literal<value, "scale"> scale, named_literal<v4i, "dim"> dim, literal<v2i> screen_size,
                 std::array<buffer<value>, 3> positions, std::array<buffer<value>, 3> velocities,
                 buffer<value_i> terminated,
                 std::array<buffer<value>, 3> positions_out, std::array<buffer<value>, 3> velocities_out, literal<value_i> ray_count, literal<value> frac, literal<value> slice_width, literal<value> step,
                 buffer<render_ray_info>& render_out)
{
    ctx.add_function("buffer_index", buffer_index_f<value, 3>);
    ctx.add_function("buffer_indexh", buffer_index_f<value_h, 3>);
    ctx.add_function("buffer_read_linear", buffer_read_linear_f<value, 3>);

    ctx.ignored_variables.push_back(frac.name);

    ctx.order = 1;
    ctx.uses_linear = true;

    //ctx.exec("int lidx = get_global_id(0)");

    //value_i lidx = "lidx";

    value_i x = declare(ctx, value_i{"get_global_id(0)"});
    value_i y = declare(ctx, value_i{"get_global_id(1)"});

    value_i lidx = y * screen_size.get().x() + x;

    ctx.exec(if_s(lidx >= ray_count, return_s));

    ctx.exec(if_s(terminated[lidx] > 0, return_s));

    value local_frac = declare(ctx, frac.get());

    ctx.ignored_variables.push_back(type_to_string(local_frac));

    v3f pos = {positions[0][lidx], positions[1][lidx], positions[2][lidx]};
    v3f vel = {velocities[0][lidx], velocities[1][lidx], velocities[2][lidx]};

    auto w2v = [&](v3f in)
    {
        v3i centre = (dim.get().xyz() - 1)/2;

        return (in / scale) + (v3f)centre;
    };

    v3f voxel_pos = w2v(pos);

    value universe_size = ((dim.get().x()-1)/2).convert<float>() * scale;

    value u_sq = universe_size * universe_size * 0.95f * 0.95f;

    int index_table[3][3] = {{0, 1, 2},
                             {1, 3, 4},
                             {2, 4, 5}};

    metric<value, 3, 3> Yij;
    tensor<value, 3, 3> Kij;
    tensor<value, 3> gB;
    value gA;

    value until_out = frac.get() * slice_width;
    value steps = ceil(until_out / step.get());

    v3f loop_pos = declare(ctx, pos);
    v3f loop_vel = declare(ctx, vel);

    ctx.position_override = {type_to_string(loop_pos[0]), type_to_string(loop_pos[1]), type_to_string(loop_pos[2])};

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            int tidx = index_table[i][j];

            Yij[i, j] = mix(buffer_index_generic(linear_Yij_1[tidx], loop_pos, dim.name), buffer_index_generic(linear_Yij_2[tidx], loop_pos, dim.name), local_frac);
            Kij[i, j] = mix(buffer_index_generic(linear_Kij_1[tidx], loop_pos, dim.name), buffer_index_generic(linear_Kij_2[tidx], loop_pos, dim.name), local_frac);
        }

        gB[i] = mix(buffer_index_generic(linear_gB_1[i], loop_pos, dim.name), buffer_index_generic(linear_gB_2[i], loop_pos, dim.name), local_frac);
    }

    gA = mix(buffer_index_generic(linear_gA_1, loop_pos, dim.name), buffer_index_generic(linear_gA_2, loop_pos, dim.name), local_frac);

    ctx.pin(Kij);

    tensor<value, 3> dx;
    tensor<value, 3> V_upper_diff;

    {
        inverse_metric<value, 3, 3> iYij = Yij.invert();

        ctx.pin(iYij);

        tensor<value, 3, 3, 3> full_christoffel2 = christoffel_symbols_2(ctx, Yij, iYij);

        ctx.pin(full_christoffel2);

        tensor<value, 3> V_upper = loop_vel;

        value length_sq = dot_metric(V_upper, V_upper, Yij);

        value length = sqrt(fabs(length_sq));

        V_upper = V_upper / length;

        dx = gA * V_upper - gB;

        for(int i=0; i < 3; i++)
        {
            V_upper_diff.idx(i) = 0;

            for(int j=0; j < 3; j++)
            {
                value kjvk = 0;

                for(int k=0; k < 3; k++)
                {
                    kjvk += Kij.idx(j, k) * V_upper.idx(k);
                }

                value christoffel_sum = 0;

                for(int k=0; k < 3; k++)
                {
                    christoffel_sum += full_christoffel2.idx(i, j, k) * V_upper.idx(k);
                }

                value dlog_gA = diff1(ctx, gA, j) / gA;

                V_upper_diff.idx(i) += gA * V_upper.idx(j) * (V_upper.idx(i) * (dlog_gA - kjvk) + 2 * raise_index(Kij, iYij, 0).idx(i, j) - christoffel_sum)
                                       - iYij.idx(i, j) * diff1(ctx, gA, j) - V_upper.idx(j) * diff1(ctx, gB.idx(i), j);

            }
        }
    }

    value_i hit_type = declare(ctx, value_i{-1});

    value frac_increment = step.get() / slice_width.get();

    //ctx.exec("for(int i=0; i < " + type_to_string(steps) + "; i++) {");

    ctx.exec(for_s("idx", value_i(0), (value_i)steps, value_i("idx++")));

    {
        v3f dpos = declare(ctx, dx);
        v3f dvel = declare(ctx, V_upper_diff);

        value pos_sq = loop_pos.squared_length();

        value escape_cond = pos_sq >= u_sq;
        value ingested_cond = dpos.squared_length() < 0.2f * 0.2f;

        ctx.exec(if_s(escape_cond,
                        (assign(hit_type, value_i{0}),
                         break_s)
                      ));

        ctx.exec(if_s(ingested_cond,
                      (assign(hit_type, value_i{1}),
                       break_s)
                      ));

        ctx.exec(assign(loop_pos, loop_pos + dpos * step.get()));
        ctx.exec(assign(loop_vel, loop_vel + dvel * step.get()));

        ctx.exec(assign(local_frac, clamp(local_frac - frac_increment, value{0.f}, value{1.f})));
    }

    ctx.exec(for_end());

    //ctx.exec("}");

    ctx.exec("if(" + type_to_string(x==128 && y == 128) + "){printf(\"%i\", " + type_to_string(hit_type) + ");}");

    render_ray_info out = render_out[lidx];

    value_v on_terminated = (assign(terminated[lidx], value_i{1}),
                             assign(out.x.get(), x),
                             assign(out.y.get(), y),
                             assign(out.X.get(), loop_pos.x()),
                             assign(out.Y.get(), loop_pos.y()),
                             assign(out.Z.get(), loop_pos.z()),
                             assign(out.dX.get(), dx.x()),
                             assign(out.dY.get(), dx.y()),
                             assign(out.dZ.get(), dx.z()),
                             assign(out.hit_type.get(), hit_type),
                             assign(out.R.get(), value{0}),
                             assign(out.G.get(), value{0}),
                             assign(out.B.get(), value{0}),
                             assign(out.background_power.get(), value{1}),
                             assign(out.zp1.get(), value{1}));

    ctx.exec(assign(out.x.get(), x));
    ctx.exec(assign(out.y.get(), y));
    ctx.exec(assign(out.R.get(), value{0}));
    ctx.exec(assign(out.G.get(), value{1}));
    ctx.exec(assign(out.B.get(), value{0}));
    ctx.exec(assign(out.hit_type.get(), value_i{1}));

    ctx.exec(if_s(hit_type != value_i{-1},
                  on_terminated
                  ));

    ctx.exec(assign(positions_out[0][lidx], loop_pos.x()));
    ctx.exec(assign(positions_out[1][lidx], loop_pos.y()));
    ctx.exec(assign(positions_out[2][lidx], loop_pos.z()));

    ctx.exec(assign(velocities_out[0][lidx], loop_vel.x()));
    ctx.exec(assign(velocities_out[1][lidx], loop_vel.y()));
    ctx.exec(assign(velocities_out[2][lidx], loop_vel.z()));
}

void build_raytracing_kernels(cl::context& clctx, base_bssn_args& bssn_args)
{
    {
        equation_context ectx;

        cl::kernel kern = single_source::make_dynamic_kernel_for(clctx, ectx, get_raytraced_quantities, "get_raytraced_quantities", "", bssn_args);

        clctx.register_kernel("get_raytraced_quantities", kern);
    }

    {
        equation_context ectx;

        cl::kernel kern = single_source::make_kernel_for(clctx, ectx, init_slice_rays, "init_slice_rays", "");

        clctx.register_kernel("init_slice_rays", kern);
    }

    {
        equation_context ectx;

        cl::kernel kern = single_source::make_kernel_for(clctx, ectx, trace_slice, "trace_slice", "");

        clctx.register_kernel("trace_slice", kern);
    }
}

raytracing_manager::raytracing_manager(cl::context& clctx, const tensor<int, 2>& screen) : render_ray_info_buf(clctx)
{
    slice_size = {64, 64, 64};
    slice_width = 5;

    render_ray_info_buf.alloc(30 * sizeof(float) * 4096 * 4096);
}

std::vector<cl::buffer> raytracing_manager::get_fresh_buffers(cl::context& clctx)
{
    std::vector<cl::buffer> ret;
    ///so, Yij: 6 components
    ///Kij: 6 components
    ///gA,
    ///gB: 3 components

    for(int i=0; i < (6+6+1+3); i++)
    {
        ret.emplace_back(clctx).alloc(sizeof(cl_float) * slice_size.x() * slice_size.y() * slice_size.z());
    }

    return ret;
}

void raytracing_manager::trace(cl::context& clctx, cl::managed_command_queue& mqueue, float scale, const tensor<int, 2>& screen, vec3f camera_pos, vec4f camera_quat)
{
    if(slices.size() == 0)
        return;

    float last_slice_time = (slices.size() - 1) * slice_width;

    std::array<cl::buffer, 6> rays_1{clctx, clctx, clctx, clctx, clctx, clctx};
    std::array<cl::buffer, 6> rays_2{clctx, clctx, clctx, clctx, clctx, clctx};
    cl::buffer terminated(clctx);

    int size = screen.x() * screen.y() * sizeof(cl_float);

    terminated.alloc(size);
    terminated.set_to_zero(mqueue);

    for(int i=0; i < 6; i++)
    {
        rays_1[i].alloc(size);
        rays_2[i].alloc(size);

        rays_1[i].set_to_zero(mqueue);
        rays_2[i].set_to_zero(mqueue);
    }

    cl_float3 ccamera_pos = {camera_pos.x(), camera_pos.y(), camera_pos.z()};
    cl_float4 ccamera_quat = {camera_quat.x(), camera_quat.y(), camera_quat.z(), camera_quat.w()};
    cl_int4 mesh_dim = {slice_size.x(), slice_size.y(), slice_size.z()};
    cl_int2 clscreen = {screen.x(), screen.y()};

    {
        cl::args args;
        args.push_back(ccamera_pos, ccamera_quat, screen);

        std::vector<cl::buffer>& which_buf = slices.back();

        for(auto& i : which_buf)
        {
            args.push_back(i);
        }


        args.push_back(scale);
        args.push_back(mesh_dim);

        for(int i=0; i < 6; i++)
        {
            args.push_back(rays_1[i]);
        }

        mqueue.exec("init_slice_rays", args, {screen.x(), screen.y()}, {8,8});
    }

    auto get_slices_for_time = [&](float time) -> std::tuple<std::vector<cl::buffer>*, std::vector<cl::buffer>*, float>
    {
        if(slices.size() == 1)
            return {&slices[0], &slices[0], 0.f};

        if(time <= 0)
            return {&slices[0], &slices[0], 0.f};

        for(int i=0; i < (int)slices.size() - 1; i++)
        {
            float current_time = i * slice_width;
            float next_time = (i + 1) * slice_width;

            if(time >= current_time && time < next_time)
            {
                float frac = (time - current_time) / (next_time - current_time);

                return {&slices[i], &slices[i+1], frac};
            }
        }

        int last = slices.size() - 1;

        std::vector<cl::buffer>* last_buf = &slices[last];

        return {last_buf, last_buf, 0.f};
    };

    float my_time = slices.size() * slice_width;
    float my_step = 2.f;

    int steps = 100;

    for(int i=0; i < steps; i++)
    {
        float current_time = my_time - my_step * i;

        {
            cl::args args;

            auto [b1, b2, frac] = get_slices_for_time(current_time);

            for(int i=0; i < b1->size(); i++)
            {
                args.push_back((*b1)[i]);
            }

            for(int i=0; i < b2->size(); i++)
            {
                args.push_back((*b2)[i]);
            }

            args.push_back(scale, mesh_dim, clscreen);

            for(int i=0; i < 6; i++)
                args.push_back(rays_1[i]);

            args.push_back(terminated);

            for(int i=0; i < 6; i++)
                args.push_back(rays_2[i]);

            cl_int rays = screen.x() * screen.y();

            args.push_back(rays);
            args.push_back(frac);
            args.push_back(slice_width);
            args.push_back(my_step);
            args.push_back(render_ray_info_buf);

            mqueue.exec("trace_slice", args, {screen.x(), screen.y()}, {8,8});

            std::swap(rays_1, rays_2);
        }
    }


    /*void trace_slice(equation_context& ctx,
                 std::array<buffer<value, 3>, 6> linear_Yij_1, std::array<buffer<value, 3>, 6> linear_Kij_1, buffer<value, 3> linear_gA_1, std::array<buffer<value, 3>, 3> linear_gB_1,
                 std::array<buffer<value, 3>, 6> linear_Yij_2, std::array<buffer<value, 3>, 6> linear_Kij_2, buffer<value, 3> linear_gA_2, std::array<buffer<value, 3>, 3> linear_gB_2,
                 named_literal<value, "scale"> scale, named_literal<v4i, "dim"> dim, literal<v2i> screen_size,
                 std::array<buffer<value>, 3> positions, std::array<buffer<value>, 3> velocities,
                 buffer<value_i> terminated,
                 std::array<buffer<value>, 3> positions_out, std::array<buffer<value>, 3> velocities_out, literal<value_i> ray_count, literal<value> frac, literal<value> slice_width, literal<value> step,
                 buffer<render_ray_info>& render_out)*/

    /*
void init_slice_rays(equation_context& ctx, literal<v3f> camera_pos, literal<v4f> camera_quat, literal<v2i> screen_size,
                     std::array<buffer<value, 3>, 6> linear_Yij_1, std::array<buffer<value, 3>, 6> linear_Kij_1, buffer<value, 3> linear_gA_1, std::array<buffer<value, 3>, 3> linear_gB_1,
                     named_literal<value, "scale"> scale, named_literal<v4i, "dim"> dim,
                     std::array<buffer<value>, 3> positions_out, std::array<buffer<value>, 3> velocities_out
                     )*/
}

void raytracing_manager::grab_buffers(cl::context& clctx, cl::managed_command_queue& mqueue, const std::vector<cl::buffer>& bufs, float scale, const tensor<cl_int, 4>& clsize, float step)
{
    if((int)slices.size() >= max_slices)
        return;

    int c2 = floor(time_elapsed / slice_width);

    if(last_grabbed != c2)
    {
        last_grabbed = c2;

        std::cout << "Grabby\n";

        tensor<int, 4> out_clsize = {slice_size.x(), slice_size.y(), slice_size.z(), 0};

        ///take a snapshot!
        cl::args args;

        for(cl::buffer b : bufs)
        {
            args.push_back(b);
        }

        args.push_back(clsize);
        args.push_back(out_clsize);

        auto bufs = get_fresh_buffers(clctx);

        for(auto& i : bufs)
        {
            args.push_back(i);
        }

        mqueue.exec("get_raytraced_quantities", args, {slice_size.x(), slice_size.y(), slice_size.z()}, {8,8,1});

        slices.push_back(bufs);
    }

    time_elapsed += step;
}
