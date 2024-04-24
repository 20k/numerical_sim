#include "raytracing.hpp"
#include "bssn.hpp"
#include "single_source.hpp"
#include "mesh_manager.hpp"

using single_source::named_buffer;
using single_source::named_literal;

void get_raytraced_quantities(single_source::argument_generator& arg_gen, equation_context& ctx, base_bssn_args& bssn_args)
{
    using namespace dual_types::implicit;

    ctx.add_function("buffer_read_linear2", buffer_read_linear_f_unpacked<value>);

    arg_gen.add(bssn_args.buffers);
    arg_gen.add<named_literal<v4i, "dim">>();
    arg_gen.add<named_literal<v4i, "out_dim">>();

    v3i in_dim = {"dim.x", "dim.y", "dim.z"};
    v3i out_dim = {"out_dim.x", "out_dim.y", "out_dim.z"};

    auto Yij_out = arg_gen.add<std::array<buffer<value_mut>, 6>>();
    auto Kij_out = arg_gen.add<std::array<buffer<value_mut>, 6>>();
    auto gA_out = arg_gen.add<buffer<value_mut>>();
    auto gB_out = arg_gen.add<std::array<buffer<value_mut>, 3>>();
    //auto slice = arg_gen.add<literal<value_i>>();

    value_i ix = declare(value_i{"get_global_id(0)"}, "ix");
    value_i iy = declare(value_i{"get_global_id(1)"}, "iy");
    value_i iz = declare(value_i{"get_global_id(2)"}, "iz");

    v3i pos = {ix, iy, iz};

    if_e(pos.x() >= out_dim.x() || pos.y() >= out_dim.y() || pos.z() >= out_dim.z(), [&]()
    {
        return_e();
    });

    v3f in_dimf = (v3f)in_dim;
    v3f out_dimf = (v3f)out_dim;

    v3f in_ratio = in_dimf / out_dimf;

    v3f upper_pos = (v3f)pos * in_ratio;

    ctx.uses_linear = true;

    declare(upper_pos.x(), "fx");
    declare(upper_pos.y(), "fy");
    declare(upper_pos.z(), "fz");

    standard_arguments args(ctx);

    ///don't need to do the slice thing, because all rays share coordinate time
    value_i idx = pos.z() * out_dim.y() * out_dim.x() + pos.y() * out_dim.x() + pos.x();
    //value_i idx = slice * out_dim.z() * out_dim.y() * out_dim.x() + pos.z() * out_dim.y() * out_dim.x() + pos.y() * out_dim.x() + pos.x();

    for(int i=0; i < 6; i++)
    {
        vec2i vidx = args.linear_indices[i];

        mut(Yij_out[i][idx]) = args.Yij[vidx.x(), vidx.y()];
        mut(Kij_out[i][idx]) = args.Kij[vidx.x(), vidx.y()];
    }

    for(int i=0; i < 3; i++)
    {
        mut(gB_out[i][idx]) = args.gB[i];
    }

    mut(gA_out[idx]) = args.gA;
}

lightray make_lightray(equation_context& ctx,
                       const tensor<value, 3>& world_position, const tensor<value, 4>& camera_quat, v2i screen_size, v2i xy,
                       const metric<value, 3, 3>& Yij, const value& gA, const tensor<value, 3>& gB)
{
    metric<value, 4, 4> real_metric = calculate_real_metric(Yij, gA, gB);

    ctx.pin(real_metric);

    return make_lightray(ctx, world_position, camera_quat, screen_size, xy, real_metric, gA, gB);
}

lightray make_lightray(equation_context& ctx,
                       const tensor<value, 3>& world_position, const tensor<value, 4>& camera_quat, v2i screen_size, v2i xy,
                       const metric<value, 4, 4>& real_metric, const value& gA, const tensor<value, 3>& gB)
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

    //#define INVERT_TIME
    #ifdef INVERT_TIME
    pixel_t = -pixel_t;
    #endif // INVERT_TIME

    vec<4, value> lightray_velocity = pixel_x + pixel_y + pixel_z + pixel_t;
    tensor<value, 4> lightray_position = {0, world_position.x(), world_position.y(), world_position.z()};

    tensor<value, 4> tensor_velocity = {lightray_velocity.x(), lightray_velocity.y(), lightray_velocity.z(), lightray_velocity.w()};

    tensor<value, 4> tensor_velocity_lowered = lower_index(tensor_velocity, real_metric, 0);

    value ku_uobsu = sum_multiply(tensor_velocity_lowered, observer_velocity);

    //ctx.add("GET_KU_UOBSU", ku_uobsu);

    ///https://arxiv.org/pdf/gr-qc/0703035.pdf todo

    tensor<value, 4> N = get_adm_hypersurface_normal_raised(gA, gB);

    value E = -sum_multiply(tensor_velocity_lowered, N);

    tensor<value, 4> adm_velocity = -((tensor_velocity / E) - N);

    lightray ret;
    ret.adm_pos = world_position;
    ret.adm_vel = {adm_velocity[1], adm_velocity[2], adm_velocity[3]};

    ret.ku_uobsu = ku_uobsu;

    ret.pos4 = lightray_position;
    ret.vel4 = tensor_velocity;

    return ret;
}

void init_slice_rays(equation_context& ctx, literal<v3f> camera_pos, literal<v4f> camera_quat, literal<v2i> screen_size,
                     std::array<buffer<value>, 6> linear_Yij_1, std::array<buffer<value>, 6> linear_Kij_1, buffer<value> linear_gA_1, std::array<buffer<value>, 3> linear_gB_1,
                     std::array<buffer<value>, 6> linear_Yij_2, std::array<buffer<value>, 6> linear_Kij_2, buffer<value> linear_gA_2, std::array<buffer<value>, 3> linear_gB_2,
                     literal<value> frac,
                     named_literal<value, "scale"> scale, named_literal<v4i, "dim"> dim,
                     std::array<buffer<value_mut>, 3> positions_out, std::array<buffer<value_mut>, 3> velocities_out
                     )
{
    using namespace dual_types::implicit;

    ctx.add_function("buffer_read_linear2", buffer_read_linear_f_unpacked<value>);

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

    auto w2v = [&](v3f in)
    {
        v3i centre = (dim.get().xyz() - 1)/2;

        return (in / scale) + (v3f)centre;
    };

    v3i dim3 = dim.get().xyz();

    v3f voxel_pos = w2v(pos);

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            int tidx = index_table[i][j];

            Yij[i, j] = mix(buffer_index_generic(linear_Yij_1[tidx], voxel_pos, dim3), buffer_index_generic(linear_Yij_2[tidx], voxel_pos, dim3), frac.get());
            Kij[i, j] = mix(buffer_index_generic(linear_Kij_1[tidx], voxel_pos, dim3), buffer_index_generic(linear_Kij_2[tidx], voxel_pos, dim3), frac.get());
        }

        gB[i] = mix(buffer_index_generic(linear_gB_1[i], voxel_pos, dim3), buffer_index_generic(linear_gB_2[i], voxel_pos, dim3), frac.get());
    }

    gA = mix(buffer_index_generic(linear_gA_1, voxel_pos, dim3), buffer_index_generic(linear_gA_2, voxel_pos, dim3), frac.get());

    v2i in_xy = {"get_global_id(0)", "get_global_id(1)"};

    v2i xy = declare(ctx, in_xy);

    if_e(xy.x() >= screen_size.get().x() || xy.y() >= screen_size.get().y(), [&]()
    {
        return_e();
    });

    lightray ray = make_lightray(ctx, camera_pos.get(), camera_quat.get(), screen_size.get(), xy, Yij, gA, gB);

    value_i out_idx = xy.y() * screen_size.get().x() + xy.x();

    ctx.exec(assign(positions_out[0][out_idx], ray.adm_pos.x()));
    ctx.exec(assign(positions_out[1][out_idx], ray.adm_pos.y()));
    ctx.exec(assign(positions_out[2][out_idx], ray.adm_pos.z()));

    ctx.exec(assign(velocities_out[0][out_idx], ray.adm_vel.x()));
    ctx.exec(assign(velocities_out[1][out_idx], ray.adm_vel.y()));
    ctx.exec(assign(velocities_out[2][out_idx], ray.adm_vel.z()));
}

struct render_ray_info : single_source::struct_base<render_ray_info>
{
    std::string type = "ray_render_info";

    literal<value_mut> X, Y, Z;
    literal<value_mut> dX, dY, dZ;

    literal<value_i_mut> hit_type;

    literal<value_mut> R, G, B;
    literal<value_mut> background_power;

    literal<value_i_mut> x, y;
    literal<value_mut> zp1;

    auto as_tuple()
    {
        return std::tie(X, Y, Z, dX, dY, dZ, hit_type, R, G, B, background_power, x, y, zp1);
    }
};

void trace_slice(equation_context& ctx,
                 std::array<buffer<value>, 6> linear_Yij_1, std::array<buffer<value>, 6> linear_Kij_1, buffer<value> linear_gA_1, std::array<buffer<value>, 3> linear_gB_1,
                 std::array<buffer<value>, 6> linear_Yij_2, std::array<buffer<value>, 6> linear_Kij_2, buffer<value> linear_gA_2, std::array<buffer<value>, 3> linear_gB_2,
                 named_literal<value, "scale"> scale, named_literal<v4i, "dim"> dim, literal<v2i> screen_size, literal<value_i> iteration,
                 std::array<buffer<value>, 3> positions, std::array<buffer<value>, 3> velocities,
                 buffer<value_i_mut> terminated,
                 std::array<buffer<value_mut>, 3> positions_out, std::array<buffer<value_mut>, 3> velocities_out, literal<value_i> ray_count, literal<value> frac, literal<value> slice_width, literal<value> step,
                 buffer<render_ray_info>& render_out)
{
    using namespace dual_types::implicit;

    ctx.add_function("buffer_read_linear2", buffer_read_linear_f_unpacked<value>);

    ctx.ignored_variables.push_back(frac.name);

    ctx.order = 1;
    ctx.uses_linear = true;

    //ctx.exec("int lidx = get_global_id(0)");

    //value_i lidx = "lidx";

    value_i x = declare(value_i{"get_global_id(0)"});
    value_i y = declare(ctx, value_i{"get_global_id(1)"});

    value_i lidx = y * screen_size.get().x() + x;

    if_e(lidx >= ray_count, [&]()
    {
        return_e();
    });

    if_e(terminated[lidx] > 0, [&]()
    {
        for(int i=0; i < 3; i++)
        {
            mut(positions_out[i][lidx]) = positions[i][lidx];
            mut(velocities_out[i][lidx]) = velocities[i][lidx];
        }

        return_e();
    });

    value_mut local_frac = declare_mut(ctx, frac.get());

    ctx.ignored_variables.push_back(type_to_string(local_frac));

    v3f pos = {positions[0][lidx], positions[1][lidx], positions[2][lidx]};
    v3f vel = {velocities[0][lidx], velocities[1][lidx], velocities[2][lidx]};

    //ctx.exec("if(" + type_to_string(x==128 && y == 128) + "){printf(\"base pos %f %f %f\", " + type_to_string(pos.x()) + "," + type_to_string(pos.y()) + "," + type_to_string(pos.z()) + ");}");

    //ctx.exec(return_v());

    auto w2v = [&](v3f in)
    {
        v3i centre = (dim.get().xyz() - 1)/2;

        return (in / scale) + (v3f)centre;
    };

    auto v2w = [&](v3f in)
    {
        v3i centre = (dim.get().xyz() - 1)/2;

        return (in - (v3f)centre) * scale;
    };

    v3f voxel_pos = w2v(pos);

    value universe_size = ((dim.get().x()-1)/2).convert<float>() * scale;

    value u_sq = universe_size * universe_size * 0.95f * 0.95f;

    int index_table[3][3] = {{0, 1, 2},
                             {1, 3, 4},
                             {2, 4, 5}};

    v3i dim3 = dim.get().xyz();

    metric<value, 3, 3> Yij;
    tensor<value, 3, 3> Kij;
    tensor<value, 3> gB;
    value gA;

    value until_out = frac.get() * slice_width;
    value steps = ceil(until_out / step.get());

    steps = max(steps, value{1.f});

    v3f_mut loop_voxel_pos = declare_mut(voxel_pos);
    v3f_mut loop_vel = declare_mut(vel);

    ctx.position_override = {type_to_string(loop_voxel_pos[0]), type_to_string(loop_voxel_pos[1]), type_to_string(loop_voxel_pos[2])};

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            int tidx = index_table[i][j];

            Yij[i, j] = mix(buffer_index_generic(linear_Yij_1[tidx], as_constant(loop_voxel_pos), dim3), buffer_index_generic(linear_Yij_2[tidx], as_constant(loop_voxel_pos), dim3), local_frac);
            Kij[i, j] = mix(buffer_index_generic(linear_Kij_1[tidx], as_constant(loop_voxel_pos), dim3), buffer_index_generic(linear_Kij_2[tidx], as_constant(loop_voxel_pos), dim3), local_frac);
        }

        gB[i] = mix(buffer_index_generic(linear_gB_1[i], as_constant(loop_voxel_pos), dim3), buffer_index_generic(linear_gB_2[i], as_constant(loop_voxel_pos), dim3), local_frac);
    }

    gA = mix(buffer_index_generic(linear_gA_1, as_constant(loop_voxel_pos), dim3), buffer_index_generic(linear_gA_2, as_constant(loop_voxel_pos), dim3), local_frac);

    ctx.pin(Kij);

    tensor<value, 3> dx;
    tensor<value, 3> V_upper_diff;

    {
        inverse_metric<value, 3, 3> iYij = Yij.invert();

        ctx.pin(iYij);

        tensor<value, 3, 3, 3> full_christoffel2 = christoffel_symbols_2(ctx, Yij, iYij);

        ctx.pin(full_christoffel2);

        tensor<value, 3> V_upper = as_constant(loop_vel);

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

    value_i_mut hit_type = declare_mut(ctx, value_i{-1});

    value frac_increment = step.get() / slice_width.get();

    //ctx.exec("for(int i=0; i < " + type_to_string(steps) + "; i++) {");

    //steps = 1;

    for_e("idx", value_i(0), value_i("idx") < (value_i)steps, value_i("idx++"),
    [&](const value_i& idx)
    {
        v3f dpos = declare(dx);
        v3f dvel = declare(V_upper_diff);

        value pos_sq = v2w(as_constant(loop_voxel_pos)).squared_length();

        value escape_cond = pos_sq >= u_sq;
        value ingested_cond = dpos.squared_length() < 0.1f * 0.1f;


        //ctx.exec("if(" + type_to_string(x==128 && y == 128) + "){printf(\"dx %f %f %f\", " + type_to_string(dpos.x()) + "," + type_to_string(dpos.y()) + "," + type_to_string(dpos.z()) + ");}");
        //ctx.exec("if(" + type_to_string(x==128 && y == 128) + "){printf(\"loop_vel %f %f %f\", " + type_to_string(loop_vel.x()) + "," + type_to_string(loop_vel.y()) + "," + type_to_string(loop_vel.z()) + ");}");

        if_e(escape_cond, [&]()
        {
            mut(hit_type) = 0;
            break_e();
        });

        if_e(ingested_cond, [&]()
        {
            mut(hit_type) = 1;
            break_e();
        });

        mut(loop_voxel_pos) = as_constant(loop_voxel_pos) + dpos * step.get() / scale;
        mut(loop_vel) = as_constant(loop_vel) + dvel * step.get();

        mut(local_frac) = clamp(local_frac - frac_increment, value{0.f}, value{1.f});
    });

    //ctx.exec("}");

    //ctx.exec("if(" + type_to_string(x==128 && y == 128) + "){printf(\"%i\", " + type_to_string((value_i)steps) + ");}");

    v3f fin_world_pos = v2w(as_constant(loop_voxel_pos));

    //ctx.exec("if(" + type_to_string(x==128 && y == 128) + "){printf(\"p2 %f %f %f\", " + type_to_string(fin_world_pos.x()) + "," + type_to_string(fin_world_pos.y()) + "," + type_to_string(fin_world_pos.z()) + ");}");

    render_ray_info out = render_out[lidx];

    mut(out.x.get()) = x;
    mut(out.y.get()) = y;
    mut(out.R.get()) = value{0};
    mut(out.G.get()) = value{1};
    mut(out.B.get()) = value{0};
    mut(out.hit_type.get()) = value_i{1};

    if_e(hit_type != value_i{-1}, [&]()
    {
        mut(terminated[lidx]) = value_i{1};
        mut(out.x.get()) = x;
        mut(out.y.get()) = y;
        mut(out.X.get()) = fin_world_pos.x();
        mut(out.Y.get()) = fin_world_pos.y();
        mut(out.Z.get()) = fin_world_pos.z();
        mut(out.dX.get()) = dx.x();
        mut(out.dY.get()) = dx.y();
        mut(out.dZ.get()) = dx.z();
        mut(out.hit_type.get()) = hit_type;
        mut(out.R.get()) = value{0};
        mut(out.G.get()) = value{0};
        mut(out.B.get()) = value{0};
        mut(out.background_power.get()) = value{1};
        mut(out.zp1.get()) = value{1};
    });

    mut(positions_out[0][lidx]) = fin_world_pos.x();
    mut(positions_out[1][lidx]) = fin_world_pos.y();
    mut(positions_out[2][lidx]) = fin_world_pos.z();

    mut(velocities_out[0][lidx]) = loop_vel.x();
    mut(velocities_out[1][lidx]) = loop_vel.y();
    mut(velocities_out[2][lidx]) = loop_vel.z();

    //ctx.exec("if(" + type_to_string(x==128 && y == 128) + "){printf(\"p3 %i %f\", " + type_to_string(hit_type) + ", " + type_to_string(u_sq) + ");}");
}

raytracing_manager::raytracing_manager(cl::context& clctx, const tensor<int, 2>& screen) : render_ray_info_buf(clctx)
{
    slice_size = {100, 100, 100};
    slice_width = 2;

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

void raytracing_manager::trace(cl::context& clctx, cl::command_queue& mqueue, float simulation_width, const tensor<int, 2>& screen, vec3f camera_pos, vec4f camera_quat, float camera_start_time)
{
    if(slices.size() == 0)
        return;

    float scale = calculate_scale(simulation_width, slice_size);

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

    float my_time = camera_start_time;

    {
        auto [b1, b2, frac] = get_slices_for_time(my_time);

        cl::args args;
        args.push_back(ccamera_pos, ccamera_quat, screen);

        for(auto& i : *b1)
        {
            args.push_back(i);
        }

        for(auto& i : *b2)
        {
            args.push_back(i);
        }

        args.push_back(frac);

        args.push_back(scale);
        args.push_back(mesh_dim);

        for(int i=0; i < 6; i++)
        {
            args.push_back(rays_1[i]);
        }

        mqueue.exec("init_slice_rays", args, {screen.x(), screen.y()}, {8,8});
    }

    //float my_time = slices.size() * slice_width;
    float my_step = 0.05f;

    int steps = 150;

    for(int i=0; i < steps; i++)
    {
        //float current_time = my_time - i * slice_width;

        //printf("Executing at time %f\n", my_time);

        {
            cl::args args;

            auto [b1, b2, frac] = get_slices_for_time(my_time);

            for(auto& i : *b1)
            {
                args.push_back(i);
            }

            for(auto& i : *b2)
            {
                args.push_back(i);
            }

            args.push_back(scale, mesh_dim, clscreen, i);

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
            //printf("Exec?\n");

            float until_out = frac * slice_width;
            float steps = ceil(until_out / my_step);

            steps = std::max(steps, 1.f);

            my_time -= steps * my_step;
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

void raytracing_manager::grab_buffers(cl::context& clctx, cl::command_queue& mqueue, const std::vector<cl::buffer>& bufs, float scale, const tensor<cl_int, 4>& clsize, float step)
{
    if((int)slices.size() >= max_slices)
        return;

    int c2 = ceil(time_elapsed / slice_width);

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

using ray4_value_mut = value_h_mut;
using ray4_value = value_h;

void get_raytraced_quantities4(single_source::argument_generator& arg_gen, equation_context& ctx, base_bssn_args& bssn_args)
{
    using namespace dual_types::implicit;

    ctx.add_function("buffer_read_linear2", buffer_read_linear_f_unpacked<value>);
    ctx.add_function("buffer_read_linearh2", buffer_read_linear_f_unpacked<value_h>);

    arg_gen.add(bssn_args.buffers);
    arg_gen.add<named_literal<v4i, "dim">>();
    arg_gen.add<named_literal<v4i, "out_dim">>();
    auto slice = arg_gen.add<named_literal<value_i, "slice">>();

    v3i in_dim = {"dim.x", "dim.y", "dim.z"};
    v3i out_dim = {"out_dim.x", "out_dim.y", "out_dim.z"};

    auto Guv_out = arg_gen.add<std::array<buffer<ray4_value_mut>, 10>>();

    value_i ix = declare(value_i{"get_global_id(0)"}, "ix");
    value_i iy = declare(value_i{"get_global_id(1)"}, "iy");
    value_i iz = declare(value_i{"get_global_id(2)"}, "iz");

    v3i pos = {ix, iy, iz};

    if_e(pos.x() >= out_dim.x() || pos.y() >= out_dim.y() || pos.z() >= out_dim.z(), [&]()
    {
        return_e();
    });

    v3f in_dimf = (v3f)in_dim;
    v3f out_dimf = (v3f)out_dim;

    v3f in_ratio = in_dimf / out_dimf;

    v3f upper_pos = (v3f)pos * in_ratio;

    ctx.uses_linear = true;

    declare(upper_pos.x(), "fx");
    declare(upper_pos.y(), "fy");
    declare(upper_pos.z(), "fz");

    standard_arguments args(ctx);

    /*std::array<buffer<ray4_value>, 6> linear_cY = {"cY0", "cY1", "cY2", "cY3", "cY4", "cY5"};
    buffer<ray4_value> linear_conformal = {"X"};
    buffer<ray4_value> linear_gA = {"gA"};
    std::array<buffer<ray4_value>, 3> linear_gB = {"gB0", "gB1", "gB2"};

    metric<value, 3, 3> Yij;
    value X = buffer_read_linear(linear_conformal, upper_pos, );
    value gA;
    tensor<value, 3> gB;*/

    value_i idx = slice.get() * out_dim.x() * out_dim.y() * out_dim.z() + pos.z() * out_dim.y() * out_dim.x() + pos.y() * out_dim.x() + pos.x();

    metric<value, 4, 4> Guv = calculate_real_metric(args.Yij, args.gA, args.gB);

    vec2i linear_indices[] = {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {1, 1}, {1, 2}, {1, 3}, {2, 2}, {2, 3}, {3, 3}};

    for(int i=0; i < 10; i++)
    {
        vec2i lidx = linear_indices[i];

        mut(Guv_out[i][idx]) = (ray4_value)Guv[lidx.x(), lidx.y()];
    }
}

raytracing4_manager::raytracing4_manager(cl::context& clctx, const tensor<int, 2>& screen) : render_ray_info_buf(clctx)
{
    slice_size = {100, 100, 100};
    slice_width = 4;

    render_ray_info_buf.alloc(30 * sizeof(float) * 2000 * 2000);

    for(int i=0; i < 10; i++)
    {
        cl::buffer buf(clctx);

        buf.alloc(sizeof(ray4_value::value_type) * slice_size.x() * slice_size.y() * slice_size.z() * max_slices);

        slice.push_back(buf);
    }
}

void raytracing4_manager::grab_buffers(cl::context& clctx, cl::command_queue& mqueue, const std::vector<cl::buffer>& bufs, float scale, const tensor<cl_int, 4>& clsize, float step)
{
    if(last_grabbed >= max_slices)
        return;

    int c2 = floor(time_elapsed / slice_width);

    if(c2 < 0)
        return;

    if(last_grabbed != c2)
    {
        std::cout << "Grabby4 " << c2 << std::endl;;

        tensor<int, 4> out_clsize = {slice_size.x(), slice_size.y(), slice_size.z(), 0};

        ///take a snapshot!
        cl::args args;

        for(cl::buffer b : bufs)
        {
            args.push_back(b);
        }

        args.push_back(clsize);
        args.push_back(out_clsize);

        args.push_back(c2);

        for(auto& i : slice)
        {
            args.push_back(i);
        }

        mqueue.exec("get_raytraced_quantities4", args, {slice_size.x(), slice_size.y(), slice_size.z()}, {8,8,1});

        last_grabbed = c2;
    }

    time_elapsed += step;
}

void raytracing4_manager::trace(cl::context& clctx, cl::command_queue& mqueue, float simulation_width, const tensor<int, 2>& screen, vec3f camera_pos, vec4f camera_quat, float camera_start_time)
{
    if(last_grabbed < 0)
        return;

    std::vector<cl::buffer> ray_props;

    for(int i=0; i < 9; i++)
    {
        ray_props.emplace_back(clctx).alloc(sizeof(cl_float) * screen.x() * screen.y());
        ray_props.back().set_to_zero(mqueue);
    }

    cl_float3 ccamera_pos = {camera_pos.x(), camera_pos.y(), camera_pos.z()};
    cl_float4 ccamera_quat = {camera_quat.x(), camera_quat.y(), camera_quat.z(), camera_quat.w()};
    cl_int4 mesh_dim = {slice_size.x(), slice_size.y(), slice_size.z(), last_grabbed + 1};
    cl_int2 clscreen = {screen.x(), screen.y()};
    float scale = calculate_scale(simulation_width, slice_size);

    {
        cl::args args;
        args.push_back(ccamera_pos, ccamera_quat, clscreen);

        for(auto& i : slice)
        {
            args.push_back(i);
        }

        args.push_back(scale);
        args.push_back(mesh_dim);
        args.push_back(camera_start_time);
        args.push_back(slice_width);

        for(auto& i : ray_props)
        {
            args.push_back(i);
        }

        mqueue.exec("init_slice_rays4", args, {screen.x(), screen.y()}, {8,8});
    }

    {
        cl::args args;

        for(auto& i : slice)
        {
            args.push_back(i);
        }

        int ray_count = screen.x() * screen.y();

        args.push_back(scale);
        args.push_back(mesh_dim);
        args.push_back(clscreen);
        args.push_back(ray_count);
        args.push_back(slice_width);

        for(auto& i : ray_props)
        {
            args.push_back(i);
        }

        args.push_back(render_ray_info_buf);

        mqueue.exec("trace_slice4", args, {screen.x(), screen.y()}, {8,8});
    }
}

v4f world_to_voxel4(v4f in, v4i dim, value slice_width, value scale)
{
    v3i centre = (dim.xyz() - 1)/2;

    v3f spatial_world = {in.y(), in.z(), in.w()};

    v3f spatial = (spatial_world / scale) + (v3f)centre;
    value time = in.x() / slice_width;

    return {time, spatial.x(), spatial.y(), spatial.z()};
}


v4f voxel_to_world4(v4f in, v4i dim, value slice_width, value scale)
{
    v3i centre = (dim.xyz() - 1)/2;

    v3f spatial_voxel = {in.y(), in.z(), in.w()};

    v3f spatial = (spatial_voxel - (v3f)centre) * scale;

    value time = in.x() * slice_width;

    return {time, spatial.x(), spatial.y(), spatial.z()};
}

void init_slice_rays4(equation_context& ctx, literal<v3f> camera_pos, literal<v4f> camera_quat, literal<v2i> screen_size,
                     std::array<buffer<ray4_value>, 10> Guv,
                     named_literal<value, "scale"> scale, named_literal<v4i, "dim"> dim,
                     literal<value> w_coord, literal<value> slice_width,
                     std::array<buffer<value_mut>, 4> positions_out, std::array<buffer<value_mut>, 4> velocities_out,
                     buffer<value_mut> ku_uobsu_out
                     )
{
    using namespace dual_types::implicit;

    ctx.add_function("buffer_read_linear2", buffer_read_linear_f_unpacked<value>);
    ctx.add_function("buffer_read_linear4", buffer_read_linear_f4<value>);

    ctx.order = 1;
    ctx.uses_linear = true;

    auto w2v4 = [&](v4f in)
    {
        return world_to_voxel4(in, dim.get(), slice_width.get(), scale.get());
    };

    v3f world_pos3 = camera_pos.get();

    ///takes in t, x, y, z, outputs t, x, y, z
    v4f pos4_txyz = w2v4({w_coord.get(), world_pos3.x(), world_pos3.y(), world_pos3.z()});

    ///x, y, z, t
    v4f pos4 = {pos4_txyz.y(), pos4_txyz.z(), pos4_txyz.w(), pos4_txyz.x()};

    v2i in_xy = {"get_global_id(0)", "get_global_id(1)"};

    v2i xy = declare(in_xy);

    if_e(xy.x() >= screen_size.get().x() || xy.y() >= screen_size.get().y(), [&]()
    {
        return_e();
    });

    int indices[4][4] = {{0, 1, 2, 3},
                         {1, 4, 5, 6},
                         {2, 5, 7, 8},
                         {3, 6, 8, 9}};

    std::array<value, 10> Guv_i;

    for(int i=0; i < 10; i++)
    {
        Guv_i[i] = buffer_read_linear(Guv[i], pos4, dim.get());
    }

    metric<value, 4, 4> Guv_built;

    for(int i=0; i < 4; i++)
    {
        for(int j=0; j < 4; j++)
        {
            Guv_built[i, j] = (value)Guv_i[indices[i][j]];
        }
    }

    ctx.pin(Guv_built);

    value fgA = 0;
    tensor<value, 3> fgB = {0,0,0};

    lightray ray = make_lightray(ctx, camera_pos.get(), camera_quat.get(), screen_size.get(), xy, Guv_built, fgA, fgB);

    value_i out_idx = xy.y() * screen_size.get().x() + xy.x();

    mut(positions_out[0][out_idx]) = w_coord.get();
    mut(positions_out[1][out_idx]) = ray.pos4.y();
    mut(positions_out[2][out_idx]) = ray.pos4.z();
    mut(positions_out[3][out_idx]) = ray.pos4.w();

    mut(velocities_out[0][out_idx]) = ray.vel4.x();
    mut(velocities_out[1][out_idx]) = ray.vel4.y();
    mut(velocities_out[2][out_idx]) = ray.vel4.z();
    mut(velocities_out[3][out_idx]) = ray.vel4.w();
    mut(ku_uobsu_out[out_idx]) = ray.ku_uobsu;
}

tensor<value, 4> txyz_to_xyzt(const tensor<value, 4>& in)
{
    return {in.y(), in.z(), in.w(), in.x()};
}

value acceleration_to_precision(const v4f& acceleration, const value& max_acceleration, value* next_ds_out)
{
    value current_acceleration_err = acceleration.length();

    value err = max_acceleration;

    //#define MIN_STEP 0.00001f
    #define MIN_STEP 0.000001f

    value max_timestep = 100000;

    current_acceleration_err = min(current_acceleration_err, max_acceleration * pow(max_timestep, 2.f));

    ///of course, as is tradition, whatever works for kerr does not work for alcubierre
    ///the sqrt error calculation is significantly better for alcubierre, largely in terms of having no visual artifacts at all
    ///whereas the pow version is nearly 2x faster for kerr
    value next_ds = sqrt(max_acceleration / current_acceleration_err);

    *next_ds_out = next_ds;

    return current_acceleration_err;
}

value calculate_ds_error(const value& current_ds, const v4f& acceleration, const value& max_acceleration, value* next_ds_out)
{
    value next_ds = 0;
    value diff = acceleration_to_precision(acceleration, max_acceleration, &next_ds);

    ///produces strictly worse results for kerr
    next_ds = 0.99f * current_ds * clamp(next_ds / current_ds, value{0.3f}, value{2.f});

    next_ds = max(next_ds, value{MIN_STEP});

    *next_ds_out = next_ds;

    value err = max_acceleration;

    return next_ds == MIN_STEP && diff > err * 1000;
}

void trace_slice4(equation_context& ctx,
                 std::array<buffer<ray4_value>, 10> Guv_4d,
                 named_literal<value, "scale"> scale, named_literal<v4i, "dim"> dim, literal<v2i> screen_size, literal<value_i> ray_count,
                 literal<value> slice_width,
                 std::array<buffer<value>, 4> positions, std::array<buffer<value>, 4> velocities, buffer<value> ku_uobsu,
                 buffer<render_ray_info>& render_out)
{
    using namespace dual_types::implicit;

    ctx.add_function("buffer_read_linear2", buffer_read_linear_f_unpacked<value>);
    ctx.add_function("buffer_read_linear4", buffer_read_linear_f4<value>);

    ctx.order = 1;
    ctx.uses_linear = true;

    value_i x = declare(value_i{"get_global_id(0)"});
    value_i y = declare(value_i{"get_global_id(1)"});

    value_i lidx = y * screen_size.get().x() + x;

    if_e(lidx >= ray_count, [&]()
    {
        return_e();
    });

    ///t, x, y, z
    v4f pos = {positions[0][lidx], positions[1][lidx], positions[2][lidx], positions[3][lidx]};
    v4f vel = {velocities[0][lidx], velocities[1][lidx], velocities[2][lidx], velocities[3][lidx]};

    ///takes in t, x, y, z -> outputs t, x, y, z
    auto w2v4 = [&](v4f in)
    {
        return world_to_voxel4(in, dim.get(), slice_width.get(), scale.get());
    };

    ///t, x, y, z
    v4f voxel_pos_txyz = w2v4(pos);

    v4f_mut loop_voxel_pos_txyz = declare_mut(voxel_pos_txyz);
    v4f_mut loop_vel = declare_mut(vel);

    value universe_size = ((dim.get().x()-1)/2).convert<float>() * scale;

    value u_sq = universe_size * universe_size * 0.95f * 0.95f;

    v4f scales = {slice_width.get(), scale.get(), scale.get(), scale.get()};

    int indices[4][4] = {{0, 1, 2, 3},
                         {1, 4, 5, 6},
                         {2, 5, 7, 8},
                         {3, 6, 8, 9}};

    auto get_Guv = [&](v4f my_voxel_pos_txyz)
    {
        metric<value, 10> Guv_lin;

        for(int i=0; i < 10; i++)
        {
            Guv_lin[i] = (value)buffer_read_linear(Guv_4d[i], txyz_to_xyzt(my_voxel_pos_txyz), dim.get());
        }

        metric<value, 4, 4> Guv;

        for(int i=0; i < 4; i++)
        {
            for(int j=0; j < 4; j++)
            {
                Guv[i, j] = Guv_lin[indices[i][j]];
            }
        }

        ctx.pin(Guv);

        return Guv;
    };

    auto do_Guv = [&](v4f my_voxel_pos_txyz, v4f my_velocity)
    {
        ///k, uv
        tensor<value, 4, 4, 4> dGuv;

        for(int k=0; k < 4; k++)
        {
            v4f directions[4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};

            for(int i=0; i < 4; i++)
            {
                for(int j=0; j < 4; j++)
                {
                    v4f offset = directions[k];

                    v4f voxel_pos_u = my_voxel_pos_txyz + offset;
                    v4f voxel_pos_l = my_voxel_pos_txyz - offset;

                    ///buffer stores x, y, z, t
                    ///coordinates are t, x, y, z. Annoying!
                    v4f shuffled_u = txyz_to_xyzt(voxel_pos_u);
                    v4f shuffled_l = txyz_to_xyzt(voxel_pos_l);

                    int linear_index = indices[i][j];

                    dGuv[k, i, j] = (value)(buffer_read_linear(Guv_4d[linear_index], shuffled_u, dim.get()) - buffer_read_linear(Guv_4d[linear_index], shuffled_l, dim.get())) / (2 * scales[k]);
                }
            }
        }

        ctx.pin(dGuv);

        auto Guv = get_Guv(my_voxel_pos_txyz);

        inverse_metric<value, 4, 4> inverted = Guv.invert();

        ctx.pin(inverted);

        tensor<value, 4, 4, 4> christoff2 = christoffel_symbols_2(inverted, dGuv);

        tensor<value, 4> acceleration;

        for(int mu=0; mu < 4; mu++)
        {
            value sum = 0;

            for(int a=0; a < 4; a++)
            {
                for(int b=0; b < 4; b++)
                {
                    sum += christoff2[mu, a, b] * my_velocity[a] * my_velocity[b];
                }
            }

            acceleration[mu] = -sum;
        }

        ctx.pin(acceleration);

        return acceleration;
    };

    value_i_mut hit_type = declare_mut(ctx, value_i{-1});

    value_i steps = 800;

    value max_accel = 0.1f;

    value_mut current_ds = declare_mut(ctx, value{0.001f});

    //value next_ds_start;
    //acceleration_to_precision(acceleration, max_accel, &next_ds_start);

    //ctx.exec(assign(current_ds, next_ds_start));

    //ctx.exec(for_b("idx", value_i(0), value_i("idx") < (value_i)steps, value_i("idx++")));

    for_e("idx", value_i(0), value_i("idx") < (value_i)steps, value_i("idx++"),
    [&](const value_i& idx)
    {
        assert(loop_voxel_pos_txyz[0].is_mutable);

        ///so, we could totally cache this is in equivalent, and there's 0 reason to deprive us of that
        ///loop_voxel_pos_pinned has a unique name, which cannot be referred to outside this scope
        ///its safe to look up loop_voxel_pos_pinned within this scope, because we're fundamentally saying
        ///i want the value of loop_voxel_pos_pinned, NOT the value of its *contents*
        ///that means that it is safe to cache
        ///MUST NOT INSPECT THE TREE WHICH MAKES IT UP, which is a very different constraint, and isn't something that is done
        ///YET
        v4f loop_voxel_pos_pinned = declare(as_constant(loop_voxel_pos_txyz));
        v4f loop_velocity_pinned = declare(as_constant(loop_vel));
        value current_ds_pinned = declare(as_constant(current_ds));

        bool can_cache = true;

        loop_voxel_pos_pinned[0].recurse_arguments([&](const value& in)
        {
            if(in.is_mutable)
                can_cache = false;
        });

        assert(can_cache);

        v4f acceleration = do_Guv(loop_voxel_pos_pinned, loop_velocity_pinned);

        v4f v_half = loop_velocity_pinned + 0.5f * acceleration * current_ds_pinned;

        v4f next_voxel_pos = loop_voxel_pos_pinned + v_half * current_ds_pinned / scales;
        v4f intermediate_next_velocity = loop_velocity_pinned + acceleration * current_ds_pinned;

        v4f next_acceleration = do_Guv(next_voxel_pos, intermediate_next_velocity);

        v4f next_velocity = v_half + 0.5f * next_acceleration * current_ds_pinned;

        //next_velocity.x() = clamp(next_velocity.x(), value{-100}, value{100});

        v4f world_pos4 = voxel_to_world4(next_voxel_pos, dim.get(), slice_width.get(), scale.get());
        v3f world_pos = {world_pos4.y(), world_pos4.z(), world_pos4.w()};

        value escape_cond = world_pos.squared_length() >= u_sq;
        //value ingested_cond = dpos.squared_length() < 0.1f * 0.1f;

        value ingested_cond = fabs(next_velocity.x()) > 100 || fabs(next_acceleration.x()) >= 100;
        //value ingested_cond = fabs(loop_vel.x()) > 1000 && fabs(dvel.x()) > 100;

        if_e(escape_cond, [&]()
        {
            mut(hit_type) = 0;
            break_e();
        });

        if_e(ingested_cond, [&]()
        {
            mut(hit_type) = 1;
            break_e();
        });

        mut(loop_voxel_pos_txyz) = next_voxel_pos;
        mut(loop_vel) = next_velocity;

        value ds_out = 0;
        calculate_ds_error(current_ds, next_acceleration, max_accel, &ds_out);

        mut(current_ds) = ds_out;
    });

    ///guarantee that loop_vel and loop_voxel_pos_txyz won't change now
    v4f loop_vel_cst = declare(as_constant(loop_vel));
    v4f loop_voxel_pos_txyz_cst = declare(as_constant(loop_voxel_pos_txyz));

    value zp1 = 1;

    {
        //auto Guv = get_Guv();

        metric<value, 4, 4> Guv;

        Guv[0, 0] = -1;

        for(int i=1; i < 4; i++)
            Guv[i, i] = 1;

        v4f observer = v4f{1, 0, 0, 0};

        v4f observer_lowered = lower_index(observer, Guv, 0);

        value top = sum_multiply(loop_vel_cst, observer_lowered);

        top = clamp(top, value{-100.f}, {100.f});

        zp1 = top / ku_uobsu[lidx];

        ///pretty sure this boils down to
        ///(-velocity.x / ray->ku_uobsu)
    }

    v4f fin_world_pos = voxel_to_world4(loop_voxel_pos_txyz_cst, dim.get(), slice_width.get(), scale.get());

    render_ray_info out = render_out[lidx];

    mut(out.x.get()) = x;
    mut(out.y.get()) = y;
    mut(out.R.get()) = value{1};
    mut(out.G.get()) = value{0};
    mut(out.B.get()) = value{0};
    mut(out.hit_type.get()) = value_i{1};

    if_e(hit_type != value_i{-1}, ctx, [&]()
    {
        mut(out.x.get()) = x;
        mut(out.y.get()) = y;
        mut(out.X.get()) = fin_world_pos.y();
        mut(out.Y.get()) = fin_world_pos.z();
        mut(out.Z.get()) = fin_world_pos.w();
        mut(out.dX.get()) = loop_vel_cst.y();
        mut(out.dY.get()) = loop_vel_cst.z();
        mut(out.dZ.get()) = loop_vel_cst.w();
        mut(out.hit_type.get()) = hit_type;
        mut(out.R.get()) = value{0};
        mut(out.G.get()) = value{0};
        mut(out.B.get()) = value{0};
        mut(out.background_power.get()) = value{1};
        mut(out.zp1.get()) = zp1;
    });
}

void build_raytracing_kernels(cl::context& clctx, base_bssn_args& bssn_args)
{
    #if 0
    single_source::make_async_dynamic_kernel_for(clctx, get_raytraced_quantities, "get_raytraced_quantities", "", bssn_args);

    single_source::make_async_kernel_for(clctx, init_slice_rays, "init_slice_rays");
    single_source::make_async_kernel_for(clctx, trace_slice, "trace_slice");

    single_source::make_async_dynamic_kernel_for(clctx, get_raytraced_quantities4, "get_raytraced_quantities4", "", bssn_args);

    single_source::make_async_kernel_for(clctx, init_slice_rays4, "init_slice_rays4");
    single_source::make_async_kernel_for(clctx, trace_slice4, "trace_slice4");
    #endif
}
