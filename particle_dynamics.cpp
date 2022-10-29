#include "particle_dynamics.hpp"
#include <geodesic/dual_value.hpp>
#include "equation_context.hpp"
#include "bssn.hpp"

particle_dynamics::particle_dynamics(cl::context& ctx) : particle_3_position{ctx, ctx}, particle_3_velocity{ctx, ctx}, adm_p{ctx}, adm_Si{ctx, ctx, ctx}, adm_Sij{ctx, ctx, ctx, ctx, ctx, ctx}, adm_S{ctx}, pd(ctx)
{

}

std::vector<buffer_descriptor> particle_dynamics::get_buffers()
{
    return std::vector<buffer_descriptor>();
}

void build_adm_geodesic(equation_context& ctx, vec3f dim)
{
    ctx.uses_linear = true;
    ctx.order = 2;
    ctx.use_precise_differentiation = false;

    standard_arguments args(ctx);

    ctx.pin(args.Kij);
    ctx.pin(args.Yij);

    float universe_length = (dim/2.f).max_elem();

    value scale = "scale";

    ctx.add("universe_size", universe_length * scale);

    tensor<value, 3> V_upper = {"V0", "V1", "V2"};

    inverse_metric<value, 3, 3> iYij = args.iYij;

    inverse_metric<value, 3, 3> icY = args.cY.invert();

    tensor<value, 3> dX = args.get_dX();

    tensor<value, 3, 3, 3> conformal_christoff2 = christoffel_symbols_2(ctx, args.cY, icY);

    tensor<value, 3, 3, 3> full_christoffel2 = get_full_christoffel2(args.get_X(), dX, args.cY, icY, conformal_christoff2);

    value length_sq = dot_metric(V_upper, V_upper, args.Yij);

    value length = sqrt(fabs(length_sq));

    V_upper = (V_upper * 1 / length);

    ///https://arxiv.org/pdf/1208.3927.pdf (28a)
    tensor<value, 3> dx = args.gA * V_upper - args.gB;

    tensor<value, 3> V_upper_diff;

    for(int i=0; i < 3; i++)
    {
        V_upper_diff.idx(i) = 0;

        for(int j=0; j < 3; j++)
        {
            value kjvk = 0;

            for(int k=0; k < 3; k++)
            {
                kjvk += args.Kij.idx(j, k) * V_upper.idx(k);
            }

            value christoffel_sum = 0;

            for(int k=0; k < 3; k++)
            {
                christoffel_sum += full_christoffel2.idx(i, j, k) * V_upper.idx(k);
            }

            value dlog_gA = diff1(ctx, args.gA, j) / args.gA;

            V_upper_diff.idx(i) += args.gA * V_upper.idx(j) * (V_upper.idx(i) * (dlog_gA - kjvk) + 2 * raise_index(args.Kij, iYij, 0).idx(i, j) - christoffel_sum)
                                   - iYij.idx(i, j) * diff1(ctx, args.gA, j) - V_upper.idx(j) * diff1(ctx, args.gB.idx(i), j);
        }
    }

    ctx.add("MASSIVE_V0Diff", V_upper_diff.idx(0));
    ctx.add("MASSIVE_V1Diff", V_upper_diff.idx(1));
    ctx.add("MASSIVE_V2Diff", V_upper_diff.idx(2));

    ctx.add("MASSIVE_X0Diff", dx.idx(0));
    ctx.add("MASSIVE_X1Diff", dx.idx(1));
    ctx.add("MASSIVE_X2Diff", dx.idx(2));
}

void particle_dynamics::init(cpu_mesh& mesh, cl::context& ctx, cl::command_queue& cqueue,         thin_intermediates_pool& pool, buffer_set& to_init)
{
    vec3i dim = mesh.dim;

    cl_int4 clsize = {dim.x(), dim.y(), dim.z(), 0};
    float scale = mesh.scale;

    uint64_t size = dim.x() * dim.y() * dim.z() * sizeof(cl_float);

    adm_p.alloc(size);

    for(int i=0; i < 3; i++)
    {
        adm_Si[i].alloc(size);
    }

    for(int i=0; i < 6; i++)
    {
        adm_Sij[i].alloc(size);
    }

    adm_S.alloc(size);

    int particle_num = 16;

    for(int i=0; i < 2; i++)
    {
        particle_3_position[i].alloc(sizeof(cl_float) * 3 * particle_num);
        particle_3_velocity[i].alloc(sizeof(cl_float) * 3 * particle_num);
    }

    float generation_radius = 0.5f * get_c_at_max()/2.f;

    ///need to use an actual rng if i'm doing anything even vaguely scientific
    std::minstd_rand0 rng(1234);

    std::vector<vec3f> positions;
    std::vector<vec3f> directions;

    for(int i=0; i < particle_num; i++)
    {
        int kk=0;

        for(; kk < 1024; kk++)
        {
            float x = rand_det_s(rng, -1.f, 1.f) * generation_radius;
            float y = rand_det_s(rng, -1.f, 1.f) * generation_radius;
            float z = rand_det_s(rng, -1.f, 1.f) * generation_radius;

            vec3f pos = {x, y, z};

            if(pos.length() >= generation_radius)
                continue;

            positions.push_back(pos);
        }

        if(kk == 1024)
            throw std::runtime_error("Did not successfully assign particle position");

        directions.push_back({1, 0, 0});
    }

    particle_3_position[0].write(cqueue, positions);

    cl::buffer initial_dirs(ctx);
    initial_dirs.alloc(sizeof(cl_float) * 3 * particle_num);
    initial_dirs.write(cqueue, directions);

    assert((int)positions.size() == particle_num);

    {
        std::string argument_string = "-I ./ -cl-std=CL2.0 ";

        vec<4, value> position = {0, "px", "py", "pz"};
        vec<3, value> direction = {"dirx", "diry", "dirz"};

        direction = direction.norm();

        equation_context ectx;
        standard_arguments args(ectx);

        metric<value, 4, 4> real_metric = calculate_real_metric(args.Yij, args.gA, args.gB);

        ectx.pin(real_metric);

        frame_basis basis = calculate_frame_basis(ectx, real_metric);

        ///todo, orient basis
        ///so, our observer is whatever we get out of the metric which isn't terribly scientific
        ///but it should be uh. Stationary?
        tetrad tet = {basis.v1, basis.v2, basis.v3, basis.v4};

        vec<4, value> velocity = get_timelike_vector(direction, 1, tet);

        ectx.add("OUT_VT", velocity.x());
        ectx.add("OUT_VX", velocity.y());
        ectx.add("OUT_VY", velocity.z());
        ectx.add("OUT_VZ", velocity.w());

        ectx.build(argument_string, "tparticleinit");

        pd = cl::program(ctx, "particle_dynamics.cl");
        pd.build(ctx, argument_string);
    }

    {
        cl::kernel kern(pd, "init_geodesics");

        cl::args args;

        for(named_buffer& i : to_init.buffers)
        {
            args.push_back(i.buf);
        }

        args.push_back(particle_3_position[0]);
        args.push_back(initial_dirs);
        args.push_back(particle_3_velocity[0]);

        args.push_back(particle_num);
        args.push_back(scale);
        args.push_back(clsize);

        kern.set_args(args);

        cqueue.exec(kern, {particle_num}, {128});
    }

    cl::copy(cqueue, particle_3_position[0], particle_3_position[1]);
    cl::copy(cqueue, particle_3_velocity[0], particle_3_velocity[1]);
}

void particle_dynamics::step(cpu_mesh& mesh, cl::context& ctx, cl::managed_command_queue& mqueue, thin_intermediates_pool& pool, buffer_set& in, buffer_set& out, buffer_set& base, float timestep, int iteration, int max_iteration)
{
    ///so. Need to take all my particles, advance them forwards in time. Some complications because I'm not going to do this in a backwards euler way, so only on the 0th iteration do we do fun things. Need to pre-swap buffers
    ///need to fill up the adm buffers from the *current* particle positions
}
