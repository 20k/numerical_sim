#include "particle_dynamics.hpp"
#include <geodesic/dual_value.hpp>
#include "equation_context.hpp"
#include "bssn.hpp"

value particle_matter_interop::calculate_adm_p(equation_context& ctx, standard_arguments& bssn_args)
{
    value adm_p = bidx("adm_p", ctx.uses_linear, false);

    return adm_p;
}

value particle_matter_interop::calculate_adm_S(equation_context& ctx, standard_arguments& bssn_args)
{
    value adm_S = bidx("adm_S", ctx.uses_linear, false);

    printf("Using particle ADM matter\n");

    return adm_S;
}

tensor<value, 3> particle_matter_interop::calculate_adm_Si(equation_context& ctx, standard_arguments& bssn_args)
{
    value adm_Si0 = bidx("adm_Si0", ctx.uses_linear, false);
    value adm_Si1 = bidx("adm_Si1", ctx.uses_linear, false);
    value adm_Si2 = bidx("adm_Si2", ctx.uses_linear, false);

    return {adm_Si0, adm_Si1, adm_Si2};
}

tensor<value, 3, 3> particle_matter_interop::calculate_adm_X_Sij(equation_context& ctx, standard_arguments& bssn_args)
{
    value adm_Sij0 = bidx("adm_Sij0", ctx.uses_linear, false);
    value adm_Sij1 = bidx("adm_Sij1", ctx.uses_linear, false);
    value adm_Sij2 = bidx("adm_Sij2", ctx.uses_linear, false);
    value adm_Sij3 = bidx("adm_Sij3", ctx.uses_linear, false);
    value adm_Sij4 = bidx("adm_Sij4", ctx.uses_linear, false);
    value adm_Sij5 = bidx("adm_Sij5", ctx.uses_linear, false);

    value X = bssn_args.get_X();

    tensor<value, 3, 3> Sij;

    std::array<int, 9> arg_table
    {
        0, 1, 2,
        1, 3, 4,
        2, 4, 5,
    };

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            int index = arg_table[i * 3 + j];

            Sij.idx(i, j) = bidx("adm_Sij" + std::to_string(index), false, false);
        }
    }

    return X * Sij;
}

particle_dynamics::particle_dynamics(cl::context& ctx) : particle_3_position{ctx, ctx}, particle_3_velocity{ctx, ctx}, pd(ctx), indices_block(ctx), weights_block(ctx), memory_alloc_count(ctx)
{
    indices_block.alloc(sizeof(cl_int) * 1024 * 1024 * 40);
    weights_block.alloc(sizeof(cl_float) * 1024 * 1024 * 40);
    memory_alloc_count.alloc(sizeof(cl_int));
}

std::vector<buffer_descriptor> particle_dynamics::get_buffers()
{
    return {{"adm_p", "dont_care", 0.f, 0, 0},
            {"adm_Si0", "dont_care", 0.f, 0, 0},
            {"adm_Si1", "dont_care", 0.f, 0, 0},
            {"adm_Si2", "dont_care", 0.f, 0, 0},
            {"adm_Sij0", "dont_care", 0.f, 0, 0},
            {"adm_Sij1", "dont_care", 0.f, 0, 0},
            {"adm_Sij2", "dont_care", 0.f, 0, 0},
            {"adm_Sij3", "dont_care", 0.f, 0, 0},
            {"adm_Sij4", "dont_care", 0.f, 0, 0},
            {"adm_Sij5", "dont_care", 0.f, 0, 0},
            {"adm_S", "dont_care", 0.f, 0, 0}};
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

    //value length_sq = dot_metric(V_upper, V_upper, args.Yij);

    //value length = sqrt(fabs(length_sq));

    //V_upper = (V_upper * 1 / length);

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

    ctx.add("V0Diff", V_upper_diff.idx(0));
    ctx.add("V1Diff", V_upper_diff.idx(1));
    ctx.add("V2Diff", V_upper_diff.idx(2));

    ctx.add("X0Diff", dx.idx(0));
    ctx.add("X1Diff", dx.idx(1));
    ctx.add("X2Diff", dx.idx(2));
}

float get_kepler_velocity(float distance_between_bodies, float my_mass, float their_mass)
{
    float R = distance_between_bodies;

    float M = my_mass + their_mass;

    float velocity = sqrt(M/R);

    return velocity;

    //float velocity = their_mass * their_mass / (R * M);
}

///https://www.mdpi.com/2075-4434/6/3/70/htm (7)

void particle_dynamics::init(cpu_mesh& mesh, cl::context& ctx, cl::command_queue& cqueue,         thin_intermediates_pool& pool, buffer_set& to_init)
{
    vec3i dim = mesh.dim;

    memory_ptrs = cl::buffer(ctx);
    counts = cl::buffer(ctx);

    memory_ptrs.value().alloc(sizeof(cl_int) * dim.x() * dim.y() * dim.z());
    counts.value().alloc(sizeof(cl_int) * dim.x() * dim.y() * dim.z());

    cl_int4 clsize = {dim.x(), dim.y(), dim.z(), 0};
    float scale = mesh.scale;

    uint64_t size = dim.x() * dim.y() * dim.z() * sizeof(cl_float);

    /*adm_p.alloc(size);

    for(int i=0; i < 3; i++)
    {
        adm_Si[i].alloc(size);
    }

    for(int i=0; i < 6; i++)
    {
        adm_Sij[i].alloc(size);
    }

    adm_S.alloc(size);*/

    particle_count = 256;

    for(int i=0; i < 2; i++)
    {
        particle_3_position[i].alloc(sizeof(cl_float) * 3 * particle_count);
        particle_3_velocity[i].alloc(sizeof(cl_float) * 3 * particle_count);
    }

    float generation_radius = 0.5f * get_c_at_max()/2.f;

    ///need to use an actual rng if i'm doing anything even vaguely scientific
    std::minstd_rand0 rng(1234);

    std::vector<vec3f> positions;
    std::vector<vec3f> directions;

    float mass = 0.01;
    float total_mass = mass * particle_count;

    for(int i=0; i < particle_count; i++)
    {
        int kk=0;

        for(; kk < 1024; kk++)
        {
            float x = rand_det_s(rng, -1.f, 1.f) * generation_radius;
            float y = rand_det_s(rng, -1.f, 1.f) * generation_radius;
            float z = rand_det_s(rng, -1.f, 1.f) * generation_radius;

            vec3f pos = {x, y, z};

            pos.z() *= 0.001f;

            float angle = atan2(pos.y(), pos.x());
            float radius = pos.length();

            if(radius >= generation_radius || radius < generation_radius * 0.1f)
                continue;

            positions.push_back(pos);

            vec2f velocity_direction = (vec2f){1, 0}.rot(angle + M_PI/2);

            //float linear_velocity = get_kepler_velocity(radius, mass, total_mass - mass) * 0.15f;

            //printf("Linear velocity %f\n", linear_velocity);

            float linear_velocity = 0.025f;

            vec2f velocity = linear_velocity * velocity_direction;

            vec3f velocity3 = {velocity.x(), velocity.y(), 0};

            directions.push_back(velocity3);

            break;
        }

        if(kk == 1024)
            throw std::runtime_error("Did not successfully assign particle position");



        //directions.push_back({0.00005, 0, 0});
    }

    particle_3_position[0].write(cqueue, positions);

    cl::buffer initial_dirs(ctx);
    initial_dirs.alloc(sizeof(cl_float) * 3 * particle_count);
    initial_dirs.write(cqueue, directions);

    assert((int)positions.size() == particle_count);

    std::string argument_string = "-I ./ -cl-std=CL2.0 ";

    {

        vec<4, value> position = {0, "px", "py", "pz"};
        vec<3, value> direction = {"dirx", "diry", "dirz"};

        //direction = direction.norm();

        equation_context ectx;
        ectx.uses_linear = true;
        standard_arguments args(ectx);

        metric<value, 4, 4> real_metric = calculate_real_metric(args.Yij, args.gA, args.gB);

        ectx.pin(real_metric);

        frame_basis basis = calculate_frame_basis(ectx, real_metric);

        ///todo, orient basis
        ///so, our observer is whatever we get out of the metric which isn't terribly scientific
        ///but it should be uh. Stationary?
        tetrad tet = {basis.v1, basis.v2, basis.v3, basis.v4};

        ectx.add("Debug_t0", basis.v2.x());
        ectx.add("Debug_t1", basis.v2.y());
        ectx.add("Debug_t2", basis.v2.z());
        ectx.add("Debug_t3", basis.v2.w());

        vec<4, value> velocity = get_timelike_vector(direction, 1, tet);

        ectx.add("OUT_VT", velocity.x());
        ectx.add("OUT_VX", velocity.y());
        ectx.add("OUT_VY", velocity.z());
        ectx.add("OUT_VZ", velocity.w());

        ectx.build(argument_string, "tparticleinit");
    }

    {
        equation_context ectx;
        build_adm_geodesic(ectx, {mesh.dim.x(), mesh.dim.y(), mesh.dim.z()});

        ectx.build(argument_string, 6);
    }

    ///relevant resources
    ///https://arxiv.org/pdf/1611.07906.pdf 16
    ///https://artscimedia.case.edu/wp-content/uploads/sites/213/2018/08/18010345/Mertens_SestoGR18.pdf
    ///https://scholarworks.rit.edu/cgi/viewcontent.cgi?article=11286&context=theses 3.81
    ///https://einsteinrelativelyeasy.com/index.php/fr/einstein/9-general-relativity/78-the-energy-momentum-tensor
    ///https://arxiv.org/pdf/1905.08890.pdf
    ///https://en.wikipedia.org/wiki/Stress%E2%80%93energy_tensor#Stress%E2%80%93energy_in_special_situations
    {
        equation_context ectx;
        standard_arguments args(ectx);

        tensor<value, 3> v_upper = {"vel.x", "vel.y", "vel.z"};

        value sum = 0;

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                sum += args.cY.idx(i, j) * v_upper.idx(i) * v_upper.idx(j);
            }
        }

        tensor<value, 4> hypersurface_normal_raised = get_adm_hypersurface_normal_raised(args.gA, args.gB);

        value paper_w = sqrt(1 + sum);

        tensor<value, 4> v_full = {0, v_upper.idx(0), v_upper.idx(1), v_upper.idx(2)};

        tensor<value, 4> u_full = v_full + paper_w * hypersurface_normal_raised;

        value lorentz = u_full.idx(0);

        tensor<value, 3> u3_upper = {u_full.idx(1), u_full.idx(2), u_full.idx(3)};

        ///wait. But this is equal to vi_lower. Ah I'm such a muppet
        tensor<value, 3> u3_lower = lower_index(u3_upper, args.Yij, 0);

        value out_adm_p = mass * lorentz * lorentz;
        tensor<value, 3> Si = mass * lorentz * u3_lower;
        tensor<value, 3, 3> Sij;

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                Sij.idx(i, j) = mass * u3_lower.idx(i) * u3_lower.idx(j);
            }
        }

        value out_adm_S = out_adm_p - mass;

        ectx.add("OUT_ADM_S", out_adm_S);
        ectx.add("OUT_ADM_SI0", Si.idx(0));
        ectx.add("OUT_ADM_SI1", Si.idx(1));
        ectx.add("OUT_ADM_SI2", Si.idx(2));
        ectx.add("OUT_ADM_SIJ0", Sij.idx(0, 0));
        ectx.add("OUT_ADM_SIJ1", Sij.idx(1, 0));
        ectx.add("OUT_ADM_SIJ2", Sij.idx(2, 0));
        ectx.add("OUT_ADM_SIJ3", Sij.idx(1, 1));
        ectx.add("OUT_ADM_SIJ4", Sij.idx(2, 1));
        ectx.add("OUT_ADM_SIJ5", Sij.idx(2, 2));
        ectx.add("OUT_ADM_P", out_adm_p);

        ectx.build(argument_string, "admmatter");
    }

    argument_string += "-DBORDER_WIDTH=" + std::to_string(BORDER_WIDTH) + " ";

    pd = cl::program(ctx, "particle_dynamics.cl");
    pd.build(ctx, argument_string);

    ctx.register_program(pd);

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

        args.push_back(particle_count);
        args.push_back(scale);
        args.push_back(clsize);

        kern.set_args(args);

        cqueue.exec(kern, {particle_count}, {128});
    }

    cl::copy(cqueue, particle_3_position[0], particle_3_position[1]);
    cl::copy(cqueue, particle_3_velocity[0], particle_3_velocity[1]);

    to_init.lookup("adm_p").buf.set_to_zero(cqueue);
    to_init.lookup("adm_Si0").buf.set_to_zero(cqueue);
    to_init.lookup("adm_Si1").buf.set_to_zero(cqueue);
    to_init.lookup("adm_Si2").buf.set_to_zero(cqueue);
    to_init.lookup("adm_Sij0").buf.set_to_zero(cqueue);
    to_init.lookup("adm_Sij1").buf.set_to_zero(cqueue);
    to_init.lookup("adm_Sij2").buf.set_to_zero(cqueue);
    to_init.lookup("adm_Sij3").buf.set_to_zero(cqueue);
    to_init.lookup("adm_Sij4").buf.set_to_zero(cqueue);
    to_init.lookup("adm_Sij5").buf.set_to_zero(cqueue);
    to_init.lookup("adm_S").buf.set_to_zero(cqueue);
}

void particle_dynamics::step(cpu_mesh& mesh, cl::context& ctx, cl::managed_command_queue& mqueue, thin_intermediates_pool& pool, buffer_set& in, buffer_set& out, buffer_set& base, float timestep, int iteration, int max_iteration)
{
    cl::buffer& memory_ptrs_val = memory_ptrs.value();
    cl::buffer& counts_val = counts.value();

    memory_alloc_count.set_to_zero(mqueue);

    ///so. Need to take all my particles, advance them forwards in time. Some complications because I'm not going to do this in a backwards euler way, so only on the 0th iteration do we do fun things. Need to pre-swap buffers
    ///need to fill up the adm buffers from the *current* particle positions
    vec3i dim = mesh.dim;
    cl_int4 clsize = {mesh.dim.x(), mesh.dim.y(), mesh.dim.z(), 0};
    float scale = mesh.scale;

    ///shit no its not correct, we need to be implicit otherwise the sources are incorrect innit. Its F(y+1)
    if(iteration == 0)
    {
        std::swap(particle_3_position[0], particle_3_position[1]);
        std::swap(particle_3_velocity[0], particle_3_velocity[1]);

        {
            cl::args args;
            args.push_back(particle_3_position[0]);
            args.push_back(particle_3_velocity[0]);
            args.push_back(particle_3_position[1]);
            args.push_back(particle_3_velocity[1]);
            args.push_back(particle_count);

            for(named_buffer& i : in.buffers)
            {
                args.push_back(i.buf);
            }

            args.push_back(scale);
            args.push_back(clsize);
            args.push_back(timestep);

            mqueue.exec("trace_geodesics", args, {particle_count}, {128});
        }
    }

    in.lookup("adm_p").buf.set_to_zero(mqueue);
    in.lookup("adm_Si0").buf.set_to_zero(mqueue);
    in.lookup("adm_Si1").buf.set_to_zero(mqueue);
    in.lookup("adm_Si2").buf.set_to_zero(mqueue);
    in.lookup("adm_Sij0").buf.set_to_zero(mqueue);
    in.lookup("adm_Sij1").buf.set_to_zero(mqueue);
    in.lookup("adm_Sij2").buf.set_to_zero(mqueue);
    in.lookup("adm_Sij3").buf.set_to_zero(mqueue);
    in.lookup("adm_Sij4").buf.set_to_zero(mqueue);
    in.lookup("adm_Sij5").buf.set_to_zero(mqueue);
    in.lookup("adm_S").buf.set_to_zero(mqueue);

    counts_val.set_to_zero(mqueue);

    ///find how many particles would be written per-cell
    {
        cl_int actually_write = 0;

        cl::args args;
        args.push_back(particle_3_position[0]);
        args.push_back(particle_count);
        args.push_back(counts_val);
        args.push_back(memory_ptrs_val);
        args.push_back(indices_block);
        args.push_back(weights_block);
        args.push_back(scale);
        args.push_back(clsize);
        args.push_back(actually_write);

        mqueue.exec("collect_particle_spheres", args, {particle_count}, {128});
    }

    ///allocate memory for each cell
    {
        cl::args args;
        args.push_back(counts_val);
        args.push_back(memory_ptrs_val);
        args.push_back(memory_alloc_count);
        args.push_back(clsize);

        mqueue.exec("allocate_particle_spheres", args, {dim.x(), dim.y(), dim.z()}, {8,8,1});
    }

    ///write the indices and weights into indices_block, and weights_block at the offsets determined by memory_ptrs_val per-cell
    {
        cl_int actually_write = 1;

        cl::args args;
        args.push_back(particle_3_position[0]);
        args.push_back(particle_count);
        args.push_back(counts_val);
        args.push_back(memory_ptrs_val);
        args.push_back(indices_block);
        args.push_back(weights_block);
        args.push_back(scale);
        args.push_back(clsize);
        args.push_back(actually_write);

        mqueue.exec("collect_particle_spheres", args, {particle_count}, {128});
    }

    ///calculate adm quantities per-cell by summing across the list of particles
    {
        cl::args args;
        args.push_back(particle_3_position[0]);
        args.push_back(particle_3_velocity[0]);
        args.push_back(counts_val);
        args.push_back(memory_ptrs_val);
        args.push_back(indices_block);
        args.push_back(weights_block);

        for(named_buffer& i : in.buffers)
        {
            args.push_back(i.buf);
        }

        args.push_back(scale);
        args.push_back(clsize);

        mqueue.exec("do_weighted_summation", args, {dim.x(), dim.y(), dim.z()}, {8,8,1});
    }
}
