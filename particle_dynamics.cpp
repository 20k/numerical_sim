#include "particle_dynamics.hpp"
#include <geodesic/dual_value.hpp>
#include "equation_context.hpp"
#include "bssn.hpp"
#include "random.hpp"
#include "spherical_integration.hpp"
#include "cache.hpp"

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

            Sij.idx(i, j) = bidx("adm_Sij" + std::to_string(index), ctx.uses_linear, false);
        }
    }

    return X * Sij;
}

particle_dynamics::particle_dynamics(cl::context& ctx) : p_data{ctx, ctx, ctx}, dirac_buf(ctx), pd(ctx), indices_block(ctx), memory_alloc_count(ctx)
{
    indices_block.alloc(max_intermediate_size * sizeof(cl_ulong));
    memory_alloc_count.alloc(sizeof(size_t));
}

std::vector<buffer_descriptor> particle_dynamics::get_utility_buffers()
{
    return {{"adm_p", "do_weighted_summation", 0.f, 0, 0},
            {"adm_Si0", "do_weighted_summation", 0.f, 0, 0},
            {"adm_Si1", "do_weighted_summation", 0.f, 0, 0},
            {"adm_Si2", "do_weighted_summation", 0.f, 0, 0},
            {"adm_Sij0", "do_weighted_summation", 0.f, 0, 0},
            {"adm_Sij1", "do_weighted_summation", 0.f, 0, 0},
            {"adm_Sij2", "do_weighted_summation", 0.f, 0, 0},
            {"adm_Sij3", "do_weighted_summation", 0.f, 0, 0},
            {"adm_Sij4", "do_weighted_summation", 0.f, 0, 0},
            {"adm_Sij5", "do_weighted_summation", 0.f, 0, 0},
            {"adm_S", "do_weighted_summation", 0.f, 0, 0}};
}

void build_lorentz(equation_context& ctx)
{
    ctx.uses_linear = true;
    ctx.order = 2;
    ctx.use_precise_differentiation = false;

    standard_arguments args(ctx);

    ctx.pin(args.Kij);
    ctx.pin(args.Yij);

    value scale = "scale";
    tensor<value, 3> V_upper = {"V0", "V1", "V2"};
    value L = value{"eq_L"} + 1;

    value diff = 0;

    for(int i=0; i < 3; i++)
    {
        value kij_sum = 0;

        for(int j=0; j < 3; j++)
        {
            kij_sum += args.Kij.idx(i, j) * V_upper.idx(j);
        }

        diff += L * V_upper.idx(i) * (args.gA * kij_sum - diff1(ctx, args.gA, i));
    }

    ctx.add("LorentzDiff", diff);
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

    #if 0
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

            value dlog_gA = diff1(ctx, args.gA, j) / max(args.gA, 1e-4f);

            V_upper_diff.idx(i) += args.gA * V_upper.idx(j) * (V_upper.idx(i) * (dlog_gA * 1 - kjvk) + 2 * raise_index(args.Kij, iYij, 0).idx(i, j) - christoffel_sum)
                                   - iYij.idx(i, j) * diff1(ctx, args.gA, j) - V_upper.idx(j) * diff1(ctx, args.gB.idx(i), j);
        }
    }

    tensor<value, 3> dbg_val;

    for(int i=0; i < 3; i++)
    {
        value sum = 0;

        for(int j=0; j < 3; j++)
        {
            sum += iYij.idx(i, j) * diff1(ctx, args.gA, j);
        }

        dbg_val.idx(i) = sum;
    }

    ctx.add("DBGA0", dbg_val.idx(0));
    ctx.add("DBGA1", dbg_val.idx(1));
    ctx.add("DBGA2", dbg_val.idx(2));

    ctx.add("DIFFK", diff1(ctx, args.K, 0));

    #endif

    #if 0
    tensor<value, 3> u_lower = {"V0", "V1", "V2"};

    inverse_metric<value, 3, 3> iYij = args.iYij;

    value sum = 0;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            sum += iYij.idx(i, j) * u_lower.idx(i) * u_lower.idx(j);
        }
    }

    value a_u0 = sqrt(1 + sum);
    //value u0 = a_u0 / args.gA;
    value i_u0 = args.gA / a_u0;

    tensor<value, 3> u_lower_diff;

    for(int i=0; i < 3; i++)
    {
        value p1 = -a_u0 * diff1(ctx, args.gA, i);

        value p2 = 0;

        for(int k=0; k < 3; k++)
        {
            p2 += u_lower.idx(k) * diff1(ctx, args.gB.idx(k), i);
        }


        value p3 = 0;

        for(int j=0; j < 3; j++)
        {
            for(int m=0; m < 3; m++)
            {
                p3 += -0.5f * i_u0 * u_lower.idx(j) * u_lower.idx(m) * diff1(ctx, iYij.idx(j, m), i);
            }
        }

        u_lower_diff.idx(i) = p1 + p2 + p3;
    }

    tensor<value, 3> dx;

    for(int j=0; j< 3; j++)
    {
        value p1 = 0;

        for(int k=0; k < 3; k++)
        {
            p1 += iYij.idx(j, k) * u_lower.idx(k) * i_u0;
        }

        dx.idx(j) = p1 - args.gB.idx(j);
    }
    #endif

    tensor<value, 3> pi = {"V0", "V1", "V2"};

    value mass = "mass";

    value X = args.get_X();

    inverse_metric<value, 3, 3> icY = args.cY.invert();

    value Ea = sqrt(mass * mass + X * trace(outer_product(pi, pi), icY));

    tensor<value, 3> dx = -args.gB + args.gA * X * (1/Ea) * symmetric_multiply(icY.to_tensor(), pi);

    ///https://arxiv.org/pdf/1904.07841.pdf 2.27
    tensor<value, 3> pjdibj;

    for(int i=0; i < 3; i++)
    {
        value sum = 0;

        for(int j=0; j < 3; j++)
        {
            sum += pi.idx(j) * diff1(ctx, args.gB.idx(j), i);
        }

        pjdibj.idx(i) = sum;
    }

    tensor<value, 3> interior_left;

    for(int i=0; i < 3; i++)
    {
        value sum = 0;

        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                for(int l=0; l < 3; l++)
                {
                    sum += icY.idx(j, l) * args.christoff2.idx(k, i, l) * pi.idx(j) * pi.idx(k);
                }
            }
        }

        interior_left.idx(i) = X * sum;
    }

    tensor<value, 3> interior_right = -0.5 * (Ea * Ea - mass * mass) * diff<3>(ctx, X)/max(X, value{1e-4f});

    tensor<value, 3> dp = -Ea * diff<3>(ctx, args.gA) + pjdibj + args.gA * (interior_left + interior_right) / Ea;

    ctx.add("V0Diff", dp.idx(0));
    ctx.add("V1Diff", dp.idx(1));
    ctx.add("V2Diff", dp.idx(2));

    ctx.add("X0Diff", dx.idx(0));
    ctx.add("X1Diff", dx.idx(1));
    ctx.add("X2Diff", dx.idx(2));
}

void particle_dynamics::add_particles(particle_data&& data)
{
    start_data = std::move(data);
}

///https://www.mdpi.com/2075-4434/6/3/70/htm (7)
///ok sweet! Next up, want to divorce particle step from field step
///ideally we'll step forwards the particles by a large timestep, and the interpolate to generate the underlying fields
///geodesic trace time >> discretisation time, so shouldn't be a problem, and that'll take us to 2m particles
///https://arxiv.org/pdf/1611.07906.pdf particle hamiltonian constraint
void particle_dynamics::init(cpu_mesh& mesh, cl::context& ctx, cl::command_queue& cqueue, thin_intermediates_pool& pool, buffer_set& to_init)
{
    vec3i dim = mesh.dim;

    memory_ptrs = cl::buffer(ctx);
    counts = cl::buffer(ctx);

    memory_ptrs.value().alloc(sizeof(cl_ulong) * dim.x() * dim.y() * dim.z());
    counts.value().alloc(sizeof(cl_ulong) * dim.x() * dim.y() * dim.z());

    cl_int4 clsize = {dim.x(), dim.y(), dim.z(), 0};
    float scale = mesh.scale;

    //auto [positions, directions, masses] = build_galaxy(*this);

    std::vector<vec3f> positions = std::move(start_data.positions);
    std::vector<vec3f> directions = std::move(start_data.velocities);
    std::vector<float> masses = std::move(start_data.masses);

    particle_count = positions.size();

    printf("Actual particle count %i\n", particle_count);

    for(int i=0; i < (int)p_data.size(); i++)
    {
        p_data[i].position.alloc(sizeof(cl_float) * 3 * particle_count);
        p_data[i].velocity.alloc(sizeof(cl_float) * 3 * particle_count);
        p_data[i].mass.alloc(sizeof(cl_float) * particle_count);
        p_data[i].lorentz.alloc(sizeof(cl_float) * particle_count);
    }

    dirac_buf.alloc(sizeof(cl_float) * particle_count);

    /*auto get_mond_velocity = [&](float r, float M, float G, float a0)
    {
        float p1 = G * M/r;

        float p2 = (1/sqrt(2.f));

        float frac = 2 * a0 / (G * M);

        float p_inner = 1 + sqrt(1 + pow(r, 4.f) * pow(frac, 2.f));

        float p3 = sqrt(p_inner);

        return sqrt(p1 * p2 * p3);
    };*/

    /*for(int i=0; i < particle_count; i++)
    {
        float radius = 0;

        do
        {
            float random_mass = random() * milky_way_mass_in_scale;

            radius = select_from_cdf(random_mass, milky_way_diameter_in_scale/2.f, cdf);
        } while(radius >= milky_way_diameter_in_scale/2.f);

        ///I have a distinct feeling we might need a sphere term in here
        float angle = random() * 2 *  M_PI;

        float z = (random() - 0.5f) * 2.f * milky_way_diameter_in_scale * 0.000001f;

        vec2f pos2 = {cos(angle) * radius, sin(angle) * radius};
        vec3f pos = {pos2.x(), pos2.y(), z};

        positions.push_back(pos);
        analytic_radius.push_back(radius);
    }*/


    ///just for debugging performance, cuts off 30ms out of a 215 ms runtime. Ie sizeable
    /*std::sort(positions.begin(), positions.end(), [](vec3f v1, vec3f v2)
    {
        return std::tie(v1.z(), v1.y(), v1.x()) < std::tie(v2.z(), v2.y(), v2.x());
    });*/

    cl::buffer positions_in(ctx);
    positions_in.alloc(p_data[0].position.alloc_size);
    positions_in.write(cqueue, positions);

    p_data[0].mass.write(cqueue, masses);

    cl::buffer initial_dirs(ctx);
    initial_dirs.alloc(sizeof(cl_float) * 3 * particle_count);
    initial_dirs.write(cqueue, directions);

    assert((int)positions.size() == particle_count);

    std::string argument_string = "-I ./ -cl-std=CL2.0 ";

    {
        equation_context ectx;
        ectx.add("MINIMUM_MASS", start_data.minimum_mass);

        ectx.build(argument_string, "massdiss");
    }

    {
        vec<4, value> position = {0, "px", "py", "pz"};
        vec<3, value> direction = {"dirx", "diry", "dirz"};

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

        /*tensor<value, 3> ten_vel = {velocity.y(), velocity.z(), velocity.w()};

        tensor<value, 3> lowered_vel = lower_index(ten_vel, args.Yij, 0);*/

        ///4 velocity
        tensor<value, 4> tensor_velocity = {velocity[0], velocity[1], velocity[2], velocity[3]};

        tensor<value, 4> velocity_lowered = lower_index(tensor_velocity, real_metric, 0);

        tensor<value, 4> adm_velocity_lowered = tensor_project_lower(velocity_lowered, args.gA, args.gB);

        tensor<value, 3> adm3_velocity_lowered = {adm_velocity_lowered.idx(1), adm_velocity_lowered.idx(2), adm_velocity_lowered.idx(3)};

        value mass = "mass";

        tensor<value, 3> momentum = mass * adm3_velocity_lowered;

        ///https://arxiv.org/pdf/1208.3927.pdf (8)
        ///pa = E(na + va)
        ///ua = G(na + va) (divide by mass)
        ///(ua/G) - na = va
        ///G = u0

        ///todo. Check nuvu = 0
        ///todo: check vivi = 1 for.. photons?
        //tensor<value, 4> adm_velocity = (tensor_velocity / tensor_velocity.idx(0)) - get_adm_hypersurface_normal_raised(args.gA, args.gB);

        /*value lorentz = tensor_velocity.idx(0);

        ectx.add("OUT_lorentz", lorentz);
        ectx.add("OUT_VX", adm_velocity.idx(1));
        ectx.add("OUT_VY", adm_velocity.idx(2));
        ectx.add("OUT_VZ", adm_velocity.idx(3));

        ectx.build(argument_string, "tparticleinit");*/

        ectx.add("OUT_lorentz", 1);
        ectx.add("OUT_VX", momentum.idx(0));
        ectx.add("OUT_VY", momentum.idx(1));
        ectx.add("OUT_VZ", momentum.idx(2));

        ectx.build(argument_string, "tparticleinit");
    }

    /*{
        equation_context ectx;
        build_lorentz(ectx);

        ectx.build(argument_string, "lorentz");
    }*/

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
    ///https://arxiv.org/pdf/1904.07841.pdf so, have to discretise the dirac delta. This paper gives explicit version
    {
        equation_context ectx;
        ectx.uses_linear = false;
        ectx.use_precise_differentiation = false;
        ectx.order = 2;
        standard_arguments args(ectx);

        //tensor<value, 3> u_lower = {"vel.x", "vel.y", "vel.z"};

        value mass = "mass";

        #if 0
        /*value sum = 0;

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                sum += args.iYij.idx(i, j) * u_lower.idx(i) * u_lower.idx(j);
            }
        }

        ///https://arxiv.org/pdf/1904.07841.pdf 2.28
        ///https://en.wikipedia.org/wiki/Four-momentum#Relation_to_four-velocity
        ///its worth noting that the Ea formulation involves no divisions by lorentz factors
        ///and Ea could be made numerically stable
        ///this formalism seems the most obviously numerically stable however
        //value Ea = sqrt(mass * mass + mass * mass * sum);
        value lorentz = sqrt(1 + fabs(sum));*/

        tensor<value, 3> v_upper = {"vel.x", "vel.y", "vel.z"};

        tensor<value, 3> v_lower = lower_index(v_upper, args.Yij, 0);

        value lorentz = value{"lorentz"} + 1;

        /*value idet = 0;

        #ifdef USE_W
        value W_impl = bidx("X", false, false);

        idet = pow(W_impl, 3);
        #else
        assert(false);
        #endif*/

        //value idet = pow(args.W_impl, 3);
        value idet = pow(args.get_X(), 3.f/2.f);

        value out_adm_p = idet * mass * lorentz;

        tensor<value, 3> Si = idet * mass * lorentz * v_lower;

        tensor<value, 3, 3> Sij;

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                //Sij.idx(i, j) = idet * (covariant_momentum.idx(i) * covariant_momentum.idx(j) / Ea);
                Sij.idx(i, j) = idet * mass * lorentz * v_lower.idx(i) * v_lower.idx(j);
            }
        }
        #endif // 0

        tensor<value, 3> pi = {"vel.x", "vel.y", "vel.z"};

        value idet = pow(args.get_X(), 3.f/2.f);

        inverse_metric<value, 3, 3> icY = args.cY.invert();

        value Ea = sqrt(mass * mass + args.get_X() * trace(outer_product(pi, pi), icY));

        value out_adm_p = idet * Ea;

        tensor<value, 3> Si = idet * pi;
        tensor<value, 3, 3> Sij = idet * outer_product(pi, pi) / Ea;

        value out_adm_S = trace(Sij, args.iYij);

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

        ectx.add("MASS_CULL_SIZE", (get_c_at_max() / 2.f) * 0.8f);

        ectx.build(argument_string, "admmatter");
    }

    argument_string += "-DBORDER_WIDTH=" + std::to_string(BORDER_WIDTH) + " ";

    pd = build_program_with_cache(ctx, "particle_dynamics.cl", argument_string);

    ctx.register_program(pd);

    {
        cl::kernel kern(pd, "init_geodesics");

        cl::args args;

        for(named_buffer& i : to_init.buffers)
        {
            args.push_back(i.buf);
        }

        args.push_back(positions_in);
        args.push_back(initial_dirs);
        args.push_back(p_data[0].mass);
        args.push_back(p_data[0].position);
        args.push_back(p_data[0].velocity);
        args.push_back(p_data[0].lorentz);

        args.push_back(particle_count);
        args.push_back(scale);
        args.push_back(clsize);

        kern.set_args(args);

        cqueue.exec(kern, {particle_count}, {128});
    }

    ///this isn't necessary is it?
    ///Todo: Remove this if it isn't used next time I'm testing particle code, because its confusing for future me
    cl::copy(cqueue, p_data[0].position, p_data[1].position);
    cl::copy(cqueue, p_data[0].velocity, p_data[1].velocity);
    cl::copy(cqueue, p_data[0].mass, p_data[1].mass);
    cl::copy(cqueue, p_data[0].lorentz, p_data[1].lorentz);

    cl::copy(cqueue, p_data[0].position, p_data[2].position);
    cl::copy(cqueue, p_data[0].velocity, p_data[2].velocity);
    cl::copy(cqueue, p_data[0].mass, p_data[2].mass);
    cl::copy(cqueue, p_data[0].lorentz, p_data[2].lorentz);

    mesh.utility_data.lookup("adm_p").buf.set_to_zero(cqueue);
    mesh.utility_data.lookup("adm_Si0").buf.set_to_zero(cqueue);
    mesh.utility_data.lookup("adm_Si1").buf.set_to_zero(cqueue);
    mesh.utility_data.lookup("adm_Si2").buf.set_to_zero(cqueue);
    mesh.utility_data.lookup("adm_Sij0").buf.set_to_zero(cqueue);
    mesh.utility_data.lookup("adm_Sij1").buf.set_to_zero(cqueue);
    mesh.utility_data.lookup("adm_Sij2").buf.set_to_zero(cqueue);
    mesh.utility_data.lookup("adm_Sij3").buf.set_to_zero(cqueue);
    mesh.utility_data.lookup("adm_Sij4").buf.set_to_zero(cqueue);
    mesh.utility_data.lookup("adm_Sij5").buf.set_to_zero(cqueue);
    mesh.utility_data.lookup("adm_S").buf.set_to_zero(cqueue);
}

void particle_dynamics::step(cpu_mesh& mesh, cl::context& ctx, cl::managed_command_queue& mqueue, thin_intermediates_pool& pool, buffer_pack& pack, float timestep, int iteration, int max_iteration)
{
    //buffer_set& in = pack.in;
    buffer_set& out = pack.out;
    buffer_set& base = pack.base;

    cl::buffer& memory_ptrs_val = memory_ptrs.value();
    cl::buffer& counts_val = counts.value();

    //memory_alloc_count.set_to_zero(mqueue);

    ///so. Need to take all my particles, advance them forwards in time. Some complications because I'm not going to do this in a backwards euler way, so only on the 0th iteration do we do fun things. Need to pre-swap buffers
    ///need to fill up the adm buffers from the *current* particle positions
    vec3i dim = mesh.dim;
    cl_int4 clsize = {mesh.dim.x(), mesh.dim.y(), mesh.dim.z(), 0};
    float scale = mesh.scale;

    ///shit no its not correct, we need to be implicit otherwise the sources are incorrect innit. Its F(y+1)
    /*if(iteration == 0)
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
    }*/

    //int in_idx = pack.in_idx;
    int out_idx = pack.out_idx;
    int base_idx = pack.base_idx;

    ///make sure to mark up the particle code!
    {
        cl::args args;
        args.push_back(p_data[base_idx].position.as_device_read_only());
        args.push_back(p_data[base_idx].mass.as_device_read_only());
        args.push_back(p_data[out_idx].mass);
        args.push_back(p_data[base_idx].mass.as_device_read_only());
        args.push_back(particle_count);

        for(named_buffer& i : base.buffers)
        {
            args.push_back(i.buf.as_device_read_only());
        }

        args.push_back(scale);
        args.push_back(clsize);
        args.push_back(timestep);

        mqueue.exec("dissipate_mass", args, {particle_count}, {128});
    }

    ///so. The collect/sort method is generally a big performance win, except for when particles are *very* densely packed together
    /*{
        cl_int actually_write = 0;

        counts_val.set_to_zero(mqueue);

        cl::args args;
        args.push_back(p_data[in_idx].position);
        args.push_back(p_data[in_idx].mass);
        args.push_back(particle_count);
        args.push_back(counts_val);
        args.push_back(memory_ptrs_val);
        args.push_back(indices_block);
        args.push_back(scale);
        args.push_back(clsize);
        args.push_back(actually_write);

        mqueue.exec("collect_geodesics", args, {particle_count}, {128});
    }

    {
        cl::args args;
        args.push_back(counts_val);
        args.push_back(memory_ptrs_val);
        args.push_back(memory_alloc_count);
        args.push_back(max_intermediate_size);
        args.push_back(clsize);

        mqueue.exec("memory_allocate", args, {dim.x(), dim.y(), dim.z()}, {8,8,1});
    }

    {
        cl_int actually_write = 1;

        cl::args args;
        args.push_back(p_data[in_idx].position);
        args.push_back(p_data[in_idx].mass);
        args.push_back(particle_count);
        args.push_back(counts_val);
        args.push_back(memory_ptrs_val);
        args.push_back(indices_block);
        args.push_back(scale);
        args.push_back(clsize);
        args.push_back(actually_write);

        mqueue.exec("collect_geodesics", args, {particle_count}, {128});
    }

    {
        cl::args args;
        args.push_back(p_data[in_idx].position.as_device_read_only());
        args.push_back(p_data[in_idx].velocity.as_device_read_only());
        args.push_back(p_data[out_idx].position.as_device_write_only());
        args.push_back(p_data[out_idx].velocity.as_device_write_only());
        args.push_back(p_data[base_idx].position.as_device_read_only());
        args.push_back(p_data[base_idx].velocity.as_device_read_only());
        args.push_back(p_data[in_idx].mass.as_device_read_only());
        args.push_back(indices_block);
        args.push_back(memory_alloc_count);
        args.push_back(particle_count);

        for(named_buffer& i : in.buffers)
        {
            args.push_back(i.buf);
        }

        args.push_back(scale);
        args.push_back(clsize);
        args.push_back(timestep);

        mqueue.exec("index_trace_geodesics", args, {particle_count}, {128});
    }*/

    /*{
        cl::args args;
        args.push_back(p_data[in_idx].position.as_device_read_only());
        args.push_back(p_data[in_idx].velocity.as_device_read_only());
        args.push_back(p_data[out_idx].position.as_device_write_only());
        args.push_back(p_data[out_idx].velocity.as_device_write_only());
        args.push_back(p_data[base_idx].position.as_device_read_only());
        args.push_back(p_data[base_idx].velocity.as_device_read_only());
        args.push_back(p_data[in_idx].mass.as_device_read_only());
        args.push_back(counts_val);
        args.push_back(memory_ptrs_val);
        args.push_back(indices_block);
        args.push_back(particle_count);

        for(named_buffer& i : in.buffers)
        {
            args.push_back(i.buf);
        }

        args.push_back(scale);
        args.push_back(clsize);
        args.push_back(timestep);

        mqueue.exec("cube_trace_geodesics", args, {dim.x(), dim.y(), dim.z()}, {8,8,1});
    }*/

    /*{
        cl::args args;
        args.push_back(p_data[in_idx].position.as_device_read_only());
        args.push_back(p_data[in_idx].velocity.as_device_read_only());
        args.push_back(p_data[in_idx].lorentz.as_device_read_only());
        args.push_back(p_data[out_idx].lorentz);
        args.push_back(p_data[base_idx].lorentz.as_device_read_only());
        args.push_back(particle_count);

        for(named_buffer& i : in.buffers)
        {
            args.push_back(i.buf.as_device_read_only());
        }

        args.push_back(scale);
        args.push_back(clsize);
        args.push_back(timestep);

        mqueue.exec("evolve_lorentz", args, {particle_count}, {128});
    }*/

    {
        cl::args args;
        args.push_back(p_data[base_idx].position.as_device_read_only());
        args.push_back(p_data[base_idx].velocity.as_device_read_only());
        args.push_back(p_data[out_idx].position);
        args.push_back(p_data[out_idx].velocity);
        args.push_back(p_data[base_idx].position.as_device_read_only());
        args.push_back(p_data[base_idx].velocity.as_device_read_only());
        args.push_back(p_data[base_idx].mass.as_device_read_only());
        args.push_back(particle_count);

        for(named_buffer& i : base.buffers)
        {
            args.push_back(i.buf.as_device_read_only());
        }

        args.push_back(scale);
        args.push_back(clsize);
        args.push_back(timestep);

        mqueue.exec("trace_geodesics", args, {particle_count}, {128});
    }

    counts_val.set_to_zero(mqueue);
    memory_alloc_count.set_to_zero(mqueue);

    ///find how many particles would be written per-cell
    {
        cl_int actually_write = 0;

        cl::args args;
        args.push_back(p_data[base_idx].position.as_device_read_only());
        args.push_back(p_data[base_idx].mass.as_device_read_only());
        args.push_back(dirac_buf.as_device_inaccessible());
        args.push_back(particle_count);
        args.push_back(counts_val);
        args.push_back(memory_ptrs_val.as_device_inaccessible());
        args.push_back(indices_block.as_device_inaccessible());

        for(named_buffer& i : base.buffers)
        {
            args.push_back(i.buf.as_device_read_only());
        }

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
        args.push_back(max_intermediate_size);
        args.push_back(clsize);

        mqueue.exec("memory_allocate", args, {dim.x(), dim.y(), dim.z()}, {8,8,1});
    }

    ///write the indices indices_block at the offsets determined by memory_ptrs_val per-cell
    {
        cl_int actually_write = 1;

        cl::args args;
        args.push_back(p_data[base_idx].position.as_device_read_only());
        args.push_back(p_data[base_idx].mass.as_device_read_only());
        args.push_back(dirac_buf.as_device_write_only());
        args.push_back(particle_count);
        args.push_back(counts_val);
        args.push_back(memory_ptrs_val.as_device_read_only());
        args.push_back(indices_block);

        for(named_buffer& i : base.buffers)
        {
            args.push_back(i.buf.as_device_read_only());
        }

        args.push_back(scale);
        args.push_back(clsize);
        args.push_back(actually_write);

        mqueue.exec("collect_particle_spheres", args, {particle_count}, {128});
    }

    ///calculate adm quantities per-cell by summing across the list of particles
    {
        cl::args args;
        args.push_back(mesh.points_set.all_points);
        args.push_back(mesh.points_set.all_count);
        args.push_back(p_data[base_idx].position.as_device_read_only());
        args.push_back(p_data[base_idx].velocity.as_device_read_only());
        args.push_back(p_data[base_idx].mass.as_device_read_only());
        args.push_back(dirac_buf.as_device_read_only());
        args.push_back(particle_count);
        args.push_back(counts_val.as_device_read_only());
        args.push_back(memory_ptrs_val.as_device_read_only());
        args.push_back(indices_block.as_device_read_only());

        for(named_buffer& i : base.buffers)
        {
            if(i.desc.modified_by == "do_weighted_summation")
                args.push_back(i.buf);
            else
                args.push_back(i.buf.as_device_read_only());
        }

        mesh.append_utility_buffers("do_weighted_summation", args);

        args.push_back(scale);
        args.push_back(clsize);

        mqueue.exec("do_weighted_summation", args, {mesh.points_set.all_count}, {128});
    }

    ///todo: not this, want to have the indices controlled from a higher level
    /*if(iteration != max_iteration)
    {
        std::swap(p_data[in_idx], p_data[out_idx]);
    }
    else
    {
        std::swap(p_data[base_idx], p_data[out_idx]);
    }*/

    if(iteration != max_iteration - 1)
    {
        std::swap(p_data[1], p_data[2]);
    }
    else
    {
        std::swap(p_data[1], p_data[0]);
    }
}

void particle_dynamics::finalise(cpu_mesh& mesh, cl::context& ctx, cl::managed_command_queue& mqueue, thin_intermediates_pool& pool, float timestep)
{

}

void particle_dynamics::save(cl::command_queue& cqueue, const std::string& directory)
{
    save_buffer(cqueue, p_data[0].position, directory + "/particle_position.bin");
    save_buffer(cqueue, p_data[0].velocity, directory + "/particle_velocity.bin");
    save_buffer(cqueue, p_data[0].mass    , directory + "/particle_mass.bin");
    save_buffer(cqueue, p_data[0].lorentz , directory + "/particle_lorentz.bin");
}

void particle_dynamics::load(cl::command_queue& cqueue, const std::string& directory)
{
    load_buffer(cqueue, p_data[0].position, directory + "/particle_position.bin");
    load_buffer(cqueue, p_data[0].velocity, directory + "/particle_velocity.bin");
    load_buffer(cqueue, p_data[0].mass    , directory + "/particle_mass.bin");
    load_buffer(cqueue, p_data[0].lorentz , directory + "/particle_lorentz.bin");
}
