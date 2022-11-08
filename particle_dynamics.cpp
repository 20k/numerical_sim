#include "particle_dynamics.hpp"
#include <geodesic/dual_value.hpp>
#include "equation_context.hpp"
#include "bssn.hpp"
#include "random.hpp"
#include "spherical_integration.hpp"

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

particle_dynamics::particle_dynamics(cl::context& ctx) : p_data{ctx, ctx, ctx}, pd(ctx), indices_block(ctx), weights_block(ctx), memory_alloc_count(ctx)
{
    indices_block.alloc(max_intermediate_size * sizeof(cl_ulong));
    weights_block.alloc(max_intermediate_size * sizeof(cl_float));
    memory_alloc_count.alloc(sizeof(size_t));
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

    #if 0
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
    #endif

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

    ctx.add("V0Diff", u_lower_diff.idx(0));
    ctx.add("V1Diff", u_lower_diff.idx(1));
    ctx.add("V2Diff", u_lower_diff.idx(2));

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

///https://arxiv.org/pdf/1705.04131.pdf 28
float matter_cdf(float m0, float r0, float rc, float r, float B = 1)
{
    return m0 * pow(sqrt(r0/rc) * r/(r + rc), 3 * B);
}

float select_from_cdf(float value_mx, float max_radius, auto cdf)
{
    float value_at_max = cdf(max_radius);

    if(value_mx >= value_at_max)
        return max_radius * 1000;

    float next_upper = max_radius;
    float next_lower = 0;

    for(int i=0; i < 50; i++)
    {
        float test_val = (next_upper + next_lower)/2.f;

        float found_val = cdf(test_val);

        if(found_val < value_mx)
        {
            next_lower = test_val;
        }
        else if(found_val > value_mx)
        {
            next_upper = test_val;
        }
        else
        {
            return test_val;
        }
    }

    //printf("Looking for %.14f found %.14f with y = %.14f\n", scaled, (next_upper + next_lower)/2.f, cdf((next_upper + next_lower)/2.f));

    return (next_upper + next_lower)/2.f;
}

///https://www.mdpi.com/2075-4434/6/3/70/htm (7)
///ok sweet! Next up, want to divorce particle step from field step
///ideally we'll step forwards the particles by a large timestep, and the interpolate to generate the underlying fields
///geodesic trace time >> discretisation time, so shouldn't be a problem, and that'll take us to 2m particles
void particle_dynamics::init(cpu_mesh& mesh, cl::context& ctx, cl::command_queue& cqueue, thin_intermediates_pool& pool, buffer_set& to_init)
{
    vec3i dim = mesh.dim;

    memory_ptrs = cl::buffer(ctx);
    counts = cl::buffer(ctx);

    memory_ptrs.value().alloc(sizeof(cl_ulong) * dim.x() * dim.y() * dim.z());
    counts.value().alloc(sizeof(cl_ulong) * dim.x() * dim.y() * dim.z());

    cl_int4 clsize = {dim.x(), dim.y(), dim.z(), 0};
    float scale = mesh.scale;

    particle_count = 1000 * 80;

    for(int i=0; i < (int)p_data.size(); i++)
    {
        p_data[i].position.alloc(sizeof(cl_float) * 3 * particle_count);
        p_data[i].velocity.alloc(sizeof(cl_float) * 3 * particle_count);
        p_data[i].mass.alloc(sizeof(cl_float) * particle_count);
    }

    float generation_radius = 0.5f * get_c_at_max()/2.f;

    ///need to use an actual rng if i'm doing anything even vaguely scientific
    std::vector<float> analytic_radius;
    std::vector<vec3f> positions;
    std::vector<vec3f> directions;
    std::vector<float> masses;

    double solar_mass = 1.98892 * pow(10., 30.);

    //double milky_way_mass = 6. * pow(10., 42.);

    double milky_way_mass = 6.43 * pow(10., 10.) * 1.16 * solar_mass;

    double C = 299792458.;
    double G = 6.67430 * pow(10., -11.);

    double milky_way_mass_in_meters = milky_way_mass * G / (C*C);

    printf("Milky mass %f\n", milky_way_mass_in_meters);

    double milky_way_diameter_in_meters = pow(10., 21.);

    double milky_way_diameter_in_scale = get_c_at_max() * 0.6f;

    double meters_to_scale = milky_way_diameter_in_scale / milky_way_diameter_in_meters;

    double milky_way_mass_in_scale = meters_to_scale * milky_way_mass_in_meters;

    printf("Milky mass numerical %.15f\n", milky_way_mass_in_scale);
    printf("Milky Radius scale %.15f\n", milky_way_diameter_in_scale);

    //float total_mass = 2;

    float total_mass = milky_way_mass_in_scale;

    float init_mass = total_mass / particle_count;

    {
        double time_for_light_to_traverse_s = milky_way_diameter_in_meters / C;
        double time_for_light_to_traverse_m = time_for_light_to_traverse_s * C;
        double time_for_light_to_traverse_scale = time_for_light_to_traverse_m * meters_to_scale;

        printf("Light Time %f\n", time_for_light_to_traverse_scale);
    }

    ///https://www.mdpi.com/2075-4434/6/3/70/htm mond galaxy info

    printf("Mass per particle %.20f\n", init_mass);

    //float init_mass = 0.000002;
    //float total_mass = mass * particle_count;

    for(uint64_t i=0; i < particle_count; i++)
    {
        masses.push_back(init_mass);
    }

    float M0 = milky_way_mass_in_scale;
    float R0 = milky_way_diameter_in_scale/5.f;
    float Rc = milky_way_diameter_in_scale/5.f;

    /*auto cdf = [&](float r)
    {
        return matter_cdf(milky_way_mass_in_scale, R0, Rc, r, 1);
    };*/

    //float integrated


    auto surface_density = [&](float r)
    {
        //float a = 1;

        //return (milky_way_mass_in_scale / (2 * M_PI * a * a)) * pow(1 + r*r/a*a, -3.f/2.f);

        float a = 1;

        return (milky_way_mass_in_scale * a / (2 * M_PI)) * pow(r*r + a*a, -3.f/2.f);
    };

    auto cdf = [&](float r)
    {
        /*auto p2 = [&](float r)
        {
            return 4 * M_PI * r * r * surface_density(r);
        };*/

        auto p3 = [&](float r)
        {
            return 2 * M_PI * r * surface_density(r);
        };

        return integrate_1d(p3, 64, r, 0.f);
    };

    auto get_mond_velocity = [&](float r, float M, float G, float a0)
    {
        float p1 = G * M/r;

        float p2 = (1/sqrt(2.f));

        float frac = 2 * a0 / (G * M);

        float p_inner = 1 + sqrt(1 + pow(r, 4.f) * pow(frac, 2.f));

        float p3 = sqrt(p_inner);

        return sqrt(p1 * p2 * p3);

        /*float b = 0.352;
        float B = 1;

        float p1 = (G * M0) / r;

        float p2 = pow(sqrt(R0/Rc) * r / (r + Rc), 3 * B);

        float p3 = (1 + b * (1 + r/R0));

        return sqrt(p1 * p2 * p3);*/
    };

    xoshiro256ss_state rng = xoshiro256ss_init(2345);

    auto random = [&]()
    {
        return uint64_to_double(xoshiro256ss(rng));
    };

    float approximate_core_mass = cdf(Rc);

    /*int core_particles = ceilf(approximate_core_mass / init_mass);

    for(int i=0; i < core_particles; i++)
    {
        vec3f random_pos;

        while(1)
        {
            random_pos = {random(), random(), random()};

            random_pos = (random_pos - 0.5f) * 2 * Rc;

            if(random_pos.length() <= Rc)
                break;
        }

        vec3f pos = random_pos;

        pos.z() *= 0.05f;

        positions.push_back(pos);
    }*/

    for(int i=0; i < particle_count; i++)
    {
        /*float radius = 0;

        while(1)
        {
            float random_val = random();

            radius = select_from_cdf(random_val, milky_way_diameter_in_scale/2.f, cdf);

            if(radius > Rc)
                break;
        }*/

        float radius = 0;

        do
        {
            float random_mass = random() * milky_way_mass_in_scale;

            radius = select_from_cdf(random_mass, milky_way_diameter_in_scale/2.f, cdf);
        } while(radius >= milky_way_diameter_in_scale/2.f || radius <= milky_way_diameter_in_scale/100.f);

        ///M
        //float mass_density = cdf(radius);

        ///I have a distinct feeling we might need a sphere term in here
        float angle = random() * 2 *  M_PI;

        float z = (random() - 0.5f) * 2.f * milky_way_diameter_in_scale * 0.000001f;

        vec2f pos2 = {cos(angle) * radius, sin(angle) * radius};
        vec3f pos = {pos2.x(), pos2.y(), z};

        positions.push_back(pos);
        analytic_radius.push_back(radius);
    }

    /*std::sort(positions.begin(), positions.end(), [](vec3f v1, vec3f v2)
    {
        return v1.length() < v2.length();
    });*/

    {
        double real_cdf_by_radius = 0;

        int which = 0;

        //for(vec3f p : positions)

        for(int i=0; i < (int)positions.size(); i++)
        {
            vec3f p = positions[i];
            float radius = analytic_radius[i];

            real_cdf_by_radius += init_mass;

            //float radius = p.length();

            //float M_r = real_cdf_by_radius;

            float M_r = cdf(radius);

            float angle = atan2(p.y(), p.x());

            vec2f velocity_direction = (vec2f){1, 0}.rot(angle + M_PI/2);

            double critical_acceleration_ms2 = 1.2 * pow(10., -8);

            double critical_acceleration_im = critical_acceleration_ms2 / (C * C); ///units of 1/meters
            double critical_acceleration_scale = critical_acceleration_im / meters_to_scale;

            float mond_velocity = get_mond_velocity(radius, M_r, 1, critical_acceleration_scale);

            //float mond_velocity = sqrt(1 * M_r / radius);

            //float mond_velocity = sqrt(1 * M_r * radius * radius * pow(radius * radius + 1 * 1, -3.f/2.f));

            if((which % 100) == 0)
            {
                printf("Velocity %f at radius %f Total Mass %.10f Analytic Mass %.10f\n", mond_velocity, radius, M_r, cdf(radius));
            }

            float linear_velocity = mond_velocity;

            vec2f velocity = linear_velocity * velocity_direction;

            vec3f velocity3 = {velocity.x(), velocity.y(), 0};

            directions.push_back(velocity3);

            which++;
        }
    }

    #if 0
    ///oh! use the actual matter distribution we get out instead of the theoretical one
    for(uint64_t i=0; i < particle_count; i++)
    {
        int kk=0;

        for(; kk < 1024; kk++)
        {
            /*float v0 = uint64_to_double(xoshiro256ss(rng));
            float v1 = uint64_to_double(xoshiro256ss(rng));
            float v2 = uint64_to_double(xoshiro256ss(rng));

            float x = (v0 - 0.5f) * 2 * generation_radius;
            float y = (v1 - 0.5f) * 2 * generation_radius;
            float z = (v2 - 0.5f) * 2 * generation_radius;

            vec3f pos = {x, y, z};

            pos.z() *= 0.1f;

            float angle = atan2(pos.y(), pos.x());
            float radius = pos.length();*/

            float random_val = random();

            float radius = select_from_cdf(random_val, milky_way_diameter_in_scale/2.f, cdf);

            ///M
            float mass_density = cdf(radius);

            ///I have a distinct feeling we might need a sphere term in here
            float angle = random() * 2 *  M_PI;

            float z = (random() - 0.5f) * 2.f * milky_way_diameter_in_scale * 0.05f;

            vec2f pos2 = {cos(angle) * radius, sin(angle) * radius};
            vec3f pos = {pos2.x(), pos2.y(), z};

            //if(radius >= generation_radius || radius < generation_radius * 0.1f)

            //if(radius >= generation_radius)
            //    continue;

            positions.push_back(pos);

            vec2f velocity_direction = (vec2f){1, 0}.rot(angle + M_PI/2);

            //float linear_velocity = get_kepler_velocity(radius, mass, total_mass - mass) * 0.15f;

            //printf("Linear velocity %f\n", linear_velocity);

            //float linear_velocity = 0.1f * pow(radius / generation_radius, 2.f);

            double critical_acceleration_ms2 = 1.2 * pow(10., -8);

            double critical_acceleration_im = critical_acceleration_ms2 / (C * C); ///units of 1/meters
            double critical_acceleration_scale = critical_acceleration_im / meters_to_scale;

            float mond_velocity = get_mond_velocity(radius, mass_density, 1, critical_acceleration_scale);

            float linear_velocity = mond_velocity;

            if(linear_velocity >= 0.5f)
            {
                printf("Umm\n");
            }

            vec2f velocity = linear_velocity * velocity_direction;

            vec3f velocity3 = {velocity.x(), velocity.y(), 0};

            directions.push_back(velocity3);

            break;
        }

        if(kk == 1024)
            throw std::runtime_error("Did not successfully assign particle position");
    }
    #endif


    {
        std::vector<vec3f> sorted_positions = positions;

        std::sort(sorted_positions.begin(), sorted_positions.end(), [](vec3f v1, vec3f v2)
        {
            return v1.length() < v2.length();
        });

        double run_mass = 0;

        int dbg_idx = 0;

        for(vec3f v : sorted_positions)
        {
            float rad = v.length();

            //if((dbg_idx % 100) == 0)
            //    printf("Current mass %.18f Expected mass %.18f at rad %.18f milky %.18f\n", run_mass / total_mass, cdf(rad) / cdf(milky_way_diameter_in_scale/2.f), rad, milky_way_diameter_in_scale);

            run_mass += init_mass;

            dbg_idx++;
        }
    }

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

        tensor<value, 3> ten_vel = {velocity.y(), velocity.z(), velocity.w()};

        tensor<value, 3> lowered_vel = lower_index(ten_vel, args.Yij, 0);

        ectx.add("OUT_VT", 1);
        ectx.add("OUT_VX", lowered_vel.idx(0));
        ectx.add("OUT_VY", lowered_vel.idx(1));
        ectx.add("OUT_VZ", lowered_vel.idx(2));

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
    ///https://arxiv.org/pdf/1904.07841.pdf so, have to discretise the dirac delta. This paper gives explicit version
    {
        equation_context ectx;
        standard_arguments args(ectx);

        tensor<value, 3> u_lower = {"vel.x", "vel.y", "vel.z"};

        value mass = "mass";

        value sum = 0;

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                sum += args.iYij.idx(i, j) * u_lower.idx(i) * u_lower.idx(j);
            }
        }

        ///https://arxiv.org/pdf/1904.07841.pdf 2.28
        ///https://en.wikipedia.org/wiki/Four-momentum#Relation_to_four-velocity
        //value Ea = sqrt(mass * mass + mass * mass * sum);
        value lorentz = sqrt(1 + fabs(sum));
        value Ea = mass * lorentz;

        tensor<value, 3> covariant_momentum = mass * u_lower; ///????

        //value idet = pow(args.W_impl, 3);
        value idet = pow(args.get_X(), 3.f/2.f);

        value out_adm_p = idet * Ea;

        tensor<value, 3> Si = idet * covariant_momentum;

        tensor<value, 3, 3> Sij;

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                //Sij.idx(i, j) = idet * (covariant_momentum.idx(i) * covariant_momentum.idx(j) / Ea);
                Sij.idx(i, j) = idet * (covariant_momentum.idx(i) * (u_lower.idx(j) / lorentz));
            }
        }

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

        args.push_back(positions_in);
        args.push_back(initial_dirs);
        args.push_back(p_data[0].position);
        args.push_back(p_data[0].velocity);

        args.push_back(particle_count);
        args.push_back(scale);
        args.push_back(clsize);

        kern.set_args(args);

        cqueue.exec(kern, {particle_count}, {128});
    }

    cl::copy(cqueue, p_data[0].position, p_data[1].position);
    cl::copy(cqueue, p_data[0].velocity, p_data[1].velocity);
    cl::copy(cqueue, p_data[0].mass, p_data[1].mass);

    cl::copy(cqueue, p_data[0].position, p_data[2].position);
    cl::copy(cqueue, p_data[0].velocity, p_data[2].velocity);
    cl::copy(cqueue, p_data[0].mass, p_data[2].mass);

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

void particle_dynamics::step(cpu_mesh& mesh, cl::context& ctx, cl::managed_command_queue& mqueue, thin_intermediates_pool& pool, buffer_pack& pack, float timestep, int iteration, int max_iteration)
{
    elapsed += timestep;

    buffer_set& in = pack.in;
    buffer_set& out = pack.out;
    buffer_set& base = pack.base;

    cl::buffer& memory_ptrs_val = memory_ptrs.value();
    cl::buffer& counts_val = counts.value();

    memory_alloc_count.set_to_zero(mqueue);

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

    int in_idx = pack.in_idx;
    int out_idx = pack.out_idx;
    int base_idx = pack.base_idx;

    bool step_particles = elapsed > get_c_at_max() * 2;

    ///make sure to mark up the particle code!
    if(step_particles)
    {
        cl::args args;
        args.push_back(p_data[in_idx].position.as_device_read_only());
        args.push_back(p_data[in_idx].velocity.as_device_read_only());
        args.push_back(p_data[in_idx].mass.as_device_read_only());
        args.push_back(p_data[out_idx].mass.as_device_write_only());
        args.push_back(p_data[base_idx].mass.as_device_read_only());
        args.push_back(particle_count);
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

    if(step_particles)
    {
        cl::args args;
        args.push_back(p_data[in_idx].position.as_device_read_only());
        args.push_back(p_data[in_idx].velocity.as_device_read_only());
        args.push_back(p_data[out_idx].position.as_device_write_only());
        args.push_back(p_data[out_idx].velocity.as_device_write_only());
        args.push_back(p_data[base_idx].position.as_device_read_only());
        args.push_back(p_data[base_idx].velocity.as_device_read_only());
        args.push_back(p_data[in_idx].mass.as_device_read_only());
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

    counts_val.set_to_zero(mqueue);
    memory_alloc_count.set_to_zero(mqueue);

    ///find how many particles would be written per-cell
    {
        cl_int actually_write = 0;

        cl::args args;
        args.push_back(p_data[in_idx].position);
        args.push_back(p_data[in_idx].mass);
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
        args.push_back(max_intermediate_size);
        args.push_back(clsize);

        mqueue.exec("memory_allocate", args, {dim.x(), dim.y(), dim.z()}, {8,8,1});
    }

    ///write the indices and weights into indices_block, and weights_block at the offsets determined by memory_ptrs_val per-cell
    {
        cl_int actually_write = 1;

        cl::args args;
        args.push_back(p_data[in_idx].position);
        args.push_back(p_data[in_idx].mass);
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
        args.push_back(p_data[in_idx].position);
        args.push_back(p_data[in_idx].velocity);
        args.push_back(p_data[in_idx].mass);
        args.push_back(particle_count);
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

    if(step_particles)
    {
        ///todo: not this, want to have the indices controlled from a higher level
        if(iteration != max_iteration)
        {
            std::swap(p_data[in_idx].position, p_data[out_idx].position);
            std::swap(p_data[in_idx].velocity, p_data[out_idx].velocity);
            std::swap(p_data[in_idx].mass, p_data[out_idx].mass);
        }
        else
        {
            std::swap(p_data[base_idx].position, p_data[out_idx].position);
            std::swap(p_data[base_idx].velocity, p_data[out_idx].velocity);
            std::swap(p_data[base_idx].mass, p_data[out_idx].mass);
        }
    }
}

void particle_dynamics::finalise(cpu_mesh& mesh, cl::context& ctx, cl::managed_command_queue& mqueue, thin_intermediates_pool& pool, float timestep)
{

}
