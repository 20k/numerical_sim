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

void build_energy(equation_context& ctx)
{
    ctx.uses_linear = true;
    ctx.order = 2;
    ctx.use_precise_differentiation = false;

    standard_arguments args(ctx);

    ctx.pin(args.Kij);
    ctx.pin(args.Yij);

    value scale = "scale";
    tensor<value, 3> V_upper = {"V0", "V1", "V2"};
    value E = "eq_E";

    value diff = 0;

    for(int i=0; i < 3; i++)
    {
        value kij_sum = 0;

        for(int j=0; j < 3; j++)
        {
            kij_sum += args.Kij.idx(i, j) * V_upper.idx(j);
        }

        diff += E * V_upper.idx(i) * (args.gA * kij_sum - diff1(ctx, args.gA, i));
    }

    ctx.add("EDiff", diff);
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

    #if 1
    //ctx.add("universe_size", universe_length * scale);

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

    /*tensor<value, 3> u_lower = {"V0", "V1", "V2"};

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
    }*/

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

double get_solar_mass_kg()
{
    return 1.98892 * std::pow(10., 30.);
}

///m/s
double get_c()
{
    return 299792458;
}

///m3 kg-1 s-2
double get_G()
{
    return 6.67430 * std::pow(10., -11.);
}

struct galaxy_params
{
    double mass_kg = 0;
    double radius_m = 0;
};

struct disk_distribution
{
    bool is_disk = true;

    double a = 3;

    double cdf(double M0, double G, double r)
    {
        /*auto surface_density = [this, M0](double r)
        {
            return (M0 * a / (2 * M_PI)) * 1./std::pow(r*r + a*a, 3./2.);

            //return (M0 / (2 * M_PI * a * a)) / pow(1 + r*r/a*a, 3./2.);
        };

        auto integral = [&surface_density](double r)
        {
            return 2 * M_PI * r * surface_density(r);
        };

        return integrate_1d(integral, 64, r, 0.);*/

        return M0 * (1 - a / sqrt(r*r + a*a));
    }

    ///https://galaxiesbook.org/chapters/II-01.-Flattened-Mass-Distributions.html 8.17 implies that kuzmin uses M0, not the CDF
    double get_velocity_at(double M0, double G, double r)
    {
        return std::sqrt(G * M0 * r * r * pow(r * r + a * a, -3./2.));
        //return std::sqrt(G * cdf(M0, G, r) * r * r * pow(r * r + a * a, -3./2.));
    }

    ///https://adsabs.harvard.edu/full/1995MNRAS.276..453B
    double get_mond_velocity_at(double M0, double G, double r, double a0)
    {
        ///they use h for a
        double u = r / a;

        double v_inf = pow(M0 * G * a0, 1./4.);

        ///I've seen this before but I can't remember what it is
        double squiggle = M0 * G / (a*a * a0);

        double p1 = v_inf * v_inf;

        double p2 = u*u / (1 + u*u);

        double divisor_part = 1 + u*u;

        double interior_1 = sqrt(1 + squiggle*squiggle / (4 * pow(divisor_part, 2)));

        double interior_2 = squiggle / (2 * divisor_part);

        return sqrt(p1 * p2 * sqrt(interior_1 + interior_2));
    }
};

/*struct spherical_distribution
{
    bool is_disk = false;

    double cdf(double M0, double G, double r)
    {
        auto density = [M0](double r)
        {
            double r0 = 1;
            double rc = 1;
            double B = 1;

            return
        };
    }
};*/

///This is not geometric units, this is scale independent
template<typename T>
struct galaxy_distribution
{
    double mass = 0;
    double max_radius = 0;

    double local_G = 0;
    double meters_to_local = 0;

    T distribution;

    double local_distance_to_meters(double r)
    {
        return r / meters_to_local;
    }

    ///M(r)
    double cdf(double r)
    {
        ///correct for cumulative sphere model
        /*auto p2 = [&](float r)
        {
            return 4 * M_PI * r * r * surface_density(r);
        };*/

        ///correct for surface density
        /*auto p3 = [&](double r)
        {
            return 2 * M_PI * r * surface_density(r);
        };

        return integrate_1d(p3, 64, r, 0.);*/

        return distribution.cdf(mass, local_G, r);
    };

    double get_velocity_at(double r)
    {
        double a0_ms2 = 1.2 * pow(10., -10.);

        double a0 = a0_ms2 * meters_to_local;

        /*double p1 = local_G * cdf(r)/r;

        double p2 = (1/sqrt(2.f));

        double frac = 2 * a0 / (local_G * cdf(r));

        double p_inner = 1 + sqrt(1 + pow(r, 4.f) * pow(frac, 2.f));

        double p3 = sqrt(p_inner);

        return std::sqrt(p1 * p2 * p3);*/

        //return std::sqrt(local_G * cdf(r) / r);

        ///https://galaxiesbook.org/chapters/II-01.-Flattened-Mass-Distributions.html 8.16
        //return std::sqrt(local_G * cdf(r) * r * r * pow(r * r + 1 * 1, -3.f/2.f));

        //return distribution.get_mond_velocity_at(mass, local_G, r, a0);
        return distribution.get_velocity_at(mass, local_G, r);
    }

    galaxy_distribution(const galaxy_params& params)
    {
        ///decide scale. Probably just crack radius between 0 and 5 because astrophysics!
        ///sun units?

        mass = params.mass_kg / get_solar_mass_kg();
        max_radius = 14.4; ///YEP

        double to_local_distance = max_radius / params.radius_m;
        double to_local_mass = mass / params.mass_kg;

        local_G = get_G() * pow(to_local_distance, 3) / to_local_mass;
        meters_to_local = to_local_distance;
    }

    double select_radius(xoshiro256ss_state& rng)
    {
        auto lambda_cdf = [&](double r)
        {
            return cdf(r);
        };

        double found_radius = 0;

        do
        {
            double random = uint64_to_double(xoshiro256ss(rng));

            double random_mass = random * mass;

            found_radius = select_from_cdf(random_mass, max_radius, lambda_cdf);
        } while(found_radius >= max_radius);

        return found_radius;
    }
};

struct numerical_params
{
    double mass = 0;
    double radius = 0;

    double mass_to_m = 0;
    double m_to_scale = 0;

    numerical_params(const galaxy_params& params)
    {
        double C = 299792458.;
        double G = 6.67430 * pow(10., -11.);

        double mass_in_m = params.mass_kg * G / (C*C);
        double radius_in_m = params.radius_m;

        double max_scale_radius = get_c_at_max() * 0.5f * 0.7f;
        double meters_to_scale = max_scale_radius / radius_in_m;

        mass = mass_in_m * meters_to_scale;
        radius = radius_in_m * meters_to_scale;

        printf("My mass %f\n", mass);

        mass_to_m = G / (C*C);
        m_to_scale = meters_to_scale;
    }

    double convert_mass_to_scale(double mass_kg)
    {
        return mass_kg * mass_to_m * m_to_scale;
    }

    double convert_distance_to_scale(double dist_m)
    {
        return dist_m * m_to_scale;
    }
};

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

    particle_count = 1000 * 60;

    for(int i=0; i < (int)p_data.size(); i++)
    {
        p_data[i].position.alloc(sizeof(cl_float) * 3 * particle_count);
        p_data[i].velocity.alloc(sizeof(cl_float) * 3 * particle_count);
        p_data[i].mass.alloc(sizeof(cl_float) * particle_count);
        p_data[i].energy.alloc(sizeof(cl_float) * particle_count);
    }

    ///need to use an actual rng if i'm doing anything even vaguely scientific
    //std::vector<float> analytic_radius;
    std::vector<vec3f> positions;
    std::vector<vec3f> directions;
    std::vector<float> masses;

    ///https://arxiv.org/abs/1607.08364
    //double milky_way_mass_kg = 6.43 * pow(10., 10.) * 1.16 * get_solar_mass_kg();
    double milky_way_mass_kg = 4 * pow(10, 11) * get_solar_mass_kg();
    //double milky_way_radius_m = 0.5f * pow(10., 21.);
    double milky_way_radius_m = 0.5 * 8 * pow(10., 20.);

    galaxy_params params;
    params.mass_kg = milky_way_mass_kg;
    params.radius_m = milky_way_radius_m;

    galaxy_distribution<disk_distribution> dist(params);

    numerical_params num_params(params);

    float init_mass = num_params.mass / particle_count;

    ///https://www.mdpi.com/2075-4434/6/3/70/htm mond galaxy info

    printf("Mass per particle %.20f\n", init_mass);

    for(uint64_t i=0; i < particle_count; i++)
    {
        masses.push_back(init_mass);
    }

    /*auto get_mond_velocity = [&](float r, float M, float G, float a0)
    {
        float p1 = G * M/r;

        float p2 = (1/sqrt(2.f));

        float frac = 2 * a0 / (G * M);

        float p_inner = 1 + sqrt(1 + pow(r, 4.f) * pow(frac, 2.f));

        float p3 = sqrt(p_inner);

        return sqrt(p1 * p2 * p3);
    };*/

    xoshiro256ss_state rng = xoshiro256ss_init(2345);

    for(int i=0; i < particle_count; i++)
    {
        double radius = dist.select_radius(rng);
        double velocity = dist.get_velocity_at(radius);

        //printf("Local Velocity %f\n", velocity);
        //printf("Local To Meters", 1/dist.meters_to_local);

        double angle = uint64_to_double(xoshiro256ss(rng)) * 2 * M_PI;

        double z = 0;

        double radius_m = dist.local_distance_to_meters(radius);
        double radius_scale = num_params.convert_distance_to_scale(radius_m);

        //double scale_radius = num_params.convert_distance_to_scale(radius);

        vec3f pos = {cos(angle) * radius_scale, sin(angle) * radius_scale, z};

        positions.push_back(pos);

        ///velocity is distance/s so should be fine
        double speed_in_ms = dist.local_distance_to_meters(velocity);
        double speed_in_c = speed_in_ms / get_c();

        vec2f velocity_direction = (vec2f){1, 0}.rot(angle + M_PI/2);

        vec2f velocity_2d = speed_in_c * velocity_direction;

        vec3f velocity_fin = {velocity_2d.x(), velocity_2d.y(), 0.f};

        directions.push_back(velocity_fin);

        assert(speed_in_c < 1);

        //printf("Velocity %f\n", speed_in_c);
        //printf("Position %f %f %f\n", pos.x(), pos.y(), pos.z());
    }

    {
        std::vector<std::pair<vec3f, vec3f>> pos_vel;
        pos_vel.reserve(particle_count);

        for(int i=0; i < particle_count; i++)
        {
            pos_vel.push_back({positions[i], directions[i]});
        }

        std::sort(pos_vel.begin(), pos_vel.end(), [](auto& i1, auto& i2)
        {
            return i1.first.squared_length() < i2.first.squared_length();
        });

        float selection_radius = 0;

        for(auto& [p, v] : pos_vel)
        {
            float p_len = p.length();
            float v_len = v.length();

            if(p_len >= selection_radius)
            {
                selection_radius += 0.25f;
                debug_velocities.push_back(v.length());
            }
        }
    }

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

        tensor<value, 3> upper_vel = {velocity.y(), velocity.z(), velocity.w()};

        //tensor<value, 3> lowered_vel = lower_index(upper_vel, args.Yij, 0);

        value lorentz = velocity.x();

        //value Ea = lorentz * mass;

        ectx.add("OUT_LORENTZ", lorentz);
        ectx.add("OUT_VT", 1);
        ectx.add("OUT_VX", upper_vel.idx(0) / lorentz);
        ectx.add("OUT_VY", upper_vel.idx(1) / lorentz);
        ectx.add("OUT_VZ", upper_vel.idx(2) / lorentz);

        ectx.build(argument_string, "tparticleinit");
    }

    {
        equation_context ectx;
        build_energy(ectx);

        ectx.build(argument_string, "energy");
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
        ectx.uses_linear = true;
        ectx.order = 2;
        ectx.use_precise_differentiation = false;

        standard_arguments args(ectx);

        /*tensor<value, 3> u_lower = {"vel.x", "vel.y", "vel.z"};

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
        value Ea = mass * lorentz;*/

        tensor<value, 3> v_upper = {"vel.x", "vel.y", "vel.z"};
        tensor<value, 3> v_lower = lower_index(v_upper, args.Yij, 0);

        value Ea = "energy";
        value mass = "mass";

        //value lorentz = energy / mass;

        //tensor<value, 3> u_lower = lower_index(v_upper, args.Yij, 0) * lorentz;

        tensor<value, 3> covariant_momentum = v_lower * Ea; ///????

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
                Sij.idx(i, j) = idet * (covariant_momentum.idx(i) * v_lower.idx(j));
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
        args.push_back(p_data[0].mass);
        args.push_back(p_data[0].position);
        args.push_back(p_data[0].velocity);
        args.push_back(p_data[0].energy);

        args.push_back(particle_count);
        args.push_back(scale);
        args.push_back(clsize);

        kern.set_args(args);

        cqueue.exec(kern, {particle_count}, {128});
    }

    cl::copy(cqueue, p_data[0].position, p_data[1].position);
    cl::copy(cqueue, p_data[0].velocity, p_data[1].velocity);
    cl::copy(cqueue, p_data[0].mass, p_data[1].mass);
    cl::copy(cqueue, p_data[0].energy, p_data[1].energy);

    cl::copy(cqueue, p_data[0].position, p_data[2].position);
    cl::copy(cqueue, p_data[0].velocity, p_data[2].velocity);
    cl::copy(cqueue, p_data[0].mass, p_data[2].mass);
    cl::copy(cqueue, p_data[0].energy, p_data[2].energy);

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

    ///make sure to mark up the particle code!
    {
        cl::args args;
        args.push_back(p_data[in_idx].position);
        args.push_back(p_data[in_idx].mass);
        args.push_back(p_data[out_idx].mass);
        args.push_back(p_data[base_idx].mass);
        args.push_back(particle_count);
        args.push_back(timestep);

        mqueue.exec("dissipate_mass", args, {particle_count}, {128});
    }

    //return;

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

    {
        cl::args args;
        args.push_back(p_data[in_idx].position);
        args.push_back(p_data[in_idx].velocity);
        args.push_back(p_data[in_idx].energy);
        args.push_back(p_data[out_idx].energy);
        args.push_back(p_data[base_idx].energy);
        args.push_back(particle_count);

        for(named_buffer& i : in.buffers)
        {
            args.push_back(i.buf);
        }

        args.push_back(scale);
        args.push_back(clsize);
        args.push_back(timestep);

        mqueue.exec("evolve_energy", args, {particle_count}, {128});
    }

    /*{
        cl::args args;
        args.push_back(p_data[in_idx].position);
        args.push_back(p_data[in_idx].velocity);
        args.push_back(p_data[out_idx].position);
        args.push_back(p_data[out_idx].velocity);
        args.push_back(p_data[base_idx].position);
        args.push_back(p_data[base_idx].velocity);
        args.push_back(p_data[in_idx].mass);
        args.push_back(particle_count);

        for(named_buffer& i : in.buffers)
        {
            args.push_back(i.buf);
        }

        args.push_back(scale);
        args.push_back(clsize);
        args.push_back(timestep);

        mqueue.exec("trace_geodesics", args, {particle_count}, {128});
    }*/

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
        args.push_back(p_data[in_idx].energy);
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

    #if 0

    ///todo: not this, want to have the indices controlled from a higher level
    if(iteration != max_iteration)
    {
        std::swap(p_data[in_idx].position, p_data[out_idx].position);
        std::swap(p_data[in_idx].velocity, p_data[out_idx].velocity);
        std::swap(p_data[in_idx].mass, p_data[out_idx].mass);
        std::swap(p_data[in_idx].energy, p_data[out_idx].energy);
    }
    else
    {
        std::swap(p_data[base_idx].position, p_data[out_idx].position);
        std::swap(p_data[base_idx].velocity, p_data[out_idx].velocity);
        std::swap(p_data[base_idx].mass, p_data[out_idx].mass);
        std::swap(p_data[base_idx].energy, p_data[out_idx].energy);
    }
    #endif
}

void particle_dynamics::finalise(cpu_mesh& mesh, cl::context& ctx, cl::managed_command_queue& mqueue, thin_intermediates_pool& pool, float timestep)
{

}
