#include "hydrodynamics.hpp"

eularian_hydrodynamics::eularian_hydrodynamics(cl::context& ctx) : hydro_st(ctx), vars(ctx){}

void eularian_hydrodynamics::grab_resources(matter_initial_vars _vars)
{
    vars = _vars;
}

std::vector<buffer_descriptor> eularian_hydrodynamics::get_buffers()
{
    std::vector<buffer_descriptor> buffers;

    buffers.push_back({"Dp_star", "evolve_hydro_all", 0.25f, 0, 1});
    buffers.push_back({"De_star", "evolve_hydro_all", 0.25f, 0, 1});
    buffers.push_back({"DcS0", "evolve_hydro_all", 0.25f, 0, 1});
    buffers.push_back({"DcS1", "evolve_hydro_all", 0.25f, 0, 1});
    buffers.push_back({"DcS2", "evolve_hydro_all", 0.25f, 0, 1});

    if(use_colour)
    {
        buffers.push_back({"dRed", "evolve_advect", 0.25f, 0, 1});
        buffers.push_back({"dGreen", "evolve_advect", 0.25f, 0, 1});
        buffers.push_back({"dBlue", "evolve_advect", 0.25f, 0, 1});
    }

    return buffers;
}

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
        assert(i.buf.alloc_size == dim.x() * dim.y() * dim.z() * sizeof(cl_float));

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

    hydro_init.push_back(vars.superimposed_tov_phi);
    hydro_init.push_back(scale);
    hydro_init.push_back(clsize);

    cl_int cl_use_colour = use_colour;

    hydro_init.push_back(cl_use_colour);

    cqueue.exec("calculate_hydrodynamic_initial_conditions", hydro_init, {dim.x(), dim.y(), dim.z()}, {8, 8, 1});

    vars = matter_initial_vars(ctx);
}

void eularian_hydrodynamics::step(cpu_mesh& mesh, cl::context& ctx, cl::command_queue& cqueue, thin_intermediates_pool& pool, buffer_pack& pack, float timestep, int iteration, int max_iteration)
{
    buffer_set& in = pack.in;
    buffer_set& out = pack.out;
    buffer_set& base = pack.base;

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
            build.push_back(buf.buf);
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
            calc_intermediates.push_back(buf.buf);
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
            visco.push_back(buf.buf);
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
            evolve.push_back(buf.buf);
        }

        for(auto& buf : out.buffers)
        {
            evolve.push_back(buf.buf);
        }

        for(auto& buf : base.buffers)
        {
            evolve.push_back(buf.buf);
        }

        for(auto& buf : intermediates)
        {
            evolve.push_back(buf);
        }

        evolve.push_back(w_buf);

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

    if(use_colour)
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
                advect.push_back(buf.buf);
            }

            advect.push_back(w_buf);

            advect.push_back(buf_base);
            advect.push_back(buf_in);
            advect.push_back(buf_out);

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

template<typename T>
inline
T chi_to_e_m6phi(const T& chi)
{
    return pow(max(chi, 0.001f), (3.f/2.f));
}

template<typename T>
inline
T chi_to_e_6phi_unclamped(const T& chi)
{
    using namespace std;

    return pow(1/(max(chi, 0.f)), (3.f/2.f));
}

template<typename T>
inline
T chi_to_e_m6phi_unclamped(const T& chi)
{
    using namespace std;

    return pow(max(chi, 0.f), (3.f/2.f));
}

#define DIVISION_TOL 0.00001f

///https://arxiv.org/pdf/gr-qc/0209102.pdf (29)
///constant_1 is chi * icYij * cSi cSj
template<typename T>
inline
T w_next_interior(const T& w_in, const T& p_star, const T& chi, const T& constant_1, float gamma, const T& e_star)
{
    ///p*^(2-G) * (w e6phi)^G-1 is equivalent to the divisor

    T em6_phi_G = pow(chi_to_e_m6phi_unclamped(chi), gamma - 1);

    T geg = gamma * pow(e_star, gamma);

    T divisor = pow(w_in, gamma - 1);

    ///so: Limits
    ///when w -> 0, w = 0
    ///when p_star -> 0, w = 0
    ///when e_star -> 0.... I think pstar and w have to tend to 0?

    ///So! I think this equation has a regular fomulation. The non finite quantities are em6_phi_G which tends to 0, geg which can be zero
    ///pstar doesn't actually matter here, though theoretically it might
    //T non_regular_interior = divisor / (divisor + em6_phi_G * geg * pow(p_star, gamma - 2));

    ///I'm not sure this equation tends to 0, but constant_1 tends to 0 because Si = p* h uk
    T non_regular_interior = dual_types::divide_with_limit(divisor, divisor + em6_phi_G * geg * pow(p_star, gamma - 2), T{0.f}, DIVISION_TOL);

    return sqrt(p_star * p_star + constant_1 * pow(non_regular_interior, 2));
    //return sqrt(p_star * p_star + constant_1 * pow(1 + em6_phi_G * geg * pow(p_star, gamma - 2) / divisor, -2));
}

float w_next_interior_nonregular(float w_in, float p_star, float chi, float constant_1, float gamma, float e_star)
{
    float geg = gamma * pow(e_star, gamma);

    float pstarwe6phipstar = p_star * pow(w_in * chi_to_e_6phi_unclamped(chi)/p_star, gamma - 1);

    return sqrt(p_star * p_star + constant_1 * pow(1 + geg / pstarwe6phipstar, -2));
}

template<typename T>
inline
T w_next(const T& w_in, const T& p_star, const T& chi, const inverse_metric<T, 3, 3>& icY, const tensor<T, 3>& cS, float gamma, const T& e_star)
{
    T constant_1 = 0;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            constant_1 += chi * icY.idx(i, j) * cS.idx(i) * cS.idx(j);
        }
    }

    return w_next_interior(w_in, p_star, chi, constant_1, gamma, e_star);
}

tensor<value, 3> calculate_dPhi(const value& chi, const tensor<value, 3>& dchi)
{
    return -dchi/(4 * max(chi, 0.001f));
}

inline
value matter_X_1(const value& X)
{
    return max(X, 0.00f);

    value LX = clamp(X, value{0.f}, value{1.f});

    float cutoff_X = 0.45f;
    float value_at_min = 0.35f;

    ///when this is 1, X == absolute min
    ///so eg if X == 0, we get 1
    ///if X == cutoff, we get 0
    value X_frac_to_absolute_min = (cutoff_X - LX) / cutoff_X;

    value modified_X_value = X_frac_to_absolute_min * (value_at_min - cutoff_X) + cutoff_X;

    return if_v(X >= cutoff_X,
         X,
         modified_X_value);

    /*float min_X = 0.4f;
    float max_X = 0.2f;

    value extra = max(X, 1.f) - min_X;

    value interpolated = (extra / (1.f - min_X)) * (max_X - min_X) + min_X;

    value interp = dual_types::if_v(extra > 0, interpolated, X);

    return interp;*/
}

inline
value matter_X_2(const value& X)
{
    return max(X, 0.00f);
}

inline
value calculate_h_with_gamma_eos(const value& eps)
{
    float Gamma = 2;

    return 1 + Gamma * eps;
}

//#define USE_MATTER

///https://arxiv.org/pdf/0812.0641.pdf just before 23
///X = e-4phi
struct matter
{
    value p_star;
    value e_star;
    tensor<value, 3> cS;

    float Gamma = 2;

    matter(equation_context& ctx)
    {
        p_star = bidx(ctx, "Dp_star", ctx.uses_linear, false);
        e_star = bidx(ctx, "De_star", ctx.uses_linear, false);

        for(int i=0; i < 3; i++)
        {
            cS.idx(i) = bidx(ctx, "DcS" + std::to_string(i), ctx.uses_linear, false);
        }

        p_star = max(p_star, 0.f);
        e_star = max(e_star, 0.f);
    }

    /*value calculate_W(const inverse_metric<value, 3, 3>& icY, const value& chi)
    {
        value W = 0.5f;

        int iterations = 5;

        for(int i=0; i < iterations; i++)
        {
            W = w_next(W, p_star, chi, icY, cS, Gamma, e_star);
        }

        return W;
    }*/

    ///??? comes from initial conditions
    value p_star_max()
    {
        return 1;
    }

    value p_star_is_degenerate()
    {
        return p_star < 1e-5f * p_star_max();
    }

    value p_star_below_e_star_threshold()
    {
        float e_factor = 1e-4f;

        return p_star < e_factor * p_star_max();
    }

    value e_star_clamped()
    {
        return min(e_star, 10 * p_star);
    }

    value calculate_p0(const value& chi, const value& W)
    {
        return divide_with_limit(chi_to_e_m6phi(chi) * p_star * p_star, W, 0.f, DIVISION_TOL);
    }

    value calculate_eps(const value& chi, const value& W)
    {
        value e_m6phi = chi_to_e_m6phi(chi);

        /*value p0 = calculate_p0(chi, W);

        value au0 = divide_with_limit(W, p_star, 0.f);

        value lhs = divide_with_limit(pow(divide_with_limit(e_star * e_m6phi, au0, 0.f), Gamma), p0, 0.f);

        return lhs;*/

        return pow(divide_with_limit(e_m6phi, W, 0.f, DIVISION_TOL), Gamma - 1) * pow(e_star, Gamma) * pow(p_star, Gamma - 2);
    }

    /*value gamma_eos(const value& p0, const value& eps)
    {
        return (Gamma - 1) * p0 * eps;
    }*/

    value calculate_p0e(const value& chi, const value& W)
    {
        value iv_au0 = divide_with_limit(p_star, W, 0.f, DIVISION_TOL);

        value e_m6phi = chi_to_e_m6phi_unclamped(chi);

        return pow(max(e_star * e_m6phi * iv_au0, 0.f), Gamma);
    }

    value gamma_eos_from_e_star(const value& chi, const value& W)
    {
        value p0e = calculate_p0e(chi, W);

        return p0e * (Gamma - 1);
    }

    value calculate_h_with_gamma_eos(const value& chi, const value& W)
    {
        return ::calculate_h_with_gamma_eos(calculate_eps(chi, W));
    }

    ///i know these to be wrong
    ///https://arxiv.org/pdf/2203.05149.pdf
    /*tensor<value, 3> get_u_lower(const value& chi, const value& W)
    {
        tensor<value, 3> ret;

        for(int i=0; i < 3; i++)
        {
            ret.idx(i) = divide_with_limit(cS.idx(i), p_star * calculate_h_with_gamma_eos(chi, W), 0.f, DIVISION_TOL);
        }

        for(int i=0; i < 3; i++)
        {
            //ret.idx(i) = dual_types::clamp(ret.idx(i), value{-0.1f}, value{0.1f});
        }

        return ret;
    }

    tensor<value, 3> get_u_upper(const inverse_metric<value, 3, 3>& icY, const value& chi, const value& W)
    {
        tensor<value, 3> ui_lower = get_u_lower(chi, W);

        return raise_index(ui_lower, chi * icY, 0);
    }*/

    #if 0
    tensor<value, 3> get_v_upper(const inverse_metric<value, 3, 3>& icY, const value& gA, const value& chi, const value& W)
    {
        value u0 = divide_with_limit(W, (p_star * gA), 0);

        tensor<value, 3> u_up = get_u_upper(icY, chi, W);

        tensor<value, 3> clamped;

        for(int i=0; i < 3; i++)
        {
            clamped.idx(i) = divide_with_limit(u_up.idx(i), u0, 0.f);

            ///todo: tensor if_v
            //clamped.idx(i) = dual_types::if_v(p_star_is_degenerate(), 0.f, u_up.idx(i) / u0);
        }

        return clamped;
    }
    #endif // 0

    ///https://arxiv.org/pdf/gr-qc/9908027.pdf 2.12
    ///except sk = p* h uj, and uhatj = h uj
    ///and w = p* a u0 not a u0
    tensor<value, 3> get_v_upper(const inverse_metric<value, 3, 3>& icY, const value& gA, const tensor<value, 3>& gB, const value& chi, const value& W)
    {
        //#define V_UPPER_PRIMARY
        #ifdef V_UPPER_PRIMARY
        value u0 = divide_with_limit(W, (p_star * gA), 0, DIVISION_TOL);

        tensor<value, 3> u_up = get_u_upper(icY, chi, W);

        tensor<value, 3> clamped;

        for(int i=0; i < 3; i++)
        {
            clamped.idx(i) = divide_with_limit(u_up.idx(i), u0, 0.f, DIVISION_TOL);

            ///todo: tensor if_v
            //clamped.idx(i) = dual_types::if_v(p_star_is_degenerate(), 0.f, u_up.idx(i) / u0);
        }

        return clamped;
        #endif // V_UPPER_PRIMARY

        #define V_UPPER_ALT
        #ifdef V_UPPER_ALT
        value h = calculate_h_with_gamma_eos(chi, W);

        tensor<value, 3> ret = -gB;

        for(int i=0; i < 3; i++)
        {
            value sum = 0;

            for(int j=0; j < 3; j++)
            {
                sum += divide_with_limit(gA * icY.idx(i, j) * cS.idx(j) * chi, W * h, 0.f, DIVISION_TOL);
            }

            ret.idx(i) += sum;
        }

        return ret;
        #endif // V_UPPER_ALT
    }

    tensor<value, 3> p_star_vi(const inverse_metric<value, 3, 3>& icY, const value& gA, const tensor<value, 3>& gB, const value& chi, const value& W)
    {
        tensor<value, 3> v_upper = get_v_upper(icY, gA, gB, chi, W);

        return p_star * v_upper;
    }

    tensor<value, 3> e_star_vi(const inverse_metric<value, 3, 3>& icY, const value& gA, const tensor<value, 3>& gB, const value& chi, const value& W)
    {
        tensor<value, 3> v_upper = get_v_upper(icY, gA, gB, chi, W);

        return e_star * v_upper;
    }

    tensor<value, 3, 3> cSk_vi(const inverse_metric<value, 3, 3>& icY, const value& gA, const tensor<value, 3>& gB, const value& chi, const value& W)
    {
        tensor<value, 3> v_upper = get_v_upper(icY, gA, gB, chi, W);

        tensor<value, 3, 3> cSk_vi;

        for(int k=0; k < 3; k++)
        {
            for(int i=0; i < 3; i++)
            {
                cSk_vi.idx(k, i) = cS.idx(k) * v_upper.idx(i);
            }
        }

        return cSk_vi;
    }

    value calculate_adm_p(const value& chi, const value& W)
    {
        //return {};

        value h = calculate_h_with_gamma_eos(chi, W);
        value em6phi = chi_to_e_m6phi_unclamped(chi);

        value p0 = calculate_p0(chi, W);
        value eps = calculate_eps(chi, W);

        return h * W * em6phi - gamma_eos_from_e_star(chi, W);
    }

    tensor<value, 3> calculate_adm_Si(const value& chi)
    {
        value em6phi = chi_to_e_m6phi_unclamped(chi);

        return cS * em6phi;
    }

    ///the reason to calculate X_Sij is that its regular in terms of chi
    tensor<value, 3, 3> calculate_adm_X_Sij(const value& chi, const value& W, const metric<value, 3, 3>& cY)
    {
        value em6phi = chi_to_e_m6phi_unclamped(chi);
        value h = calculate_h_with_gamma_eos(chi, W);

        tensor<value, 3, 3> Sij;

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                Sij.idx(i, j) = divide_with_limit(em6phi, W * h, 0.f, DIVISION_TOL) * cS.idx(i) * cS.idx(j);
            }
        }

        ///this gamma eos is the specific problem
        tensor<value, 3, 3> X_P_Yij = gamma_eos_from_e_star(chi, W) * cY.to_tensor();
        //tensor<value, 3, 3> X_P_Yij = gamma_eos(p0, eps) * cY.to_tensor();

        return Sij * chi + X_P_Yij;
    }

    value calculate_adm_S(const metric<value, 3, 3>& cY, const inverse_metric<value, 3, 3>& icY, const value& chi, const value& W)
    {
        ///so. Raise Sij with iYij, which is X * icY
        ///now I'm actually raising X * Sij which means....... i can use icY?
        ///because iYij * Sjk = X * icYij * Sjk, and icYij * X * Sjk = X * icYij * Sjk
        tensor<value, 3, 3> XSij = calculate_adm_X_Sij(chi, W, cY);

        tensor<value, 3, 3> raised = raise_index(XSij, icY, 0);

        value sum = 0;

        for(int i=0; i < 3; i++)
        {
            sum += raised.idx(i, i);
        }

        return sum;
    }

    value calculate_PQvis(equation_context& ctx, const value& gA, const tensor<value, 3>& gB, const inverse_metric<value, 3, 3>& icY, const value& chi, const value& W)
    {
        #define QUADRATIC_VISCOSITY
        #ifndef QUADRATIC_VISCOSITY
        return 0;
        #endif // QUADRATIC_VISCOSITY

        value e_m6phi = chi_to_e_m6phi_unclamped(chi);
        value e_6phi = chi_to_e_6phi(chi);

        tensor<value, 3> vk = get_v_upper(icY, gA, gB, chi, W);

        value scale = "scale";

        value dkvk = 0;

        for(int k=0; k < 3; k++)
        {
            dkvk += 2 * diff1(ctx, vk.idx(k), k);
        }

        value littledv = dkvk * scale;

        value A = divide_with_limit(pow(e_star, Gamma) * pow(p_star, Gamma - 1) * pow(e_m6phi, Gamma - 1), pow(W, Gamma - 1), 0.f);
        //value A = divide_with_limit(pow(e_star, Gamma) * pow(p_star, Gamma - 1), pow(W * e_6phi, Gamma - 1), 0.f);

        //ctx.add("DBG_A", A);

        ///[0.1, 1.0}
        value CQvis = 1.f;

        value PQvis = if_v(littledv < 0, CQvis * A * pow(littledv, 2), value{0.f});

        return PQvis;
    }

    ///I suspect we shouldn't quadratic viscosity near the event horizon, there's an infinite term to_diff
    value estar_vi_rhs(equation_context& ctx, const value& gA, const tensor<value, 3>& gB, const inverse_metric<value, 3, 3>& icY, const value& chi, const value& W)
    {
        #ifndef QUADRATIC_VISCOSITY
        return 0;
        #endif // QUADRATIC_VISCOSITY

        value e_m6phi = chi_to_e_m6phi_unclamped(chi);

        value PQvis = calculate_PQvis(ctx, gA, gB, icY, chi, W);

        tensor<value, 3> vk = get_v_upper(icY, gA, gB, chi, W);

        value sum_interior_rhs = 0;

        for(int k=0; k < 3; k++)
        {
            value to_diff = divide_with_limit(W * vk.idx(k), p_star * e_m6phi, 0.f);

            sum_interior_rhs += diff1(ctx, to_diff, k);
        }

        value p0e = calculate_p0e(chi, W);

        value degenerate = divide_with_limit(value{1}, pow(p0e, 1 - 1/Gamma), 0.f);

        /*ctx.add("DBG_IRHS", sum_interior_rhs);

        ctx.add("DBG_p0eps", p0 * eps);

        ctx.add("DINTERIOR", p0 * eps);

        ctx.add("DP2", PQvis / Gamma);

        ctx.add("DP1", degenerate);*/

        return -degenerate * (PQvis / Gamma) * sum_interior_rhs;
    }

    tensor<value, 3> cSkvi_rhs(equation_context& ctx, const inverse_metric<value, 3, 3>& icY, const value& gA, const tensor<value, 3>& gB, const value& chi, const tensor<value, 3>& dchi, const value& P, const value& W)
    {
        tensor<value, 3> dX = dchi;

        //value PQvis = calculate_PQvis(ctx, gA, gB, icY, chi, W);

        //ctx.pin(PQvis);

        value h = calculate_h_with_gamma_eos(chi, W);

        tensor<value, 3> ret;

        for(int k=0; k < 3; k++)
        {
            ret.idx(k) += -gA * divide_with_limit(value{1}, chi_to_e_m6phi(chi), 0.f, DIVISION_TOL) * diff1(ctx, P, k);

            ret.idx(k) += -W * h * diff1(ctx, gA, k);

            {
                value sum = 0;

                for(int j=0; j < 3; j++)
                {
                    sum += -cS.idx(j) * diff1(ctx, gB.idx(j), k);
                }

                ret.idx(k) += sum;
            }

            {
                value sum = 0;

                for(int i=0; i < 3; i++)
                {
                    for(int j=0; j < 3; j++)
                    {
                        sum += divide_with_limit(gA * chi * cS.idx(i) * cS.idx(j), (2 * W * h), 0.f, DIVISION_TOL) * diff1(ctx, icY.idx(i, j), k);
                    }
                }

                ret.idx(k) += sum;
            }

            ret.idx(k) += -divide_with_limit((2 * gA * h * (W * W - p_star * p_star)), W, 0.f, DIVISION_TOL) * calculate_dPhi(chi, dX).idx(k);
        }

        return ret;
    }
};

inline
value get_cacheable_W(equation_context& ctx, standard_arguments& args, matter& matt)
{
    inverse_metric<value, 3, 3> icY = args.cY.invert();

    value W = 0.5f;
    int iterations = 5;

    for(int i=0; i < iterations; i++)
    {
        ctx.pin(W);

        W = w_next(W, matt.p_star, matter_X_2(args.get_X()), icY, matt.cS, matt.Gamma, matt.e_star);
    }

    return W;
}


value eularian_matter::calculate_adm_S(equation_context& ctx, standard_arguments& args) const
{
    matter matt(ctx);

    value W = get_cacheable_W(ctx, args, matt);

    ctx.pin(W);

    return matt.calculate_adm_S(args.cY, args.cY.invert(), args.get_X(), W);
}

value eularian_matter::calculate_adm_p(equation_context& ctx, standard_arguments& args) const
{
    matter matt(ctx);

    value W = get_cacheable_W(ctx, args, matt);

    ctx.pin(W);

    return matt.calculate_adm_p(args.get_X(), W);
}

tensor<value, 3, 3> eularian_matter::calculate_adm_X_Sij(equation_context& ctx, standard_arguments& args) const
{
    matter matt(ctx);

    value W = get_cacheable_W(ctx, args, matt);

    ctx.pin(W);

    return matt.calculate_adm_X_Sij(args.get_X(), W, args.cY);
}

tensor<value, 3> eularian_matter::calculate_adm_Si(equation_context& ctx, standard_arguments& args) const
{
    matter matt(ctx);

    return matt.calculate_adm_Si(args.get_X());
}

namespace hydrodynamics
{
    void build_intermediate_variables_derivatives(equation_context& ctx)
    {
        standard_arguments args(ctx);
        matter matt(ctx);

        ctx.is_derivative_free = true;

        value sW = get_cacheable_W(ctx, args, matt);

        ctx.pin(sW);

        value pressure = matt.gamma_eos_from_e_star(matter_X_2(args.get_X()), sW);

        ctx.add("init_pressure", pressure);
        ctx.add("init_W", sW);
    }

    void build_artificial_viscosity(equation_context& ctx)
    {
        standard_arguments args(ctx);
        matter matt(ctx);

        value sW = bidx(ctx, "hW", ctx.uses_linear, false);

        inverse_metric<value, 3, 3> icY = args.cY.invert();

        value PQvis = matt.calculate_PQvis(ctx, args.gA, args.gB, icY, matter_X_2(args.get_X()), sW);

        ctx.add("init_artificial_viscosity", PQvis);
    }

    void build_equations(equation_context& ctx)
    {
        standard_arguments args(ctx);
        matter matt(ctx);

        inverse_metric<value, 3, 3> icY = args.cY.invert();

        value sW = bidx(ctx, "hW", ctx.uses_linear, false);

        tensor<value, 3, 3> cSk_vi = matt.cSk_vi(icY, args.gA, args.gB, matter_X_2(args.get_X()), sW);

        tensor<value, 3> p_star_vi = matt.p_star_vi(icY, args.gA, args.gB, matter_X_2(args.get_X()), sW);
        tensor<value, 3> e_star_vi = matt.e_star_vi(icY, args.gA, args.gB, matter_X_2(args.get_X()), sW);

        value P = bidx(ctx, "pressure", ctx.uses_linear, false);

        value lhs_dtp_star = 0;

        for(int i=0; i < 3; i++)
        {
            lhs_dtp_star += diff1(ctx, p_star_vi.idx(i), i);
        }

        value lhs_dte_star = 0;

        for(int i=0; i < 3; i++)
        {
            lhs_dte_star += diff1(ctx, e_star_vi.idx(i), i);
        }

        //#define IMMEDIATE_UPDATE
        #ifdef IMMEDIATE_UPDATE
        matter matt2 = matt;
        matt2.p_star = "fin_p_star";
        #else
        matter& matt2 = matt;
        #endif

        //value sW = get_cacheable_W(ctx, args, matt2);

        value rhs_dte_star = matt2.estar_vi_rhs(ctx, args.gA, args.gB, icY, matter_X_1(args.get_X()), sW);

        /*ctx.add("DBG_RHS_DTESTAR", rhs_dte_star);

        ctx.add("DBG_LHS_DTESTAR", lhs_dte_star);
        ctx.add("DBG_ESTARVI0", e_star_vi.idx(0));
        ctx.add("DBG_ESTARVI1", e_star_vi.idx(1));
        ctx.add("DBG_ESTARVI2", e_star_vi.idx(2));*/

        tensor<value, 3> lhs_dtSk;

        for(int k=0; k < 3; k++)
        {
            value sum = 0;

            for(int i=0; i < 3; i++)
            {
                sum += diff1(ctx, cSk_vi.idx(k, i), i);
            }

            lhs_dtSk.idx(k) = sum;
        }

        tensor<value, 3> rhs_dtSk = matt2.cSkvi_rhs(ctx, icY, args.gA, args.gB, matter_X_2(args.get_X()), args.get_dX(), P, sW);

        value dtp_star = -lhs_dtp_star;
        value dte_star = -lhs_dte_star + rhs_dte_star;

        ctx.add("lhs_dtsk0", -lhs_dtSk.idx(0));
        ctx.add("rhs_dtsk0", rhs_dtSk.idx(0));

        tensor<value, 3> dtSk = -lhs_dtSk + rhs_dtSk;

        ctx.add("init_dtp_star", dtp_star);
        ctx.add("init_dte_star", dte_star);

        for(int i=0; i < 3; i++)
        {
            ctx.add("init_dtSk" + std::to_string(i), dtSk.idx(i));
        }

        //ctx.add("e_star_p_limit", matt.p_star_below_e_star_threshold());
        //ctx.add("e_star_p_value", matt.e_star_clamped());
        ctx.add("p_star_max", matt.p_star_max());
    }

    void build_advection(equation_context& ctx)
    {
        standard_arguments args(ctx);
        matter matt(ctx);

        inverse_metric<value, 3, 3> icY = args.cY.invert();

        value sW = bidx(ctx, "hW", ctx.uses_linear, false);

        tensor<value, 3> vi_upper = matt.get_v_upper(icY, args.gA, args.gB, matter_X_2(args.get_X()), sW);

        value quantity = max(bidx(ctx, "quantity_in", ctx.uses_linear, false), 0.f);

        value fin = 0;

        for(int i=0; i < 3; i++)
        {
            value to_diff = quantity * vi_upper.idx(i);

            fin += diff1(ctx, to_diff, i);
        }

        /*value fin = 0;

        for(int i=0; i < 3; i++)
        {
            fin += vi_upper.idx(i) * diff1(ctx, quantity, i);
        }*/

        ctx.add("HYDRO_ADVECT", -fin);
    }
}

void test_w()
{
    {
        float w = 0.5f;
        float w1 = 0.5f;

        for(int i=0; i < 50; i++)
        {
            w =              w_next_interior(w, 0.234f, 1.12f, 0.25f, 2.f, 0.1f);
            w1 = w_next_interior_nonregular(w1, 0.234f, 1.12f, 0.25f, 2.f, 0.1f);
        }

        assert(approx_equal(w, w1));

        printf("reg check %f %f\n", w, w1);
    }

    {
        float w = 0.5f;

        for(int i=0; i < 50; i++)
        {
            ///by the property that w = p*au0, perhaps set to 0 if p* < crit
            w = w_next_interior(w, 0.f, 0.f, 0.f, 2, 0.f);

            assert(isfinite(w));
        }
    }

    /*{
        for(float p1 = 0; p1 <= 1; p1 += 0.01f)
        {
            for(float p2 = 0; p2 <= 1; p2 += 0.01f)
            {
                for(float p3 = 0; p3 <= 1; p3 += 0.01f)
                {
                    for(float p4 = 0; p4 <= 1; p4 += 0.01f)
                    {
                        float w = 0.5f;

                        for(int i=0; i < 6; i++)
                        {
                            w = w_next_interior(w, p1, p2, p3, 2, p4);
                            assert(isfinite(w));
                        }
                    }
                }
            }
        }
    }*/
}
