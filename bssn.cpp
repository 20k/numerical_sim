#include "bssn.hpp"
#include "single_source.hpp"
#include "bitflags.cl"
#include "spherical_decomposition.hpp"
#include "util.hpp"
#include <thread>
#include <toolkit/clock.hpp>

using single_source::named_buffer;
using single_source::named_literal;

value matter_meta_interop::calculate_adm_S(equation_context& ctx, standard_arguments& bssn_args) const
{
    value ret;

    for(auto& i : sub_interop)
    {
        ret += i->calculate_adm_S(ctx, bssn_args);
    }

    return ret;
}

value matter_meta_interop::calculate_adm_p(equation_context& ctx, standard_arguments& bssn_args) const
{
    value ret;

    for(auto& i : sub_interop)
    {
        ret += i->calculate_adm_p(ctx, bssn_args);
    }

    return ret;
}

tensor<value, 3, 3> matter_meta_interop::calculate_adm_X_Sij(equation_context& ctx, standard_arguments& bssn_args) const
{
    tensor<value, 3, 3> ret;

    for(auto& i : sub_interop)
    {
        ret += i->calculate_adm_X_Sij(ctx, bssn_args);
    }

    return ret;
}

tensor<value, 3> matter_meta_interop::calculate_adm_Si(equation_context& ctx, standard_arguments& bssn_args) const
{
    tensor<value, 3> ret;

    for(auto& i : sub_interop)
    {
        ret += i->calculate_adm_Si(ctx, bssn_args);
    }

    return ret;
}

std::array<value_i, 4> setup(equation_context& ctx, buffer<tensor<value_us, 4>> points, value_i point_count, tensor<value_i, 4> dim, buffer<value_us> order_ptr);

float get_backwards_euler_relax_parameter()
{
    return 1.f;
}

///in, base, dvalue, dt
value backwards_euler_relax(const value& ynp1k, const value& yn, const value& f_ynp1k, const value& dt)
{
    float relax = get_backwards_euler_relax_parameter();

    return ynp1k * (1 - relax) + (yn + f_ynp1k * dt) * relax;
}

void calculate_christoffel_symbol(single_source::argument_generator& arg_gen, equation_context& ctx, base_bssn_args bssn_args)
{
    arg_gen.add(bssn_args.buffers);

    auto points = arg_gen.add<buffer<tensor<value_us, 4>>>();
    auto point_count = arg_gen.add<literal<value_i>>();
    auto order_ptr = arg_gen.add<named_buffer<value_us, "order_ptr">>();

    arg_gen.add<named_literal<value, "scale">>();
    auto dim = arg_gen.add<named_literal<tensor<value_i, 4>, "dim">>();

    named_buffer<value_mut, "cGi0"> cGi0;
    named_buffer<value_mut, "cGi1"> cGi1;
    named_buffer<value_mut, "cGi2"> cGi2;

    auto [ix, iy, iz, index] = setup(ctx, points, point_count, dim, order_ptr);

    standard_arguments args(ctx);

    auto icY = args.cY.invert();

    tensor<value, 3, 3, 3> christoff2 = christoffel_symbols_2(ctx, args.cY, icY);

    tensor<value, 3> cGi;

    for(int i=0; i < 3; i++)
    {
        value sum = 0;

        for(int m=0; m < 3; m++)
        {
            for(int n=0; n < 3; n++)
            {
                sum += icY[m, n] * christoff2[i, m, n];
            }
        }

        cGi[i] = sum;
    }

    ctx.exec(assign(cGi0[index], cGi[0]));
    ctx.exec(assign(cGi1[index], cGi[1]));
    ctx.exec(assign(cGi2[index], cGi[2]));
}

void bssn::init(equation_context& ctx, const metric<value, 3, 3>& Yij, const tensor<value, 3, 3>& Aij, const value& gA)
{
    vec2i linear_indices[6] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};

    ///https://arxiv.org/pdf/gr-qc/0206072.pdf see 10
    ///https://arxiv.org/pdf/gr-qc/9810065.pdf, 11
    ///phi

    value Y = Yij.det();
    //value conformal_factor = (1/12.f) * log(Y);
    //ctx.pin(conformal_factor);

    value gB0 = 0;
    value gB1 = 0;
    value gB2 = 0;

    tensor<value, 3> cGi;
    value K = 0;

    ///https://arxiv.org/pdf/gr-qc/0206072.pdf (58)
    //value X = exp(-4 * conformal_factor);

    #ifndef USE_W
    ///X also eq (conformal + u)^-4, aka psi^-4
    value X = pow(Y, -1.f/3.f);

    tensor<value, 3, 3> cAij = X * Aij;
    metric<value, 3, 3> cYij = X * Yij;
    #else
    value X = pow(Y, -1.f/6.f);

    tensor<value, 3, 3> cAij = X * X * Aij;
    metric<value, 3, 3> cYij = X * X * Yij;
    #endif

    ///need to do the same thing for Aij. Think the extrinsic curvature near the centre is screwed
    #define FORCE_FLAT
    #ifdef FORCE_FLAT
    cYij = get_flat_metric<value, 3>();
    #endif // FORCE_FLAT

    for(int i=0; i < 6; i++)
    {
        vec2i index = linear_indices[i];

        std::string y_name = "init_cY" + std::to_string(i);

        value val = cYij.idx(index.x(), index.y());

        if(i == 0)
            val = val - CY0_ADD;

        if(i == 3)
            val = val - CY3_ADD;

        if(i == 5)
            val = val - CY5_ADD;

        ctx.add(y_name, val);
    }

    for(int i=0; i < 6; i++)
    {
        ctx.add("init_cA" + std::to_string(i), cAij.idx(linear_indices[i].x(), linear_indices[i].y()));
    }

    ctx.add("init_cGi0", cGi.idx(0));
    ctx.add("init_cGi1", cGi.idx(1));
    ctx.add("init_cGi2", cGi.idx(2));

    ctx.add("init_K", K);
    ctx.add("init_X", X - X_ADD);

    ctx.add("init_gA", gA - GA_ADD);
    ctx.add("init_gB0", gB0);
    ctx.add("init_gB1", gB1);
    ctx.add("init_gB2", gB2);

    //#define USE_GBB
    #ifdef USE_GBB
    value gBB0 = 0;
    value gBB1 = 0;
    value gBB2 = 0;

    ctx.add("init_gBB0", gBB0);
    ctx.add("init_gBB1", gBB1);
    ctx.add("init_gBB2", gBB2);
    #endif // USE_GBB

    #ifdef USE_W
    ctx.add("X_IS_ACTUALLY_W", 1);
    #endif

    #ifdef DAMP_C
    ctx.add("DAMPED_CONSTRAINTS", 1);
    #endif // DAMP_C

    standard_arguments args(ctx);

    ctx.add("GET_X", args.get_X());
}

void bssn::init(equation_context& ctx, const metric<value, 4, 4>& Guv, const tensor<value, 4, 4, 4>& dGuv)
{
    metric<value, 3, 3> Yij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            Yij[i, j] = Guv[i+1, j+1];
        }
    }

    tensor<value, 3, 3, 3> Yij_derivatives;

    for(int k=0; k < 3; k++)
    {
        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                Yij_derivatives[k, i, j] = dGuv[k+1, i+1, j+1];
            }
        }
    }

    tensor<value, 3, 3, 3> Yij_christoffel = christoffel_symbols_2(Yij.invert(), Yij_derivatives);

    ctx.pin(Yij_christoffel);

    auto covariant_derivative_low_vec_e = [&](const tensor<value, 3>& lo, const tensor<value, 3, 3>& dlo)
    {
        ///DcXa
        tensor<value, 3, 3> ret;

        for(int a=0; a < 3; a++)
        {
            for(int c=0; c < 3; c++)
            {
                value sum = 0;

                for(int b=0; b < 3; b++)
                {
                    sum += Yij_christoffel[b, c, a] * lo[b];
                }

                ret[c, a] = dlo[c, a] - sum;
            }
        }

        return ret;
    };

    tensor<value, 3> gB_lower;
    tensor<value, 3, 3> dgB_lower;

    for(int i=0; i < 3; i++)
    {
        gB_lower[i] = Guv[0, i+1];

        for(int k=0; k < 3; k++)
        {
            dgB_lower[k, i] = dGuv[k+1, 0, i+1];
        }
    }

    tensor<value, 3> gB = raise_index(gB_lower, Yij.invert(), 0);

    ctx.pin(gB);

    value gB_sum = sum_multiply(gB, gB_lower);

    ///g00 = nini - n^2
    ///g00 - nini = -n^2
    ///-g00 + nini = n^2
    ///n = sqrt(-g00 + nini)
    value gA = sqrt(-Guv[0, 0] + gB_sum);

    ///https://clas.ucdenver.edu/math-clinic/sites/default/files/attached-files/master_project_mach_.pdf 4-19a
    tensor<value, 3, 3> DigBj = covariant_derivative_low_vec_e(gB_lower, dgB_lower);

    ctx.pin(DigBj);

    tensor<value, 3, 3> Kij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            Kij[i, j] = (1/(2 * gA)) * (DigBj[i, j] + DigBj[j, i] - dGuv[0, i+1, j+1]);
        }
    }


    value X = pow(Yij.det(), -1/3.f);
    metric<value, 3, 3> cY = X * Yij;
    value K = trace(Kij, Yij.invert());

    inverse_metric<value, 3, 3> icY = cY.invert();

    ///Kij = (1/X) * (cAij + 1/3 Yij K)
    ///X Kij = cAij + 1/3 Yij K
    ///X Kij - 1/3 Yij K = cAij

    tensor<value, 3, 3> cA = X * Kij - (1.f/3.f) * cY.to_tensor() * K;

    tensor<value, 3> cGi;

    vec2i linear_indices[6] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};

    for(int i=0; i < 6; i++)
    {
        vec2i index = linear_indices[i];

        std::string y_name = "init_cY" + std::to_string(i);

        value val = cY.idx(index.x(), index.y());

        if(i == 0)
            val = val - CY0_ADD;

        if(i == 3)
            val = val - CY3_ADD;

        if(i == 5)
            val = val - CY5_ADD;

        ctx.add(y_name, val);
    }

    for(int i=0; i < 6; i++)
    {
        ctx.add("init_cA" + std::to_string(i), cA.idx(linear_indices[i].x(), linear_indices[i].y()));
    }

    ctx.add("init_cGi0", cGi.idx(0));
    ctx.add("init_cGi1", cGi.idx(1));
    ctx.add("init_cGi2", cGi.idx(2));

    ctx.add("init_K", K);

    #ifdef USE_W
    ctx.add("init_X", sqrt(X) - X_ADD);
    #else
    ctx.add("init_X", X - X_ADD);
    #endif

    ctx.add("init_gA", gA - GA_ADD);
    ctx.add("init_gB0", gB[0]);
    ctx.add("init_gB1", gB[1]);
    ctx.add("init_gB2", gB[2]);

    #ifdef USE_W
    ctx.add("X_IS_ACTUALLY_W", 1);
    #endif

    #ifdef DAMP_C
    ctx.add("DAMPED_CONSTRAINTS", 1);
    #endif // DAMP_C

}

///returns DcTab
///my covariant derivative functions are an absolute mess
tensor<value, 3, 3, 3> covariant_derivative_low_tensor(equation_context& ctx, const tensor<value, 3, 3>& mT, const metric<value, 3, 3>& met, const inverse_metric<value, 3, 3>& inverse)
{
    tensor<value, 3, 3, 3> christoff2 = christoffel_symbols_2(ctx, met, inverse);

    tensor<value, 3, 3, 3> ret;

    for(int a=0; a < 3; a++)
    {
        for(int b=0; b < 3; b++)
        {
            for(int c=0; c < 3; c++)
            {
                value sum = 0;

                for(int d=0; d < 3; d++)
                {
                    sum += -christoff2.idx(d, c, a) * mT.idx(d, b) - christoff2.idx(d, c, b) * mT.idx(a, d);
                }

                ret.idx(c, a, b) = diff1(ctx, mT.idx(a, b), c) + sum;
            }
        }
    }

    return ret;
}

///https://iopscience.iop.org/article/10.1088/1361-6382/ac7e16/pdf 2.8 would be a nicer formulation
tensor<value, 3> bssn::calculate_momentum_constraint(matter_interop& interop, equation_context& ctx, bool use_matter)
{
    standard_arguments args(ctx);

    inverse_metric<value, 3, 3> icY = args.cY.invert();
    auto unpinned_icY = args.unpinned_cY.invert();
    //ctx.pin(icY);

    value X_clamped = max(args.get_X(), 0.001f);

    tensor<value, 3> Mi;

    #if 0
    tensor<value, 3, 3, 3> dmni = covariant_derivative_low_tensor(ctx, args.cA, args.cY, icY);

    tensor<value, 3, 3> mixed_cAij = raise_index(args.cA, icY, 0);

    tensor<value, 3> ji_lower = interop.calculate_adm_Si(ctx, args);

    tensor<value, 3> dX = args.get_dX();

    for(int i=0; i < 3; i++)
    {
        value s1 = 0;

        for(int m=0; m < 3; m++)
        {
            for(int n=0; n < 3; n++)
            {
                s1 += icY.idx(m, n) * dmni.idx(m, n, i);
            }
        }

        value s2 = -(2.f/3.f) * diff1(ctx, args.K, i);

        value s3 = 0;

        for(int m=0; m < 3; m++)
        {
            s3 += -(3.f/2.f) * mixed_cAij.idx(m, i) * dX.idx(m) / X_clamped;
        }

        /*Mi.idx(i) = dual_if(args.X <= 0.001f,
        []()
        {
            return 0.f;
        },
        [&]()
        {
            return s1 + s2 + s3;
        });*/

        Mi.idx(i) = s1 + s2 + s3;

        if(use_matter)
        {
            Mi.idx(i) += -8 * M_PI * ji_lower.idx(i);
        }
    }
    #endif

    ///https://arxiv.org/pdf/1205.5111v1.pdf (54)
    value X = args.get_X();
    tensor<value, 3> dX = args.get_dX();

    tensor<value, 3, 3> aij_raised = raise_index(args.unpinned_cA, unpinned_icY, 1);

    tensor<value, 3> dPhi = -dX / (4 * max(X, 0.0001f));

    for(int i=0; i < 3; i++)
    {
        value s1 = 0;

        for(int j=0; j < 3; j++)
        {
            s1 += diff1(ctx, aij_raised.idx(i, j), j);
        }

        value s2 = 0;

        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                s2 += -0.5f * icY.idx(j, k) * diff1(ctx, args.unpinned_cA.idx(j, k), i);
            }
        }

        value s3 = 0;

        for(int j=0; j < 3; j++)
        {
            s3 += 6 * dPhi.idx(j) * aij_raised.idx(i, j);
        }

        value p4 = -(2.f/3.f) * diff1(ctx, args.K, i);

        Mi.idx(i) = s1 + s2 + s3 + p4;

        if(use_matter)
        {
            tensor<value, 3> ji_lower = interop.calculate_adm_Si(ctx, args);

            Mi.idx(i) += -8 * (float)M_PI * ji_lower.idx(i);
        }
    }

    return Mi;
}

value bssn::calculate_hamiltonian_constraint(const matter_interop& interop, equation_context& ctx, bool use_matter)
{
    standard_arguments args(ctx);

    inverse_metric<value, 3, 3> icY = args.cY.invert();

    ctx.pin(icY);

    tensor<value, 3, 3, 3> christoff1 = christoffel_symbols_1(ctx, args.unpinned_cY);

    ctx.pin(christoff1);
    ctx.pin(args.christoff2);

    tensor<value, 3, 3> xgARij = calculate_xgARij(ctx, args, icY, christoff1, args.christoff2);

    tensor<value, 3, 3> Rij = xgARij / max(args.get_X() * args.gA, 0.0001f);

    value R = trace(Rij, args.iYij);

    tensor<value, 3, 3> aIJ = raise_both(args.cA, icY);

    value aij_aIJ;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            aij_aIJ += args.cA.idx(i, j) * aIJ.idx(i, j);
        }
    }

    value ret = R + (2.f/3.f) * args.K * args.K - aij_aIJ;

    if(use_matter)
    {
        ret += -16.f * (float)M_PI * interop.calculate_adm_p(ctx, args);
    }

    return ret;
}

value get_kc()
{
    #ifdef DAMP_C
    return 10.f;
    #else
    return 0.f;
    #endif
}

///https://arxiv.org/pdf/gr-qc/0401076.pdf
//#define DAMP_HAMILTONIAN

std::array<value_i, 4> setup(equation_context& ctx, buffer<tensor<value_us, 4>> points, value_i point_count, tensor<value_i, 4> dim, buffer<value_us> order_ptr)
{
    using namespace dual_types::implicit;

    value_i local_idx = declare(value_i{"get_global_id(0)"}, "lidx");

    if_e(local_idx >= point_count, [&]()
    {
        return_e();
    });

    value_i ix = declare(points[local_idx].x().convert<int>(), "ix");
    value_i iy = declare(points[local_idx].y().convert<int>(), "iy");
    value_i iz = declare(points[local_idx].z().convert<int>(), "iz");

    ///((k) * dim.x * dim.y + (j) * dim.x + (i))

    value_i index = declare(iz * dim.x() * dim.y() + iy * dim.x() + ix, "index");

    //ctx.exec("prefetch(&order_ptr[index], 1)");

    dual_types::side_effect(ctx, "prefetch(&cY0[index], 1)");
    dual_types::side_effect(ctx, "prefetch(&cY1[index], 1)");
    dual_types::side_effect(ctx, "prefetch(&cY2[index], 1)");
    dual_types::side_effect(ctx, "prefetch(&cY3[index], 1)");
    dual_types::side_effect(ctx, "prefetch(&cY4[index], 1)");
    dual_types::side_effect(ctx, "prefetch(&cY5[index], 1)");

    dual_types::side_effect(ctx, "prefetch(&cA0[index], 1)");
    dual_types::side_effect(ctx, "prefetch(&cA1[index], 1)");
    dual_types::side_effect(ctx, "prefetch(&cA2[index], 1)");
    dual_types::side_effect(ctx, "prefetch(&cA3[index], 1)");
    dual_types::side_effect(ctx, "prefetch(&cA4[index], 1)");
    dual_types::side_effect(ctx, "prefetch(&cA5[index], 1)");

    dual_types::side_effect(ctx, "prefetch(&cGi0[index], 1)");
    dual_types::side_effect(ctx, "prefetch(&cGi1[index], 1)");
    dual_types::side_effect(ctx, "prefetch(&cGi2[index], 1)");
    dual_types::side_effect(ctx, "prefetch(&X[index], 1)");
    dual_types::side_effect(ctx, "prefetch(&K[index], 1)");
    dual_types::side_effect(ctx, "prefetch(&gA[index], 1)");
    dual_types::side_effect(ctx, "prefetch(&gB0[index], 1)");
    dual_types::side_effect(ctx, "prefetch(&gB1[index], 1)");
    dual_types::side_effect(ctx, "prefetch(&gB2[index], 1)");

    value_i order = declare((value_i)order_ptr[index], "order");

    value_i lD_FULL = (int)D_FULL;
    value_i lD_LOW = (int)D_LOW;

    ///note to self the d_full check here is bad
    value_i is_bad = ((order & lD_FULL) == 0) && ((order & lD_LOW) == 0);

    if_e(is_bad, [&]()
    {
        return_e();
    });

    return {ix, iy, iz, index};
}

template<typename base, single_source::impl::fixed_string str>
struct bssn_arg_pack
{
    std::array<named_buffer<base, str + "cY">, 6> cY;
    std::array<named_buffer<base, str + "cA">, 6> cA;
    std::array<named_buffer<base, str + "cGi">, 3> cGi;
    named_buffer<base, str + "K"> K;
    named_buffer<base, str + "X"> X;
    named_buffer<base, str + "gA"> gA;
    std::array<named_buffer<base, str + "gB">, 3> gB;

    #ifdef USE_GBB
    std::array<named_buffer<base, str + "gBB">, 3> gBB;
    #endif // USE_GBB

    bssn_arg_pack()
    {
        for(int i=0; i < 6; i++)
        {
            cY[i].name = cY[i].name + std::to_string(i);
            cA[i].name = cA[i].name + std::to_string(i);
        }

        for(int i=0; i < 3; i++)
        {
            cGi[i].name = cGi[i].name + std::to_string(i);
            gB[i].name = gB[i].name + std::to_string(i);

            #ifdef USE_GBB
            gBB[i].name = gBB[i].name + std::to_string(i);
            #endif // USE_GBB
        }
    }
};

struct all_args
{
    buffer<tensor<value_us, 4>> points;
    literal<value_i> point_count;

    bssn_arg_pack<value, ""> in;
    bssn_arg_pack<value_mut, "o"> out;
    bssn_arg_pack<value, "base_"> base;

    std::array<named_buffer<value, "momentum">, 3> momentum;
    std::array<named_buffer<half_type, "dcYij">, 18> dcYij; std::array<named_buffer<half_type, "digA">, 3> digA;
    std::array<named_buffer<half_type, "digB">, 9> digB; std::array<named_buffer<half_type, "dX">, 3> dX;
    named_literal<value, "scale"> scale;
    named_literal<tensor<value_i, 4>, "dim"> dim;
    literal<value> timestep;
    named_buffer<value_us, "order_ptr"> order_ptr;

    all_args(single_source::argument_generator& arg_gen, base_bssn_args& bssn_args, base_utility_args& utility_args)
    {
        arg_gen.add(points, point_count);

        auto non_mut_buffers = bssn_args.buffers;

        for(auto& i : non_mut_buffers)
            i.is_constant = true;

        arg_gen.add(non_mut_buffers);
        arg_gen.add(std::string{"o"}, bssn_args.buffers);
        arg_gen.add(std::string{"base_"}, non_mut_buffers);

        arg_gen.add(momentum);
        arg_gen.add(dcYij, digA, digB, dX);

        auto non_mut_utility = utility_args.buffers;

        for(auto& i : non_mut_utility)
            i.is_constant = true;

        arg_gen.add(utility_args.buffers);

        arg_gen.add(scale, dim, timestep, order_ptr);
    }
};

struct exec_builder_base
{
    virtual void start(standard_arguments& args, equation_context& ctx, const matter_interop& interop, bool use_matter, const simulation_modifications& mod){}
    virtual void execute(equation_context& ctx, all_args& all){}

    virtual ~exec_builder_base(){}
};

template<typename T, auto U, auto V>
struct exec_builder : exec_builder_base
{
    T dt;

    void start(standard_arguments& args, equation_context& ctx, const matter_interop& interop, bool use_matter, const simulation_modifications& mod) override
    {
        dt = U(args, ctx, interop, use_matter, mod);
    }

    void execute(equation_context& ctx, all_args& all) override
    {
        V(ctx, all, dt);
    }

    virtual ~exec_builder(){}
};

tensor<value, 6> get_dtcYij(standard_arguments& args, equation_context& ctx, const matter_interop& interop, bool use_matter, const simulation_modifications& mod)
{
    metric<value, 3, 3> unpinned_cY = args.unpinned_cY;

    //ctx.pin(args.cY);

    tensor<value, 3> bigGi_lower = lower_index(args.bigGi, args.cY, 0);
    ///Oh no. These are associated with Y, not cY
    tensor<value, 3> gB_lower = lower_index(args.gB, args.cY, 0);

    ctx.pin(args.cY);

    //ctx.pin(bigGi_lower);
    ctx.pin(gB_lower);

    tensor<value, 3, 3> lie_cYij = lie_derivative_weight(ctx, args.gB, unpinned_cY);

    ///https://arxiv.org/pdf/gr-qc/0511048.pdf (1)
    ///https://scholarworks.rit.edu/cgi/viewcontent.cgi?article=11286&context=theses 3.66
    tensor<value, 3, 3> dtcYij = -2 * args.gA * trace_free(args.cA, args.cY, args.cY.invert()) + lie_cYij;

    value damp_factor = get_kc()/3.f;

    //damp_factor = min(damp_factor, 0.3f/value{"timestep"});

    dtcYij += -damp_factor * args.gA * args.cY.to_tensor() * log(args.cY.det());

    ///this specifically is incredibly low
    if(mod.hamiltonian_cY_damp)
        dtcYij += mod.hamiltonian_cY_damp.value().val * args.gA * args.cY.to_tensor() * -bssn::calculate_hamiltonian_constraint(interop, ctx, use_matter);

    ///http://eanam6.khu.ac.kr/presentations/7-5.pdf check this

    if(mod.sigma)
    {
        ///https://arxiv.org/pdf/1205.5111v1.pdf 46
        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                value sigma = mod.sigma.value().val;

                dtcYij.idx(i, j) += sigma * 0.5f * (gB_lower.idx(i) * bigGi_lower.idx(j) + gB_lower.idx(j) * bigGi_lower.idx(i));

                dtcYij.idx(i, j) += -(1.f/5.f) * args.cY.idx(i, j) * sum_multiply(args.gB, bigGi_lower);
            }
        }
    }

    ///pretty sure https://arxiv.org/pdf/0711.3575v1.pdf 2.21 is equivalent, and likely a LOT faster
    if(mod.mod_cY1)
    {
        tensor<value, 3, 3> cD = covariant_derivative_low_vec(ctx, bigGi_lower, args.christoff2);

        ctx.pin(cD);

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                float cK = mod.mod_cY1.value().val;

                dtcYij.idx(i, j) += cK * args.gA * 0.5f * (cD.idx(i, j) + cD.idx(j, i));
            }
        }
    }

    ///it looks like this might cause issues in the hydrodynamics
    if(mod.mod_cY2)
    {
        tensor<value, 3, 3> d_cGi;

        for(int m=0; m < 3; m++)
        {
            tensor<dual, 3, 3, 3> d_dcYij;

            #define FORWARD_DIFFERENTIATION
            #ifdef FORWARD_DIFFERENTIATION
            metric<dual, 3, 3> d_cYij;

            for(int i=0; i < 3; i++)
            {
                for(int j=0; j < 3; j++)
                {
                    d_cYij[i, j].real = args.cY[i, j];
                    d_cYij[i, j].dual = args.dcYij[m, i, j];
                }
            }

            ctx.pin(d_cYij);

            auto dicY = d_cYij.invert();

            ctx.pin(dicY);

            #else
            std::vector<std::pair<value, value>> derivatives;

            for(int i=0; i < 3; i++)
            {
                for(int j=0; j < 3; j++)
                {
                    derivatives.push_back({args.cY[i, j], args.dcYij[m, i, j]});
                }
            }

            auto icY = args.cY.invert();

            inverse_metric<dual, 3, 3> dicY;

            for(int i=0; i < 3; i++)
            {
                for(int j=0; j < 3; j++)
                {
                    ///perform analytic differentiation, where the variable is args.cY[i, j]
                    dicY[i, j] = icY[i, j].dual2(derivatives);
                }
            }

            #endif // FORWARD_DIFFERENTIATION

            for(int k=0; k < 3; k++)
            {
                for(int i=0; i < 3; i++)
                {
                    for(int j=0; j < 3; j++)
                    {
                        d_dcYij[k, i, j].real = args.dcYij[k, i, j];
                        d_dcYij[k, i, j].dual = diff1(ctx, args.dcYij[k, i, j], m);
                    }
                }
            }

            ctx.pin(d_dcYij);

            auto d_christoff2 = christoffel_symbols_2(dicY, d_dcYij);

            ctx.pin(d_christoff2);

            tensor<dual, 3> dcGi_G;

            for(int i=0; i < 3; i++)
            {
                dual sum = 0;

                for(int j=0; j < 3; j++)
                {
                    for(int k=0; k < 3; k++)
                    {
                        sum += dicY[j, k] * d_christoff2[i, j, k];
                    }
                }

                dcGi_G[i] = sum;
            }

            ctx.pin(dcGi_G);

            for(int i=0; i < 3; i++)
            {
                d_cGi[m, i] = diff1(ctx, args.cGi[i], m) - dcGi_G[i].dual;
            }
        }

        tensor<value, 3, 3> cD = covariant_derivative_high_vec(ctx, args.bigGi, d_cGi, args.christoff2);

        ctx.pin(cD);

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                value sum = 0;

                for(int k=0; k < 3; k++)
                {
                    sum += 0.5f * (args.cY[k, i] * cD[k, j] + args.cY[k, j] * cD[k, i]);
                }

                float cK = mod.mod_cY2.value().val;

                dtcYij.idx(i, j) += cK * args.gA * sum;
            }
        }
    }

    tensor<value, 6> dt = {
        dtcYij.idx(0, 0),
        dtcYij.idx(1, 0),
        dtcYij.idx(2, 0),
        dtcYij.idx(1, 1),
        dtcYij.idx(1, 2),
        dtcYij.idx(2, 2)
    };

    //ctx.pin(dt);

    return dt;
}

void finish_cY(equation_context& ctx, all_args& all, tensor<value, 6>& dtcY)
{
    using namespace dual_types::implicit;

    value_i index = "index";

    for(int i=0; i < 6; i++)
    {
        mut(all.out.cY[i][index]) = backwards_euler_relax(all.in.cY[i][index], all.base.cY[i][index], dtcY[i], all.timestep);
    }
}

exec_builder<tensor<value, 6>, get_dtcYij, finish_cY> cYexec;

tensor<value, 3, 3> bssn::calculate_xgARij(equation_context& ctx, standard_arguments& args, const inverse_metric<value, 3, 3>& icY, const tensor<value, 3, 3, 3>& christoff1, const tensor<value, 3, 3, 3>& christoff2)
{
    value gA_X = args.gA_X;

    tensor<value, 3, 3> cRij;

    tensor<value, 3> derived_cGi = args.derived_cGi;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value s1 = 0;

            for(int l=0; l < 3; l++)
            {
                for(int m=0; m < 3; m++)
                {
                    s1 = s1 + -0.5f * icY.idx(l, m) * diff2(ctx, args.unpinned_cY.idx(i, j), m, l, args.dcYij.idx(m, i, j), args.dcYij.idx(l, i, j));
                }
            }

            value s2 = 0;

            for(int k=0; k < 3; k++)
            {
                s2 = s2 + 0.5f * (args.cY.idx(k, i) * diff1(ctx, args.cGi.idx(k), j) + args.cY.idx(k, j) * diff1(ctx, args.cGi.idx(k), i));
            }

            value s3 = 0;

            for(int k=0; k < 3; k++)
            {
                s3 = s3 + 0.5f * derived_cGi.idx(k) * (christoff1.idx(i, j, k) + christoff1.idx(j, i, k));
            }

            value s4 = 0;

            for(int m=0; m < 3; m++)
            {
                for(int l=0; l < 3; l++)
                {
                    value inner1 = 0;
                    value inner2 = 0;

                    for(int k=0; k < 3; k++)
                    {
                        inner1 = inner1 + 0.5f * (2 * christoff2.idx(k, l, i) * christoff1.idx(j, k, m) + 2 * christoff2.idx(k, l, j) * christoff1.idx(i, k, m));
                    }

                    for(int k=0; k < 3; k++)
                    {
                        inner2 = inner2 + christoff2.idx(k, i, m) * christoff1.idx(k, l, j);
                    }

                    s4 = s4 + icY.idx(l, m) * (inner1 + inner2);
                }
            }

            cRij.idx(i, j) = s1 + s2 + s3 + s4;
        }
    }

    #ifndef USE_W
    tensor<value, 3> dX = args.get_dX();

    ///this needs to be fixed if we're using W
    tensor<value, 3, 3> cov_div_X = double_covariant_derivative(ctx, args.get_X(), args.dX_impl, christoff2);
    ctx.pin(cov_div_X);

    ///https://indico.cern.ch/event/505595/contributions/1183661/attachments/1332828/2003830/sperhake.pdf
    tensor<value, 3, 3> xgARphiij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value s1 = 0;
            value s2 = 0;

            for(int m=0; m < 3; m++)
            {
                for(int n=0; n < 3; n++)
                {
                    s1 += icY.idx(m, n) * cov_div_X.idx(n, m);
                    s2 += icY.idx(m, n) * dX.idx(m) * dX.idx(n);
                }
            }

            value s3 = (1/2.f) * (args.gA * cov_div_X.idx(j, i) - gA_X * (1/2.f) * dX.idx(i) * dX.idx(j));

            s1 = args.gA * (args.cY.idx(i, j) / 2.f) * s1;
            s2 = gA_X * (args.cY.idx(i, j) / 2.f) * -(3.f/2.f) * s2;

            xgARphiij.idx(i, j) = s1 + s2 + s3;
        }
    }
    #else
    ///https://arxiv.org/pdf/1307.7391.pdf (9)
    tensor<value, 3, 3> xgARphiij;

    tensor<value, 3, 3> didjW;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            ///dcd uses the notation i;j
            didjW.idx(i, j) = double_covariant_derivative(ctx, args.W_impl, args.dW_impl, christoff2).idx(j, i);
        }
    }

    value W = args.W_impl;
    tensor<value, 3> dW = args.dW_calc;

    value p2 = -2 * sum_multiply(dW, raise_index(dW, icY, 0));
    value p3 = W * sum_multiply(icY.to_tensor(), didjW);

    ///https://iopscience.iop.org/article/10.1088/1361-6382/ac7e16/pdf (2.6)
    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value p1 = (args.gA * W) * didjW.idx(i, j);

            xgARphiij.idx(i, j) = p1 + args.cY.idx(i, j) * (args.gA * (p2 + p3));
        }
    }
    #endif

    tensor<value, 3, 3> xgARij = xgARphiij + args.get_X() * args.gA * cRij;

    ctx.pin(xgARij);

    return xgARij;
}

value calculate_hamiltonian(const metric<value, 3, 3>& cY, const inverse_metric<value, 3, 3>& icY, const metric<value, 3, 3>& Yij, const inverse_metric<value, 3, 3>& iYij, const tensor<value, 3, 3>& Rij, const value& K, const tensor<value, 3, 3>& cA)
{
    value R = trace(Rij, iYij);

    tensor<value, 3, 3> aIJ = raise_both(cA, icY);

    value aij_aIJ;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            aij_aIJ += cA.idx(i, j) * aIJ.idx(i, j);
        }
    }

    float D = 4;

    ///https://indico.cern.ch/event/505595/contributions/1183661/attachments/1332828/2003830/sperhake.pdf
    return R + ((D - 2) / (D - 1)) * K*K - aij_aIJ;
}

value calculate_R_from_hamiltonian(const value& K, const tensor<value, 3, 3>& cA, const inverse_metric<value, 3, 3>& icY)
{
    tensor<value, 3, 3> aIJ = raise_both(cA, icY);

    value aij_aIJ;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            aij_aIJ += cA.idx(i, j) * aIJ.idx(i, j);
        }
    }

    return -((2.f/3.f) * K * K - aij_aIJ);
}

value calculate_hamiltonian(equation_context& ctx, standard_arguments& args)
{
    auto icY = args.cY.invert();

    tensor<value, 3, 3, 3> christoff1 = christoffel_symbols_1(ctx, args.unpinned_cY);

    tensor<value, 3, 3> xgARij = bssn::calculate_xgARij(ctx, args, icY, christoff1, args.christoff2);

    return calculate_hamiltonian(args.cY, icY, args.Yij, args.iYij, (xgARij / (max(args.get_X(), 0.001f) * args.gA)), args.K, args.cA);
}

tensor<value, 6> get_dtcAij(standard_arguments& args, equation_context& ctx, const matter_interop& interop, bool use_matter, const simulation_modifications& mod)
{
    value scale = "scale";

    ctx.pin(args.derived_cGi);

    inverse_metric<value, 3, 3> icY = args.cY.invert();

    //tensor<value, 3, 3, 3> christoff1 = gpu_christoffel_symbols_1(ctx, args.cY);
    tensor<value, 3, 3, 3> christoff2 = args.christoff2;

    tensor<value, 3, 3, 3> christoff1;

    ///Gak Ckbc
    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                value sum = 0;

                for(int d = 0; d < 3; d++)
                {
                    sum += args.cY.idx(i, d) * christoff2.idx(d, j, k);
                }

                christoff1.idx(i, j, k) = sum;
            }
        }
    }

    ctx.pin(christoff1);
    ctx.pin(christoff2);

    metric<value, 3, 3> cY = args.cY;

    ctx.pin(icY);

    tensor<value, 3, 3> cA = args.cA;

    auto unpinned_cA = args.unpinned_cA;

    ///the christoffel symbol
    tensor<value, 3> cGi = args.cGi;

    value gA = args.gA;

    tensor<value, 3> gB = args.gB;

    value X = args.get_X();
    value K = args.K;

    tensor<value, 3> dX = args.get_dX();

    tensor<value, 3> derived_cGi = args.derived_cGi;

    tensor<value, 3, 3> xgARij = bssn::calculate_xgARij(ctx, args, icY, christoff1, christoff2);

    ctx.pin(xgARij);

    tensor<value, 3, 3> Xdidja;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value Xderiv = X * double_covariant_derivative(ctx, args.gA, args.digA, args.christoff2).idx(j, i);
            //value Xderiv = X * gpu_covariant_derivative_low_vec(ctx, args.digA, cY, icY).idx(j, i);

            value s2 = 0.5f * (dX.idx(i) * diff1(ctx, gA, j) + dX.idx(j) * diff1(ctx, gA, i));

            value s3 = 0;

            for(int m=0; m < 3; m++)
            {
                for(int n=0; n < 3; n++)
                {
                    s3 += icY.idx(m, n) * dX.idx(m) * diff1(ctx, gA, n);
                }
            }

            Xdidja.idx(i, j) = Xderiv + s2 + -0.5f * cY.idx(i, j) * s3;
        }
    }

    ctx.pin(Xdidja);

    ///recover Yij from X and cYij
    ///https://arxiv.org/pdf/gr-qc/0511048.pdf
    ///https://arxiv.org/pdf/gr-qc/9810065.pdf
    ///X = exp(-4 phi)
    ///consider trying to eliminate via https://arxiv.org/pdf/gr-qc/0206072.pdf (27). I think this is what you're meant to do
    ///to eliminate the dependency on the non conformal metric entirely. This would improve stability quite significantly
    ///near the puncture

    ///Aki G^kj
    tensor<value, 3, 3> mixed_cAij = raise_index(cA, icY, 0);

    ctx.pin(mixed_cAij);

    ///not sure dtcaij is correct, need to investigate
    tensor<value, 3, 3> dtcAij;

    ///https://indico.cern.ch/event/505595/contributions/1183661/attachments/1332828/2003830/sperhake.pdf replaced with definition under bssn aux
    tensor<value, 3, 3> with_trace = -Xdidja + xgARij;

    tensor<value, 3, 3> without_trace = trace_free(with_trace, cY, icY);

    tensor<value, 3, 3> symmetric_momentum_deriv;

    if(mod.momentum_damping2)
    {
        tensor<value, 3, 3> momentum_deriv;

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                momentum_deriv.idx(i, j) = diff1(ctx, args.momentum_constraint.idx(i), j);
            }
        }

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                symmetric_momentum_deriv.idx(i, j) = 0.5f * (momentum_deriv.idx(i, j) + momentum_deriv.idx(j, i));
            }
        }

        ctx.pin(symmetric_momentum_deriv);
    }

    tensor<value, 3, 3> BiMj_TF;

    if(mod.aij_sigma)
    {
        tensor<value, 3> Mi = args.momentum_constraint;

        tensor<value, 3> gB_lower = lower_index(gB, cY, 0);

        tensor<value, 3, 3> BiMj;

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                BiMj.idx(i, j) = gB_lower.idx(i) * Mi.idx(j);
            }
        }

        BiMj_TF = trace_free(BiMj, cY, icY);
    }

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value sum = 0;

            for(int k=0; k < 3; k++)
            {
                sum = sum + cA.idx(i, k) * mixed_cAij.idx(k, j);
            }

            ///so
            ///the trace is calculated as iYij Vij, where Vij is whatever
            ///if Yij = cYij / X
            ///https://en.wikipedia.org/wiki/Invertible_matrix#Other_properties
            ///then iYij = = X * icYij
            ///the trace is the sum X * icYij * Vij
            ///making something trace free is denoted as Vij - (1/3) metij * V, where V = trace
            ///= Vij - (1/3) Yij * V
            ///= Vij - (1/3) (cYij / X) * V
            ///but the trace is the sum of something multiplied by X
            ///= Vij - (1/3) cYij (icYkl Vkl)
            ///therefore I think constant factor multiplications to the metric make no difference to the trace calculation, so we can use
            ///cY here instead of Yij

            ///not convinced its correct to push x inside of trace free?
            ///what if the riemann quantity is made trace free by cY instead of Yij like I assumed?
            value p1 = without_trace.idx(i, j);

            value p2 = gA * (K * cA.idx(i, j) - 2 * sum);

            value p3 = lie_derivative_weight(ctx, gB, unpinned_cA).idx(i, j);

            if(i == 0 && j == 0)
            {
                ctx.add("debug_p1", p1);
                ctx.add("debug_p2", p2);
                ctx.add("debug_p3", p3);
            }

            dtcAij.idx(i, j) = p1 + p2 + p3;

            if(mod.classic_momentum_damping)
            {
                float Ka = mod.classic_momentum_damping.value().val;

                dtcAij.idx(i, j) += Ka * gA * 0.5f *
                                                    (covariant_derivative_low_vec(ctx, args.momentum_constraint, args.unpinned_cY, icY).idx(i, j)
                                                     + covariant_derivative_low_vec(ctx, args.momentum_constraint, args.unpinned_cY, icY).idx(j, i));
            }

            if(mod.momentum_damping2)
            {
                value F_a = scale;

                if(mod.momentum_damping2.value().use_lapse)
                    F_a = scale * gA;

                ///https://arxiv.org/pdf/1205.5111v1.pdf (56)
                dtcAij.idx(i, j) += scale * F_a * trace_free(symmetric_momentum_deriv, cY, icY).idx(i, j);
            }

            if(mod.aij_sigma)
            {
                float sigma = mod.aij_sigma.value().val;

                dtcAij.idx(i, j) += (-3.f/5.f) * sigma * BiMj_TF.idx(i, j);
            }

            ///matter
            if(use_matter)
            {
                tensor<value, 3, 3> xSij = interop.calculate_adm_X_Sij(ctx, args);

                tensor<value, 3, 3> xgASij = trace_free(-8 * (float)M_PI * gA * xSij, cY, icY);

                //ctx.add("DBGXGA", xgASij.idx(0, 0));
                //ctx.add("Debug_cS0", args.matt.cS.idx(0));

                dtcAij.idx(i, j) += xgASij.idx(i, j);
            }
        }
    }

    value damp_factor = get_kc()/3.f;
    //damp_factor = min(damp_factor, 0.3f/value{"timestep"});

    dtcAij += -damp_factor * args.gA * args.cY.to_tensor() * trace(args.cA, args.cY.invert());

    if(mod.hamiltonian_cA_damp)
        dtcAij += -mod.hamiltonian_cA_damp.value().val * args.gA * args.cA * -bssn::calculate_hamiltonian_constraint(interop, ctx, use_matter);

    if(mod.cA_damp)
    {
        ///https://arxiv.org/pdf/gr-qc/0204002.pdf 4.3
        value bigGi_diff = 0;

        for(int i=0; i < 3; i++)
        {
            bigGi_diff += diff1(ctx, args.bigGi.idx(i), i);
        }

        float k8 = mod.cA_damp.value().val;

        dtcAij += -k8 * args.gA * args.get_X() * args.cY.to_tensor() * bigGi_diff;
    }

    tensor<value, 6> dt = {
        dtcAij.idx(0, 0),
        dtcAij.idx(1, 0),
        dtcAij.idx(2, 0),
        dtcAij.idx(1, 1),
        dtcAij.idx(1, 2),
        dtcAij.idx(2, 2)
    };

    //ctx.pin(dt);

    return dt;
}

void finish_cA(equation_context& ctx, all_args& all, tensor<value, 6>& dtcA)
{
    using namespace dual_types::implicit;

    value_i index = "index";

    //ctx.pin(dtcA);

    for(int i=0; i < 6; i++)
    {
        mut(all.out.cA[i][index]) = backwards_euler_relax(all.in.cA[i][index], all.base.cA[i][index], dtcA[i], all.timestep);
        //ctx.exec(assign(all.out.cA[i][index], all.base.cA[i][index] + all.timestep * dtcA[i]));
    }
}

exec_builder<tensor<value, 6>, get_dtcAij, finish_cA> cAexec;

tensor<value, 3> get_dtcGi(standard_arguments& args, equation_context& ctx, const matter_interop& interop, bool use_matter, const simulation_modifications& mod)
{
    inverse_metric<value, 3, 3> icY = args.cY.invert();

    tensor<value, 3, 3, 3> christoff2 = args.christoff2;

    ctx.pin(christoff2);

    metric<value, 3, 3> cY = args.cY;

    inverse_metric<value, 3, 3> unpinned_icY = args.unpinned_cY.invert();

    tensor<value, 3, 3> cA = args.cA;

    ///the christoffel symbol
    tensor<value, 3> cGi = args.cGi;

    value gA = args.gA;

    tensor<value, 3> gB = args.gB;

    value X = args.get_X();
    tensor<value, 3> dX = args.get_dX();
    value K = args.K;

    tensor<value, 3, 3> icAij = raise_both(cA, icY);

    value gA_X = args.gA_X;

    ///these seem to suffer from oscillations
    tensor<value, 3> dtcGi;

    tensor<value, 3> derived_cGi = args.derived_cGi;

    ///https://arxiv.org/pdf/gr-qc/0511048.pdf

    ///https://arxiv.org/pdf/1205.5111v1.pdf 49
    ///made it to 58 with this
    #define CHRISTOFFEL_49
    #ifdef CHRISTOFFEL_49
    tensor<value, 3> Yij_Kj;

    if(mod.christoff_modification_1)
    {
        //tensor<value, 3, 3> littlekij = unpinned_icY.to_tensor() * K;

        tensor<dual, 3, 3, 3> dicY;

        for(int k=0; k < 3; k++)
        {
            unit_metric<dual, 3, 3> cYk;

            for(int i=0; i < 3; i++)
            {
                for(int j=0; j < 3; j++)
                {
                    dual d;
                    d.real = args.unpinned_cY.idx(i, j);
                    d.dual = diff1(ctx, args.unpinned_cY.idx(i, j), k);

                    cYk.idx(i, j) = d;
                }
            }

            inverse_metric<dual, 3, 3> icYk = cYk.invert();

            for(int i=0; i < 3; i++)
            {
                for(int j=0; j < 3; j++)
                {
                    dicY.idx(k, i, j) = icYk.idx(i, j);
                }
            }
        }

        for(int i=0; i < 3; i++)
        {
            value sum = 0;

            for(int j=0; j < 3; j++)
            {
                sum += icY.idx(i, j) * diff1(ctx, K, j) + K * dicY.idx(j, i, j).dual;
                //sum += diff1(ctx, littlekij.idx(i, j), j);
            }

            Yij_Kj.idx(i) = sum + args.K * derived_cGi.idx(i);
        }
    }
    else
    {
        for(int i=0; i < 3; i++)
        {
            value sum = 0;

            for(int j=0; j < 3; j++)
            {
                sum += icY.idx(i, j) * diff1(ctx, args.K, j);
            }

            Yij_Kj.idx(i) = sum;
        }
    }

    for(int i=0; i < 3; i++)
    {
        value s1 = 0;

        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                s1 += 2 * gA * christoff2.idx(i, j, k) * icAij.idx(j, k);
            }
        }

        value s2 = 2 * gA * -(2.f/3.f) * Yij_Kj.idx(i);

        #ifndef USE_W
        value s3 = 0;

        for(int j=0; j < 3; j++)
        {
            s3 += 2 * (-1.f/4.f) * gA_X * 6 * icAij.idx(i, j) * dX.idx(j);
        }
        #else
        value s3 = 0;

        for(int j=0; j < 3; j++)
        {
            s3 += icAij.idx(i, j) * 2 * args.dW_calc.idx(j);
        }

        s3 = 2 * (-1.f/4.f) * gA / max(args.W_impl, 0.0001f) * 6 * s3;
        #endif

        value s4 = 0;

        for(int j=0; j < 3; j++)
        {
            s4 += icAij.idx(i, j) * diff1(ctx, gA, j);
        }

        s4 = -2 * s4;

        value s5 = 0;

        for(int j=0; j < 3; j++)
        {
            s5 += upwind_differentiate(ctx, gB.idx(j), cGi.idx(i), j);
        }

        value s6 = 0;

        for(int j=0; j < 3; j++)
        {
            s6 += -derived_cGi.idx(j) * args.digB.idx(j, i);
        }

        value s7 = 0;

        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                //s7 += icY.idx(j, k) * hacky_differentiate(args.digB.idx(k, i), j);
                s7 += icY.idx(j, k) * diff2(ctx, args.gB.idx(i), k, j, args.digB.idx(k, i), args.digB.idx(j, i));
            }
        }

        value s8 = 0;

        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                //s8 += (1.f/3.f) * icY.idx(i, j) * hacky_differentiate(args.digB.idx(k, k), j);
                s8 += icY.idx(i, j) * diff2(ctx, args.gB.idx(k), k, j, args.digB.idx(k, k), args.digB.idx(j, k));
            }
        }

        s8 = (1.f/3.f) * s8;

        value s9 = 0;

        for(int k=0; k < 3; k++)
        {
            s9 += args.digB.idx(k, k);
        }

        s9 = (2.f/3.f) * s9 * derived_cGi.idx(i);

        ///this is the only instanced of derived_cGi that might want to be regular cGi
        //value s10 = (2.f/3.f) * -2 * gA * K * derived_cGi.idx(i);

        dtcGi.idx(i) = s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9;

        ///https://arxiv.org/pdf/1205.5111v1.pdf 50
        ///made it to 70+ and then i got bored, but the simulation was meaningfully different
        if(mod.christoff_modification_2)
        {
            auto step = [](const value& in)
            {
                return if_v((value_i)(in >= 0.f), value{1.f}, value{0.f});
            };

            value bkk = 0;

            for(int k=0; k < 3; k++)
            {
                bkk += args.digB.idx(k, k);
            }

            float E = mod.christoff_modification_2.value().val;

            value lambdai = (2.f/3.f) * (bkk - 2 * gA * K)
                            - args.digB.idx(i, i)
                            - (2.f/5.f) * gA * raise_index(cA, icY, 1).idx(i, i);

            dtcGi.idx(i) += -(1 + E) * step(lambdai) * lambdai * args.bigGi.idx(i);

        }

        if(use_matter)
        {
            tensor<value, 3> ji_lower = interop.calculate_adm_Si(ctx, args);

            value sum = 0;

            for(int j=0; j < 3; j++)
            {
                sum += icY.idx(i, j) * ji_lower.idx(j);
            }

            dtcGi.idx(i) += gA * -2 * 8 * (float)M_PI * sum;
        }
    }
    #endif // CHRISTOFFEL_49

    ///todo: test 2.22 https://arxiv.org/pdf/0711.3575.pdf
    if(mod.ybs)
    {
        value E = mod.ybs.value().val;

        value sum = 0;

        for(int k=0; k < 3; k++)
        {
            sum += diff1(ctx, args.gB.idx(k), k);
        }

        dtcGi += (-2.f/3.f) * (E + 1) * args.bigGi * sum;
    }

    ///https://arxiv.org/pdf/gr-qc/0204002.pdf table 2, think case E2 is incorrectly labelled
    if(mod.mod_cGi)
    {
        float mcGicst = mod.mod_cGi.value().val;

        dtcGi += mcGicst * gA * args.bigGi;
    }

    //ctx.pin(dtcGi);

    return dtcGi;
}

void finish_cGi(equation_context& ctx, all_args& all, tensor<value, 3>& dtcGi)
{
    using namespace dual_types::implicit;

    value_i index = "index";

    for(int i=0; i < 3; i++)
    {
        mut(all.out.cGi[i][index]) = backwards_euler_relax(all.in.cGi[i][index], all.base.cGi[i][index], dtcGi[i], all.timestep);

        //ctx.exec(assign(all.out.cGi[i][index], all.base.cGi[i][index] + all.timestep * dtcGi[i]));
    }
}

exec_builder<tensor<value, 3>, get_dtcGi, finish_cGi> cGiexec;

value get_dtK(standard_arguments& args, equation_context& ctx, const matter_interop& interop, bool use_matter, const simulation_modifications& mod)
{
    inverse_metric<value, 3, 3> icY = args.cY.invert();

    metric<value, 3, 3> cY = args.cY;

    ctx.pin(icY);

    tensor<value, 3, 3> cA = args.cA;

    ///the christoffel symbol
    tensor<value, 3> cGi = args.cGi;

    value gA = args.gA;

    tensor<value, 3> gB = args.gB;

    value X = args.get_X();
    tensor<value, 3> dX = args.get_dX();
    value K = args.K;

    tensor<value, 3, 3> Xdidja;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value Xderiv = X * double_covariant_derivative(ctx, args.gA, args.digA, args.christoff2).idx(j, i);
            //value Xderiv = X * gpu_covariant_derivative_low_vec(ctx, args.digA, cY, icY).idx(j, i);

            value s2 = 0.5f * (dX.idx(i) * diff1(ctx, gA, j) + dX.idx(j) * diff1(ctx, gA, i));

            value s3 = 0;

            for(int m=0; m < 3; m++)
            {
                for(int n=0; n < 3; n++)
                {
                    s3 += icY.idx(m, n) * dX.idx(m) * diff1(ctx, gA, n);
                }
            }

            Xdidja.idx(i, j) = Xderiv + s2 + -0.5f * cY[i, j] * s3;
        }
    }

    tensor<value, 3, 3> icAij = raise_both(cA, icY);

    value dtK = sum(tensor_upwind(ctx, gB, K)) - sum_multiply(icY.to_tensor(), Xdidja) + gA * (sum_multiply(icAij, cA) + (1/3.f) * K * K);

    if(use_matter)
    {
        value matter_s = interop.calculate_adm_S(ctx, args);
        value matter_p = interop.calculate_adm_p(ctx, args);

        dtK += (8 * (float)M_PI / 2) * gA * (matter_s + matter_p);
    }

    //ctx.pin(dtK);

    return dtK;
}

void finish_K(equation_context& ctx, all_args& all, value& dtK)
{
    using namespace dual_types::implicit;

    value_i index = "index";

    mut(all.out.K[index]) = backwards_euler_relax(all.in.K[index], all.base.K[index], dtK, all.timestep);

    //ctx.exec(assign(all.out.K[index], all.base.K[index] + all.timestep * dtK));
}

exec_builder<value, get_dtK, finish_K> Kexec;

value get_dtX(standard_arguments& args, equation_context& ctx, const matter_interop& interop, bool use_matter, const simulation_modifications& mod)
{
    tensor<value, 3> linear_dB;

    for(int i=0; i < 3; i++)
    {
        linear_dB.idx(i) = diff1(ctx, args.gB.idx(i), i);
    }

    #ifndef USE_W
    value dtX = (2.f/3.f) * args.get_X() * (args.gA * args.K - sum(linear_dB)) + sum(tensor_upwind(ctx, args.gB, args.get_X()));

    //ctx.pin(dtX);

    return dtX;
    #else
    value dtW = (1.f/3.f) * args.W_impl * (args.gA * args.K - sum(linear_dB)) + sum(tensor_upwind(ctx, args.gB, args.W_impl));

    //ctx.pin(dtW);

    return dtW;
    #endif // USE_W
}

void finish_X(equation_context& ctx, all_args& all, value& dtX)
{
    using namespace dual_types::implicit;

    value_i index = "index";

    mut(all.out.X[index]) = backwards_euler_relax(all.in.X[index], all.base.X[index], dtX, all.timestep);

    //ctx.exec(assign(all.out.X[index], all.base.X[index] + all.timestep * dtX));
}

exec_builder<value, get_dtX, finish_X> Xexec;

value get_dtgA(standard_arguments& args, equation_context& ctx, const matter_interop& interop, bool use_matter, const simulation_modifications& mod)
{
    value dtgA = 0;

    if(mod.lapse.advect)
    {
        dtgA += lie_derivative(ctx, args.gB, args.gA);
    }

    if(std::holds_alternative<lapse_conditions::one_plus_log>(mod.lapse.type))
    {
        dtgA += -2 * args.gA * args.K;
    }

    if(std::holds_alternative<lapse_conditions::harmonic>(mod.lapse.type))
    {
        dtgA += -args.gA * args.gA * args.K;
    }

    if(std::holds_alternative<lapse_conditions::shock_avoiding>(mod.lapse.type))
    {
        dtgA += -(8.f/3.f) * args.gA * args.K / (3 - args.gA);
    }

    return dtgA;
}

void finish_gA(equation_context& ctx, all_args& all, value& dtgA)
{
    using namespace dual_types::implicit;

    value_i index = "index";

    value next = backwards_euler_relax(all.in.gA[index], all.base.gA[index], dtgA, all.timestep);

    //next = max(next, value{0.f});

    mut(all.out.gA[index]) = next;
}

exec_builder<value, get_dtgA, finish_gA> gAexec;

tensor<value, 3> get_dtgB(standard_arguments& args, equation_context& ctx, const matter_interop& interop, bool use_matter, const simulation_modifications& mod)
{
    inverse_metric<value, 3, 3> icY = args.cY.invert();

    value X = args.get_X();

    tensor<value, 3> bjdjbi;

    for(int i=0; i < 3; i++)
    {
        value v = 0;

        for(int j=0; j < 3; j++)
        {
           v += upwind_differentiate(ctx, args.gB.idx(j), args.gB.idx(i), j);
        }

        bjdjbi.idx(i) = v;
    }


    #ifdef VDAMP_1
    ///so
    ///X = (1/12) * log(det)
    //value det = exp(12 * X);

    ///(bl^4 * kron) = Yij
    ///
    //value conformal_factor = pow(det, 1.f/16.f);

    /*value phi = log(X) / -4.f;

    ///https://arxiv.org/pdf/gr-qc/0206072.pdf (10)
    value psi = exp(phi);*/

    //value psi = pow(X, -1.f/4.f);
    //value ipsi = pow(psi, -2.f);

    ///https://arxiv.org/pdf/0912.3125.pdf
    ///https://www.wolframalpha.com/input?i=%28e%5E%28log%28x%29%2F-4%29%29%5E-2
    value ipsi2 = sqrt(X);

    float hat_r0 = 1.31;

    ///https://arxiv.org/pdf/0912.3125.pdf(4)
    value Ns_r = 0;

    {
        value sum = 0;

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                sum += icY.idx(i, j) * diff1(ctx, ipsi2, i) * diff1(ctx, ipsi2, j);
            }
        }

        Ns_r = hat_r0 * sqrt(sum) / pow(1 - ipsi2, 2);
    }
    #endif

    //#define VDAMP_2
    #ifdef VDAMP_2
    ///https://arxiv.org/pdf/1009.0292.pdf
    value Ns_r = 0;

    {
        float R0 = 1.31f;

        value W = sqrt(X);

        float a = 2;
        float b = 2;

        value sum = 0;

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                sum += icY.idx(i, j) * diff1(ctx, W, i) * diff1(ctx, W, j);
            }
        }

        Ns_r = R0 * sqrt(sum) / pow(1 - pow(W, a), b);
    }

    #endif

    #define STATIC_DAMP
    #ifdef STATIC_DAMP
    value Ns_r = mod.shift.N;
    #endif

    value N = max(Ns_r, 0.5f);

    #ifndef USE_GBB
    ///https://arxiv.org/pdf/gr-qc/0605030.pdf 26
    ///todo: remove this
    tensor<value, 3> dtgB = (3.f/4.f) * args.derived_cGi - N * args.gB;

    if(mod.shift.advect)
    {
        dtgB += bjdjbi;
    }

    //dtgB = {0,0,0};

    tensor<value, 3> dtgBB;
    dtgBB.idx(0) = 0;
    dtgBB.idx(1) = 0;
    dtgBB.idx(2) = 0;

    #else

    tensor<value, 3> bjdjBi;

    for(int i=0; i < 3; i++)
    {
        value v = 0;

        for(int j=0; j < 3; j++)
        {
           v += upwind_differentiate(ctx, args.gB.idx(j), args.gBB.idx(i), j);
        }

        bjdjBi.idx(i) = v;
    }

    tensor<value, 3> christoffd;

    for(int i=0; i < 3; i++)
    {
        value sum = 0;

        for(int j=0; j < 3; j++)
        {
            sum += args.gB.idx(j) * diff1(ctx, args.cGi.idx(i), j);
        }

        christoffd.idx(i) = sum;
    }

    tensor<value, 3> dtcGi;
    dtcGi.idx(0).make_value("f_dtcGi0");
    dtcGi.idx(1).make_value("f_dtcGi1");
    dtcGi.idx(2).make_value("f_dtcGi2");

    tensor<value, 3> dtgB;
    tensor<value, 3> dtgBB;

    ///https://arxiv.org/pdf/gr-qc/0511048.pdf (11)
    /*for(int i=0; i < 3; i++)
    {
        dtgB.idx(i) = (3.f/4.f) * args.gBB.idx(i) + bjdjbi.idx(i);
    }*/

    /*#ifdef PAPER_0610128
    float N = 1;

    dtgB = (3.f/4.f) * args.gBB;

    dtgBB = dtcGi - N * args.gBB;
    #else*/

    #define USE_GBB1
    #ifdef USE_GBB1
    dtgB = (3.f/4.f) * args.gBB + bjdjbi;

    dtgBB = dtcGi - N * args.gBB + bjdjBi - christoffd;
    #endif

    //#define USE_GBB2
    #ifdef USE_GBB2
    dtgB = args.gBB;

    dtgBB = (3.f/4.f) * dtcGi - N * args.gBB;
    #endif

    //#define USE_SINGLE_GBB
    #ifdef USE_SINGLE_GBB
    dtgB = (3.f/4.f) * args.gBB;

    dtgBB = args.gA * args.gA * dtcGi - N * args.gBB;
    #endif

    //#endif // PAPER_0610128
    #endif // USE_GBB

    //ctx.pin(dtgB);

    return dtgB;
}

void finish_gB(equation_context& ctx, all_args& all, tensor<value, 3>& dtgB)
{
    using namespace dual_types::implicit;

    value_i index = "index";

    for(int i=0; i < 3; i++)
    {
        mut(all.out.gB[i][index]) = backwards_euler_relax(all.in.gB[i][index], all.base.gB[i][index], dtgB[i], all.timestep);

        //ctx.exec(assign(all.out.gB[i][index], all.base.gB[i][index] + all.timestep * dtgB[i]));
    }
}

exec_builder<tensor<value, 3>, get_dtgB, finish_gB> gBexec;

void build_kernel(single_source::argument_generator& arg_gen, equation_context& ctx, const matter_interop* interop, bool use_matter, base_bssn_args& bssn_args, base_utility_args& utility_args, std::vector<exec_builder_base*> execs, vec3i dim, simulation_modifications mod)
{
    std::cout << "Start build\n";

    ctx.dynamic_drop = true;

    all_args all(arg_gen, bssn_args, utility_args);

    tensor<value_i, 4> ddim = all.dim.get();

    (void)setup(ctx, all.points, all.point_count.get(), ddim, all.order_ptr);

    standard_arguments args(ctx);

    for(int i=0; i < (int)execs.size(); i++)
    {
        steady_timer time;

        execs[i]->start(args, ctx, *interop, use_matter, mod);
        execs[i]->execute(ctx, all);

        std::cout << "Elapsed " << time.get_elapsed_time_s() * 1000 << "ms" << std::endl;
    }

    std::cout << "End build\n";
}

void bssn::build(cl::context& clctx, const matter_interop& interop, bool use_matter, base_bssn_args bssn_args, base_utility_args utility_args, vec3i dim, simulation_modifications mod)
{
    std::vector<exec_builder_base*> b = {&cAexec, &Xexec, &Kexec, &gAexec, &gBexec, &cYexec, &cGiexec};

    single_source::make_async_dynamic_kernel_for(clctx, build_kernel, "evolve_1", "", &interop, use_matter, bssn_args, utility_args, b, dim, mod);

    single_source::make_async_dynamic_kernel_for(clctx, calculate_christoffel_symbol, "calculate_christoffel_symbol", "", bssn_args);
}
