#include "bssn.hpp"
#include "single_source.hpp"
#include "bitflags.cl"
#include "spherical_decomposition.hpp"
#include "util.hpp"

value matter_meta_interop::calculate_adm_S(equation_context& ctx, standard_arguments& bssn_args)
{
    value ret;

    for(auto& i : sub_interop)
    {
        ret += i->calculate_adm_S(ctx, bssn_args);
    }

    return ret;
}

value matter_meta_interop::calculate_adm_p(equation_context& ctx, standard_arguments& bssn_args)
{
    value ret;

    for(auto& i : sub_interop)
    {
        ret += i->calculate_adm_p(ctx, bssn_args);
    }

    return ret;
}

tensor<value, 3, 3> matter_meta_interop::calculate_adm_X_Sij(equation_context& ctx, standard_arguments& bssn_args)
{
    tensor<value, 3, 3> ret;

    for(auto& i : sub_interop)
    {
        ret += i->calculate_adm_X_Sij(ctx, bssn_args);
    }

    return ret;
}

tensor<value, 3> matter_meta_interop::calculate_adm_Si(equation_context& ctx, standard_arguments& bssn_args)
{
    tensor<value, 3> ret;

    for(auto& i : sub_interop)
    {
        ret += i->calculate_adm_Si(ctx, bssn_args);
    }

    return ret;
}

void bssn::init(equation_context& ctx, const metric<value, 3, 3>& Yij, const tensor<value, 3, 3>& Aij, const value& gA)
{
    ctx.add_function("buffer_index", buffer_index_f<value, 3>);
    ctx.add_function("buffer_indexh", buffer_index_f<value_h, 3>);

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

        ctx.add(y_name, cYij.idx(index.x(), index.y()));
    }

    for(int i=0; i < 6; i++)
    {
        ctx.add("init_cA" + std::to_string(i), cAij.idx(linear_indices[i].x(), linear_indices[i].y()));
    }

    ctx.add("init_cGi0", cGi.idx(0));
    ctx.add("init_cGi1", cGi.idx(1));
    ctx.add("init_cGi2", cGi.idx(2));

    ctx.add("init_K", K);
    ctx.add("init_X", X);

    ctx.add("init_gA", gA);
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

value bssn::calculate_hamiltonian_constraint(matter_interop& interop, equation_context& ctx, bool use_matter)
{
    standard_arguments args(ctx);

    inverse_metric<value, 3, 3> icY = args.cY.invert();

    tensor<value, 3, 3, 3> christoff1 = christoffel_symbols_1(ctx, args.unpinned_cY);

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

#define STANDARD_ARGS(x) named_buffer<value, 3, #x"cY0"> x##cY0, named_buffer<value, 3, #x"cY1"> x##cY1, named_buffer<value, 3, #x"cY2"> x##cY2, named_buffer<value, 3, #x"cY3"> x##cY3, named_buffer<value, 3, #x"cY4"> x##cY4, named_buffer<value, 3, #x"cY5"> x##cY5, \
                   named_buffer<value, 3, #x"cA0"> x##cA0, named_buffer<value, 3, #x"cA1"> x##cA1, named_buffer<value, 3, #x"cA2"> x##cA2, named_buffer<value, 3, #x"cA3"> x##cA3, named_buffer<value, 3, #x"cA4"> x##cA4, named_buffer<value, 3, #x"cA5"> x##cA5, \
                   named_buffer<value, 3, #x"cGi0"> x##cGi0, named_buffer<value, 3, #x"cGi1"> x##cGi1, named_buffer<value, 3, #x"cGi2"> x##cGi2, named_buffer<value, 3, #x"K"> x##K, named_buffer<value, 3, #x"X"> x##X, named_buffer<value, 3, #x"gA"> x##gA, \
                   named_buffer<value, 3, #x"gB0"> x##gB0, named_buffer<value, 3, #x"gB1"> x##gB1, named_buffer<value, 3, #x"gB2"> x##gB2

#ifdef USE_HALF_INTERMEDIATE
using half_type = value_h;
#else
using half_type = value;
#endif

using namespace single_source;

std::array<value_i, 4> setup(equation_context& ctx, buffer<tensor<value_us, 4>, 3> points, value_i point_count, const tensor<value_i, 4>& dim, const buffer<value_us, 3>& order_ptr)
{
    ctx.add_function("buffer_index", buffer_index_f<value, 3>);
    ctx.add_function("buffer_indexh", buffer_index_f<value_h, 3>);

    ctx.exec("int lidx = get_global_id(0)");

    value_i local_idx = "lidx";

    ctx.exec(if_s(local_idx >= point_count, return_s));

    value_i tix = points[local_idx].x().convert<int>();
    value_i tiy = points[local_idx].y().convert<int>();
    value_i tiz = points[local_idx].z().convert<int>();

    ctx.exec("int ix = " + type_to_string(tix) + ";");
    ctx.exec("int iy = " + type_to_string(tiy) + ";");
    ctx.exec("int iz = " + type_to_string(tiz) + ";");

    value_i ix = "ix";
    value_i iy = "iy";
    value_i iz = "iz";

    ///((k) * dim.x * dim.y + (j) * dim.x + (i))

    value_i c_index = iz * dim.x() * dim.y() + iy * dim.x() + ix;

    ctx.exec("int index = " + type_to_string(c_index) + ";");

    value_i index = "index";

    //ctx.exec("prefetch(&order_ptr[index], 1)");
    ctx.exec("prefetch(&cY0[index], 1)");
    ctx.exec("prefetch(&cY1[index], 1)");
    ctx.exec("prefetch(&cY2[index], 1)");
    ctx.exec("prefetch(&cY3[index], 1)");
    ctx.exec("prefetch(&cY4[index], 1)");
    ctx.exec("prefetch(&cY5[index], 1)");

    ctx.exec("prefetch(&cA0[index], 1)");
    ctx.exec("prefetch(&cA1[index], 1)");
    ctx.exec("prefetch(&cA2[index], 1)");
    ctx.exec("prefetch(&cA3[index], 1)");
    ctx.exec("prefetch(&cA4[index], 1)");
    ctx.exec("prefetch(&cA5[index], 1)");

    ctx.exec("prefetch(&cGi0[index], 1)");
    ctx.exec("prefetch(&cGi1[index], 1)");
    ctx.exec("prefetch(&cGi2[index], 1)");
    ctx.exec("prefetch(&X[index], 1)");
    ctx.exec("prefetch(&K[index], 1)");
    ctx.exec("prefetch(&gA[index], 1)");
    ctx.exec("prefetch(&gB0[index], 1)");
    ctx.exec("prefetch(&gB1[index], 1)");
    ctx.exec("prefetch(&gB2[index], 1)");

    value_i c_order = order_ptr[index].convert<int>();

    ctx.exec("int order = " + type_to_string(c_order) + ";");

    value_i order = "order";

    value_i lD_FULL = (int)D_FULL;
    value_i lD_LOW = (int)D_LOW;

    value_i is_bad = ((order & lD_FULL) == 0) && ((order & lD_LOW) == 0);

    ctx.exec(if_s(is_bad, return_s));

    return {ix, iy, iz, index};
}

template<single_source::impl::fixed_string str>
struct bssn_arg_pack
{
    std::array<named_buffer<value, 3, str + "cY">, 6> cY;
    std::array<named_buffer<value, 3, str + "cA">, 6> cA;
    std::array<named_buffer<value, 3, str + "cGi">, 3> cGi;
    named_buffer<value, 3, str + "K"> K;
    named_buffer<value, 3, str + "X"> X;
    named_buffer<value, 3, str + "gA"> gA;
    std::array<named_buffer<value, 3, str + "gB">, 3> gB;

    #ifdef USE_GBB
    std::array<named_buffer<value, 3, str + "gBB">, 3> gBB;
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
    buffer<tensor<value_us, 4>, 3> points;
    literal<value_i> point_count;

    bssn_arg_pack<""> in;
    bssn_arg_pack<"o"> out;
    bssn_arg_pack<"base_"> base;

    std::array<named_buffer<value, 3, "momentum">, 3> momentum;
    std::array<named_buffer<half_type, 3, "dcYij">, 18> dcYij; std::array<named_buffer<half_type, 3, "digA">, 3> digA;
    std::array<named_buffer<half_type, 3, "digB">, 9> digB; std::array<named_buffer<half_type, 3, "dX">, 3> dX;
    named_literal<value, "scale"> scale;
    named_literal<tensor<value_i, 4>, "dim"> dim;
    literal<value> timestep;
    named_buffer<value_us, 3, "order_ptr"> order_ptr;

    all_args(argument_generator& arg_gen, base_bssn_args& bssn_args, base_utility_args& utility_args)
    {
        arg_gen.add(points, point_count);

        arg_gen.add(bssn_args.buffers);
        arg_gen.add(std::string{"o"}, bssn_args.buffers);
        arg_gen.add(std::string{"base_"}, bssn_args.buffers);

        arg_gen.add(momentum);
        arg_gen.add(dcYij, digA, digB, dX);

        arg_gen.add(utility_args.buffers);

        arg_gen.add(scale, dim, timestep, order_ptr);
    }
};

struct exec_builder_base
{
    virtual void start(standard_arguments& args, equation_context& ctx, matter_interop& interop, bool use_matter){}
    virtual void execute(equation_context& ctx, all_args& all){}

    virtual ~exec_builder_base(){}
};

template<typename T, auto U, auto V>
struct exec_builder : exec_builder_base
{
    T dt;

    void start(standard_arguments& args, equation_context& ctx, matter_interop& interop, bool use_matter) override
    {
        dt = U(args, ctx, interop, use_matter);
    }

    void execute(equation_context& ctx, all_args& all) override
    {
        V(ctx, all, dt);
    }

    virtual ~exec_builder(){}
};

tensor<value, 6> get_dtcYij(standard_arguments& args, equation_context& ctx, matter_interop& interop, bool use_matter)
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
    #ifdef DAMP_HAMILTONIAN
    dtcYij += 0.01f * args.gA * args.cY.to_tensor() * -bssn::calculate_hamiltonian_constraint(interop, ctx, use_matter);
    #endif

    ///http://eanam6.khu.ac.kr/presentations/7-5.pdf check this
    ///makes it to 50 with this enabled
    //#define USE_DTCYIJ_MODIFICATION
    #ifdef USE_DTCYIJ_MODIFICATION
    ///https://arxiv.org/pdf/1205.5111v1.pdf 46
    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value sigma = 4/5.f;

            dtcYij.idx(i, j) += sigma * 0.5f * (gB_lower.idx(i) * bigGi_lower.idx(j) + gB_lower.idx(j) * bigGi_lower.idx(i));

            dtcYij.idx(i, j) += -(1.f/5.f) * args.cY.idx(i, j) * sum_multiply(args.gB, bigGi_lower);
        }
    }
    #endif // USE_DTCYIJ_MODIFICATION


    ///pretty sure https://arxiv.org/pdf/0711.3575v1.pdf 2.21 is equivalent, and likely a LOT faster
    //#define MOD_CY
    #ifdef MOD_CY
    tensor<value, 3, 3> cD = covariant_derivative_low_vec(ctx, bigGi_lower, args.christoff2);

    ctx.pin(cD);

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            float cK = -0.035f;

            dtcYij.idx(i, j) += cK * args.gA * 0.5f * (cD.idx(i, j) + cD.idx(j, i));
        }
    }
    #endif

    #define MOD_CY2
    #ifdef MOD_CY2
    tensor<value, 3, 3> d_cGi;

    for(int m=0; m < 3; m++)
    {
        tensor<dual, 3, 3, 3> d_dcYij;
        metric<dual, 3, 3> d_cYij;

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                d_cYij[i, j].real = args.cY[i, j];
                d_cYij[i, j].dual = args.dcYij[m, i, j];
            }
        }

        auto icY = d_cYij.invert();

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

        auto d_christoff2 = christoffel_symbols_2(icY, d_dcYij);

        tensor<dual, 3> dcGi_G;

        for(int i=0; i < 3; i++)
        {
            dual sum = 0;

            for(int j=0; j < 3; j++)
            {
                for(int k=0; k < 3; k++)
                {
                    sum += icY[j, k] * d_christoff2[i, j, k];
                }
            }

            dcGi_G[i] = sum;
        }

        for(int i=0; i < 3; i++)
        {
            d_cGi[m, i] = diff1(ctx, args.cGi[i], m) - dcGi_G[i].dual;
        }
    }

    tensor<value, 3, 3> cD = covariant_derivative_high_vec(ctx, args.bigGi, d_cGi, args.christoff2);

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value sum = 0;

            for(int k=0; k < 3; k++)
            {
                sum += 0.5f * (args.cY[k, i] * cD[k, j] + args.cY[k, j] * cD[k, i]);
            }

            float cK = -0.035f;

            dtcYij.idx(i, j) += cK * args.gA * sum;
        }
    }
    #endif // MOD_CY2

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
    value_i index = "index";

    for(int i=0; i < 6; i++)
    {
        ctx.exec(assign(all.out.cY[i][index], all.base.cY[i][index] + all.timestep * dtcY[i]));
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
            value p1 = W * didjW.idx(i, j);

            xgARphiij.idx(i, j) = args.gA * (p1 + args.cY.idx(i, j) * (p2 + p3));
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

tensor<value, 6> get_dtcAij(standard_arguments& args, equation_context& ctx, matter_interop& interop, bool use_matter)
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
                    value v = icY.idx(m, n) * dX.idx(m) * diff1(ctx, gA, n);

                    s3 += v;
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

    #ifdef BETTERDAMP_DTCAIJ
    tensor<value, 3, 3> momentum_deriv;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            momentum_deriv.idx(i, j) = diff1(ctx, args.momentum_constraint.idx(i), j);
        }
    }

    tensor<value, 3, 3> symmetric_momentum_deriv;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            symmetric_momentum_deriv.idx(i, j) = 0.5f * (momentum_deriv.idx(i, j) + momentum_deriv.idx(j, i));
        }
    }

    ctx.pin(symmetric_momentum_deriv);

    #endif // BETTERDAMP_DTCAIJ

    #ifdef AIJ_SIGMA
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

    tensor<value, 3, 3> BiMj_TF = trace_free(BiMj, cY, icY);
    #endif // AIJ_SIGMA

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

            #ifdef DAMP_DTCAIJ
            float Ka = 0.01f;

            dtcAij.idx(i, j) += Ka * gA * 0.5f *
                                                (covariant_derivative_low_vec(ctx, args.momentum_constraint, args.unpinned_cY, icY).idx(i, j)
                                                 + covariant_derivative_low_vec(ctx, args.momentum_constraint, args.unpinned_cY, icY).idx(j, i));
            #endif // DAMP_DTCAIJ

            #ifdef BETTERDAMP_DTCAIJ
            value F_a = scale * gA;

            ///https://arxiv.org/pdf/1205.5111v1.pdf (56)
            dtcAij.idx(i, j) += scale * F_a * trace_free(symmetric_momentum_deriv, cY, icY).idx(i, j);
            #endif // BETTERDAMP_DTCAIJ

            #ifdef AIJ_SIGMA
            float sigma = 0.25f;

            dtcAij.idx(i, j) += (-3.f/5.f) * sigma * BiMj_TF.idx(i, j);
            #endif // AIJ_SIGMA

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

    #ifdef DAMP_HAMILTONIAN
    dtcAij += -0.5f * args.gA * args.cA * -bssn::calculate_hamiltonian_constraint(interop, ctx, use_matter);
    #endif

    //#define MOD_CA
    #ifdef MOD_CA
    ///https://arxiv.org/pdf/gr-qc/0204002.pdf 4.3
    value bigGi_diff = 0;

    for(int i=0; i < 3; i++)
    {
        bigGi_diff += diff1(ctx, args.bigGi.idx(i), i);
    }

    float k8 = 1.f;

    dtcAij += -k8 * args.gA * args.get_X() * args.cY.to_tensor() * bigGi_diff;

    #endif // MOD_CA

    ///https://arxiv.org/pdf/gr-qc/0204002.pdf todo: 4.3

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
    value_i index = "index";

    //ctx.pin(dtcA);

    for(int i=0; i < 6; i++)
    {
        ctx.exec(assign(all.out.cA[i][index], all.base.cA[i][index] + all.timestep * dtcA[i]));
    }
}

exec_builder<tensor<value, 6>, get_dtcAij, finish_cA> cAexec;

tensor<value, 3> get_dtcGi(standard_arguments& args, equation_context& ctx, matter_interop& interop, bool use_matter)
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

    ///PAPER_12055111_SUBST

    tensor<value, 3> Yij_Kj;

    #define PAPER_1205_5111
    #ifdef PAPER_1205_5111
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
    #else
    for(int i=0; i < 3; i++)
    {
        value sum = 0;

        for(int j=0; j < 3; j++)
        {
            sum += icY.idx(i, j) * diff1(ctx, args.K, j);
        }

        Yij_Kj.idx(i) = sum;
    }
    #endif // PAPER_1205_5111

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
            s3 += 2 * (-1.f/4.f) * gA / max(args.W_impl, 0.0001f) * 6 * icAij.idx(i, j) * 2 * args.dW_calc.idx(j);
        }
        #endif

        value s4 = 0;

        for(int j=0; j < 3; j++)
        {
            s4 += -2 * icAij.idx(i, j) * diff1(ctx, gA, j);
        }

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
                s8 += (1.f/3.f) * icY.idx(i, j) * diff2(ctx, args.gB.idx(k), k, j, args.digB.idx(k, k), args.digB.idx(j, k));
            }
        }

        value s9 = 0;

        for(int k=0; k < 3; k++)
        {
            s9 += (2.f/3.f) * args.digB.idx(k, k) * derived_cGi.idx(i);
        }

        ///this is the only instanced of derived_cGi that might want to be regular cGi
        //value s10 = (2.f/3.f) * -2 * gA * K * derived_cGi.idx(i);

        dtcGi.idx(i) = s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9;

        ///https://arxiv.org/pdf/1205.5111v1.pdf 50
        ///made it to 70+ and then i got bored, but the simulation was meaningfully different
        #define EQ_50
        #ifdef EQ_50

        auto step = [](const value& in)
        {
            return if_v(in >= 0.f, value{1.f}, value{0.f});
        };

        value bkk = 0;

        for(int k=0; k < 3; k++)
        {
            bkk += args.digB.idx(k, k);
        }

        float E = 1;

        value lambdai = (2.f/3.f) * (bkk - 2 * gA * K)
                        - args.digB.idx(i, i)
                        - (2.f/5.f) * gA * raise_index(cA, icY, 1).idx(i, i);

        dtcGi.idx(i) += -(1 + E) * step(lambdai) * lambdai * args.bigGi.idx(i);
        #endif // EQ_50

        ///todo: test 2.22 https://arxiv.org/pdf/0711.3575.pdf
        //#define YBS
        #ifdef YBS
        value E = 1;

        {
            value sum = 0;

            for(int k=0; k < 3; k++)
            {
                sum += diff1(ctx, args.gB.idx(k), k);
            }

            dtcGi.idx(i) += (-2.f/3.f) * (E + 1) * args.bigGi.idx(i) * sum;
        }
        #endif // YBS

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

    ///https://arxiv.org/pdf/gr-qc/0204002.pdf table 2, think case E2 is incorrectly labelled
    //#define MOD_CGI
    #ifdef MOD_CGI
    float mcGicst = -0.1f;

    dtcGi += mcGicst * gA * args.bigGi;
    #endif // MOD_CGI

    //ctx.pin(dtcGi);

    return dtcGi;
}

void finish_cGi(equation_context& ctx, all_args& all, tensor<value, 3>& dtcGi)
{
    value_i index = "index";

    for(int i=0; i < 3; i++)
    {
        ctx.exec(assign(all.out.cGi[i][index], all.base.cGi[i][index] + all.timestep * dtcGi[i]));
    }
}

exec_builder<tensor<value, 3>, get_dtcGi, finish_cGi> cGiexec;

value get_dtK(standard_arguments& args, equation_context& ctx, matter_interop& interop, bool use_matter)
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
                    value v = -0.5f * cY.idx(i, j) * icY.idx(m, n) * dX.idx(m) * diff1(ctx, gA, n);

                    s3 += v;
                }
            }

            Xdidja.idx(i, j) = Xderiv + s2 + s3;
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
    value_i index = "index";

    ctx.exec(assign(all.out.K[index], all.base.K[index] + all.timestep * dtK));
}

exec_builder<value, get_dtK, finish_K> Kexec;

value get_dtX(standard_arguments& args, equation_context& ctx, matter_interop& interop, bool use_matter)
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
    value_i index = "index";

    ctx.exec(assign(all.out.X[index], all.base.X[index] + all.timestep * dtX));
}

exec_builder<value, get_dtX, finish_X> Xexec;

value get_dtgA(standard_arguments& args, equation_context& ctx, matter_interop& interop, bool use_matter)
{
    ///https://arxiv.org/pdf/gr-qc/0206072.pdf (94) is bad
    value dtgA = lie_derivative(ctx, args.gB, args.gA) * 1 - 2 * args.gA * args.K;

    /*value dibi = 0;

    for(int i=0; i < 3; i++)
    {
        dibi += diff1(ctx, args.gB.idx(i), i);
    }*/

    ///shock
    ///-a^2 f(a) A
    ///f(a) = (8/3)/(a(3 - a))
    ///-a * (8/3) * A / (3 - a)

    //value dtgA = lie_derivative(ctx, args.gB, args.gA) + dibi * 0 - args.gA * (8.f/3.f) * args.K / (3 - args.gA);

    //dtgA = 0;

    //ctx.pin(dtgA);

    return dtgA;
}

void finish_gA(equation_context& ctx, all_args& all, value& dtgA)
{
    value_i index = "index";

    ctx.exec(assign(all.out.gA[index], all.base.gA[index] + all.timestep * dtgA));
}

exec_builder<value, get_dtgA, finish_gA> gAexec;

tensor<value, 3> get_dtgB(standard_arguments& args, equation_context& ctx, matter_interop& interop, bool use_matter)
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
    value Ns_r = 2.f;
    #endif

    value N = max(Ns_r, 0.5f);

    #ifndef USE_GBB
    ///https://arxiv.org/pdf/gr-qc/0605030.pdf 26
    ///todo: remove this
    tensor<value, 3> dtgB = (3.f/4.f) * args.derived_cGi + bjdjbi * 1 - N * args.gB;

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
    value_i index = "index";

    for(int i=0; i < 3; i++)
    {
        ctx.exec(assign(all.out.gB[i][index], all.base.gB[i][index] + all.timestep * dtgB[i]));
    }
}

exec_builder<tensor<value, 3>, get_dtgB, finish_gB> gBexec;

void build_kernel(argument_generator& arg_gen, equation_context& ctx, matter_interop& interop, bool use_matter, base_bssn_args& bssn_args, base_utility_args& utility_args, std::vector<exec_builder_base*> execs)
{
    all_args all(arg_gen, bssn_args, utility_args);

    (void)setup(ctx, all.points, all.point_count.get(), all.dim.get(), all.order_ptr);

    standard_arguments args(ctx);

    for(int i=0; i < (int)execs.size(); i++)
    {
        execs[i]->start(args, ctx, interop, use_matter);
        execs[i]->execute(ctx, all);
    }

    ctx.fix_buffers();
}

void get_raytraced_quantities(argument_generator& arg_gen, equation_context& ctx, base_bssn_args& bssn_args)
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

void trace_slice(equation_context& ctx,
                 std::array<buffer<value, 3>, 6> linear_Yij_1, std::array<buffer<value, 3>, 6> linear_Kij_1, buffer<value, 3> linear_gA_1, std::array<buffer<value, 3>, 3> linear_gB_1,
                 std::array<buffer<value, 3>, 6> linear_Yij_2, std::array<buffer<value, 3>, 6> linear_Kij_2, buffer<value, 3> linear_gA_2, std::array<buffer<value, 3>, 3> linear_gB_2,
                 named_literal<value, "scale"> scale, named_literal<v4i, "dim"> dim,
                 std::array<buffer<value>, 3> positions, std::array<buffer<value>, 3> velocities,
                 std::array<buffer<value>, 3> positions_out, std::array<buffer<value>, 3> velocities_out, literal<value_i> ray_count, literal<value> frac, literal<value> slice_width, literal<value> step)
{
    ctx.add_function("buffer_index", buffer_index_f<value, 3>);
    ctx.add_function("buffer_indexh", buffer_index_f<value_h, 3>);
    ctx.add_function("buffer_read_linear", buffer_read_linear_f<value, 3>);

    ctx.ignored_variables.push_back(frac.name);

    ctx.order = 1;
    ctx.uses_linear = true;

    ctx.exec("int lidx = get_global_id(0)");

    value_i lidx = "lidx";

    ctx.exec(if_s(lidx >= ray_count, return_s));

    v3f pos = {positions[0][lidx], positions[1][lidx], positions[2][lidx]};
    v3f vel = {velocities[0][lidx], velocities[1][lidx], velocities[2][lidx]};

    auto w2v = [&](v3f in)
    {
        v3i centre = (dim.get().xyz() - 1)/2;

        return (in / scale) + (v3f)centre;
    };

    v3f voxel_pos = w2v(pos);

    value universe_length = ((dim.get().x()-1)/2).convert<float>();

    int index_table[3][3] = {{0, 1, 2},
                             {1, 3, 4},
                             {2, 4, 5}};

    metric<value, 3, 3> Yij;
    tensor<value, 3, 3> Kij;
    tensor<value, 3> gB;
    value gA;

    value steps = floor(slice_width.get() / step.get());

    v3f loop_pos = declare(ctx, pos);
    v3f loop_vel = declare(ctx, vel);

    ctx.position_override = {type_to_string(loop_pos[0]), type_to_string(loop_pos[1]), type_to_string(loop_pos[2])};

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            int tidx = index_table[i][j];

            Yij[i, j] = mix(buffer_index_generic(linear_Yij_1[tidx], loop_pos, dim.name), buffer_index_generic(linear_Yij_2[tidx], loop_pos, dim.name), frac.get());
            Kij[i, j] = mix(buffer_index_generic(linear_Kij_1[tidx], loop_pos, dim.name), buffer_index_generic(linear_Kij_2[tidx], loop_pos, dim.name), frac.get());
        }

        gB[i] = mix(buffer_index_generic(linear_gB_1[i], loop_pos, dim.name), buffer_index_generic(linear_gB_2[i], loop_pos, dim.name), frac.get());
    }

    gA = mix(buffer_index_generic(linear_gA_1, loop_pos, dim.name), buffer_index_generic(linear_gA_2, loop_pos, dim.name), frac.get());

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

    ctx.exec("for(int i=0; i < " + type_to_string(steps) + "; i++) {");

    {
        v3f dpos = declare(ctx, dx);
        v3f dvel = declare(ctx, V_upper_diff);

        ctx.exec(assign(loop_pos, loop_pos + dpos * step.get()));
        ctx.exec(assign(loop_vel, loop_vel + dvel * step.get()));
    }

    ctx.exec("}");

    ctx.exec(assign(positions_out[0][lidx], loop_pos.x()));
    ctx.exec(assign(positions_out[1][lidx], loop_pos.y()));
    ctx.exec(assign(positions_out[2][lidx], loop_pos.z()));

    ctx.exec(assign(velocities_out[0][lidx], loop_vel.x()));
    ctx.exec(assign(velocities_out[1][lidx], loop_vel.y()));
    ctx.exec(assign(velocities_out[2][lidx], loop_vel.z()));
}

void bssn::build(cl::context& clctx, matter_interop& interop, bool use_matter, base_bssn_args& bssn_args, base_utility_args& utility_args)
{
    {
        std::vector<exec_builder_base*> b = {&cAexec, &Xexec, &Kexec, &gAexec, &gBexec, &cYexec, &cGiexec};

        equation_context ectx;

        cl::kernel kern = single_source::make_dynamic_kernel_for(clctx, ectx, build_kernel, "evolve_1", "", interop, use_matter, bssn_args, utility_args, b);

        clctx.register_kernel("evolve_1", kern);
    }

    {
        equation_context ectx;

        cl::kernel kern = single_source::make_dynamic_kernel_for(clctx, ectx, get_raytraced_quantities, "get_raytraced_quantities", "", bssn_args);

        clctx.register_kernel("get_raytraced_quantities", kern);
    }

    /*{
        equation_context ectx;

        cl::kernel kern = single_source::make_kernel_for(clctx, ectx, init_slice_rays, "init_slice_rays", "");

        clctx.register_kernel("init_slice_rays", kern);
    }*/

    /*{
        equation_context ectx;

        cl::kernel kern = single_source::make_kernel_for(clctx, ectx, trace_slice, "trace_slice", "");

        clctx.register_kernel("trace_slice", kern);
    }*/
}
