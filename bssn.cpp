#include "bssn.hpp"

void bssn::init(equation_context& ctx, const metric<value, 3, 3>& Yij, const tensor<value, 3, 3>& Aij, const value& gA)
{
    vec2i linear_indices[6] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};

    value Y = Yij.det();

    value conformal_factor = (1/12.f) * log(Y);

    ctx.pin(conformal_factor);

    value gB0 = 0;
    value gB1 = 0;
    value gB2 = 0;

    tensor<value, 3> cGi;
    value K = 0;

    ///https://arxiv.org/pdf/gr-qc/0206072.pdf (58)

    value X = exp(-4 * conformal_factor);

    tensor<value, 3, 3> cAij = X * Aij;

    tensor<value, 3, 3> cYij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            cYij.idx(i, j) = (i == j) ? 1 : 0;
        }
    }

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
}

void bssn::build_cY(equation_context& ctx)
{
    standard_arguments args(ctx);

    metric<value, 3, 3> unpinned_cY = args.cY;

    ctx.pin(args.cY);

    tensor<value, 3> bigGi_lower = lower_index(args.bigGi, args.cY, 0);
    tensor<value, 3> gB_lower = lower_index(args.gB, args.cY, 0);

    ctx.pin(bigGi_lower);
    ctx.pin(gB_lower);

    tensor<value, 3, 3> lie_cYij = lie_derivative_weight(ctx, args.gB, unpinned_cY);

    ///https://arxiv.org/pdf/gr-qc/0511048.pdf (1)
    ///https://scholarworks.rit.edu/cgi/viewcontent.cgi?article=11286&context=theses 3.66
    tensor<value, 3, 3> dtcYij = -2 * args.gA * args.cA + lie_cYij;

    ///makes it to 50 with this enabled
    #define USE_DTCYIJ_MODIFICATION
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

    for(int i=0; i < 6; i++)
    {
        std::string name = "dtcYij" + std::to_string(i);

        vec2i idx = args.linear_indices[i];

        ctx.add(name, dtcYij.idx(idx.x(), idx.y()));
    }
}

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
                    s1 = s1 + -0.5f * icY.idx(l, m) * diff2(ctx, args.cY.idx(i, j), m, l, args.dcYij.idx(m, i, j), args.dcYij.idx(l, i, j));
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

    tensor<value, 3, 3> cov_div_X = double_covariant_derivative(ctx, args.X, args.dX, args.cY, icY, christoff2);
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
                    s2 += icY.idx(m, n) * args.dX.idx(m) * args.dX.idx(n);
                }
            }

            value s3 = (1/2.f) * (args.gA * cov_div_X.idx(j, i) - gA_X * (1/2.f) * args.dX.idx(i) * args.dX.idx(j));

            s1 = args.gA * (args.cY.idx(i, j) / 2.f) * s1;
            s2 = gA_X * (args.cY.idx(i, j) / 2.f) * -(3.f/2.f) * s2;

            xgARphiij.idx(i, j) = s1 + s2 + s3;
        }
    }

    tensor<value, 3, 3> xgARij = xgARphiij + args.X * args.gA * cRij;

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

    tensor<value, 3, 3, 3> christoff1 = christoffel_symbols_1(ctx, args.cY);

    tensor<value, 3, 3> xgARij = bssn::calculate_xgARij(ctx, args, icY, christoff1, args.christoff2);

    return calculate_hamiltonian(args.cY, icY, args.Yij, args.iYij, (xgARij / (max(args.X, 0.001f) * args.gA)), args.K, args.cA);
}

void bssn::build_cA(matter_interop& interop, equation_context& ctx, bool use_matter)
{
    standard_arguments args(ctx);

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

    unit_metric<value, 3, 3> cY = args.cY;

    inverse_metric<value, 3, 3> unpinned_icY = cY.invert();

    ctx.pin(icY);

    tensor<value, 3, 3> cA = args.cA;

    auto unpinned_cA = cA;

    ///the christoffel symbol
    tensor<value, 3> cGi = args.cGi;

    value gA = args.gA;

    tensor<value, 3> gB = args.gB;

    value X = args.X;
    value K = args.K;

    tensor<value, 3> derived_cGi = args.derived_cGi;

    tensor<value, 3, 3> xgARij = bssn::calculate_xgARij(ctx, args, icY, christoff1, christoff2);

    ctx.pin(xgARij);

    tensor<value, 3, 3> Xdidja;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value Xderiv = X * double_covariant_derivative(ctx, args.gA, args.digA, cY, icY, args.christoff2).idx(j, i);
            //value Xderiv = X * gpu_covariant_derivative_low_vec(ctx, args.digA, cY, icY).idx(j, i);

            value s2 = 0.5f * (diff1(ctx, X, i) * diff1(ctx, gA, j) + diff1(ctx, X, j) * diff1(ctx, gA, i));

            value s3 = 0;

            for(int m=0; m < 3; m++)
            {
                for(int n=0; n < 3; n++)
                {
                    value v = icY.idx(m, n) * diff1(ctx, X, m) * diff1(ctx, gA, n);

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
            float Ka = 0.0001f;

            dtcAij.idx(i, j) += Ka * gA * 0.5f *
                                                (gpu_covariant_derivative_low_vec(ctx, args.momentum_constraint, cY, icY).idx(i, j)
                                                 + gpu_covariant_derivative_low_vec(ctx, args.momentum_constraint, cY, icY).idx(j, i));
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

                tensor<value, 3, 3> xgASij = trace_free(-8 * M_PI * gA * xSij, cY, icY);

                //ctx.add("DBGXGA", xgASij.idx(0, 0));
                //ctx.add("Debug_cS0", args.matt.cS.idx(0));

                dtcAij.idx(i, j) += xgASij.idx(i, j);
            }
        }
    }

    for(int i=0; i < 6; i++)
    {
        std::string name = "dtcAij" + std::to_string(i);

        vec2i idx = args.linear_indices[i];

        ctx.add(name, dtcAij.idx(idx.x(), idx.y()));
    }
}

void bssn::build_cGi(matter_interop& interop, equation_context& ctx, bool use_matter)
{
    standard_arguments args(ctx);

    inverse_metric<value, 3, 3> icY = args.cY.invert();

    tensor<value, 3, 3, 3> christoff2 = args.christoff2;

    ctx.pin(christoff2);

    unit_metric<value, 3, 3> cY = args.cY;

    inverse_metric<value, 3, 3> unpinned_icY = cY.invert();

    tensor<value, 3, 3> cA = args.cA;

    ///the christoffel symbol
    tensor<value, 3> cGi = args.cGi;

    value gA = args.gA;

    tensor<value, 3> gB = args.gB;

    value X = args.X;
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
    tensor<value, 3, 3> littlekij = unpinned_icY.to_tensor() * K;

    ///PAPER_12055111_SUBST

    tensor<value, 3> Yij_Kj;

    #define PAPER_1205_5111
    #ifdef PAPER_1205_5111
    for(int i=0; i < 3; i++)
    {
        value sum = 0;

        for(int j=0; j < 3; j++)
        {
            sum += diff1(ctx, littlekij.idx(i, j), j);
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

        value s3 = 0;

        for(int j=0; j < 3; j++)
        {
            s3 += 2 * (-1.f/4.f) * gA_X * 6 * icAij.idx(i, j) * diff1(ctx, X, j);
        }

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
            return dual_if(in >= 0,
            [](){return 1;},
            [](){return 0;});

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

            dtcGi.idx(i) += gA * -2 * 8 * M_PI * sum;
        }
    }
    #endif // CHRISTOFFEL_49

    for(int i=0; i < 3; i++)
    {
        std::string name = "dtcGi" + std::to_string(i);

        ctx.add(name, dtcGi.idx(i));
    }
}

void bssn::build_K(matter_interop& interop, equation_context& ctx, bool use_matter)
{
    standard_arguments args(ctx);

    inverse_metric<value, 3, 3> icY = args.cY.invert();

    unit_metric<value, 3, 3> cY = args.cY;

    ctx.pin(icY);

    tensor<value, 3, 3> cA = args.cA;

    ///the christoffel symbol
    tensor<value, 3> cGi = args.cGi;

    value gA = args.gA;

    tensor<value, 3> gB = args.gB;

    value X = args.X;
    value K = args.K;

    tensor<value, 3, 3> Xdidja;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value Xderiv = X * double_covariant_derivative(ctx, args.gA, args.digA, cY, icY, args.christoff2).idx(j, i);
            //value Xderiv = X * gpu_covariant_derivative_low_vec(ctx, args.digA, cY, icY).idx(j, i);

            value s2 = 0.5f * (diff1(ctx, X, i) * diff1(ctx, gA, j) + diff1(ctx, X, j) * diff1(ctx, gA, i));

            value s3 = 0;

            for(int m=0; m < 3; m++)
            {
                for(int n=0; n < 3; n++)
                {
                    value v = -0.5f * cY.idx(i, j) * icY.idx(m, n) * diff1(ctx, X, m) * diff1(ctx, gA, n);

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

        dtK += (8 * M_PI / 2) * gA * (matter_s + matter_p);

        /*value h = calculate_h_with_gamma_eos(chi, W);
        value em6phi = chi_to_e_m6phi(chi);

        value p0 = calculate_p0(chi, W);
        value eps = calculate_eps(chi, W);

        return h * W * em6phi - gamma_eos(p0, eps);*/

        /*value h = args.matt.calculate_h_with_gamma_eos(X, args.matt.stashed_W);
        value em6phi = chi_to_e_m6phi(X);
        value p0 = args.matt.calculate_p0(X, args.matt.stashed_W);
        value eps = args.matt.calculate_eps(X, args.matt.stashed_W);

        ctx.add("Dbg_matter_s", matter_s);
        ctx.add("Dbg_matter_p", matter_p);

        ctx.add("Dbg_h", h);
        ctx.add("Dbg_em", em6phi);
        ctx.add("Dbg_p0", p0);
        ctx.add("Dbg_eps", eps);
        ctx.add("Dbg_X", X);*/
    }

    ctx.add("dtK", dtK);
}

void bssn::build_X(equation_context& ctx)
{
    standard_arguments args(ctx);

    tensor<value, 3> linear_dB;

    for(int i=0; i < 3; i++)
    {
        linear_dB.idx(i) = diff1(ctx, args.gB.idx(i), i);
    }

    value dtX = (2.f/3.f) * args.X * (args.gA * args.K - sum(linear_dB)) + sum(tensor_upwind(ctx, args.gB, args.X));

    ctx.add("dtX", dtX);
}

void bssn::build_gA(equation_context& ctx)
{
    standard_arguments args(ctx);

    //value bl_s = "(init_BL_val)";
    //value bl = bl_s + 1;

    ///https://arxiv.org/pdf/gr-qc/0206072.pdf (94)
    ///this breaks immediately
    //int m = 4;
    //value dtgA = lie_derivative(ctx, args.gB, args.gA) - 2 * args.gA * args.K * pow(bl, m);

    value dtgA = lie_derivative(ctx, args.gB, args.gA) - 2 * args.gA * args.K;

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

    ctx.add("dtgA", dtgA);
}

void bssn::build_gB(equation_context& ctx)
{
    standard_arguments args(ctx);

    inverse_metric<value, 3, 3> icY = args.cY.invert();

    value X = args.X;

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
    value Ns_r = 2;
    #endif

    value N = max(Ns_r, 0.5f);

    #ifndef USE_GBB
    ///https://arxiv.org/pdf/gr-qc/0605030.pdf 26
    ///todo: remove this
    tensor<value, 3> dtgB = (3.f/4.f) * args.derived_cGi + bjdjbi - N * args.gB;

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

    //#endif // PAPER_0610128
    #endif // USE_GBB

    for(int i=0; i < 3; i++)
    {
        std::string name = "dtgB" + std::to_string(i);

        ctx.add(name, dtgB.idx(i));
    }

    for(int i=0; i < 3; i++)
    {
        std::string name = "dtgBB" + std::to_string(i);

        ctx.add(name, dtgBB.idx(i));
    }
}

struct ccz4_args
{
    value gA;
    tensor<value, 3> gB;
    tensor<value, 3> gBB;

    unit_metric<value, 3, 3> cY;
    tensor<value, 3, 3> cA;

    value W;
    value K;
    value theta;

    tensor<value, 3> cGi_hat;

    tensor<value, 3, 3, 3> dcYij;
    tensor<value, 3, 3> digB;
    tensor<value, 3> digA;
    tensor<value, 3> dW;

    inverse_metric<value, 3, 3> iYij;

    vec2i linear_indices[6] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};

    int index_table[3][3] = {{0, 1, 2},
                             {1, 3, 4},
                             {2, 4, 5}};

    ccz4_args(equation_context& ctx)
    {
        bool interpolate = ctx.uses_linear;

        gA = bidx("gA", interpolate, false);

        gA = max(gA, 0.f);

        gB.idx(0) = bidx("gB0", interpolate, false);
        gB.idx(1) = bidx("gB1", interpolate, false);
        gB.idx(2) = bidx("gB2", interpolate, false);

        gBB.idx(0) = bidx("gBB0", interpolate, false);
        gBB.idx(1) = bidx("gBB1", interpolate, false);
        gBB.idx(2) = bidx("gBB2", interpolate, false);

        cGi_hat.idx(0) = bidx("cGi0", interpolate, false);
        cGi_hat.idx(1) = bidx("cGi1", interpolate, false);
        cGi_hat.idx(2) = bidx("cGi2", interpolate, false);

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

                cY.idx(i, j) = bidx("cY" + std::to_string(index), interpolate, false);
            }
        }

        //cY.idx(2, 2) = (1 + cY.idx(1, 1) * cY.idx(0, 2) * cY.idx(0, 2) - 2 * cY.idx(0, 1) * cY.idx(1, 2) * cY.idx(0, 2) + cY.idx(0, 0) * cY.idx(1, 2) * cY.idx(1, 2)) / (cY.idx(0, 0) * cY.idx(1, 1) - cY.idx(0, 1) * cY.idx(0, 1));

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                int index = arg_table[i * 3 + j];

                cA.idx(i, j) = bidx("cA" + std::to_string(index), interpolate, false);
            }
        }

        W = bidx("X", interpolate, false);
        K = bidx("K", interpolate, false);

        theta = bidx("constraint_theta", interpolate, false);

        for(int k=0; k < 3; k++)
        {
            for(int i=0; i < 3; i++)
            {
                for(int j=0; j < 3; j++)
                {
                    int symmetric_index = index_table[i][j];

                    int final_index = k + symmetric_index * 3;

                    dcYij.idx(k, i, j) = bidx("dcYij" + std::to_string(final_index), interpolate, true);
                }
            }
        }

        digA.idx(0) = bidx("digA0", interpolate, true);
        digA.idx(1) = bidx("digA1", interpolate, true);
        digA.idx(2) = bidx("digA2", interpolate, true);

        dW.idx(0) = bidx("dX0", interpolate, true);
        dW.idx(1) = bidx("dX1", interpolate, true);
        dW.idx(2) = bidx("dX2", interpolate, true);

        ///derivative
        for(int i=0; i < 3; i++)
        {
            ///value
            for(int j=0; j < 3; j++)
            {
                int idx = i + j * 3;

                digB.idx(i, j)  = bidx("digB" + std::to_string(idx), interpolate, true);
            }
        }

        iYij = W * W * cY.invert();
    }

    tensor<value, 3> get_cGi(equation_context& ctx)
    {
        /*inverse_metric<value, 3, 3> icY = cY.invert();

        tensor<value, 3, 3, 3> lchristoff2 = christoffel_symbols_2(ctx, cY, icY);

        tensor<value, 3> cGi;

        for(int i=0; i < 3; i++)
        {
            value sum = 0;

            for(int j=0; j < 3; j++)
            {
                for(int k=0; k < 3; k++)
                {
                    sum += icY.idx(j, k) * lchristoff2.idx(i, j, k);
                }
            }

            cGi.idx(i) = sum;
        }

        return cGi;*/

        inverse_metric<value, 3, 3> icY = cY.invert();

        tensor<value, 3> cGi;

        for(int i=0; i < 3; i++)
        {
            value sum = 0;

            for(int j=0; j < 3; j++)
            {
                for(int k=0; k < 3; k++)
                {
                    for(int l = 0; l < 3; l++)
                    {
                        sum += icY.idx(i, j) * icY.idx(k, l) * dcYij.idx(l, j, k);
                    }
                }
            }

            cGi.idx(i) = sum;
        }

        return cGi;
    }

    ///https://arxiv.org/pdf/1307.7391.pdf (11)
    tensor<value, 3> get_cYij_Zj(equation_context& ctx)
    {
        tensor<value, 3> cGi = get_cGi(ctx);

        return (cGi_hat - cGi) / 2.f;
    }

    ///https://inspirehep.net/files/6d0b069dcb3f4d485539aa0cba4f67e2
    tensor<value, 3> get_Zi_raised(equation_context& ctx)
    {
        ///so, we're 'raising' Zj by the conformal metric
        ///to actually raise it, we need to multiply by w^2
        tensor<value, 3> cYij_Zj = get_cYij_Zj(ctx);

        return W * W * cYij_Zj;
    }

    tensor<value, 3> get_Zi_lowered(equation_context& ctx)
    {
        ///well, we could lower get_Zi_raised
        ///which would involve summing by (cY / W^2)
        ///which would cancel the W^2 above
        tensor<value, 3> cYij_Zj = get_cYij_Zj(ctx);

        tensor<value, 3> result;

        for(int i=0; i < 3; i++)
        {
            value sum = 0;

            for(int j=0; j < 3; j++)
            {
                sum += cY.idx(i, j) * cYij_Zj.idx(j);
            }

            result.idx(i) = sum;
        }

        return result;
    }
};

void ccz4::init(equation_context& ctx, const metric<value, 3, 3>& Yij, const tensor<value, 3, 3>& Aij, const value& gA)
{
    vec2i linear_indices[6] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};

    value Y = Yij.det();

    value gB0 = 0;
    value gB1 = 0;
    value gB2 = 0;

    tensor<value, 3> cGi;
    value K = 0;

    ///https://arxiv.org/pdf/gr-qc/0206072.pdf (58)

    value W = pow(Y, -1.f/6.f);

    tensor<value, 3, 3> cAij = W * W * Aij;
    metric<value, 3, 3> cYij = W * W * Yij;

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
    ctx.add("init_X", W);

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

    ctx.add("USE_THETA", 1);
    ctx.add("init_constraint_theta", 0);
}

tensor<value, 3, 3> calculate_Rij(equation_context& ctx, ccz4_args& args)
{
    tensor<value, 3, 3> cRij;

    inverse_metric<value, 3, 3> icY = args.cY.invert();

    tensor<value, 3> cGi = args.get_cGi(ctx);

    tensor<value, 3, 3, 3> christoff1 = christoffel_symbols_1(ctx, args.cY);
    tensor<value, 3, 3, 3> christoff2 = christoffel_symbols_2(ctx, args.cY, icY);

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value s1 = 0;

            for(int l=0; l < 3; l++)
            {
                for(int m=0; m < 3; m++)
                {
                    s1 = s1 + -0.5f * icY.idx(l, m) * diff2(ctx, args.cY.idx(i, j), m, l, args.dcYij.idx(m, i, j), args.dcYij.idx(l, i, j));
                }
            }

            value s2 = 0;

            for(int k=0; k < 3; k++)
            {
                s2 = s2 + 0.5f * (args.cY.idx(k, i) * diff1(ctx, cGi.idx(k), j) + args.cY.idx(k, j) * diff1(ctx, cGi.idx(k), i));
            }

            value s3 = 0;

            for(int k=0; k < 3; k++)
            {
                s3 = s3 + 0.5f * cGi.idx(k) * (christoff1.idx(i, j, k) + christoff1.idx(j, i, k));
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

    value i_W_sq = 1/max(args.W * args.W, 0.0001f);

    tensor<value, 3, 3> didjW;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            ///dcd uses the notation i;j
            didjW.idx(i, j) = double_covariant_derivative(ctx, args.W, args.dW, args.cY, icY, christoff2).idx(j, i);
        }
    }

    tensor<value, 3> dW = args.dW;

    tensor<value, 3, 3> rphiij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value i1 = didjW.idx(i, j);

            value s2 = 0;

            for(int l=0; l < 3; l++)
            {
                s2 += raise_index(didjW, icY, 0).idx(l, l);
            }

            value i2 = args.cY.idx(i, j) * s2;

            value left = args.W * (i1 + i2);

            value right_sum = sum_multiply(raise_index(dW, icY, 0), dW);

            value right = -2 * args.cY.idx(i, j) * right_sum;

            value full = i_W_sq * (left + right);

            rphiij.idx(i, j) = full;
        }
    }

    return cRij + rphiij;
}

namespace
{
    value W_to_X(const value& W)
    {
        return W*W;
    }

    tensor<value, 3> dW_to_dX(const value& W, const tensor<value, 3>& dW)
    {
        return 2 * W * dW;
    }
}

void ccz4::build_cY(equation_context& ctx)
{
    ccz4_args args(ctx);

    tensor<value, 3, 3> lie_cYij = lie_derivative_weight(ctx, args.gB, args.cY);

    ///https://arxiv.org/pdf/gr-qc/0511048.pdf (1)
    ///https://scholarworks.rit.edu/cgi/viewcontent.cgi?article=11286&context=theses 3.66
    tensor<value, 3, 3> dtcYij = -2 * args.gA * trace_free(args.cA, args.cY, args.cY.invert()) + lie_cYij;

    for(int i=0; i < 6; i++)
    {
        std::string name = "dtcYij" + std::to_string(i);

        vec2i idx = args.linear_indices[i];

        ctx.add(name, dtcYij.idx(idx.x(), idx.y()));
    }
}

tensor<value, 3, 3, 3> get_full_christoffel2(const value& W, const tensor<value, 3>& dW, const unit_metric<value, 3, 3>& cY, const tensor<value, 3, 3, 3>& christoff2)
{
    value X = W_to_X(W);
    tensor<value, 3> dX = dW_to_dX(W, dW);

    value clamped_X = max(X, 0.0001f);

    inverse_metric<value, 3, 3> icY = cY.invert();

    tensor<value, 3, 3, 3> result = christoff2;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                value rhs_sum = 0;

                for(int m=0; m < 3; m++)
                {
                    rhs_sum += -cY.idx(j, k) * icY.idx(i, m) * dX.idx(m);
                }

                value kron_ik = (i == k) ? 1 : 0;
                value kron_ij = (i == j) ? 1 : 0;

                result.idx(i, j, k) += -(1.f/(2 * clamped_X)) * (kron_ik * dX.idx(j) + kron_ij * dX.idx(k) + rhs_sum);
            }
        }
    }

    return result;
}

tensor<value, 3, 3> calculate_didja(equation_context& ctx)
{
    /*tensor<value, 3, 3> Xdidja;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value Xderiv = X * double_covariant_derivative(ctx, args.gA, args.digA, cY, icY, args.christoff2).idx(j, i);
            //value Xderiv = X * gpu_covariant_derivative_low_vec(ctx, args.digA, cY, icY).idx(j, i);

            value s2 = 0.5f * (diff1(ctx, X, i) * diff1(ctx, gA, j) + diff1(ctx, X, j) * diff1(ctx, gA, i));

            value s3 = 0;

            for(int m=0; m < 3; m++)
            {
                for(int n=0; n < 3; n++)
                {
                    value v = icY.idx(m, n) * diff1(ctx, X, m) * diff1(ctx, gA, n);

                    s3 += v;
                }
            }

            Xdidja.idx(i, j) = Xderiv + s2 + -0.5f * cY.idx(i, j) * s3;
        }
    }

    ctx.pin(Xdidja);*/

    ccz4_args args(ctx);

    inverse_metric<value, 3, 3> icY = args.cY.invert();

    tensor<value, 3, 3, 3> christoff2 = christoffel_symbols_2(ctx, args.cY, icY);

    tensor<value, 3, 3> didja;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value s1 = double_covariant_derivative(ctx, args.gA, args.digA, args.cY, icY, christoff2).idx(j, i);

            value X = W_to_X(args.W);
            tensor<value, 3> dX = dW_to_dX(args.W, args.dW);

            value clamped_X = 1/max(X, 0.0001f);

            value s2 = (1/clamped_X) * 0.5f * (dX.idx(i) * args.digA.idx(j) + dX.idx(j) * args.digA.idx(i));

            value i3 = 0;

            for(int m=0; m < 3; m++)
            {
                for(int n=0; n < 3; n++)
                {
                    value v = icY.idx(m, n) * dX.idx(m) * diff1(ctx, args.gA, n);

                    i3 += v;
                }
            }

            value s3 = -(1 / (2 * clamped_X)) * args.cY.idx(i, j) * i3;

            didja.idx(i, j) = s1 + s2 + s3;
        }
    }

    return didja;
}

void ccz4::build_cA(matter_interop& interop, equation_context& ctx, bool use_matter)
{
    ccz4_args args(ctx);

    inverse_metric<value, 3, 3> icY = args.cY.invert();

    tensor<value, 3, 3> Rij = calculate_Rij(ctx, args);

    tensor<value, 3> Zi_lower = args.get_Zi_lowered(ctx);

    tensor<value, 3, 3> ZiDj = covariant_derivative_low_vec(ctx, Zi_lower, args.cY, icY);

    /*///for the tensor DcDa, this returns idx(a, c)
    template<typename T, int N>
    inline
    tensor<T, N, N> covariant_derivative_low_vec(differentiator& ctx, const tensor<T, N>& v_in, const metric<T, N, N>& met, const inverse_metric<T, N, N>& inverse)
    */

    tensor<value, 3, 3> DiDja = calculate_didja(ctx);

    tensor<value, 3, 3> with_trace;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value s1 = -DiDja.idx(i, j);

            value s2 = args.gA * (Rij.idx(i, j) + ZiDj.idx(j, i) + ZiDj.idx(i, j));

            with_trace.idx(i, j) = s1 + s2;
        }
    }

    tensor<value, 3, 3> TF = args.W * args.W * trace_free(with_trace, args.cY, icY);

    tensor<value, 3, 3> part_2 = args.gA * args.cA * (args.K - 2 * args.theta);

    tensor<value, 3, 3> part_3;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            value sum = 0;

            for(int l=0; l < 3; l++)
            {
                sum += -2 * args.gA * args.cA.idx(i, l) * raise_index(args.cA, icY, 1).idx(j, l);
            }

            part_3.idx(i, j) = sum;
        }
    }

    tensor<value, 3, 3> dtcAij = TF + part_2 + part_3 + lie_derivative_weight(ctx, args.gB, args.cA);

    for(int i=0; i < 6; i++)
    {
        std::string name = "dtcAij" + std::to_string(i);

        vec2i idx = args.linear_indices[i];

        ctx.add(name, dtcAij.idx(idx.x(), idx.y()));
    }
}

void ccz4::build_W(equation_context& ctx)
{
    ccz4_args args(ctx);

    value dkbk = 0;

    for(int i=0; i < 3; i++)
    {
        dkbk += diff1(ctx, args.gB.idx(i), i);
    }

    value bkdw = 0;

    for(int i=0; i < 3; i++)
    {
        bkdw += args.gB.idx(i) * diff1(ctx, args.W, i);
    }

    value dtW = (1.f/3.f) * args.gA * args.W * args.K - (1.f/3.f) * args.W * dkbk + bkdw;

    ctx.add("dtW", dtW);
}

value get_R(equation_context& ctx, ccz4_args& args)
{
    tensor<value, 3, 3> Rij = calculate_Rij(ctx, args);

    return trace(Rij, args.cY.invert());
}

value get_k1(equation_context& ctx)
{
    return 0;
}

value get_k2(equation_context& ctx)
{
    return 0;
}

value get_k3(equation_context& ctx)
{
    return 0;
}

void ccz4::build_K(matter_interop& interop, equation_context& ctx, bool use_matter)
{
    ccz4_args args(ctx);

    inverse_metric<value, 3, 3> icY = args.cY.invert();

    tensor<value, 3, 3> DiDja = calculate_didja(ctx);

    tensor<value, 3, 3> raised_didja = -raise_index(DiDja, icY, 0);

    value p1 = 0;

    for(int i=0; i < 3; i++)
    {
        p1 += raised_didja.idx(i, i);
    }

    tensor<value, 3> Zi_upper = args.get_Zi_raised(ctx);

    tensor<value, 3, 3> ZiDj = covariant_derivative_low_vec(ctx, Zi_upper, args.cY, icY);

    value ZiDj_sum = 0;

    for(int i=0; i < 3; i++)
    {
        ZiDj_sum += ZiDj.idx(i, i);
    }

    value R = get_R(ctx, args);

    value p2 = args.gA * (R + 2 * ZiDj_sum + args.K * args.K - 2 * args.theta * args.K);

    value p3 = 0;

    for(int j=0; j < 3; j++)
    {
        p3 += args.gB.idx(j) * diff1(ctx, args.K, j);
    }

    value p4 = -3 * args.gA * get_k1(ctx) * (1 + get_k2(ctx)) * args.theta;

    value dtK = p1 + p2 + p3 + p4;

    ctx.add("dtK", dtK);
}

void ccz4::build_theta(equation_context& ctx)
{
    ccz4_args args(ctx);

    inverse_metric<value, 3, 3> icY = args.cY.invert();

    tensor<value, 3> Zi_upper = args.get_Zi_raised(ctx);

    tensor<value, 3, 3> ZiDj = covariant_derivative_low_vec(ctx, Zi_upper, args.cY, icY);

    value ZiDj_sum = 0;

    for(int i=0; i < 3; i++)
    {
        ZiDj_sum += ZiDj.idx(i, i);
    }

    value R = get_R(ctx, args);

    value aij_aIJ = 0;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            aij_aIJ += args.cA.idx(i, j) * raise_both(args.cA, icY).idx(i, j);
        }
    }

    value bkdktheta = 0;

    for(int k=0; k < 3; k++)
    {
        bkdktheta += args.gB.idx(k) * diff1(ctx, args.theta, k);
    }

    ///dtconstraint_theta
    value dtconstraint_theta = 0.5f * args.gA * (R + 2 * ZiDj_sum - aij_aIJ + (2.f/3.f) * args.K * args.K - 2 * args.theta * args.K)
                               - sum_multiply(Zi_upper, args.digA)
                               + bkdktheta
                               - args.gA * get_k1(ctx) * (2 + get_k2(ctx)) * args.theta;

    ctx.add("dtconstraint_theta", dtconstraint_theta);
}

void ccz4::build_cGi_hat(matter_interop& interop, equation_context& ctx, bool use_matter)
{
    ccz4_args args(ctx);

    inverse_metric<value, 3, 3> icY = args.cY.invert();

    tensor<value, 3, 3> aIJ = raise_both(args.cA, icY);

    tensor<value, 3, 3, 3> christoff2 = christoffel_symbols_2(ctx, args.cY, icY);

    tensor<value, 3> Zi_lower = args.get_Zi_lowered(ctx);

    tensor<value, 3> cGi = args.get_cGi(ctx);

    value dkbk = 0;

    for(int k=0; k < 3; k++)
    {
        dkbk += args.digB.idx(k, k);
    }

    tensor<value, 3> dtcG_hat;

    for(int i=0; i < 3; i++)
    {
        value p1 = 0;

        {
            value s1 = 0;

            for(int j=0; j < 3; j++)
            {
                for(int k=0; k < 3; k++)
                {
                    s1 += christoff2.idx(i, j, k) * aIJ.idx(j, k);
                }
            }

            value s2 = 0;

            for(int j=0; j < 3; j++)
            {
                s2 += -3.f * aIJ.idx(i, j) * diff1(ctx, args.W, j) / max(args.W, 0.0001f);
            }

            value s3 = 0;

            for(int j=0; j < 3; j++)
            {
                s3 += -(2.f/3.f) * icY.idx(i, j) * diff1(ctx, args.K, j);
            }

            p1 = 2 * args.gA * (s1 + s2 + s3);
        }

        value p2 = 0;

        for(int k=0; k < 3; k++)
        {
            p2 += 2 * icY.idx(k, i) * (
                                       args.gA * diff1(ctx, args.theta, k)
                                       - args.theta * diff1(ctx, args.gA, k)
                                       - (2.f/3.f) * args.gA * args.K * Zi_lower.idx(k));
        }

        value p3 = 0;

        for(int j=0; j < 3; j++)
        {
            p3 += -2 * aIJ.idx(i, j) * diff1(ctx, args.gA, j);
        }

        value p4 = 0;

        for(int k=0; k < 3; k++)
        {
            for(int l=0; l < 3; l++)
            {
                p4 += icY.idx(k, l) * diff2(ctx, args.gB.idx(i), l, k, args.digB.idx(l, i), args.digB.idx(k, i));
            }
        }

        value p5 = 0;

        for(int k=0; k < 3; k++)
        {
            for(int l=0; l < 3; l++)
            {
                p5 += (1.f/3.f) * icY.idx(i, k) * diff2(ctx, args.gB.idx(l), l, k, args.digB.idx(l, l), args.digB.idx(k, l));
            }
        }

        value p6 = (2.f/3.f) * cGi.idx(i) * dkbk;

        value p7 = 0;

        for(int k=0; k < 3; k++)
        {
            p7 += -cGi.idx(k) * args.digB.idx(k, i);
        }

        value p8_1 = 0;
        value p8_2 = 0;

        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                p8_1 += 2 * get_k3(ctx) * (2.f/3.f) * icY.idx(i, j) * Zi_lower.idx(j) * args.digB.idx(k, k);
                p8_2 += 2 * get_k3(ctx) * -icY.idx(j, k) * Zi_lower.idx(j) * args.digB.idx(k, i);
            }
        }

        value p9 = 0;

        for(int k=0; k < 3; k++)
        {
            p9 += args.gB.idx(k) * diff1(ctx, args.cGi_hat.idx(i), k);
        }

        value p10 = -2 * args.gA * get_k1(ctx) * args.get_cYij_Zj(ctx).idx(i);

        dtcG_hat.idx(i) = p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8_1 + p8_2 + p9 + p10;
    }

    ctx.add("dtcGi0", dtcG_hat.idx(0));
    ctx.add("dtcGi1", dtcG_hat.idx(1));
    ctx.add("dtcGi2", dtcG_hat.idx(2));
}
