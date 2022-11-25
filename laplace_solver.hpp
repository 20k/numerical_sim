#ifndef LAPLACE_SOLVER_HPP_INCLUDED
#define LAPLACE_SOLVER_HPP_INCLUDED

#include <geodesic/dual_value.hpp>
#include <toolkit/opencl.hpp>
#include "equation_context.hpp"

struct laplace_data
{
    value boundary;
    value rhs;
    cl::buffer aij_aIJ;
    cl::buffer ppw2p;
    cl::buffer nonconformal_pH;
    std::vector<std::pair<std::string, value>> extras;
    equation_context ectx;

    laplace_data(cl::buffer& in_aij_aIJ, cl::buffer& in_ppw2p, cl::buffer& in_nonconformal_pH) : aij_aIJ(in_aij_aIJ), ppw2p(in_ppw2p), nonconformal_pH(in_nonconformal_pH){}
};

cl::buffer laplace_solver(cl::context& clcltx, cl::command_queue& cqueue, laplace_data& data, float scale, vec3i dim, float err = 0.0001f);

struct sandwich_data
{
    sandwich_data(cl::context& ctx) : u_arg(ctx){}

    value gA_phi_rhs;
    value gB0_rhs;
    value gB1_rhs;
    value gB2_rhs;
    value u_to_phi;

    value djbj;

    cl::buffer u_arg;
};

struct sandwich_result
{
    sandwich_result(cl::context& ctx) : gB0(ctx), gB1(ctx), gB2(ctx), gA(ctx){}

    cl::buffer gB0;
    cl::buffer gB1;
    cl::buffer gB2;
    cl::buffer gA;
};

sandwich_result sandwich_solver(cl::context& clcltx, cl::command_queue& cqueue, const sandwich_data& data, float scale, vec3i dim, float err = 0.0001f);

#endif // LAPLACE_SOLVER_HPP_INCLUDED
