#ifndef RAYTRACING_HPP_INCLUDED
#define RAYTRACING_HPP_INCLUDED

#include <geodesic/dual_value.hpp>
#include "single_source.hpp"
#include <vec/tensor.hpp>
#include <toolkit/opencl.hpp>

struct equation_context;
struct base_bssn_args;

struct lightray
{
    tensor<value, 4> pos4;
    tensor<value, 4> vel4;

    tensor<value, 3> adm_pos;
    tensor<value, 3> adm_vel;
    value ku_uobsu;
};

lightray make_lightray(equation_context& ctx,
                       const tensor<value, 3>& world_position, const tensor<value, 4>& camera_quat, v2i screen_size, v2i xy,
                       const metric<value, 3, 3>& Yij, const value& gA, const tensor<value, 3>& gB);

void build_raytracing_kernels(cl::context& clctx, base_bssn_args& bssn_args);

#endif // RAYTRACING_HPP_INCLUDED
