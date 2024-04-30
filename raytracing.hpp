#ifndef RAYTRACING_HPP_INCLUDED
#define RAYTRACING_HPP_INCLUDED

#include <vec/value.hpp>
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

lightray make_lightray(equation_context& ctx,
                       const tensor<value, 3>& world_position, const tensor<value, 4>& camera_quat, v2i screen_size, v2i xy,
                       const metric<value, 4, 4>& Guv, const value& gA, const tensor<value, 3>& gB);

void build_raytracing_kernels(cl::context& clctx, base_bssn_args& bssn_args);

///store raw adm variables, *not* bssn. Uses a list of buffers, which it linearly interpolates between with a fixed step size
struct raytracing_manager
{
    float time_elapsed = 0;
    int last_grabbed = 0;

    vec3i slice_size;
    float slice_width = 0;
    int max_slices = 80;
    std::vector<std::vector<cl::buffer>> slices;

    cl::buffer render_ray_info_buf;

    raytracing_manager(cl::context& clctx, const tensor<int, 2>& screen);

    std::vector<cl::buffer> get_fresh_buffers(cl::context& clctx);
    void trace(cl::context& clctx, cl::command_queue& mqueue, float simulation_width, const tensor<int, 2>& screen, vec3f camera_pos, vec4f camera_quat, float camera_start_time);
    void grab_buffers(cl::context& clctx, cl::command_queue& mqueue, const std::vector<cl::buffer>& bufs, float scale, const tensor<cl_int, 4>& clsize, float step);
};

///4x4, reduced to 10 components. Stored as a 4d hypercube
struct raytracing4_manager
{
    float time_elapsed = 0;
    int last_grabbed = -1;

    ///x, y, z, using the cube strategy here
    vec3i slice_size;
    float slice_width = 0;
    int max_slices = 75;
    ///using one fat slice per buffer
    std::vector<cl::buffer> slice;

    cl::buffer render_ray_info_buf;

    raytracing4_manager(cl::context& clctx, const tensor<int, 2>& screen);

    void trace(cl::context& clctx, cl::command_queue& mqueue, float simulation_width, const tensor<int, 2>& screen, vec3f camera_pos, vec4f camera_quat, float camera_start_time);
    void grab_buffers(cl::context& clctx, cl::command_queue& mqueue, const std::vector<cl::buffer>& bufs, float scale, const tensor<cl_int, 4>& clsize, float step);
};

void process_geodesics(equation_context& ctx);
void loop_geodesics4(equation_context& ctx);
void loop_geodesics(equation_context& ctx, vec3f dim);

#endif // RAYTRACING_HPP_INCLUDED
