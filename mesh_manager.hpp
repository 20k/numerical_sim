#ifndef MESH_MANAGER_HPP_INCLUDED
#define MESH_MANAGER_HPP_INCLUDED

#include <cl/cl.h>
#include <vec/vec.hpp>
#include <toolkit/opencl.hpp>
#include "ref_counted.hpp"

template<typename T>
inline
float calculate_scale(float c_at_max, const T& size)
{
    return c_at_max / size.largest_elem();
}

inline
float get_c_at_max()
{
    return 55.f * (251.f/300.f);
}

struct named_buffer
{
    cl::buffer buf;

    std::string name;
    std::string modified_by;

    named_buffer(cl::context& ctx) : buf(ctx){}
};

struct buffer_set
{
    #ifndef USE_GBB
    static constexpr int buffer_count = 12+9;
    #else
    static constexpr int buffer_count = 12 + 9 + 3;
    #endif

    std::vector<named_buffer> buffers;

    buffer_set(cl::context& ctx, vec3i size);

    named_buffer& lookup(const std::string& name);
};

struct gpu_mesh
{
    cl_int4 centre;
    cl_int4 dim;
};

struct evolution_points
{
    //cl::buffer first_derivative_points;
    //cl::buffer second_derivative_points;
    cl::buffer border_points;
    cl::buffer all_points;
    cl::buffer order;

    //int first_count = 0;
    //int second_count = 0;
    int all_count = 0;
    int border_count = 0;

    evolution_points(cl::context& ctx) : border_points(ctx), all_points(ctx), order(ctx){}
};

struct thin_intermediates_pool
{
    std::vector<ref_counted_buffer> pool;

    ref_counted_buffer request(cl::context& ctx, cl::managed_command_queue& cqueue, vec3i size, int element_size);
};

struct cpu_mesh_settings
{
    bool use_half_intermediates = false;
    bool calculate_momentum_constraint = false;
};

struct cpu_mesh
{
    cpu_mesh_settings sett;

    vec3i centre;
    vec3i dim;

    int resolution_scale = 1;
    float scale = 1;

    int which_data = 0;

    std::array<buffer_set, 2> data;
    buffer_set scratch;

    evolution_points points_set;

    cl::buffer sponge_positions;
    cl_int sponge_positions_count = 0;

    std::array<cl::buffer, 3> momentum_constraint;

    float dissipate_low = 0.25;
    float dissipate_high = 0.25;
    float dissipate_gauge = 0.25;

    float dissipate_caijyy = dissipate_high;

    #ifdef NO_CAIJYY
    dissipate_caijyy = 0;
    #endif // NO_CAIJYY

    std::array<float, buffer_set::buffer_count> dissipation_coefficients
    {
        dissipate_low, dissipate_low, dissipate_low, dissipate_low, dissipate_low, dissipate_low, //cY
        dissipate_high, dissipate_high, dissipate_high, dissipate_caijyy, dissipate_high, dissipate_high, //cA
        dissipate_low, dissipate_low, dissipate_low, //cGi
        dissipate_high, //K
        dissipate_low, //X
        dissipate_gauge, //gA
        dissipate_gauge, dissipate_gauge, dissipate_gauge, //gB
        #ifdef USE_GBB
        dissipate_gauge, dissipate_gauge, dissipate_gauge, //gBB
        #endif // USE_GBB
    };

    cpu_mesh(cl::context& ctx, cl::command_queue& cqueue, vec3i _centre, vec3i _dim, cpu_mesh_settings _sett);

    void flip();

    buffer_set& get_input();
    buffer_set& get_output();
    buffer_set& get_scratch(int which);

    void init(cl::command_queue& cqueue, cl::buffer& u_arg);

    ref_counted_buffer get_thin_buffer(cl::context& ctx, cl::managed_command_queue& cqueue, thin_intermediates_pool& pool);

    ///returns buffers and intermediates
    std::pair<std::vector<cl::buffer>, std::vector<ref_counted_buffer>> full_step(cl::context& ctx, cl::command_queue& main_queue, cl::managed_command_queue& mqueue, float timestep, thin_intermediates_pool& pool, cl::buffer& u_arg);
};

#endif // MESH_MANAGER_HPP_INCLUDED
