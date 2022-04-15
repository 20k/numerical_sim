#ifndef MESH_MANAGER_HPP_INCLUDED
#define MESH_MANAGER_HPP_INCLUDED

#include <cl/cl.h>
#include <vec/vec.hpp>
#include <toolkit/opencl.hpp>

template<typename T>
inline
float calculate_scale(float c_at_max, const T& size)
{
    return c_at_max / size.largest_elem();
}

inline
float get_c_at_max()
{
    return 60;
}

struct buffer_set
{
    #ifndef USE_GBB
    static constexpr int buffer_count = 12+9;
    #else
    static constexpr int buffer_count = 12 + 9 + 3;
    #endif

    std::vector<cl::buffer> buffers;

    buffer_set(cl::context& ctx, vec3i size);
};

struct gpu_mesh
{
    cl_int4 centre;
    cl_int4 dim;
};

struct evolution_points
{
    cl::buffer first_derivative_points;
    cl::buffer second_derivative_points;

    int first_count = 0;
    int second_count = 0;

    evolution_points(cl::context& ctx) : first_derivative_points(ctx), second_derivative_points(ctx){}
};

struct thin_intermediates_pool
{
    struct buffer_descriptor
    {
        cl::buffer buf;
        int id = 0;
        vec3i size;
        int element_size = 0;

        buffer_descriptor(cl::context& ctx) : buf(ctx){}
    };

    std::vector<buffer_descriptor> pool;

    cl::buffer request(cl::context& ctx, cl::managed_command_queue& cqueue, int id, vec3i size, int element_size);
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

    std::array<std::string, buffer_set::buffer_count> buffer_names
    {
        "cY0", "cY1", "cY2", "cY3", "cY4", "cY5",
        "cA0", "cA1", "cA2", "cA3", "cA4", "cA5",
        "cGi0", "cGi1", "cGi2",
        "K", "X", "gA",
        "gB0", "gB1", "gB2",
        #ifdef USE_GBB
        "gBB0", "gBB1", "gBB2",
        #endif // USE_GBB
    };

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

    cl::buffer get_thin_buffer(cl::context& ctx, cl::managed_command_queue& cqueue, thin_intermediates_pool& pool, int id);

    ///returns buffers and intermediates
    std::pair<std::vector<cl::buffer>, std::vector<cl::buffer>> full_step(cl::context& ctx, cl::command_queue& main_queue, cl::managed_command_queue& mqueue, float timestep, thin_intermediates_pool& pool, cl::buffer& u_arg);
};

#endif // MESH_MANAGER_HPP_INCLUDED
