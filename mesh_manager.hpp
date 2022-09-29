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
    return 45.f * (251.f/300.f);
}

///todo: do this inherity
struct named_buffer
{
    cl::buffer buf;

    std::string name;
    std::string modified_by;
    float dissipation_coeff = 0.f;
    bool matter_term = false;
    float asymptotic_value = 0;
    float wave_speed = 1;

    named_buffer(cl::context& ctx) : buf(ctx){}
};

struct buffer_set_cfg
{
    bool use_matter = false;
    bool use_matter_colour = false;
    bool use_gBB = false;
};

struct buffer_set
{
    std::vector<named_buffer> buffers;

    buffer_set(cl::context& ctx, vec3i size, buffer_set_cfg cfg);

    named_buffer& lookup(const std::string& name);
};

struct colour_set
{
    std::vector<named_buffer> buffers;

    colour_set(cl::context& ctx, vec3i size, buffer_set_cfg cfg);
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

struct hydro_state
{
    cl::buffer should_evolve;

    hydro_state(cl::context& ctx) : should_evolve(ctx){}
};

evolution_points generate_evolution_points(cl::context& ctx, cl::command_queue& cqueue, float scale, vec3i size);

struct thin_intermediates_pool
{
    std::vector<ref_counted_buffer> pool;

    ref_counted_buffer request(cl::context& ctx, cl::managed_command_queue& cqueue, vec3i size, int element_size);
};

struct cpu_mesh_settings
{
    bool use_half_intermediates = false;
    bool calculate_momentum_constraint = false;
    bool use_matter = false;
    bool use_matter_colour = false;
    bool use_gBB = false;
};

struct matter_initial_vars
{
    std::array<cl::buffer, 6> bcAij;
    cl::buffer superimposed_tov_phi;

    cl::buffer pressure_buf;
    cl::buffer rho_buf;
    cl::buffer rhoH_buf;
    cl::buffer p0_buf;
    std::array<cl::buffer, 3> Si_buf;
    std::array<cl::buffer, 3> colour_buf;

    ///there must be a better way of doing this, c++ pls
    matter_initial_vars(cl::context& ctx) : bcAij{ctx, ctx, ctx, ctx, ctx, ctx}, superimposed_tov_phi{ctx},
                                            pressure_buf{ctx}, rho_buf{ctx}, rhoH_buf{ctx}, p0_buf{ctx}, Si_buf{ctx, ctx, ctx}, colour_buf{ctx, ctx, ctx}
    {

    }

    void clear(cl::context& ctx)
    {
        auto clr = [&](cl::buffer& b)
        {
            b = cl::buffer(ctx);
        };

        for(auto& i : bcAij)
            clr(i);

        clr(superimposed_tov_phi);

        clr(pressure_buf);
        clr(rho_buf);
        clr(rhoH_buf);
        clr(p0_buf);

        for(auto& i : Si_buf)
            clr(i);

        for(auto& i : colour_buf)
            clr(i);
    }
};

struct cpu_mesh
{
    cpu_mesh_settings sett;

    vec3i centre;
    vec3i dim;

    int resolution_scale = 1;
    float scale = 1;

    std::array<buffer_set, 3> data;
    std::array<colour_set, 3> colours;

    hydro_state hydro_st;

    evolution_points points_set;

    std::array<cl::buffer, 3> momentum_constraint;

    static constexpr float dissipate_low = 0.25;
    static constexpr float dissipate_high = 0.25;
    static constexpr float dissipate_gauge = 0.25;

    cpu_mesh(cl::context& ctx, cl::command_queue& cqueue, vec3i _centre, vec3i _dim, cpu_mesh_settings _sett, evolution_points& points);

    void init(cl::command_queue& cqueue, cl::buffer& u_arg, matter_initial_vars& vars);

    void step_hydro(cl::context& ctx, cl::managed_command_queue& cqueue, thin_intermediates_pool& pool, int idx_in, int idx_out, int idx_base, float timestep, int iteration);

    ref_counted_buffer get_thin_buffer(cl::context& ctx, cl::managed_command_queue& cqueue, thin_intermediates_pool& pool);

    std::vector<ref_counted_buffer> get_derivatives_of(cl::context& ctx, buffer_set& bufs, cl::managed_command_queue& mqueue, thin_intermediates_pool& pool);

    ///returns buffers and intermediates
    std::pair<std::vector<cl::buffer>, std::vector<ref_counted_buffer>> full_step(cl::context& ctx, cl::command_queue& main_queue, cl::managed_command_queue& mqueue, float timestep, thin_intermediates_pool& pool);

    void clean_buffer(cl::managed_command_queue& mqueue, cl::buffer& in, cl::buffer& out, cl::buffer& base, float asym, float speed, float timestep);
};

#endif // MESH_MANAGER_HPP_INCLUDED
