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

struct buffer_descriptor
{
    std::string name;
    std::string modified_by;
    float dissipation_coeff = 0.f;
    float asymptotic_value = 0;
    float wave_speed = 1;
};

///todo: do this inherity
struct named_buffer
{
    cl::buffer buf;
    buffer_descriptor desc;

    named_buffer(cl::context& ctx) : buf(ctx){}
};

struct buffer_set
{
    std::vector<named_buffer> buffers;

    buffer_set(cl::context& ctx, vec3i size, const std::vector<buffer_descriptor>& in_buffers);

    named_buffer& lookup(const std::string& name);
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
    //bool use_gBB = false;
};

struct cpu_mesh;

struct plugin
{
    virtual void init(cpu_mesh& mesh, cl::context& ctx, cl::command_queue& cqueue,         thin_intermediates_pool& pool, buffer_set& to_init);
    virtual void step(cpu_mesh& mesh, cl::context& ctx, cl::managed_command_queue& mqueue, thin_intermediates_pool& pool, buffer_set& in, buffer_set& out, buffer_set& base, float timestep);

    ~plugin(){}
};

template<typename T>
struct basic_pool
{
    std::map<std::string, T> elements;

    template<typename U>
    T& get_named(U&& u, const std::string& name)
    {
        auto it = elements.find(name);

        if(it == elements.end())
        {
            elements.insert({name, u()});

            return elements.find(name)->second;
        }

        return it->second;
    }
};

///buffer, thin
using step_callback = std::function<void(cl::managed_command_queue&, std::vector<cl::buffer>&, std::vector<ref_counted_buffer>&)>;

struct cpu_mesh
{
    cpu_mesh_settings sett;

    vec3i centre;
    vec3i dim;

    int resolution_scale = 1;
    float scale = 1;

    std::array<buffer_set, 3> data;

    basic_pool<buffer_set> free_data;

    evolution_points points_set;

    std::array<cl::buffer, 3> momentum_constraint;

    static constexpr float dissipate_low = 0.25;
    static constexpr float dissipate_high = 0.25;
    static constexpr float dissipate_gauge = 0.25;

    cpu_mesh(cl::context& ctx, cl::command_queue& cqueue, vec3i _centre, vec3i _dim, cpu_mesh_settings _sett, evolution_points& points, const std::vector<buffer_descriptor>& buffers);

    void init(cl::command_queue& cqueue, cl::buffer& u_arg, std::array<cl::buffer, 6>& bcAij);

    ref_counted_buffer get_thin_buffer(cl::context& ctx, cl::managed_command_queue& cqueue, thin_intermediates_pool& pool);

    std::vector<ref_counted_buffer> get_derivatives_of(cl::context& ctx, buffer_set& bufs, cl::managed_command_queue& mqueue, thin_intermediates_pool& pool);

    ///returns buffers and intermediates
    void full_step(cl::context& ctx, cl::command_queue& main_queue, cl::managed_command_queue& mqueue, float timestep, thin_intermediates_pool& pool, step_callback callback);

    void clean_buffer(cl::managed_command_queue& mqueue, cl::buffer& in, cl::buffer& out, cl::buffer& base, float asym, float speed, float timestep);
};

#endif // MESH_MANAGER_HPP_INCLUDED
