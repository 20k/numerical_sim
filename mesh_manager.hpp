#ifndef MESH_MANAGER_HPP_INCLUDED
#define MESH_MANAGER_HPP_INCLUDED

#include <cl/cl.h>
#include <vec/vec.hpp>
#include <toolkit/opencl.hpp>
#include "ref_counted.hpp"
#include <nlohmann/json.hpp>

template<typename T>
inline
float calculate_scale(float c_at_max, const T& size)
{
    return c_at_max / size.largest_elem();
}

inline
float get_c_at_max()
{
    return 25.f;
}

inline
float get_timestep(float c_at_max, vec3i size)
{
    float timestep_at_base_c = 0.035;

    float ratio_at_base = 30.f/255.f;
    float new_ratio = c_at_max / size.largest_elem();

    return 0.035f * (new_ratio / ratio_at_base);
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
    bool currently_physical = false;
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
};

struct cpu_mesh;

struct buffer_pack
{
    buffer_set& in;
    buffer_set& out;
    buffer_set& base;

    int in_idx = 0;
    int out_idx = 0;
    int base_idx = 0;

    buffer_pack(buffer_set& _in, buffer_set& _out, buffer_set& _base, int _in_idx, int _out_idx, int _base_idx) : in(_in), out(_out), base(_base)
    {
        in_idx = _in_idx;
        out_idx = _out_idx;
        base_idx = _base_idx;
    }
};

struct plugin
{
    virtual std::vector<buffer_descriptor> get_buffers(){return std::vector<buffer_descriptor>();}
    virtual std::vector<buffer_descriptor> get_utility_buffers(){return std::vector<buffer_descriptor>();}
    virtual void init(cpu_mesh& mesh, cl::context& ctx, cl::command_queue& cqueue,         thin_intermediates_pool& pool, buffer_set& to_init){assert(false);}
    virtual void step(cpu_mesh& mesh, cl::context& ctx, cl::managed_command_queue& mqueue, thin_intermediates_pool& pool, buffer_pack& pack, float timestep, int iteration, int max_iteration){assert(false);}
    virtual void finalise(cpu_mesh& mesh, cl::context& ctx, cl::managed_command_queue& mqueue, thin_intermediates_pool& pool, float timestep) {}
    virtual void save(cl::command_queue& cqueue, const std::string& directory){assert(false);}
    virtual void load(cl::command_queue& cqueue, const std::string& directory){assert(false);}

    virtual ~plugin(){}
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

void save_buffer(cl::command_queue& cqueue, cl::buffer& buf, const std::string& where);
void load_buffer(cl::command_queue& cqueue, cl::buffer& into, const std::string& from);

///buffer, thin
using step_callback = std::function<void(cl::managed_command_queue&, std::vector<cl::buffer>&, std::vector<ref_counted_buffer>&)>;

struct cpu_mesh
{
    cpu_mesh_settings sett;

    vec3i centre;
    vec3i dim;

    int resolution_scale = 1;
    float scale = 1;

    std::map<int, buffer_set> data;
    buffer_set utility_data;

    evolution_points points_set;

    ///should really be utility buffers
    std::array<cl::buffer, 3> momentum_constraint;

    float elapsed_time = 0;

    static constexpr float dissipate_low = 0.25;
    static constexpr float dissipate_high = 0.25;
    static constexpr float dissipate_gauge = 0.25;

    cpu_mesh(cl::context& ctx, cl::command_queue& cqueue, vec3i _centre, vec3i _dim, cpu_mesh_settings _sett, evolution_points& points, const std::vector<buffer_descriptor>& buffers, const std::vector<buffer_descriptor>& utility_buffers, std::vector<plugin*> _plugins);

    void init(cl::context& ctx, cl::command_queue& cqueue, thin_intermediates_pool& pool, cl::buffer& u_arg, std::array<cl::buffer, 6>& bcAij);

    ref_counted_buffer get_thin_buffer(cl::context& ctx, cl::managed_command_queue& cqueue, thin_intermediates_pool& pool);

    std::vector<ref_counted_buffer> get_derivatives_of(cl::context& ctx, buffer_set& bufs, cl::managed_command_queue& mqueue, thin_intermediates_pool& pool);

    ///returns buffers and intermediates
    void full_step(cl::context& ctx, cl::command_queue& main_queue, cl::managed_command_queue& mqueue, float timestep, thin_intermediates_pool& pool, step_callback callback);

    void clean_buffer(cl::managed_command_queue& mqueue, cl::buffer& in, cl::buffer& out, cl::buffer& base, float asym, float speed, float timestep);

    buffer_set& get_buffers(cl::context& ctx, cl::managed_command_queue& mqueue, int index);
    void append_utility_buffers(const std::string& kernel_name, cl::args& args);

    nlohmann::json load(cl::command_queue& cqueue, const std::string& directory);
    void save(cl::command_queue& cqueue, const std::string& directory, nlohmann::json& extra);

    std::vector<plugin*> plugins;

    bool first_step = true;

private:
    std::vector<buffer_descriptor> buffers;
};

#endif // MESH_MANAGER_HPP_INCLUDED
