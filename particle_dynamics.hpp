#ifndef PARTICLE_DYNAMICS_HPP_INCLUDED
#define PARTICLE_DYNAMICS_HPP_INCLUDED

#include "mesh_manager.hpp"
#include "bssn.hpp"

struct particle_data
{
    std::vector<vec3f> positions;
    std::vector<vec3f> velocities;
    std::vector<float> masses;

    std::vector<float> debug_velocities;
    std::vector<float> debug_analytic_mass;
    std::vector<float> debug_real_mass;

    float minimum_mass = 0;
    float particle_brightness = 1.f;

    void calculate_minimum_mass()
    {
        minimum_mass = FLT_MAX;

        for(float m : masses)
        {
            minimum_mass = std::min(fabs(m), minimum_mass);
        }
    }
};

struct particle_matter_interop : matter_interop
{
    virtual value               calculate_adm_S(equation_context& ctx, standard_arguments& bssn_args) const override;
    virtual value               calculate_adm_p(equation_context& ctx, standard_arguments& bssn_args) const override;
    virtual tensor<value, 3, 3> calculate_adm_X_Sij(equation_context& ctx, standard_arguments& bssn_args) const override;
    virtual tensor<value, 3>    calculate_adm_Si(equation_context& ctx, standard_arguments& bssn_args) const override;
};

struct particle_buffer
{
    cl::buffer position;
    cl::buffer velocity;
    cl::buffer mass;
    cl::buffer lorentz;

    particle_buffer(cl::context& ctx) : position(ctx), velocity(ctx), mass(ctx), lorentz(ctx){}
};

struct particle_dynamics : plugin
{
    cl_int particle_count = 0;
    std::array<particle_buffer, 3> p_data;

    cl::buffer indices_block;
    cl::buffer memory_alloc_count;

    std::optional<cl::buffer> memory_ptrs;
    std::optional<cl::buffer> counts;

    particle_data start_data;

    /*cl::buffer adm_p;
    std::array<cl::buffer, 3> adm_Si;
    std::array<cl::buffer, 6> adm_Sij;
    cl::buffer adm_S;*/

    cl_int max_intermediate_size = int{1024} * 1024 * 400;

    cl::program pd;

    particle_dynamics(cl::context& ctx);

    void add_particles(particle_data&& data);

    //virtual std::vector<buffer_descriptor> get_buffers() override;
    virtual std::vector<buffer_descriptor> get_utility_buffers() override;

    virtual void init(cpu_mesh& mesh, cl::context& ctx, cl::command_queue& cqueue,         thin_intermediates_pool& pool, buffer_set& to_init) override;
    virtual void step(cpu_mesh& mesh, cl::context& ctx, cl::command_queue& mqueue, thin_intermediates_pool& pool, buffer_pack& pack, float timestep, int iteration, int max_iteration) override;
    virtual void finalise(cpu_mesh& mesh, cl::context& ctx, cl::command_queue& mqueue, thin_intermediates_pool& pool, float timestep) override;

    virtual void load(cl::command_queue& cqueue, const std::string& directory) override;
    virtual void save(cl::command_queue& cqueue, const std::string& directory) override;
};

void build_dirac_sample(equation_context& ctx);

#endif // PARTICLE_DYNAMICS_HPP_INCLUDED
