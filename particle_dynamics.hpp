#ifndef PARTICLE_DYNAMICS_HPP_INCLUDED
#define PARTICLE_DYNAMICS_HPP_INCLUDED

#include "mesh_manager.hpp"
#include "bssn.hpp"

struct particle_matter_interop : matter_interop
{
    virtual value               calculate_adm_S(equation_context& ctx, standard_arguments& bssn_args) override;
    virtual value               calculate_adm_p(equation_context& ctx, standard_arguments& bssn_args) override;
    virtual tensor<value, 3, 3> calculate_adm_X_Sij(equation_context& ctx, standard_arguments& bssn_args) override;
    virtual tensor<value, 3>    calculate_adm_Si(equation_context& ctx, standard_arguments& bssn_args) override;
};

struct particle_dynamics : plugin
{
    int particle_count = 0;
    std::array<cl::buffer, 2> particle_3_position;
    std::array<cl::buffer, 2> particle_3_velocity;

    cl::buffer indices_block;
    cl::buffer weights_block;
    cl::buffer memory_alloc_count;

    std::optional<cl::buffer> memory_ptrs;
    std::optional<cl::buffer> counts;

    /*cl::buffer adm_p;
    std::array<cl::buffer, 3> adm_Si;
    std::array<cl::buffer, 6> adm_Sij;
    cl::buffer adm_S;*/

    cl::program pd;

    particle_dynamics(cl::context& ctx);

    virtual std::vector<buffer_descriptor> get_buffers() override;

    virtual void init(cpu_mesh& mesh, cl::context& ctx, cl::command_queue& cqueue,         thin_intermediates_pool& pool, buffer_set& to_init) override;
    virtual void step(cpu_mesh& mesh, cl::context& ctx, cl::managed_command_queue& mqueue, thin_intermediates_pool& pool, buffer_set& in, buffer_set& out, buffer_set& base, float timestep, int iteration, int max_iteration) override;
};

#endif // PARTICLE_DYNAMICS_HPP_INCLUDED
