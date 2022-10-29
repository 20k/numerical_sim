#ifndef PARTICLE_DYNAMICS_HPP_INCLUDED
#define PARTICLE_DYNAMICS_HPP_INCLUDED

#include "mesh_manager.hpp"

struct particle_dynamics : plugin
{
    std::array<cl::buffer, 2> particle_3_position;
    std::array<cl::buffer, 2> particle_3_velocity;

    cl::buffer adm_p;
    std::array<cl::buffer, 3> adm_Si;
    std::array<cl::buffer, 6> adm_Sij;
    cl::buffer adm_S;

    cl::program pd;

    particle_dynamics(cl::context& ctx);

    virtual std::vector<buffer_descriptor> get_buffers() override;

    virtual void init(cpu_mesh& mesh, cl::context& ctx, cl::command_queue& cqueue,         thin_intermediates_pool& pool, buffer_set& to_init) override;
    virtual void step(cpu_mesh& mesh, cl::context& ctx, cl::managed_command_queue& mqueue, thin_intermediates_pool& pool, buffer_set& in, buffer_set& out, buffer_set& base, float timestep, int iteration, int max_iteration) override;
};

#endif // PARTICLE_DYNAMICS_HPP_INCLUDED
