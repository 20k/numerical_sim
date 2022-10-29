#include "particle_dynamics.hpp"

particle_dynamics::particle_dynamics(cl::context& ctx) : particles{ctx, ctx}, adm_p{ctx}, adm_Si{ctx, ctx, ctx}, adm_Sij{ctx, ctx, ctx, ctx, ctx, ctx}, adm_S{ctx}
{

}

std::vector<buffer_descriptor> particle_dynamics::get_buffers()
{
    return std::vector<buffer_descriptor>();
}


void particle_dynamics::init(cpu_mesh& mesh, cl::context& ctx, cl::command_queue& cqueue,         thin_intermediates_pool& pool, buffer_set& to_init)
{

}

void particle_dynamics::step(cpu_mesh& mesh, cl::context& ctx, cl::managed_command_queue& mqueue, thin_intermediates_pool& pool, buffer_set& in, buffer_set& out, buffer_set& base, float timestep, int iteration, int max_iteration)
{

}
