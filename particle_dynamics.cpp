#include "particle_dynamics.hpp"

particle_dynamics::particle_dynamics(cl::context& ctx) : particle_3_position{ctx, ctx}, particle_3_velocity{ctx, ctx}, adm_p{ctx}, adm_Si{ctx, ctx, ctx}, adm_Sij{ctx, ctx, ctx, ctx, ctx, ctx}, adm_S{ctx}
{

}

std::vector<buffer_descriptor> particle_dynamics::get_buffers()
{
    return std::vector<buffer_descriptor>();
}


void particle_dynamics::init(cpu_mesh& mesh, cl::context& ctx, cl::command_queue& cqueue,         thin_intermediates_pool& pool, buffer_set& to_init)
{
    vec3i dim = mesh.dim;

    uint64_t size = dim.x() * dim.y() * dim.z() * sizeof(cl_float);

    adm_p.alloc(size);

    for(int i=0; i < 3; i++)
    {
        adm_Si[i].alloc(size);
    }

    for(int i=0; i < 6; i++)
    {
        adm_Sij[i].alloc(size);
    }

    adm_S.alloc(size);

    int particle_num = 1024;

    for(int i=0; i < 2; i++)
    {
        particle_3_position[i].alloc(sizeof(cl_float) * 3 * particle_num);
        particle_3_velocity[i].alloc(sizeof(cl_float) * 3 * particle_num);
    }
}

void particle_dynamics::step(cpu_mesh& mesh, cl::context& ctx, cl::managed_command_queue& mqueue, thin_intermediates_pool& pool, buffer_set& in, buffer_set& out, buffer_set& base, float timestep, int iteration, int max_iteration)
{

}
