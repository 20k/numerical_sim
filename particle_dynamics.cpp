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

    float generation_radius = 0.5f * get_c_at_max()/2.f;

    ///need to use an actual rng if i'm doing anything even vaguely scientific
    std::minstd_rand0 rng(1234);

    std::vector<vec3f> positions;

    for(int i=0; i < particle_num; i++)
    {
        int kk=0;

        for(; kk < 1024; kk++)
        {
            float x = rand_det_s(rng, -1.f, 1.f) * generation_radius;
            float y = rand_det_s(rng, -1.f, 1.f) * generation_radius;
            float z = rand_det_s(rng, -1.f, 1.f) * generation_radius;

            vec3f pos = {x, y, z};

            if(pos.length() >= generation_radius)
                continue;

            positions.push_back(pos);
        }

        if(kk == 1024)
            throw std::runtime_error("Did not successfully assign particle position");
    }

    assert((int)positions.size() == particle_num);
}

void particle_dynamics::step(cpu_mesh& mesh, cl::context& ctx, cl::managed_command_queue& mqueue, thin_intermediates_pool& pool, buffer_set& in, buffer_set& out, buffer_set& base, float timestep, int iteration, int max_iteration)
{

}
