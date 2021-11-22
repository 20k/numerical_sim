#ifndef MESH_MANAGER_HPP_INCLUDED
#define MESH_MANAGER_HPP_INCLUDED

#include <cl/cl.h>
#include <vec/vec.hpp>
#include <toolkit/opencl.hpp>
#include <iostream>

template<typename T>
inline
float calculate_scale(float c_at_max, const T& size)
{
    return c_at_max / size.largest_elem();
}

inline
float get_c_at_max()
{
    return 65.f * (251.f/300.f);
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

    cl::buffer request(cl::context& ctx, cl::command_queue& cqueue, int id, vec3i size, int element_size);
};

struct cpu_mesh_settings
{
    bool use_half_intermediates = false;
    bool calculate_momentum_constraint = false;
};

struct cpu_mesh
{
    cpu_mesh_settings sett;

    vec3f centre;
    vec3i dim;

    int resolution_scale = 1;
    float scale = 1;

    int which_data = 0;

    std::array<buffer_set, 2> data;
    buffer_set scratch;

    evolution_points points_set;

    cl::buffer sponge_positions;
    cl_int sponge_positions_count = 0;

    cl::buffer u_arg;

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

    cpu_mesh(cl::context& ctx, cl::command_queue& cqueue, vec3f _centre, vec3i _dim, cpu_mesh_settings _sett);

    void flip();

    buffer_set& get_input();
    buffer_set& get_output();
    buffer_set& get_scratch(int which);

    void init(cl::command_queue& cqueue, cl::buffer& in_u_arg);

    cl::buffer get_thin_buffer(cl::context& ctx, cl::command_queue& cqueue, thin_intermediates_pool& pool, int id);

    ///returns buffers and intermediates
    std::pair<std::vector<cl::buffer>, std::vector<cl::buffer>> full_step(cl::context& ctx, cl::command_queue& cqueue, float timestep, thin_intermediates_pool& pool);
};

struct cpu_topology
{
    float resolution_multiplier = 1;

    vec3f world_pos;
    vec3f world_dim;
};

struct grid_topology
{
    vec3f world_pos;
    vec3i grid_dim;
};

struct grid_topology_builder
{
    float resolution_multiplier = 1;

    vec3i grid_tl, grid_br;
    vec3f world_tl, world_br;

    void expand(vec3f direction)
    {
        /*vec3f as_float = {direction.x(), direction.y(), direction.z()};

        ///ok so
        ///if we have a half extents of eg {2,2,2} at position {0,0,0}
        ///and we expand *upwards* by {2, 0, 0}
        ///the new size is clearly {4, 2, 2}
        ///but the position must migrate to compensate, because we want it to stay the same

        world_pos += -(as_float / resolution_multiplier) / 2.f;*/

        vec3f world_displacement = direction / resolution_multiplier;

        if(direction.x() > 0)
        {
            grid_br.x() += direction.x();
            world_br.x() += world_displacement.x();
        }

        if(direction.y() > 0)
        {
            grid_br.y() += direction.y();
            world_br.y() += world_displacement.y();
        }

        if(direction.z() > 0)
        {
            grid_br.z() += direction.z();
            world_br.z() += world_displacement.z();
        }

        if(direction.x() < 0)
        {
            grid_tl.x() += direction.x();
            world_tl.x() += world_displacement.x();
        }

        if(direction.y() < 0)
        {
            grid_tl.y() += direction.y();
            world_tl.y() += world_displacement.y();
        }

        if(direction.z() < 0)
        {
            grid_tl.z() += direction.z();
            world_tl.z() += world_displacement.z();
        }
    }

    grid_topology build()
    {
        vec3f centre = (world_br + world_tl)/2.f;
        vec3i dim = grid_br - grid_tl;

        grid_topology t;
        t.world_pos = centre;
        t.grid_dim = dim;

        return t;
    }
};

inline
vec3i adjacent(const cpu_topology& s1, const cpu_topology& s2)
{
    vec3f half_size_1 = s1.world_dim/2.f;
    vec3f half_size_2 = s2.world_dim/2.f;

    vec3f max_sep = half_size_1 + half_size_2;

    vec3f diff = fabs(s2.world_pos - s1.world_pos);

    auto approx_same = [](float v1, float v2)
    {
        return v1 < v2 * 1.05f && v1 >= v2 * 0.95f;
    };

    vec3i ret;

    if(approx_same(diff.x(), max_sep.x()))
    {
        if(s1.world_pos.x() < s2.world_pos.x())
            ret.x() = -1;
        else
            ret.x() = 1;
    }

    if(approx_same(diff.y(), max_sep.y()))
    {
        if(s1.world_pos.y() < s2.world_pos.y())
            ret.y() = -1;
        else
            ret.y() = 1;
    }

    if(approx_same(diff.z(), max_sep.z()))
    {
        if(s1.world_pos.z() < s2.world_pos.z())
            ret.z() = -1;
        else
            ret.z() = 1;
    }

    return ret;
}

inline
std::vector<grid_topology> generate_boundary_topology(const std::vector<cpu_topology>& top_in)
{
    int boundary_width = 4 * 2;

    std::vector<grid_topology_builder> tops;

    for(const cpu_topology& ct : top_in)
    {
        grid_topology_builder val;
        val.resolution_multiplier = ct.resolution_multiplier;

        vec3f grid_float = ct.world_dim * ct.resolution_multiplier + (vec3f){1,1,1};

        val.grid_tl = {0,0,0};
        val.grid_br = {grid_float.x(), grid_float.y(), grid_float.z()};

        val.world_tl = ct.world_pos - ct.world_dim/2.f;
        val.world_br = ct.world_pos + ct.world_dim/2.f;

        tops.push_back(val);
    }

    for(int i=0; i < (int)tops.size(); i++)
    {
        for(int j=i+1; j < (int)tops.size(); j++)
        {
            const cpu_topology& c1 = top_in[i];
            const cpu_topology& c2 = top_in[j];

            vec3i adj = adjacent(c1, c2);

            if(adj.x() == 0 && adj.y() == 0 && adj.z() == 0)
                continue;

            grid_topology_builder& g1 = tops[i];
            grid_topology_builder& g2 = tops[j];

            vec3f dir_as_float = {adj.x(), adj.y(), adj.z()};

            std::cout << "ADJ " << dir_as_float << std::endl;

            ///so, if adj.x() < 0, then c1 is to the left of c2
            ///which means c1 needs to expand rightwards, and c2 leftwards
            ///both by boundary_width

            g1.expand(-dir_as_float * boundary_width);
            g2.expand(dir_as_float * boundary_width);
        }
    }

    std::vector<grid_topology> ret;

    for(auto& i : tops)
    {
        ret.push_back(i.build());
    }

    return ret;
}

cl::buffer iterate_u(cl::context& ctx, cl::command_queue& cqueue, vec3i size, vec3f mesh_position, float c_at_max);

struct cpu_mesh_manager
{
    cpu_mesh_settings sett;

    std::vector<grid_topology> layout;
    grid_topology centre_layout;

    std::vector<cpu_mesh*> meshes;
    cpu_mesh* centre = nullptr;

    cl::buffer central_u;

    cpu_mesh_manager(cl::context& ctx, cl::command_queue& cqueue, cpu_mesh_settings _sett) : central_u(ctx)
    {
        sett = _sett;

        cpu_topology t1;
        t1.world_dim = {280, 280, 280};
        t1.world_pos = {0,0,0};

        cpu_topology t2;
        t2.world_dim = {30, 280, 280};
        t2.world_pos = {-(t1.world_dim.x() - 1.f)/2.f - (t2.world_dim.x() - 1.f)/2.f, 0.f, 0.f};

        layout = generate_boundary_topology({t1, t2});
        centre_layout = layout[0];

        for(grid_topology& i : layout)
        {
            std::cout << "POS " << i.world_pos << std::endl;
            std::cout << "DIM " << i.grid_dim << std::endl;
        }

        central_u = iterate_u(ctx, cqueue, centre_layout.grid_dim, centre_layout.world_pos, get_c_at_max());
    }

    void init(cl::context& ctx, cl::command_queue& cqueue)
    {
        centre = new cpu_mesh(ctx, cqueue, centre_layout.world_pos, centre_layout.grid_dim, sett);

        centre->init(cqueue, central_u);

        meshes.push_back(centre);

        for(int i=1; i < (int)layout.size(); i++)
        {
            const grid_topology& top = layout[i];

            cpu_mesh* mesh = new cpu_mesh(ctx, cqueue, top.world_pos, top.grid_dim, sett);

            meshes.push_back(mesh);

            cl::buffer temp_u_arg(ctx);
            temp_u_arg.alloc(mesh->dim.x() * mesh->dim.y() * mesh->dim.z() * sizeof(cl_float));
            temp_u_arg.fill(cqueue, 1.f);

            mesh->init(cqueue, temp_u_arg);
        }
    }

    template<typename T>
    void full_step_all(cl::context& ctx, cl::command_queue& cqueue, float timestep, thin_intermediates_pool& pool, T&& callback)
    {
        for(int i=0; i < (int)meshes.size(); i++)
        {
            auto [last_valid_thin_base, last_valid_thin] = meshes[i]->full_step(ctx, cqueue, timestep, pool);

            callback(meshes[i], last_valid_thin_base, last_valid_thin, i);
        }
    }
};

#endif // MESH_MANAGER_HPP_INCLUDED
