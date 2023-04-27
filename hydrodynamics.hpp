#ifndef HYDRODYNAMICS_HPP_INCLUDED
#define HYDRODYNAMICS_HPP_INCLUDED

#include "mesh_manager.hpp"
#include "bssn.hpp"

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
};

struct hydro_state
{
    cl::buffer should_evolve;

    hydro_state(cl::context& ctx) : should_evolve(ctx){}
};

struct eularian_hydrodynamics : plugin
{
    hydro_state hydro_st;
    matter_initial_vars vars;
    bool use_colour = false;

    eularian_hydrodynamics(cl::context& ctx);

    void grab_resources(matter_initial_vars _vars);

    virtual std::vector<buffer_descriptor> get_buffers() override;
    virtual void init(cpu_mesh& mesh, cl::context& ctx, cl::command_queue& cqueue,         thin_intermediates_pool& pool, buffer_set& to_init) override;
    virtual void step(cpu_mesh& mesh, cl::context& ctx, cl::managed_command_queue& mqueue, thin_intermediates_pool& pool, buffer_pack& pack, float timestep, int iteration, int max_iteration) override;

    virtual void load(cl::command_queue& cqueue, const std::string& directory) override {}
    virtual void save(cl::command_queue& cqueue, const std::string& directory) override {}
};

template<typename T>
inline
T chi_to_e_6phi(const T& chi)
{
    return pow(1/(max(chi, 0.001f)), (3.f/2.f));
}

struct eularian_matter : matter_interop
{
    virtual value calculate_adm_S(equation_context& ctx, standard_arguments& args) override;
    virtual value calculate_adm_p(equation_context& ctx, standard_arguments& args) override;
    virtual tensor<value, 3, 3> calculate_adm_X_Sij(equation_context& ctx, standard_arguments& args) override;
    virtual tensor<value, 3> calculate_adm_Si(equation_context& ctx, standard_arguments& args) override;
};

namespace hydrodynamics
{
    void build_intermediate_variables_derivatives(equation_context& ctx);
    void build_artificial_viscosity(equation_context& ctx);
    void build_equations(equation_context& ctx);
    void build_advection(equation_context& ctx);
}

void test_w();

#endif // HYDRODYNAMICS_HPP_INCLUDED
