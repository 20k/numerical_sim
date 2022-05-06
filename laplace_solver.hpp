#ifndef LAPLACE_SOLVER_HPP_INCLUDED
#define LAPLACE_SOLVER_HPP_INCLUDED

#include <geodesic/dual_value.hpp>
#include <toolkit/opencl.hpp>

struct laplace_data
{
    value boundary;
    value rhs;
    value buffer_in;
    value buffer_out;

    std::vector<std::pair<std::string, std::string>> args_in;
    std::vector<std::pair<std::string, std::string>> args_out;
};

cl::buffer laplace_solver(cl::context& clcltx, cl::command_queue& cqueue, const laplace_data& data, float scale, vec3i dim, float err = 0.0001f);

#endif // LAPLACE_SOLVER_HPP_INCLUDED
