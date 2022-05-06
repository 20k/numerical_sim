#ifndef LAPLACE_SOLVER_HPP_INCLUDED
#define LAPLACE_SOLVER_HPP_INCLUDED

#include <geodesic/dual_value.hpp>
#include <toolkit/opencl.hpp>

cl::buffer laplace_solver(cl::context& clcltx, cl::command_queue& cqueue, cl::buffer& gpu_holes, const value& rhs, const value& boundary, float scale, vec3i dim, float err = 0.0001f);

#endif // LAPLACE_SOLVER_HPP_INCLUDED
