#ifndef GRAVITATIONAL_WAVES_HPP_INCLUDED
#define GRAVITATIONAL_WAVES_HPP_INCLUDED

#include <CL/cl.h>
#include <toolkit/opencl.hpp>
#include <array>
#include <mutex>
#include <optional>
#include <vector>
#include <vec/vec.hpp>
#include <geodesic/dual_value.hpp>
#include "async_read_queue.hpp"

struct ref_counted_buffer;

struct gravitational_wave_manager
{
    cl_int4 wave_pos;
    vec3i simulation_size;

    std::vector<cl_ushort4> raw_harmonic_points;

    cl::command_queue read_queue;
    cl::buffer harmonic_points;

    async_read_queue<cl_float2> arq;

    uint32_t next_buffer = 0;

    int elements = 0;

    gravitational_wave_manager(cl::context& ctx, vec3i _simulation_size, float c_at_max, float scale);

    void issue_extraction(cl::managed_command_queue& cqueue, std::vector<cl::buffer>& buffers, std::vector<ref_counted_buffer>& thin_intermediates, float scale, const vec<4, cl_int>& clsize);
    std::vector<dual_types::complex<float>> process();

    int calculated_extraction_pixel = 0;
};

#endif // GRAVITATIONAL_WAVES_HPP_INCLUDED
