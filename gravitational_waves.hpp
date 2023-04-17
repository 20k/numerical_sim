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

struct ref_counted_buffer;

struct gravitational_wave_manager
{
    struct callback_data
    {
        gravitational_wave_manager* me = nullptr;
        cl_float2* read = nullptr;
    };

    cl_int4 wave_pos;
    vec3i simulation_size;

    std::vector<cl::buffer> wave_buffers;
    std::vector<std::pair<cl::event, cl_float2*>> gpu_data_in_flight;
    std::vector<cl_float2*> pending_unprocessed_data;

    std::vector<cl_ushort4> raw_harmonic_points;

    cl::command_queue read_queue;
    cl::buffer harmonic_points;

    uint32_t next_buffer = 0;

    int elements = 0;

    gravitational_wave_manager(cl::context& ctx, vec3i _simulation_size, float c_at_max, float scale);

    void issue_extraction(cl::managed_command_queue& cqueue, std::vector<cl::buffer>& buffers, std::vector<ref_counted_buffer>& thin_intermediates, float scale, const vec<4, cl_int>& clsize);
    std::vector<dual_types::complex<float>> process();

    int calculated_extraction_pixel = 0;
};

#endif // GRAVITATIONAL_WAVES_HPP_INCLUDED
