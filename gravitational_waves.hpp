#ifndef GRAVITATIONAL_WAVES_HPP_INCLUDED
#define GRAVITATIONAL_WAVES_HPP_INCLUDED

#include <CL/cl.h>
#include <toolkit/opencl.hpp>
#include <array>
#include <mutex>
#include <optional>
#include <vector>
#include <vec/vec.hpp>

struct gravitational_wave_manager
{
    struct callback_data
    {
        gravitational_wave_manager* me = nullptr;
        cl_float2* read = nullptr;
    };

    int extract_pixel = 70;
    cl_int4 wave_pos;
    vec3i simulation_size;

    std::array<cl::buffer, 3> wave_buffers;
    std::vector<cl_float2*> pending_unprocessed_data;
    std::mutex lock;
    std::optional<cl::event> last_event;

    std::vector<cl_ushort4> raw_harmonic_points;
    cl::buffer harmonic_points;

    cl::command_queue read_queue;

    uint32_t next_buffer = 0;

    int elements = 0;

    gravitational_wave_manager(cl::context& ctx, vec3i _simulation_size, float c_at_max, float scale);

    static void callback(cl_event event, cl_int event_command_status, void* user_data);
    void issue_extraction(cl::command_queue& cqueue, std::vector<cl::buffer>& buffers, std::vector<cl::buffer>& thin_intermediates, float scale, const vec<4, cl_int>& clsize, cl::gl_rendertexture& tex);
    std::vector<float> process();
};

#endif // GRAVITATIONAL_WAVES_HPP_INCLUDED
