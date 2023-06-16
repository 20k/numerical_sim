#include "gravitational_waves.hpp"
#include <geodesic/dual_value.hpp>
#include "legendre_nodes.h"
#include "legendre_weights.h"
#include "spherical_integration.hpp"
#include "spherical_harmonics.hpp"
#include "spherical_decomposition.hpp"
#include "ref_counted.hpp"

#define INTEGRATION_N 64

inline
vec3f dim_to_centre(vec3i dim)
{
    vec3i even = dim - 1;

    return {even.x()/2.f, even.y()/2.f, even.z()/2.f};
}

dual_types::complex<float> get_harmonic(const std::vector<cl_ushort4>& points, const std::vector<dual_types::complex<float>>& vals, vec3i dim, int extract_pixel, int l, int m)
{
    std::map<int, std::map<int, std::map<int, float>>> real_value_map;
    std::map<int, std::map<int, std::map<int, float>>> imaginary_value_map;

    assert(points.size() == vals.size());

    for(int i=0; i < (int)points.size(); i++)
    {
        cl_ushort4 point = points[i];

        real_value_map[point.s[0]][point.s[1]][point.s[2]] = vals[i].real;
        imaginary_value_map[point.s[0]][point.s[1]][point.s[2]] = vals[i].imaginary;
    }

    vec3f centre = dim_to_centre(dim);

    auto to_integrate = [&](const vec3f& pos)
    {
        float real = linear_interpolate(real_value_map, pos, dim);
        float imaginary = linear_interpolate(imaginary_value_map, pos, dim);

        return dual_types::complex<float>(real, imaginary);
    };

    return spherical_decompose_complex_cartesian_function(to_integrate, -2, l, m, centre, (float)extract_pixel, INTEGRATION_N);
}

gravitational_wave_manager::gravitational_wave_manager(cl::context& ctx, vec3i _simulation_size, float c_at_max, float scale) :
    read_queue(ctx, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE), harmonic_points{ctx},
    arq(ctx, read_queue)
{
    simulation_size = _simulation_size;

    calculated_extraction_pixel = floor(simulation_size.x()/2.f) - 20;

    printf("Extracting at pixel %i\n", calculated_extraction_pixel);

    raw_harmonic_points = get_spherical_integration_points(simulation_size, calculated_extraction_pixel);

    harmonic_points.alloc(sizeof(cl_ushort4) * raw_harmonic_points.size());
    harmonic_points.write(read_queue, raw_harmonic_points);
    read_queue.block();

    arq.start(ctx, raw_harmonic_points.size());
}

void gravitational_wave_manager::issue_extraction(cl::managed_command_queue& cqueue, std::vector<cl::buffer>& buffers, std::vector<ref_counted_buffer>& thin_intermediates, float scale, const vec<4, cl_int>& clsize)
{
    cl::buffer next = arq.fetch_next_buffer();

    cl::args waveform_args;

    cl_int point_count = raw_harmonic_points.size();

    waveform_args.push_back(harmonic_points.as_device_read_only());
    waveform_args.push_back(point_count);

    for(auto& i : buffers)
    {
        waveform_args.push_back(i.as_device_read_only());
    }

    for(auto& i : thin_intermediates)
    {
        waveform_args.push_back(i.as_device_read_only());
    }

    waveform_args.push_back(scale);
    waveform_args.push_back(clsize);
    waveform_args.push_back(next);
    //waveform_args.push_back(tex);

    cl::event kernel_event = cqueue.exec("extract_waveform", waveform_args, {point_count}, {128});

    arq.issue(next, kernel_event);
}

std::vector<dual_types::complex<float>> gravitational_wave_manager::process()
{
    std::vector<std::vector<cl_float2>> to_process = arq.process();

    std::vector<dual_types::complex<float>> complex_harmonics;

    for(const std::vector<cl_float2>& data : to_process)
    {
        std::vector<dual_types::complex<float>> as_vector;

        for(cl_float2 d : data)
        {
            as_vector.push_back({d.s[0], d.s[1]});
        }

        dual_types::complex<float> harmonic = get_harmonic(raw_harmonic_points, as_vector, simulation_size, (float)calculated_extraction_pixel, 2, 2);

        if(!isnanf(harmonic.real) && !isnanf(harmonic.imaginary))
            complex_harmonics.push_back(harmonic);
    }

    return complex_harmonics;
}
