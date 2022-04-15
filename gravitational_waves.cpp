#include "gravitational_waves.hpp"
#include <geodesic/dual_value.hpp>
#include "legendre_nodes.h"
#include "legendre_weights.h"

int64_t factorial(int i)
{
    if(i == 0)
        return 1;

    return i * factorial(i - 1);
}

dual_types::complex<float> expi(float val)
{
    return dual_types::complex<float>(cos(val), sin(val));
}

///https://arxiv.org/pdf/1906.03877.pdf 8
///aha!
///https://arxiv.org/pdf/0709.0093.pdf
///https://arxiv.org/pdf/gr-qc/0610128.pdf 40
///at last! A non horrible reference and non gpl reference for negative spin!
///this is where the cactus code comes from as well
template<typename T>
dual_types::complex<T> sYlm_2(int s, int l, int m, T theta, T phi)
{
    thread_local std::map<std::tuple<int, int, int, T, T>, dual_types::complex<T>> cache;

    if(auto found_it = cache.find(std::tuple{s, l, m, theta, phi}); found_it != cache.end())
        return found_it->second;

    auto dlms = [](T theta, int l, int m, int s)
    {
        int k1 = std::max(0, m - s);
        int k2 = std::min(l + m, l - s);

        T sum = 0;

        for(int k=k1; k <= k2; k++)
        {
            float cp1 = (double)(pow(-1, k) * sqrt((double)(factorial(l + m) * factorial(l - m) * factorial(l + s) * factorial(l - s)))) /
                        ((double)(factorial(l + m - k) * factorial(l - s - k) * factorial(k) * factorial(k + s - m)));

            assert(isfinite(cp1));

            T cp2 = pow(cos(theta/2.f), 2 * l + m - s - 2 * k);
            T cp3 = pow(sin(theta/2.f), 2 * k + s - m);

            sum = sum + cp1 * cp2 * cp3;
        }

        return sum;
    };

    T coeff = pow(-1, s) * sqrt((2 * l + 1) / (4 * M_PI));

    dual_types::complex<T> ret = coeff * dlms(theta, l, m, -s) * expi(m * phi);

    cache[std::tuple{s, l, m, theta, phi}] = ret;

    return ret;
}

///https://pomax.github.io/bezierinfo/legendre-gauss.html
///https://cbeentjes.github.io/files/Ramblings/QuadratureSphere.pdf
///http://homepage.divms.uiowa.edu/~atkinson/papers/SphereQuad1982.pdf

template<typename T>
auto integrate(float lowerbound, float upperbound, const T& f_x, int n)
{
    using variable_type = decltype(f_x(0.f));

    variable_type sum = 0;

    std::vector<float> weights = get_legendre_weights(n);
    std::vector<float> nodes = get_legendre_nodes(n);

    for(int i=0; i < n; i++)
    {
        float wi = weights[i];
        float xi = nodes[i];

        float final_val = ((upperbound - lowerbound)/2.f) * xi + (upperbound + lowerbound) / 2.f;

        auto func_eval = wi * f_x(final_val);

        sum = sum + func_eval;
    }

    return ((upperbound - lowerbound) / 2.f) * sum;
}

template<typename T>
inline
auto integrate_1d_raw(const T& func, int n, float upper, float lower)
{
    std::vector<float> weights = get_legendre_weights(n);
    std::vector<float> nodes = get_legendre_nodes(n);

    using variable_type = decltype(func(0.f));

    variable_type sum = 0;

    for(int j=0; j < n; j++)
    {
        float w = weights[j];
        float xj = nodes[j];

        float value = ((upper - lower)/2.f) * xj + (upper + lower) / 2.f;

        auto func_eval = w * func(value);

        sum = sum + func_eval;
    }

    return ((upper - lower) / 2.f) * sum;
}

template<typename T>
inline
auto integrate_1d(const T& func, int n, float upper, float lower)
{
    using variable_type = decltype(func(0.f));
    variable_type sum =  0;

    int pieces = 1;
    float step = (upper - lower) / pieces;

    for(int i=0; i < pieces; i++)
    {
        sum += integrate_1d_raw(func, n, i * step + lower, (i + 1) * step + lower);
    }

    return sum;
}

template<typename T>
auto spherical_integrate(const T& f_theta_phi, int n)
{
    using variable_type = decltype(f_theta_phi(0.f, 0.f));

    variable_type sum = 0;

    std::vector<float> weights = get_legendre_weights(n);
    std::vector<float> nodes = get_legendre_nodes(n);

    float iupper = 2 * M_PI;
    float ilower = 0;

    float jupper = M_PI;
    float jlower = 0;

    ///https://cbeentjes.github.io/files/Ramblings/QuadratureSphere.pdf7 7
    ///0 -> 2pi, phi
    auto outer_integral = [&](float phi)
    {
        auto inner_integral = [&](float theta){return f_theta_phi(theta, phi);};

        return integrate_1d(inner_integral, n, jupper, jlower);
    };

    return integrate_1d(outer_integral, n, iupper, ilower);
}

///this isn't correct at all. The integration might be fine, but we can't take the spherical harmonics of a constant
/*inline
dual_types::complex<float> get_harmonic(const dual_types::complex<float>& value, int l, int m)
{
    auto func = [&](float theta, float phi)
    {
        dual_types::complex<float> harmonic = sYlm(2, l, m, theta, phi);

        //printf("Hreal %f\n", harmonic.real);

        return value * harmonic;
    };

    int n = 16;

    dual_types::complex<float> harmonic = spherical_integrate(func, n);

    return harmonic;
}*/

float linear_interpolate(const std::map<int, std::map<int, std::map<int, float>>>& vals_map, vec3f pos, vec3i dim)
{
    vec3f floored = floor(pos);

    vec3i ipos = (vec3i){floored.x(), floored.y(), floored.z()};

    auto index = [&](vec3i lpos)
    {
        assert(lpos.x() >= 0 && lpos.y() >= 0 && lpos.z() >= 0 && lpos.x() < dim.x() && lpos.y() < dim.y() && lpos.z() < dim.z());

        //std::cout << "MAPV " << vals_map.at(lpos.x()).at(lpos.y()).at(lpos.z()) << std::endl;

        return vals_map.at(lpos.x()).at(lpos.y()).at(lpos.z());

        //return vals[lpos.z() * dim.x() * dim.y() + lpos.y() * dim.x() + lpos.x()];
    };

    auto c000 = index(ipos + (vec3i){0,0,0});
    auto c100 = index(ipos + (vec3i){1,0,0});

    auto c010 = index(ipos + (vec3i){0,1,0});
    auto c110 = index(ipos + (vec3i){1,1,0});

    auto c001 = index(ipos + (vec3i){0,0,1});
    auto c101 = index(ipos + (vec3i){1,0,1});

    auto c011 = index(ipos + (vec3i){0,1,1});
    auto c111 = index(ipos + (vec3i){1,1,1});

    vec3f frac = pos - floored;

    auto c00 = c000 * (1 - frac.x()) + c100 * frac.x();
    auto c01 = c001 * (1 - frac.x()) + c101 * frac.x();

    auto c10 = c010 * (1 - frac.x()) + c110 * frac.x();
    auto c11 = c011 * (1 - frac.x()) + c111 * frac.x();

    auto c0 = c00 * (1 - frac.y()) + c10 * frac.y();
    auto c1 = c01 * (1 - frac.y()) + c11 * frac.y();

    return c0 * (1 - frac.z()) + c1 * frac.z();
}

vec3f dim_to_centre(vec3i dim)
{
    vec3i even = dim - 1;

    return {even.x()/2.f, even.y()/2.f, even.z()/2.f};
}

float get_harmonic_extraction_radius(int extract_pixel)
{
    return extract_pixel;
}

std::vector<cl_ushort4> get_harmonic_extraction_points(vec3i dim, int extract_pixel)
{
    std::vector<vec3i> ret_as_int;

    float rad = get_harmonic_extraction_radius(extract_pixel);
    vec3f centre = dim_to_centre(dim);

    auto func = [&](float theta, float phi)
    {
        vec3f pos = {rad * cos(phi) * sin(theta), rad * sin(phi) * sin(theta), rad * cos(theta)};

        pos += centre;

        vec3f ff0 = floor(pos);

        vec3i f0 = {ff0.x(), ff0.y(), ff0.z()};

        ret_as_int.push_back(f0);
        ret_as_int.push_back({f0.x() + 1, f0.y(), f0.z()});
        ret_as_int.push_back({f0.x(), f0.y() + 1, f0.z()});
        ret_as_int.push_back({f0.x(), f0.y(), f0.z() + 1});

        ret_as_int.push_back({f0.x() + 1, f0.y() + 1, f0.z()});
        ret_as_int.push_back({f0.x() + 1, f0.y(), f0.z() + 1});
        ret_as_int.push_back({f0.x(), f0.y() + 1, f0.z() + 1});

        ret_as_int.push_back({f0.x() + 1, f0.y() + 1, f0.z() + 1});

        return 0.f;
    };

    int n = 64;

    (void)spherical_integrate(func, n);

    std::vector<cl_ushort4> ret;

    for(vec3i i : ret_as_int)
    {
        ret.push_back({i.x(), i.y(), i.z(), 0});
    }

    std::sort(ret.begin(), ret.end(), [](cl_ushort4 p1, cl_ushort4 p2)
    {
        return std::tie(p1.s[2], p1.s[1], p1.s[0]) < std::tie(p2.s[2], p2.s[1], p2.s[0]);
    });

    return ret;
}

float get_harmonic(const std::vector<cl_ushort4>& points, const std::vector<dual_types::complex<float>>& vals, vec3i dim, int extract_pixel, int l, int m)
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

    float rad = get_harmonic_extraction_radius(extract_pixel);

    vec3f centre = dim_to_centre(dim);

    auto func = [&](float theta, float phi)
    {
        dual_types::complex<float> harmonic = sYlm_2(-2, l, m, theta, phi);

        dual_types::complex<float> conj = conjugate(harmonic);

        //printf("Hreal %f\n", harmonic.real);

        vec3f pos = {rad * cos(phi) * sin(theta), rad * sin(phi) * sin(theta), rad * cos(theta)};

        pos += centre;

        float interpolated_real = linear_interpolate(real_value_map, pos, dim);
        float interpolated_imaginary = linear_interpolate(imaginary_value_map, pos, dim);

        //printf("interpolated %f %f\n", interpolated.real, interpolated.imaginary);

        float scalar_product = interpolated_real * conj.real + interpolated_imaginary * conj.imaginary;

        return scalar_product;
    };

    int n = 64;

    float harmonic = spherical_integrate(func, n);

    //printf("Harmonic %f\n", harmonic);

    return harmonic;
}

gravitational_wave_manager::gravitational_wave_manager(cl::context& ctx, vec3i _simulation_size, float c_at_max, float scale) :
    wave_buffers{ctx, ctx, ctx}, read_queue(ctx, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE), harmonic_points{ctx}
{
    simulation_size = _simulation_size;

    float r_extract = c_at_max/3;

    raw_harmonic_points = get_harmonic_extraction_points(simulation_size, extract_pixel);

    elements = raw_harmonic_points.size();
    harmonic_points.alloc(sizeof(cl_ushort4) * elements);
    harmonic_points.write(read_queue, raw_harmonic_points);
    read_queue.block();

    for(int i=0; i < (int)wave_buffers.size(); i++)
    {
        wave_buffers[i].alloc(sizeof(cl_float2) * elements);
    }
}

void gravitational_wave_manager::callback(cl_event event, cl_int event_command_status, void* user_data)
{
    callback_data& data = *(callback_data*)user_data;

    if(event_command_status != CL_COMPLETE)
        return;

    std::lock_guard guard(data.me->lock);
    data.me->pending_unprocessed_data.push_back((cl_float2*)data.read);

    delete ((callback_data*)user_data);
}

void gravitational_wave_manager::issue_extraction(cl::command_queue& cqueue, std::vector<cl::buffer>& buffers, std::vector<cl::buffer>& thin_intermediates, float scale, const vec<4, cl_int>& clsize, cl::gl_rendertexture& tex)
{
    cl::args waveform_args;

    cl_int point_count = raw_harmonic_points.size();

    waveform_args.push_back(harmonic_points);
    waveform_args.push_back(point_count);

    for(auto& i : buffers)
    {
        waveform_args.push_back(i);
    }

    for(auto& i : thin_intermediates)
    {
        waveform_args.push_back(i);
    }

    cl::buffer& next = wave_buffers[(next_buffer % 3)];
    next_buffer++;

    waveform_args.push_back(scale);
    waveform_args.push_back(clsize);
    waveform_args.push_back(next);
    waveform_args.push_back(tex);

    cl::event kernel_event = cqueue.exec("extract_waveform", waveform_args, {point_count}, {128});

    cl_float2* next_data = new cl_float2[elements];

    cl::event data = next.read_async(read_queue, (char*)next_data, elements * sizeof(cl_float2), {kernel_event});

    callback_data* cb_data = new callback_data;
    cb_data->me = this;
    cb_data->read = next_data;

    data.set_completion_callback(callback, cb_data);

    if(last_event.has_value())
    {
        last_event.value().block();
    }

    last_event = data;
}

std::vector<float> gravitational_wave_manager::process()
{
    std::vector<float> real_harmonic;

    std::vector<cl_float2*> to_process;

    {
        std::lock_guard guard(lock);

        to_process = std::move(pending_unprocessed_data);
        pending_unprocessed_data.clear();
    }

    for(cl_float2* vec : to_process)
    {
        std::vector<dual_types::complex<float>> as_vector;

        for(int i=0; i < elements; i++)
        {
            as_vector.push_back({vec[i].s[0], vec[i].s[1]});
        }

        float harmonic = get_harmonic(raw_harmonic_points, as_vector, simulation_size, extract_pixel, 2, 2);

        if(!isnanf(harmonic))
            real_harmonic.push_back(harmonic);

        delete [] vec;
    }

    return real_harmonic;
}
