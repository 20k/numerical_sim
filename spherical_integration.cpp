#include "spherical_integration.hpp"
#include "spherical_decomposition.hpp"

inline
vec3f dim_to_centre(vec3i dim)
{
    vec3i even = dim - 1;

    return {even.x()/2.f, even.y()/2.f, even.z()/2.f};
}

float linear_interpolate(const std::map<int, std::map<int, std::map<int, float>>>& vals_map, vec3f pos, vec3i dim)
{
    vec3f floored = floor(pos);

    vec3i ipos = (vec3i){floored.x(), floored.y(), floored.z()};

    auto index = [&](vec3i lpos)
    {
        assert(lpos.x() >= 0 && lpos.y() >= 0 && lpos.z() >= 0 && lpos.x() < dim.x() && lpos.y() < dim.y() && lpos.z() < dim.z());

        //std::cout << "MAPV " << vals_map.at(lpos.x()).at(lpos.y()).at(lpos.z()) << std::endl;

        return vals_map.at(lpos.x()).at(lpos.y()).at(lpos.z());
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


#define INTEGRATION_N 64

std::vector<cl_ushort4> get_spherical_integration_points(vec3i dim, int extract_pixel)
{
    std::vector<vec3i> ret_as_int;

    vec3f centre = dim_to_centre(dim);

    auto func = [&](vec3f pos)
    {
        vec3f ff0 = floor(pos);

        vec3i f0 = {(int)ff0.x(), (int)ff0.y(), (int)ff0.z()};

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

    (void)spherical_decompose_cartesian_function(func, -2, 2, 2, centre, (float)extract_pixel, INTEGRATION_N);

    std::vector<cl_ushort4> ret;

    for(vec3i i : ret_as_int)
    {
        ret.push_back({(cl_ushort)i.x(), (cl_ushort)i.y(), (cl_ushort)i.z(), 0});
    }

    std::sort(ret.begin(), ret.end(), [](cl_ushort4 p1, cl_ushort4 p2)
    {
        return std::tie(p1.s[2], p1.s[1], p1.s[0]) < std::tie(p2.s[2], p2.s[1], p2.s[0]);
    });

    auto eq_shorts = [](cl_ushort4 p1, cl_ushort4 p2){return p1.s[0] == p2.s[0] && p1.s[1] == p2.s[1] && p1.s[2] == p2.s[2];};

    printf("Pre deduplicate adm size %i\n", (int)ret.size());

    ret.erase(std::unique(ret.begin(), ret.end(), eq_shorts), ret.end());

    printf("Post deduplicate adm size %i\n", (int)ret.size());

    return ret;
}

integrator::integrator(cl::context& ctx, vec3i _dim, float _scale, cl::command_queue& _read_queue) : read_queue(_read_queue), arq(ctx, read_queue), gpu_points(ctx)
{
    scale = _scale;
    dim = _dim;
    extract_pixel = floor((dim.x() - 1)/2.f) - 20;

    printf("Extract at %i\n", extract_pixel);

    points = get_spherical_integration_points(dim, extract_pixel);

    printf("Hiyas\n");

    arq.start(ctx, points.size());

    printf("Pre write\n");

    gpu_points.alloc(sizeof(cl_ushort4) * points.size());
    gpu_points.write(read_queue, points);
    read_queue.block();

    printf("Done write\n");
}

std::vector<float> integrator::integrate()
{
    std::vector<float> ret;

    std::vector<std::vector<float>> values = arq.process();

    vec3f centre = dim_to_centre(dim);

    for(const std::vector<float>& vals : values)
    {
        std::map<int, std::map<int, std::map<int, float>>> value_map;

        assert(points.size() == vals.size());

        for(int i=0; i < (int)points.size(); i++)
        {
            cl_ushort4 point = points[i];

            value_map[point.s[0]][point.s[1]][point.s[2]] = vals[i];
        }

        float world_radius = extract_pixel * scale;

        auto to_integrate = [&](vec3f pos)
        {
            pos += centre;

            return linear_interpolate(value_map, pos, dim);
        };

        float integrated = cartesian_integrate(to_integrate, INTEGRATION_N, extract_pixel, world_radius);

        ret.push_back(integrated);
    }

    return ret;
}
