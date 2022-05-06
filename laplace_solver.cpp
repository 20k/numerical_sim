#include "laplace_solver.hpp"
#include "mesh_manager.hpp"
#include "equation_context.hpp"

void check_symmetry(const std::string& debug_name, cl::command_queue& cqueue, cl::buffer& arg, vec<4, cl_int> size)
{
    //#define CHECK_SYMMETRY
    #ifdef CHECK_SYMMETRY
    std::cout << debug_name << std::endl;

    cl::args check;
    check.push_back(arg);
    check.push_back(size);

    cqueue.exec("check_z_symmetry", check, {size.x(), size.y(), size.z()}, {8, 8, 1});

    cqueue.block();
    #endif // CHECK_SYMMETRY
}

std::vector<cl::buffer> solve_for_u(cl::context& ctx, cl::command_queue& cqueue, cl::kernel& setup, cl::kernel& iterate,
                       vec<4, cl_int> base_size, float c_at_max, int scale_factor, std::optional<std::vector<cl::buffer>> base, cl_float etol, int dimensions)
{
    vec<4, cl_int> reduced_clsize = ((base_size - 1) / scale_factor) + 1;

    std::array<std::vector<cl::buffer>, 2> reduced_u_args;

    if(base.has_value())
        reduced_u_args[0] = base.value();
    else
    {
        for(int i=0; i < dimensions; i++)
        {
            cl::buffer& buf = reduced_u_args[0].emplace_back(ctx);
            buf.alloc(reduced_clsize.x() * reduced_clsize.y() * reduced_clsize.z() * sizeof(cl_float));
        }
    }

    for(int i=0; i < dimensions; i++)
    {
        cl::buffer& buf = reduced_u_args[1].emplace_back(ctx);
        buf.alloc(reduced_clsize.x() * reduced_clsize.y() * reduced_clsize.z() * sizeof(cl_float));
    }

    int which_reduced = 0;

    if(!base.has_value())
    {
        for(int i=0; i < dimensions; i++)
        {
            cl::args initial_u_args;
            initial_u_args.push_back(reduced_u_args[0][i]);
            initial_u_args.push_back(reduced_clsize);

            setup.set_args(initial_u_args);

            cqueue.exec(setup, {reduced_clsize.x(), reduced_clsize.y(), reduced_clsize.z()}, {8, 8, 1}, {});
        }
    }

    for(int i=0; i < dimensions; i++)
    {
        cl::copy(cqueue, reduced_u_args[0][i], reduced_u_args[1][i]);
    }

    int N = 8000;

    #ifdef GPU_PROFILE
    N = 1000;
    #endif // GPU_PROFILE

    #ifdef QUICKSTART
    N = 200;
    #endif // QUICKSTART

    //cl_int still_going = 0;

    std::array<cl::buffer, 2> still_going{ctx, ctx};

    cl_int one = 1;

    for(int i=0; i < 2; i++)
    {
        still_going[i].alloc(sizeof(cl_int));
        still_going[i].fill(cqueue, one);
    }

    int which_still_going = 0;

    for(int i=0; i < N; i++)
    {
        float local_scale = calculate_scale(c_at_max, reduced_clsize);

        cl::args iterate_u_args;

        for(int d = 0; d < dimensions; d++)
        {
            iterate_u_args.push_back(reduced_u_args[which_reduced][d]);
        }

        for(int d = 0; d < dimensions; d++)
        {
            iterate_u_args.push_back(reduced_u_args[(which_reduced + 1) % 2][d]);
        }

        iterate_u_args.push_back(local_scale);
        iterate_u_args.push_back(reduced_clsize);
        iterate_u_args.push_back(still_going[which_still_going]);
        iterate_u_args.push_back(still_going[(which_still_going + 1) % 2]);
        iterate_u_args.push_back(etol);

        iterate.set_args(iterate_u_args);

        cqueue.exec(iterate, {reduced_clsize.x(), reduced_clsize.y(), reduced_clsize.z()}, {8, 8, 1}, {});

        if(((i % 50) == 0) && still_going[(which_still_going + 1) % 2].read<cl_int>(cqueue)[0] == 0)
            break;

        still_going[which_still_going].set_to_zero(cqueue);

        which_reduced = (which_reduced + 1) % 2;
        which_still_going = (which_still_going + 1) % 2;
    }

    for(int i=0; i < dimensions; i++)
    {
        check_symmetry("post_iterate", cqueue, reduced_u_args[which_reduced][i], reduced_clsize);
    }

    return reduced_u_args[which_reduced];
}

#ifdef OLD_FAST_U
cl::buffer upscale_u(cl::context& ctx, cl::command_queue& cqueue, cl::buffer& source_buffer, vec<4, cl_int> base_size, int upscale_scale, int source_scale)
{
    vec<4, cl_int> reduced_clsize = ((base_size - 1) / source_scale) + 1;
    vec<4, cl_int> upper_clsize = ((base_size - 1) / upscale_scale) + 1;

    check_symmetry("pre_iterate", cqueue, source_buffer, reduced_clsize);

    cl::buffer u_arg(ctx);
    u_arg.alloc(upper_clsize.x() * upper_clsize.y() * upper_clsize.z() * sizeof(cl_float));

    cl::args upscale_args;
    upscale_args.push_back(source_buffer);
    upscale_args.push_back(u_arg);
    upscale_args.push_back(reduced_clsize);
    upscale_args.push_back(upper_clsize);

    cqueue.exec("upscale_u", upscale_args, {upper_clsize.x(), upper_clsize.y(), upper_clsize.z()}, {8, 8, 1});

    check_symmetry("post_upscale", cqueue, u_arg, upper_clsize);

    return u_arg;
}

cl::buffer iterate_u(cl::context& ctx, cl::command_queue& cqueue, vec3i size, float c_at_max)
{
    vec<4, cl_int> clsize = {size.x(), size.y(), size.z(), 0};

    std::optional<cl::buffer> last;

    for(int i=2; i >= 0; i--)
    {
        int up_size = pow(2, i+1);
        int current_size = pow(2, i);

        cl::buffer reduced = solve_for_u(ctx, cqueue, clsize, c_at_max, up_size, last);

        cl::buffer upscaled = upscale_u(ctx, cqueue, reduced, clsize, current_size, up_size);

        last = upscaled;
    }

    return solve_for_u(ctx, cqueue, clsize, c_at_max, 1, last);
}
#endif // OLD_FAST_U

std::vector<cl::buffer> extract_u_region(cl::context& ctx, cl::command_queue& cqueue, cl::kernel& extract,
                            std::vector<cl::buffer>& in, float c_at_max_in, float c_at_max_out, vec<4, cl_int> clsize)
{
    std::vector<cl::buffer> ret;

    for(cl::buffer& buf : in)
    {
        cl::buffer& out = ret.emplace_back(ctx);
        out.alloc(buf.alloc_size);

        cl::args upscale_args;
        upscale_args.push_back(buf);
        upscale_args.push_back(out);
        upscale_args.push_back(c_at_max_in);
        upscale_args.push_back(c_at_max_out);
        upscale_args.push_back(clsize);

        extract.set_args(upscale_args);

        cqueue.exec(extract, {clsize.x(), clsize.y(), clsize.z()}, {8, 8, 1}, {});

        check_symmetry("extract_u_region", cqueue, out, clsize);
    }

    return ret;
}

std::vector<cl::buffer> iterate_u(cl::context& ctx, cl::command_queue& cqueue, cl::kernel& setup, cl::kernel& iterate, cl::kernel& extract,
                     vec3i size, float c_at_max, cl_float etol, int dimensions)
{
    float boundaries[4] = {c_at_max, c_at_max * 4, c_at_max * 8, c_at_max * 16};

    vec<4, cl_int> clsize = {size.x(), size.y(), size.z(), 0};

    std::optional<std::vector<cl::buffer>> last;

    for(int i=2; i >= 0; i--)
    {
        float current_boundary = boundaries[i + 1];
        float next_boundary = boundaries[i];

        std::vector<cl::buffer> reduced = solve_for_u(ctx, cqueue, setup, iterate, clsize, current_boundary, 1, last, etol, dimensions);

        std::vector<cl::buffer> extracted = extract_u_region(ctx, cqueue, extract, reduced, current_boundary, next_boundary, clsize);

        last = extracted;
    }

    return solve_for_u(ctx, cqueue, setup, iterate, clsize, c_at_max, 1, last, etol, dimensions);
}

std::vector<cl::buffer> laplace_solver(cl::context& clctx, cl::command_queue& cqueue, const laplace_data& data, float scale, vec3i dim, float err)
{
    equation_context ctx;

    ctx.add("U_BASE", dual_types::apply("buffer_index", "u_offset_in", "ix", "iy", "iz", "dim"));
    ctx.add("U_RHS", data.rhs);
    ctx.add("U_BOUNDARY", data.boundary);

    if(data.dimension == 1)
        ctx.add("LAPLACE_SOLVE_1", 1);

    if(data.dimension == 3)
        ctx.add("LAPLACE_SOLVE_3", 1);

    std::string local_build_str = "-I ./ -O3 -cl-std=CL2.0 -cl-mad-enable -cl-finite-math-only ";

    ctx.build(local_build_str, "UNUSEDLAPLACE");

    cl::program u_program(clctx, "u_solver.cl");
    u_program.build(clctx, local_build_str);

    cl::kernel setup(u_program, "setup_u_offset");
    cl::kernel iterate(u_program, "iterative_u_solve");
    cl::kernel extract(u_program, "extract_u_region");
    cl::kernel upscale(u_program, "upscale_u");

    float c_at_max = scale * dim.largest_elem();

    return iterate_u(clctx, cqueue, setup, iterate, extract, dim, c_at_max, err, data.dimension);
}
