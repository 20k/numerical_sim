#include "common.cl"

///if we're a 1, and around us is something that's not a 1, we're a border point
bool is_exact_border_point(float x, float y, float z, float scale, int4 dim)
{
    if(sponge_damp_coeff(x, y, z, scale, dim) < 1)
        return 0;

    int3 points[6] = {{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};

    for(int i=0; i < 6; i++)
    {
        int3 offset = points[i];

         if(sponge_damp_coeff(x + offset.x, y + offset.y, z + offset.z, scale, dim) < 1)
            return 1;
    }

    return 0;
}

///if we're surrounded entirely by 1s, we're a deep boundary point
bool is_deep_boundary_point(float x, float y, float z, float scale, int4 dim)
{
    return sponge_damp_coeff(x, y, z, scale, dim) == 1 && !is_exact_border_point(x, y, z, scale, dim);
}

bool is_low_order_evolved_point(float x, float y, float z, float scale, int4 dim)
{
    if(is_exact_border_point(x, y, z, scale, dim) == 1)
        return 0;

    if(sponge_damp_coeff(x, y, z, scale, dim) == 1)
        return 0;

    #pragma unroll
    for(int iz=-BORDER_WIDTH; iz <= BORDER_WIDTH; iz++)
    {
        #pragma unroll
        for(int iy=-BORDER_WIDTH; iy <= BORDER_WIDTH; iy++)
        {
            #pragma unroll
            for(int ix=-BORDER_WIDTH; ix <= BORDER_WIDTH; ix++)
            {
                if(is_exact_border_point(x + ix, y + iy, z + iz, scale, dim) == 1)
                    return 1;
            }
        }
    }

    return 0;
}

bool is_regular_order_evolved_point(float x, float y, float z, float scale, int4 dim)
{
    if(sponge_damp_coeff(x, y, z, scale, dim) == 1)
        return 0;

    #pragma unroll
    for(int iz=-BORDER_WIDTH; iz <= BORDER_WIDTH; iz++)
    {
        #pragma unroll
        for(int iy=-BORDER_WIDTH; iy <= BORDER_WIDTH; iy++)
        {
            #pragma unroll
            for(int ix=-BORDER_WIDTH; ix <= BORDER_WIDTH; ix++)
            {
                if(is_exact_border_point(x + ix, y + iy, z + iz, scale, dim) == 1)
                    return 0;
            }
        }
    }

    return 1;
}

bool valid_point(float ix, float iy, float iz, float scale, int4 dim)
{
    return is_regular_order_evolved_point(ix, iy, iz, scale, dim) ||
       is_low_order_evolved_point(ix, iy, iz, scale, dim) ||
       is_exact_border_point(ix, iy, iz, scale, dim);
}

__kernel
void generate_evolution_points(__global ushort4* points_1st, __global int* point_count_1st,
                               __global ushort4* points_2nd, __global int* point_count_2nd,
                               __global ushort4* points_border, __global int* point_count_border,
                               __global ushort4* points_all, __global int* point_count_all,
                               __global ushort* order_ptr,
                               float scale, int4 dim)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    if(is_regular_order_evolved_point(ix, iy, iz, scale, dim) ||
       is_low_order_evolved_point(ix, iy, iz, scale, dim))
    {
        int idx = atomic_inc(point_count_1st);

        points_1st[idx].xyz = (ushort3)(ix, iy, iz);
    }

    if(is_regular_order_evolved_point(ix, iy, iz, scale, dim) ||
       is_low_order_evolved_point(ix, iy, iz, scale, dim))
    {
        int idx = atomic_inc(point_count_2nd);

        points_2nd[idx].xyz = (ushort3)(ix, iy, iz);
    }

    if(is_regular_order_evolved_point(ix, iy, iz, scale, dim) ||
       is_low_order_evolved_point(ix, iy, iz, scale, dim) ||
       is_exact_border_point(ix, iy, iz, scale, dim))
    {
        int idx = atomic_inc(point_count_all);

        points_all[idx].xyz = (ushort3)(ix, iy, iz);
    }

    int index = IDX(ix, iy, iz);

    if(is_regular_order_evolved_point(ix, iy, iz, scale, dim))
    {
        order_ptr[index] = D_FULL;
    }

    if(is_low_order_evolved_point(ix, iy, iz, scale, dim))
    {
        order_ptr[index] = D_LOW;
    }

    if(is_exact_border_point(ix, iy, iz, scale, dim))
    {
        int border_idx = atomic_inc(point_count_border);

        points_border[border_idx].xyz = (ushort3)(ix, iy, iz);

        bool valid_px = valid_point(ix+1, iy, iz, scale, dim);// && valid_point(ix+2, iy, iz, scale, dim);
        bool valid_nx = valid_point(ix-1, iy, iz, scale, dim);// && valid_point(ix-2, iy, iz, scale, dim);

        bool valid_py = valid_point(ix, iy+1, iz, scale, dim);// && valid_point(ix, iy+2, iz, scale, dim);
        bool valid_ny = valid_point(ix, iy-1, iz, scale, dim);// && valid_point(ix, iy-2, iz, scale, dim);

        bool valid_pz = valid_point(ix, iy, iz+1, scale, dim);// && valid_point(ix, iy, iz+2, scale, dim);
        bool valid_nz = valid_point(ix, iy, iz-1, scale, dim);// && valid_point(ix, iy, iz-2, scale, dim);

        if(!valid_px && !valid_nx)
        {
            printf("Error! No valid point x for %i %i %i\n", ix, iy, iz);
        }

        if(!valid_py && !valid_ny)
        {
            printf("Error! No valid point x for %i %i %i\n", ix, iy, iz);
        }

        if(!valid_pz && !valid_nz)
        {
            printf("Error! No valid point x for %i %i %i\n", ix, iy, iz);
        }

        ushort out = 0;

        if(valid_px && valid_nx)
        {
            out |= D_BOTH_PX;
        }
        else if(valid_px)
        {
            out |= D_ONLY_PX;
        }

        if(valid_py && valid_ny)
        {
            out |= D_BOTH_PY;
        }
        else if(valid_py)
        {
            out |= D_ONLY_PY;
        }

        if(valid_pz && valid_nz)
        {
            out |= D_BOTH_PZ;
        }
        else if(valid_pz)
        {
            out |= D_ONLY_PZ;
        }

        order_ptr[index] = out;
    }
}
