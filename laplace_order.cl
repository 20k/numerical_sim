#ifndef LAPLACE_ORDER_CL_INCLUDE
#define LAPLACE_ORDER_CL_INCLUDE

#include "common.cl"

__kernel
void generate_order(__global ushort* order_ptr, int4 dim)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    bool valid_px = ix != dim.x - 1;
    bool valid_py = iy != dim.y - 1;
    bool valid_pz = iz != dim.z - 1;

    bool valid_nx = ix != 0;
    bool valid_ny = iy != 0;
    bool valid_nz = iz != 0;

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

    order_ptr[IDX(ix,iy,iz)] = out;
}

#endif // LAPLACE_ORDER_CL_INCLUDE
