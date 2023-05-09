#include "common.cl"

__kernel
void fetch_linear_value(__global float* buffer, __global float* single_out, float4 position, int4 dim)
{
    if(get_global_id(0) > 1)
        return;

    *single_out = buffer_read_linear(buffer, position.xyz, dim);
}
