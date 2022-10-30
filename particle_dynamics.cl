#include "evolution_common.cl"
#include "common.cl"
#include "transform_position.cl"

__kernel
void init_geodesics(STANDARD_ARGS(), __global float* positions3, __global float* initial_dirs3, __global float* velocities3, int geodesic_count, float scale, int4 dim)
{
    int idx = get_global_id(0);

    if(idx >= geodesic_count)
        return;

    float px = positions3[idx * 3 + 0];
    float py = positions3[idx * 3 + 1];
    float pz = positions3[idx * 3 + 2];

    float3 as_voxel = world_to_voxel((float3)(px, py, pz), dim, scale);

    as_voxel = clamp(as_voxel, (float3)(BORDER_WIDTH,BORDER_WIDTH,BORDER_WIDTH), (float3)(dim.x, dim.y, dim.z) - BORDER_WIDTH - 1);

    float fx = as_voxel.x;
    float fy = as_voxel.y;
    float fz = as_voxel.z;

    float dirx = initial_dirs3[idx * 3 + 0];
    float diry = initial_dirs3[idx * 3 + 1];
    float dirz = initial_dirs3[idx * 3 + 2];

    float vx = 0;
    float vy = 0;
    float vz = 0;

    {
        float TEMPORARIEStparticleinit;

        vx = OUT_VX;
        vy = OUT_VY;
        vz = OUT_VZ;

        printf("Tet %f %f %f %f\n", Debug_t0, Debug_t1, Debug_t2, Debug_t3);
    }

    velocities3[idx * 3 + 0] = vx;
    velocities3[idx * 3 + 1] = vy;
    velocities3[idx * 3 + 2] = vz;

    printf("Vel %f %f %f Dir %f %f %f\n", vx, vy, vz, dirx, diry, dirz);
}

///this returns the change in X, which is not velocity
///its unfortunate that position, aka X, and the conformal factor are called the same thing here
///the reason why these functions use out parameters is to work around a significant optimisation failure in AMD's opencl compiler
void velocity_to_XDiff(float3* out, float3 Xpos, float3 vel, float scale, int4 dim, STANDARD_ARGS())
{
    float3 voxel_pos = world_to_voxel(Xpos, dim, scale);

    ///isn't this already handled internally?
    voxel_pos = clamp(voxel_pos, (float3)(BORDER_WIDTH,BORDER_WIDTH,BORDER_WIDTH), (float3)(dim.x, dim.y, dim.z) - BORDER_WIDTH - 1);

    float fx = voxel_pos.x;
    float fy = voxel_pos.y;
    float fz = voxel_pos.z;

    float V0 = vel.x;
    float V1 = vel.y;
    float V2 = vel.z;

    float TEMPORARIES6;

    float d0 = X0Diff;
    float d1 = X1Diff;
    float d2 = X2Diff;

    *out = (float3){d0, d1, d2};
}

void calculate_V_derivatives(float3* out, float3 Xpos, float3 vel, float scale, int4 dim, STANDARD_ARGS())
{
    float3 voxel_pos = world_to_voxel(Xpos, dim, scale);

    ///isn't this already handled internally?
    voxel_pos = clamp(voxel_pos, (float3)(BORDER_WIDTH,BORDER_WIDTH,BORDER_WIDTH), (float3)(dim.x, dim.y, dim.z) - BORDER_WIDTH - 1);

    float fx = voxel_pos.x;
    float fy = voxel_pos.y;
    float fz = voxel_pos.z;

    float V0 = vel.x;
    float V1 = vel.y;
    float V2 = vel.z;

    float TEMPORARIES6;

    float d0 = V0Diff;
    float d1 = V1Diff;
    float d2 = V2Diff;

    *out = (float3){d0, d1, d2};
}

__kernel
void trace_geodesics(__global float* positions_in, __global float* velocities_in, __global float* positions_out, __global float* velocities_out, int geodesic_count, STANDARD_ARGS(), float scale, int4 dim, float timestep)
{
    int idx = get_global_id(0);

    if(idx >= geodesic_count)
        return;

    float3 Xpos = {positions_in[idx * 3 + 0], positions_in[idx * 3 + 1], positions_in[idx * 3 + 2]};
    float3 vel = {velocities_in[idx * 3 + 0], velocities_in[idx * 3 + 1], velocities_in[idx * 3 + 2]};

    float3 accel;
    calculate_V_derivatives(&accel, Xpos, vel, scale, dim, GET_STANDARD_ARGS());

    float3 XDiff;
    velocity_to_XDiff(&XDiff, Xpos, vel, scale, dim, GET_STANDARD_ARGS());

    //printf("In vel %f %f %f\n", vel.x, vel.y, vel.z);
    //printf("In accel %f %f %f\n", accel.x, accel.y, accel.z);

    Xpos += XDiff * timestep;
    vel += accel * timestep;

    positions_out[idx * 3 + 0] = Xpos.x;
    positions_out[idx * 3 + 1] = Xpos.y;
    positions_out[idx * 3 + 2] = Xpos.z;

    velocities_out[idx * 3 + 0] = vel.x;
    velocities_out[idx * 3 + 1] = vel.y;
    velocities_out[idx * 3 + 2] = vel.z;
}

/*float3 world_to_voxel_noround(float3 in, int4 dim, float scale)
{
    float3 centre = (float3)((dim.x - 1) / 2, (dim.y - 1)/2, (dim.z - 1)/2);

    return (in/scale) + centre
}*/

float3 voxel_to_world_unrounded(float3 pos, int4 dim, float scale)
{
    float3 centre = {(dim.x - 1)/2, (dim.y - 1)/2, (dim.z - 1)/2};

    return (pos - centre) * scale;
}

float get_f_sp(float r_rs)
{
    float f_sp = 0;

    if(r_rs <= 1)
    {
        f_sp = 1.f - (3.f/2.f) * r_rs * r_rs + (3.f/4.f) * pow(r_rs, 3.f);
    }

    else if(r_rs <= 2)
    {
        f_sp = (1.f/4.f) * pow(2 - r_rs, 3.f);
    }
    else
    {
        f_sp = 0;
    }

    return f_sp;
}

/*__kernel
void distribute_particle_spheres(__global float* positions, int geodesic_count, __global int* counts, float scale, int4 dim)
{
    int idx = get_global_id(0);

    if(idx >= geodesic_count)
        return;

    float3 world_pos = (float3)(positions_in[idx * 3 + 0], positions_in[idx * 3 + 1], positions_in[idx * 3 + 2]);

    float rs = 2 * scale;

    int spread = 6;

    for(int zz=-spread; zz <= spread; zz++)
    {
        for(int yy=-spread; yy <= spread; yy++)
        {
            for(int xx=-spread; xx <= spread; xx++)
            {
                int ix = xx + ocx;
                int iy = yy + ocy;
                int iz = zz + ocz;

                float3 cell_wp = voxel_to_world_unrounded((float3)(ix, iy, iz), dim, scale);

                float to_centre_distance = fast_length(cell_wp - world_pos);

                ///https://arxiv.org/pdf/1611.07906.pdf 20
                float r_rs = to_centre_distance / rs;

                float f_sp = get_f_sp(r_rs);

                if(f_sp == 0)
                    continue;

                int dim_index = IDX(ix,iy,iz);

                atomic_add(&counts[dim_index], 1);
            }
        }
    }
}*/

__kernel
void allocate_particle_spheres(__global int* counts, __global int* memory_ptrs, __global int* memory_allocator, int4 dim)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    int index = IDX(ix,iy,iz);

    int my_count = counts[index];
    int my_memory = atomic_add(memory_allocator, my_count);

    memory_ptrs[index] = my_memory;
}

__kernel
void collect_particle_spheres(__global float* positions, int geodesic_count, __global int* collected_count, __global int* memory_ptrs, __global int* collected_indices, __global float* collected_weights, float scale, int4 dim, int actually_write)
{
    int idx = get_global_id(0);

    if(idx >= geodesic_count)
        return;

    float3 world_pos = (float3)(positions[idx * 3 + 0], positions[idx * 3 + 1], positions[idx * 3 + 2]);

    float3 voxel_pos = world_to_voxel(world_pos, dim, scale);

    voxel_pos = clamp(voxel_pos, (float3)(BORDER_WIDTH,BORDER_WIDTH,BORDER_WIDTH), (float3)(dim.x, dim.y, dim.z) - BORDER_WIDTH - 1);

    int ocx = floor(voxel_pos.x);
    int ocy = floor(voxel_pos.y);
    int ocz = floor(voxel_pos.z);

    float rs = 2 * scale;

    int spread = 6;

    float total_weight = 0;

    if(actually_write)
    {
        for(int zz=-spread; zz <= spread; zz++)
        {
            for(int yy=-spread; yy <= spread; yy++)
            {
                for(int xx=-spread; xx <= spread; xx++)
                {
                    int ix = xx + ocx;
                    int iy = yy + ocy;
                    int iz = zz + ocz;

                    float3 cell_wp = voxel_to_world_unrounded((float3)(ix, iy, iz), dim, scale);

                    float to_centre_distance = fast_length(cell_wp - world_pos);

                    ///https://arxiv.org/pdf/1611.07906.pdf 20
                    float r_rs = to_centre_distance / rs;

                    float f_sp = get_f_sp(r_rs);

                    total_weight += f_sp;
                }
            }
        }
    }

    for(int zz=-spread; zz <= spread; zz++)
    {
        for(int yy=-spread; yy <= spread; yy++)
        {
            for(int xx=-spread; xx <= spread; xx++)
            {
                int ix = xx + ocx;
                int iy = yy + ocy;
                int iz = zz + ocz;

                float3 cell_wp = voxel_to_world_unrounded((float3)(ix, iy, iz), dim, scale);

                float to_centre_distance = fast_length(cell_wp - world_pos);

                ///https://arxiv.org/pdf/1611.07906.pdf 20
                float r_rs = to_centre_distance / rs;

                float f_sp = get_f_sp(r_rs);

                if(f_sp == 0)
                    continue;

                int my_index = atomic_inc(&collected_count[IDX(ix,iy,iz)]);

                if(actually_write)
                {
                    int my_memory_offset = memory_ptrs[IDX(ix,iy,iz)];

                    collected_indices[my_memory_offset + my_index] = idx;
                    collected_weights[my_memory_offset + my_index] = total_weight;
                }
            }
        }
    }
}

__kernel
void do_weighted_summation(__global float* positions, __global float* velocities, __global int* collected_count, __global int* memory_ptrs, __global int* collected_indices, __global float* collected_weights, STANDARD_ARGS(), float scale, int4 dim)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    int index = IDX(ix,iy,iz);

    int my_count = counts[index];
    int my_memory_start = memory_ptrs[index];

    float rs = 2 * scale;

    for(int i=0; i < my_count; i++)
    {
        int gidx = i + my_memory_start;

        int geodesic_idx = collected_indices[gidx];
        float total_weight_factor = collected_weights[gidx];

        if(total_weight_factor == 0)
            continue;

        float3 world_pos = (float3)(positions[geodesic_idx * 3 + 0], positions[geodesic_idx * 3 + 1], positions[geodesic_idx * 3 + 2]);
        float3 vel = (float3)(velocities[geodesic_idx * 3 + 0], velocities[geodesic_idx * 3 + 1], velocities[geodesic_idx * 3 + 2]);

        float3 cell_wp = voxel_to_world_unrounded((float3)(ix, iy, iz), dim, scale);

        float to_centre_distance = fast_length(cell_wp - world_pos);

        float to_centre_distance = fast_length(cell_wp - world_pos);

        ///https://arxiv.org/pdf/1611.07906.pdf 20
        float r_rs = to_centre_distance / rs;

        float f_sp = get_f_sp(r_rs) / total_weight_factor;

        if(f_sp == 0)
            continue;

        float weight = f_sp;

        {
            float TEMPORARIESadmmatter;

            float vadm_S = OUT_ADM_S;
            float vadm_Si0 = OUT_ADM_SI0;
            float vadm_Si1 = OUT_ADM_SI1;
            float vadm_Si2 = OUT_ADM_SI2;
            float vadm_Sij0 = OUT_ADM_SIJ0;
            float vadm_Sij1 = OUT_ADM_SIJ1;
            float vadm_Sij2 = OUT_ADM_SIJ2;
            float vadm_Sij3 = OUT_ADM_SIJ3;
            float vadm_Sij4 = OUT_ADM_SIJ4;
            float vadm_Sij5 = OUT_ADM_SIJ5;
            float vadm_p = OUT_ADM_P;

            adm_S[index] += vadm_S * weight;
            adm_Si0[index] += vadm_Si0 * weight;
            adm_Si1[index] += vadm_Si1 * weight;
            adm_Si2[index] += vadm_Si2 * weight;
            adm_Sij0[index] += vadm_Sij0 * weight;
            adm_Sij1[index] += vadm_Sij1 * weight;
            adm_Sij2[index] += vadm_Sij2 * weight;
            adm_Sij3[index] += vadm_Sij3 * weight;
            adm_Sij4[index] += vadm_Sij4 * weight;
            adm_Sij5[index] += vadm_Sij5 * weight;
            adm_p[index] += vadm_p * weight;
        }
    }
}

__kernel
void build_matter_sources(__global float* positions_in, __global float* velocities_in, int geodesic_count, STANDARD_ARGS(), float scale, int4 dim)
{
    int idx = get_global_id(0);
    int local_size = get_local_size(0);

    int num = ceil((float)geodesic_count / local_size);

    for(int i=0; i < num; i++)
    {
        int gidx = idx + i * local_size;

        float3 world_pos = (float3)(positions_in[gidx * 3 + 0], positions_in[gidx * 3 + 1], positions_in[gidx * 3 + 2]);
        float3 vel = (float3)(velocities_in[gidx * 3 + 0], velocities_in[gidx * 3 + 1], velocities_in[gidx * 3 + 2]);

        //printf("World Pos %f %f %f\n", world_pos.x, world_pos.y, world_pos.z);

        float3 voxel_pos = world_to_voxel(world_pos, dim, scale);

        voxel_pos = clamp(voxel_pos, (float3)(BORDER_WIDTH,BORDER_WIDTH,BORDER_WIDTH), (float3)(dim.x, dim.y, dim.z) - BORDER_WIDTH - 1);

        /*int ix = round(voxel_pos.x);
        int iy = round(voxel_pos.y);
        int iz = round(voxel_pos.z);*/

        int ocx = floor(voxel_pos.x);
        int ocy = floor(voxel_pos.y);
        int ocz = floor(voxel_pos.z);

        ///ensure that we're always smeared across several boxes
        float rs = 2 * scale;

        //printf("Rs %f\n", rs);

        int spread = 6;

        float max_contrib = 0;

        for(int zz=-spread; zz <= spread; zz++)
        {
            for(int yy=-spread; yy <= spread; yy++)
            {
                for(int xx=-spread; xx <= spread; xx++)
                {
                    int ix = xx + ocx;
                    int iy = yy + ocy;
                    int iz = zz + ocz;

                    float3 cell_wp = voxel_to_world_unrounded((float3)(ix, iy, iz), dim, scale);

                    float to_centre_distance = fast_length(cell_wp - world_pos);

                    //float weight = 1 - max(to_centre_distance / rs, 1.f);

                    ///https://arxiv.org/pdf/1611.07906.pdf 20
                    float r_rs = to_centre_distance / rs;

                    float f_sp = get_f_sp(r_rs);

                    max_contrib += f_sp;
                }
            }
        }

        for(int zz=-spread; zz <= spread; zz++)
        {
            for(int yy=-spread; yy <= spread; yy++)
            {
                for(int xx=-spread; xx <= spread; xx++)
                {
                    int ix = xx + ocx;
                    int iy = yy + ocy;
                    int iz = zz + ocz;

                    float3 cell_wp = voxel_to_world_unrounded((float3)(ix, iy, iz), dim, scale);

                    float to_centre_distance = fast_length(cell_wp - world_pos);

                    //float weight = 1 - max(to_centre_distance / rs, 1.f);

                    ///https://arxiv.org/pdf/1611.07906.pdf 20
                    float r_rs = to_centre_distance / rs;

                    float f_sp = get_f_sp(r_rs);

                    //f_sp = f_sp/(M_PI * pow(rs, 3.f));

                    float weight = f_sp / max_contrib;

                    if(weight == 0)
                        continue;

                    {
                        float TEMPORARIESadmmatter;

                        float vadm_S = OUT_ADM_S;
                        float vadm_Si0 = OUT_ADM_SI0;
                        float vadm_Si1 = OUT_ADM_SI1;
                        float vadm_Si2 = OUT_ADM_SI2;
                        float vadm_Sij0 = OUT_ADM_SIJ0;
                        float vadm_Sij1 = OUT_ADM_SIJ1;
                        float vadm_Sij2 = OUT_ADM_SIJ2;
                        float vadm_Sij3 = OUT_ADM_SIJ3;
                        float vadm_Sij4 = OUT_ADM_SIJ4;
                        float vadm_Sij5 = OUT_ADM_SIJ5;
                        float vadm_p = OUT_ADM_P;

                        int index = IDX(ix,iy,iz);

                        adm_S[index] += vadm_S * weight;
                        adm_Si0[index] += vadm_Si0 * weight;
                        adm_Si1[index] += vadm_Si1 * weight;
                        adm_Si2[index] += vadm_Si2 * weight;
                        adm_Sij0[index] += vadm_Sij0 * weight;
                        adm_Sij1[index] += vadm_Sij1 * weight;
                        adm_Sij2[index] += vadm_Sij2 * weight;
                        adm_Sij3[index] += vadm_Sij3 * weight;
                        adm_Sij4[index] += vadm_Sij4 * weight;
                        adm_Sij5[index] += vadm_Sij5 * weight;
                        adm_p[index] += vadm_p * weight;
                    }
                }
            }
        }
    }
}
