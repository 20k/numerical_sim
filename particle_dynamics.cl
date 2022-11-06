#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

#include "evolution_common.cl"
#include "common.cl"
#include "transform_position.cl"

#define GET_IDX(x, i) (x * 3 + i)

#define MASS_CUTOFF 0.00000001

__kernel
void init_geodesics(STANDARD_ARGS(), __global float* positions3_in, __global float* initial_dirs3, __global float* positions3_out, __global float* velocities3_out, ulong geodesic_count, float scale, int4 dim)
{
    size_t idx = get_global_id(0);

    if(idx >= geodesic_count)
        return;

    float px = positions3_in[idx * 3 + 0];
    float py = positions3_in[idx * 3 + 1];
    float pz = positions3_in[idx * 3 + 2];

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
    }

    positions3_out[GET_IDX(idx, 0)] = px;
    positions3_out[GET_IDX(idx, 1)] = py;
    positions3_out[GET_IDX(idx, 2)] = pz;

    ///https://arxiv.org/pdf/1611.07906.pdf (11)
    ///only if not using u formalism!!
    velocities3_out[GET_IDX(idx, 0)] = vx;
    velocities3_out[GET_IDX(idx, 1)] = vy;
    velocities3_out[GET_IDX(idx, 2)] = vz;
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
void dissipate_mass(__global float* positions, __global float* mass_in, __global float* mass_out, __global float* mass_base, ulong geodesic_count, float timestep)
{
    size_t idx = get_global_id(0);

    if(idx >= geodesic_count)
        return;

    if(mass_in[idx] <= MASS_CUTOFF)
    {
        mass_out[idx] = 0;
        return;
    }

    float3 Xpos = {positions[GET_IDX(idx, 0)], positions[GET_IDX(idx, 1)], positions[GET_IDX(idx, 2)]};

    if(fast_length(Xpos) >= MASS_CULL_SIZE)
    {
        mass_out[idx] = 0;
    }
    else
    {
        mass_out[idx] = mass_in[idx];
    }
}

__kernel
void trace_geodesics(__global float* positions_in, __global float* velocities_in,
                     __global float* positions_out, __global float* velocities_out,
                     __global float* positions_base, __global float* velocities_base,
                     __global float* masses,
                     ulong geodesic_count, STANDARD_ARGS(), float scale, int4 dim, float timestep)
{
    size_t idx = get_global_id(0);

    if(idx >= geodesic_count)
        return;

    if(masses[idx] <= MASS_CUTOFF)
        return;

    float3 Xpos = {positions_in[GET_IDX(idx, 0)], positions_in[GET_IDX(idx, 1)], positions_in[GET_IDX(idx, 2)]};
    float3 vel = {velocities_in[GET_IDX(idx, 0)], velocities_in[GET_IDX(idx, 1)], velocities_in[GET_IDX(idx, 2)]};

    float3 accel;
    calculate_V_derivatives(&accel, Xpos, vel, scale, dim, GET_STANDARD_ARGS());

    float3 XDiff;
    velocity_to_XDiff(&XDiff, Xpos, vel, scale, dim, GET_STANDARD_ARGS());

    //printf("In vel %f %f %f\n", vel.x, vel.y, vel.z);
    //printf("In accel %f %f %f\n", accel.x, accel.y, accel.z);

    //Xpos += XDiff * timestep;
    //vel += accel * timestep;

    float3 dXpos = XDiff * timestep;
    float3 dvel = accel * timestep;

    float3 base_Xpos = {positions_base[GET_IDX(idx, 0)], positions_base[GET_IDX(idx, 1)], positions_base[GET_IDX(idx, 2)]};
    float3 base_vel = {velocities_base[GET_IDX(idx, 0)], velocities_base[GET_IDX(idx, 1)], velocities_base[GET_IDX(idx, 2)]};

    float3 out_Xpos = base_Xpos + dXpos;
    float3 out_vel = base_vel + dvel;

    positions_out[GET_IDX(idx, 0)] = out_Xpos.x;
    positions_out[GET_IDX(idx, 1)] = out_Xpos.y;
    positions_out[GET_IDX(idx, 2)] = out_Xpos.z;

    velocities_out[GET_IDX(idx, 0)] = out_vel.x;
    velocities_out[GET_IDX(idx, 1)] = out_vel.y;
    velocities_out[GET_IDX(idx, 2)] = out_vel.z;
}

__kernel
void index_trace_geodesics(__global float* positions_in, __global float* velocities_in,
                           __global float* positions_out, __global float* velocities_out,
                           __global float* positions_base, __global float* velocities_base,
                           __global float* masses,
                           __global ulong* collected_indices, __global ulong* memory_alloc_count,
                          ulong geodesic_count, STANDARD_ARGS(), float scale, int4 dim, float timestep)
{
    size_t base_idx = get_global_id(0);

    if(base_idx >= geodesic_count)
        return;

    if(base_idx >= *memory_alloc_count)
        return;

    ulong idx = collected_indices[base_idx];

    if(idx >= geodesic_count)
    {
        printf("Critical error in index trace %li %li %li\n", idx, geodesic_count, *memory_alloc_count);
        return;
    }

    if(masses[idx] <= MASS_CUTOFF)
        return;

    float3 Xpos = {positions_in[GET_IDX(idx, 0)], positions_in[GET_IDX(idx, 1)], positions_in[GET_IDX(idx, 2)]};
    float3 vel = {velocities_in[GET_IDX(idx, 0)], velocities_in[GET_IDX(idx, 1)], velocities_in[GET_IDX(idx, 2)]};

    float3 accel;
    calculate_V_derivatives(&accel, Xpos, vel, scale, dim, GET_STANDARD_ARGS());

    float3 XDiff;
    velocity_to_XDiff(&XDiff, Xpos, vel, scale, dim, GET_STANDARD_ARGS());

    //printf("In vel %f %f %f\n", vel.x, vel.y, vel.z);
    //printf("In accel %f %f %f\n", accel.x, accel.y, accel.z);

    //Xpos += XDiff * timestep;
    //vel += accel * timestep;

    float3 dXpos = XDiff * timestep;
    float3 dvel = accel * timestep;

    float3 base_Xpos = {positions_base[GET_IDX(idx, 0)], positions_base[GET_IDX(idx, 1)], positions_base[GET_IDX(idx, 2)]};
    float3 base_vel = {velocities_base[GET_IDX(idx, 0)], velocities_base[GET_IDX(idx, 1)], velocities_base[GET_IDX(idx, 2)]};

    float3 out_Xpos = base_Xpos + dXpos;
    float3 out_vel = base_vel + dvel;

    positions_out[GET_IDX(idx, 0)] = out_Xpos.x;
    positions_out[GET_IDX(idx, 1)] = out_Xpos.y;
    positions_out[GET_IDX(idx, 2)] = out_Xpos.z;

    velocities_out[GET_IDX(idx, 0)] = out_vel.x;
    velocities_out[GET_IDX(idx, 1)] = out_vel.y;
    velocities_out[GET_IDX(idx, 2)] = out_vel.z;
}

__kernel
void cube_trace_geodesics(__global float* positions_in, __global float* velocities_in,
                          __global float* positions_out, __global float* velocities_out,
                          __global float* positions_base, __global float* velocities_base,
                          __global float* masses,
                          __global ulong* counts, __global ulong* memory_ptrs, __global ulong* collected_indices,
                          ulong geodesic_count, STANDARD_ARGS(), float scale, int4 dim, float timestep)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    int3 ipos = {ix, iy, iz};

    if(any(ipos < BORDER_WIDTH) || any(ipos >= dim.xyz - BORDER_WIDTH - 1))
        return;

    int buffer_index = IDX(ix,iy,iz);

    ulong my_memory_base = memory_ptrs[buffer_index];

    ulong my_count = counts[buffer_index];

    if(my_count == 0)
        return;

    for(ulong i=0; i < my_count; i++)
    {
        ulong memory_index = i + my_memory_base;

        ulong idx = collected_indices[memory_index];

        {

            float3 Xpos = {positions_in[GET_IDX(idx, 0)], positions_in[GET_IDX(idx, 1)], positions_in[GET_IDX(idx, 2)]};
            float3 vel = {velocities_in[GET_IDX(idx, 0)], velocities_in[GET_IDX(idx, 1)], velocities_in[GET_IDX(idx, 2)]};

            float3 accel;
            calculate_V_derivatives(&accel, Xpos, vel, scale, dim, GET_STANDARD_ARGS());

            float3 XDiff;
            velocity_to_XDiff(&XDiff, Xpos, vel, scale, dim, GET_STANDARD_ARGS());

            float3 dXpos = XDiff * timestep;
            float3 dvel = accel * timestep;

            float3 base_Xpos = {positions_base[GET_IDX(idx, 0)], positions_base[GET_IDX(idx, 1)], positions_base[GET_IDX(idx, 2)]};
            float3 base_vel = {velocities_base[GET_IDX(idx, 0)], velocities_base[GET_IDX(idx, 1)], velocities_base[GET_IDX(idx, 2)]};

            float3 out_Xpos = base_Xpos + dXpos;
            float3 out_vel = base_vel + dvel;

            positions_out[GET_IDX(idx, 0)] = out_Xpos.x;
            positions_out[GET_IDX(idx, 1)] = out_Xpos.y;
            positions_out[GET_IDX(idx, 2)] = out_Xpos.z;

            velocities_out[GET_IDX(idx, 0)] = out_vel.x;
            velocities_out[GET_IDX(idx, 1)] = out_vel.y;
            velocities_out[GET_IDX(idx, 2)] = out_vel.z;
        }
    }
}

__kernel
void collect_geodesics(__global float* positions, __global float* masses, ulong geodesic_count, __global ulong* counts, __global ulong* memory_ptrs, __global ulong* collected_indices, float scale, int4 dim, int actually_write)
{
    size_t idx = get_global_id(0);

    if(idx >= geodesic_count)
        return;

    if(masses[idx] <= MASS_CUTOFF)
        return;

    float3 Xpos = {positions[GET_IDX(idx, 0)], positions[GET_IDX(idx, 1)], positions[GET_IDX(idx, 2)]};

    float3 voxel_pos = world_to_voxel(Xpos, dim, scale);

    voxel_pos = clamp(voxel_pos, (float3)(BORDER_WIDTH,BORDER_WIDTH,BORDER_WIDTH), (float3)(dim.x, dim.y, dim.z) - BORDER_WIDTH - 1);

    int ix = (int)voxel_pos.x;
    int iy = (int)voxel_pos.y;
    int iz = (int)voxel_pos.z;

    /*ix /= 4;
    iy /= 2;
    iz /= 2;

    if(ix < 0 || iy < 0 || iz < 0)
        printf("WTF\n");*/

    int buffer_index = IDX(ix,iy,iz);

    ulong my_offset = atom_inc(&counts[buffer_index]);

    if(actually_write)
    {
        ulong my_memory_ptr = memory_ptrs[buffer_index];

        ulong my_index = my_offset + my_memory_ptr;

        collected_indices[my_index] = idx;
    }
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

///this kernel is unnecessarily 3d
__kernel
void memory_allocate(__global ulong* counts, __global ulong* memory_ptrs, __global ulong* memory_allocator, ulong max_memory, int4 dim)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    int index = IDX(ix,iy,iz);

    ulong my_count = counts[index];

    ulong my_memory = 0;

    if(my_count > 0)
        my_memory = atom_add(memory_allocator, my_count);

    if(my_memory + my_count > max_memory)
    {
        printf("Overflow in allocate\n");
        my_memory = 0;
    }

    memory_ptrs[index] = my_memory;

    counts[index] = 0;
}

__kernel
void collect_particle_spheres(__global float* positions, __global float* masses, ulong geodesic_count, __global ulong* collected_counts, __global ulong* memory_ptrs, __global ulong* collected_indices, __global float* collected_weights, float scale, int4 dim, int actually_write)
{
    int idx = get_global_id(0);

    if(idx >= geodesic_count)
        return;

    if(masses[idx] <= MASS_CUTOFF)
        return;

    float3 world_pos = {positions[GET_IDX(idx, 0)], positions[GET_IDX(idx, 1)], positions[GET_IDX(idx, 2)]};

    float3 voxel_pos = world_to_voxel(world_pos, dim, scale);

    voxel_pos = clamp(voxel_pos, (float3)(BORDER_WIDTH,BORDER_WIDTH,BORDER_WIDTH), (float3)(dim.x, dim.y, dim.z) - BORDER_WIDTH - 1);

    int ocx = floor(voxel_pos.x);
    int ocy = floor(voxel_pos.y);
    int ocz = floor(voxel_pos.z);

    float rs = scale;

    int spread = 3;

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

                    if(ix < 0 || iy < 0 || iz < 0 || ix >= dim.x || iy >= dim.y || iz >= dim.z)
                        continue;

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

                if(ix < 0 || iy < 0 || iz < 0 || ix >= dim.x || iy >= dim.y || iz >= dim.z)
                    continue;

                float3 cell_wp = voxel_to_world_unrounded((float3)(ix, iy, iz), dim, scale);

                float to_centre_distance = fast_length(cell_wp - world_pos);

                ///https://arxiv.org/pdf/1611.07906.pdf 20
                float r_rs = to_centre_distance / rs;

                float f_sp = get_f_sp(r_rs);

                if(f_sp == 0)
                    continue;

                //total_weight = M_PI * pow(rs, 3);

                ulong my_index = atom_inc(&collected_counts[IDX(ix,iy,iz)]);

                if(actually_write)
                {
                    ulong my_memory_offset = memory_ptrs[IDX(ix,iy,iz)];

                    collected_indices[my_memory_offset + my_index] = idx;
                    collected_weights[my_memory_offset + my_index] = total_weight;
                }
            }
        }
    }
}

__kernel
void do_weighted_summation(__global float* positions, __global float* velocities, __global float* masses, ulong geodesic_count, __global ulong* collected_counts, __global ulong* memory_ptrs, __global ulong* collected_indices, __global float* collected_weights, STANDARD_ARGS(), float scale, int4 dim)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    int index = IDX(ix,iy,iz);

    ulong my_count = collected_counts[index];
    ulong my_memory_start = memory_ptrs[index];

    float vadm_S = 0;
    float vadm_Si0 = 0;
    float vadm_Si1 = 0;
    float vadm_Si2 = 0;
    float vadm_Sij0 = 0;
    float vadm_Sij1 = 0;
    float vadm_Sij2 = 0;
    float vadm_Sij3 = 0;
    float vadm_Sij4 = 0;
    float vadm_Sij5 = 0;
    float vadm_p = 0;

    float rs = scale;

    for(ulong i=0; i < my_count; i++)
    {
        ulong gidx = i + my_memory_start;

        ulong geodesic_idx = collected_indices[gidx];

        float mass = masses[geodesic_idx];

        if(mass <= MASS_CUTOFF)
            continue;

        float total_weight_factor = collected_weights[gidx];

        if(total_weight_factor == 0)
            continue;

        float3 world_pos = {positions[GET_IDX(geodesic_idx, 0)], positions[GET_IDX(geodesic_idx, 1)], positions[GET_IDX(geodesic_idx, 2)]};
        float3 vel = {velocities[GET_IDX(geodesic_idx, 0)], velocities[GET_IDX(geodesic_idx, 1)], velocities[GET_IDX(geodesic_idx, 2)]};

        float3 cell_wp = voxel_to_world_unrounded((float3)(ix, iy, iz), dim, scale);

        float to_centre_distance = fast_length(cell_wp - world_pos);

        ///https://arxiv.org/pdf/1611.07906.pdf 20
        float r_rs = to_centre_distance / rs;

        float f_sp = get_f_sp(r_rs) / total_weight_factor;

        if(f_sp == 0)
            continue;

        float weight = f_sp;

        /*float3 vector_from_particle = cell_wp - world_pos;

        float weight = 1;*/

        {
            //float gamma = lorentzs[geodesic_idx];

            float TEMPORARIESadmmatter;

            vadm_S += OUT_ADM_S * weight;
            vadm_Si0 += OUT_ADM_SI0 * weight;
            vadm_Si1 += OUT_ADM_SI1 * weight;
            vadm_Si2 += OUT_ADM_SI2 * weight;
            vadm_Sij0 += OUT_ADM_SIJ0 * weight;
            vadm_Sij1 += OUT_ADM_SIJ1 * weight;
            vadm_Sij2 += OUT_ADM_SIJ2 * weight;
            vadm_Sij3 += OUT_ADM_SIJ3 * weight;
            vadm_Sij4 += OUT_ADM_SIJ4 * weight;
            vadm_Sij5 += OUT_ADM_SIJ5 * weight;
            vadm_p += OUT_ADM_P * weight;

            /*if(vadm_p > 0)
            {
                printf("Pos %i %i %i\n", ix, iy, iz);
            }*/

            ///138 128 106
            ///55 105 106
            ///48 110 107
            //if(ix == 138 && iy == 128 && iz == 106)
            /*if(ix == 50 && iy == 110 && iz == 105)
            {
                printf("Adm p %f i %i max %i lorentz %f lazy_det %f\n", OUT_ADM_P, i, my_count, calculated_gamma, lazy_det);
            }*/
        }
    }

    if(vadm_p > 0 || (vadm_p == 0 && adm_p[index] != 0))
    {
        adm_S[index] = vadm_S;
        adm_Si0[index] = vadm_Si0;
        adm_Si1[index] = vadm_Si1;
        adm_Si2[index] = vadm_Si2;
        adm_Sij0[index] = vadm_Sij0;
        adm_Sij1[index] = vadm_Sij1;
        adm_Sij2[index] = vadm_Sij2;
        adm_Sij3[index] = vadm_Sij3;
        adm_Sij4[index] = vadm_Sij4;
        adm_Sij5[index] = vadm_Sij5;
        adm_p[index] = vadm_p;
    }
}
