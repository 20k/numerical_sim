#include "common.cl"
#include "transform_position.cl"

struct matter_data
{
    float4 position;
    float4 linear_momentum;
    float4 angular_momentum;
    float4 colour;
    float mass;
    float compactness;
};

#ifdef INITIAL_BCAIJ
__kernel
void calculate_bcAij(__global float* tov_phi,
                     __global float* bcAij0,  __global float* bcAij1,  __global float* bcAij2,  __global float* bcAij3,  __global float* bcAij4,  __global float* bcAij5,
                     float scale, int4 dim)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    if(ix < 1 || iy < 1 || iz < 1 || ix >= dim.x - 1 || iy >= dim.y - 1 || iz >= dim.z - 1)
        return;

    float3 offset = transform_position(ix, iy, iz, dim, scale);

    float ox = offset.x;
    float oy = offset.y;
    float oz = offset.z;

    float TEMPORARIESbcaij;

    bcAij0[IDX(ix,iy,iz)] = B_BCAIJ_0;
    bcAij1[IDX(ix,iy,iz)] = B_BCAIJ_1;
    bcAij2[IDX(ix,iy,iz)] = B_BCAIJ_2;
    bcAij3[IDX(ix,iy,iz)] = B_BCAIJ_3;
    bcAij4[IDX(ix,iy,iz)] = B_BCAIJ_4;
    bcAij5[IDX(ix,iy,iz)] = B_BCAIJ_5;
}
#endif // INITIAL_BCAIJ

#ifdef INITIAL_BCAIJ_2
__kernel
void calculate_bcAij(__global struct matter_data* data,
                     __global float* tov_phi,
                     __global float* bcAij0,  __global float* bcAij1,  __global float* bcAij2,  __global float* bcAij3,  __global float* bcAij4,  __global float* bcAij5,
                     float scale, int4 dim)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    if(ix < 1 || iy < 1 || iz < 1 || ix >= dim.x - 1 || iy >= dim.y - 1 || iz >= dim.z - 1)
        return;

    float3 offset = transform_position(ix, iy, iz, dim, scale);

    float ox = offset.x;
    float oy = offset.y;
    float oz = offset.z;

    float TEMPORARIESbcaij;

    bcAij0[IDX(ix,iy,iz)] = B_BCAIJ_0;
    bcAij1[IDX(ix,iy,iz)] = B_BCAIJ_1;
    bcAij2[IDX(ix,iy,iz)] = B_BCAIJ_2;
    bcAij3[IDX(ix,iy,iz)] = B_BCAIJ_3;
    bcAij4[IDX(ix,iy,iz)] = B_BCAIJ_4;
    bcAij5[IDX(ix,iy,iz)] = B_BCAIJ_5;
}
#endif // INITIAL_BCAIJ_2

#ifdef INITIAL_PPW2P_2
__kernel
void calculate_ppw2p(__global struct matter_data* data,
                     __global float* tov_phi,
                     __global float* ppw2p,
                     float scale, int4 dim)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    if(ix < 1 || iy < 1 || iz < 1 || ix >= dim.x - 1 || iy >= dim.y - 1 || iz >= dim.z - 1)
        return;

    float3 offset = transform_position(ix, iy, iz, dim, scale);

    float ox = offset.x;
    float oy = offset.y;
    float oz = offset.z;

    float TEMPORARIESppw2p;

    ppw2p[IDX(ix,iy,iz)] = B_PPW2P;
}
#endif // INITIAL_PPW2P_2

#ifdef ACCUM_MATTER_VARIABLES
__kernel
void accum_matter_variables(__global float* tov_phi, ///named so because of convention for code gen
                            __global float* bcAij0_in,  __global float* bcAij1_in,  __global float* bcAij2_in,  __global float* bcAij3_in,  __global float* bcAij4_in,  __global float* bcAij5_in,
                            __global float* ppw2p_in,

                            __global float* tov_phi_accum,
                            __global float* bcAij0,  __global float* bcAij1,  __global float* bcAij2,  __global float* bcAij3,  __global float* bcAij4,  __global float* bcAij5,
                            __global float* ppw2p,
                            float scale, int4 dim)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    if(ix < 1 || iy < 1 || iz < 1 || ix >= dim.x - 1 || iy >= dim.y - 1 || iz >= dim.z - 1)
        return;

    float3 offset = transform_position(ix, iy, iz, dim, scale);

    float ox = offset.x;
    float oy = offset.y;
    float oz = offset.z;

    float TEMPORARIESaccum;

    bcAij0[IDX(ix,iy,iz)] += bcAij0_in[IDX(ix,iy,iz)];
    bcAij1[IDX(ix,iy,iz)] += bcAij1_in[IDX(ix,iy,iz)];
    bcAij2[IDX(ix,iy,iz)] += bcAij2_in[IDX(ix,iy,iz)];
    bcAij3[IDX(ix,iy,iz)] += bcAij3_in[IDX(ix,iy,iz)];
    bcAij4[IDX(ix,iy,iz)] += bcAij4_in[IDX(ix,iy,iz)];
    bcAij5[IDX(ix,iy,iz)] += bcAij5_in[IDX(ix,iy,iz)];
    ppw2p[IDX(ix,iy,iz)] += ppw2p_in[IDX(ix,iy,iz)];

    float super_tov_phi = B_TOV_PHI;

    if(super_tov_phi != 0)
        tov_phi_accum[IDX(ix,iy,iz)] = super_tov_phi;
}
#endif

#ifdef ACCUM_BLACK_HOLE_VARIABLES
__kernel
void accum_black_hole_variables(
                            __global float* bcAij0_in,  __global float* bcAij1_in,  __global float* bcAij2_in,  __global float* bcAij3_in,  __global float* bcAij4_in,  __global float* bcAij5_in,
                            __global float* bcAij0,  __global float* bcAij1,  __global float* bcAij2,  __global float* bcAij3,  __global float* bcAij4,  __global float* bcAij5,
                            float scale, int4 dim)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    if(ix < 1 || iy < 1 || iz < 1 || ix >= dim.x - 1 || iy >= dim.y - 1 || iz >= dim.z - 1)
        return;

    float TEMPORARIESaccum;

    bcAij0[IDX(ix,iy,iz)] += bcAij0_in[IDX(ix,iy,iz)];
    bcAij1[IDX(ix,iy,iz)] += bcAij1_in[IDX(ix,iy,iz)];
    bcAij2[IDX(ix,iy,iz)] += bcAij2_in[IDX(ix,iy,iz)];
    bcAij3[IDX(ix,iy,iz)] += bcAij3_in[IDX(ix,iy,iz)];
    bcAij4[IDX(ix,iy,iz)] += bcAij4_in[IDX(ix,iy,iz)];
    bcAij5[IDX(ix,iy,iz)] += bcAij5_in[IDX(ix,iy,iz)];
}
#endif

#ifdef CALCULATE_AIJ_AIJ
__kernel
void calculate_aij_aIJ(__global float* bcAij0, __global float* bcAij1, __global float* bcAij2, __global float* bcAij3, __global float* bcAij4, __global float* bcAij5,
                       __global float* aij_aIJ,
                       float scale, int4 dim)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    if(ix < 1 || iy < 1 || iz < 1 || ix >= dim.x - 1 || iy >= dim.y - 1 || iz >= dim.z - 1)
        return;

    aij_aIJ[IDX(ix,iy,iz)] = B_AIJ_AIJ;
}
#endif // CALCULATE_AIJ_AIJ

#ifdef ALL_MATTER_VARIABLES
__kernel
void multi_accumulate(__global struct matter_data* data,
                      __global float* pressure, __global float* rho, __global float* rhoH, __global float* p0,
                      __global float* Si0, __global float* Si1, __global float* Si2, __global float* colour0, __global float* colour1, __global float* colour2,
                      __global float* u_value, __global float* tov_phi, float scale, int4 dim)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    float3 offset = transform_position(ix, iy, iz, dim, scale);

    float ox = offset.x;
    float oy = offset.y;
    float oz = offset.z;

    float TEMPORARIESmultiaccumulate;

    int index = IDX(ix,iy,iz);

    pressure[index] += ACCUM_PRESSURE;
    rho[index] += ACCUM_RHO;
    rhoH[index] += ACCUM_RHOH;
    p0[index] += ACCUM_P0;
    Si0[index] += ACCUM_SI0;
    Si1[index] += ACCUM_SI1;
    Si2[index] += ACCUM_SI2;
    colour0[index] += ACCUM_COLOUR0;
    colour1[index] += ACCUM_COLOUR1;
    colour2[index] += ACCUM_COLOUR2;
}
#endif // ALL_MATTER_VARIABLES

#ifdef INITIAL_PARTICLES
#ifdef USE_64_BIT
#define ITYPE ulong
#define AADD(x, y) atom_add(x, y)
#define AINC(x) atom_inc(x)
#else
#define ITYPE int
#define AADD(x, y) atomic_add(x, y)
#define AINC(x) atomic_inc(x)
#endif

#define GET_IDX(x, i) (x * 3 + i)

#include "particle_dynamics_common.cl"


float3 voxel_to_world_unrounded(float3 pos, int4 dim, float scale)
{
    float3 centre = {(dim.x - 1)/2, (dim.y - 1)/2, (dim.z - 1)/2};

    return (pos - centre) * scale;
}

__kernel
void collect_particles(__global float* positions, ITYPE geodesic_count, __global ITYPE* collected_counts, __global ITYPE* memory_ptrs, __global ITYPE* collected_indices, float scale, int4 dim, int actually_write)
{
    size_t idx = get_global_id(0);

    if(idx >= geodesic_count)
        return;

    float3 world_pos = {positions[GET_IDX(idx, 0)], positions[GET_IDX(idx, 1)], positions[GET_IDX(idx, 2)]};

    float3 voxel_pos = world_to_voxel(world_pos, dim, scale);

    int ocx = floor(voxel_pos.x);
    int ocy = floor(voxel_pos.y);
    int ocz = floor(voxel_pos.z);

    float radius = get_particle_radius(scale);

    int spread = ceil(radius / scale) + 3;

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

                float f_sp = dirac_disc(to_centre_distance, radius);

                if(f_sp == 0)
                    continue;

                ITYPE my_index = AINC(&collected_counts[IDX(ix,iy,iz)]);

                if(actually_write)
                {
                    ITYPE my_memory_offset = memory_ptrs[IDX(ix,iy,iz)];

                    collected_indices[my_memory_offset + my_index] = idx;
                }
            }
        }
    }
}


///this kernel is unnecessarily 3d
__kernel
void memory_allocate(__global ITYPE* counts, __global ITYPE* memory_ptrs, __global ITYPE* memory_allocator, ITYPE max_memory, ulong work_size)
{
    size_t index = get_global_id(0);

    if(index >= work_size)
        return;

    ITYPE my_count = counts[index];

    ITYPE my_memory = 0;

    if(my_count > 0)
        my_memory = AADD(memory_allocator, my_count);

    if(my_memory + my_count > max_memory)
    {
        printf("Overflow in allocate in initial conditions\n");
        my_memory = 0;
    }

    memory_ptrs[index] = my_memory;
    counts[index] = 0;
}

///calculates mass * lorentz * dirac
///does not contain the X aka phi term
__kernel
void calculate_E_without_conformal(__global float* positions, __global float* masses, __global float* lorentz_in, __global float* adm_p, ITYPE geodesic_count, __global ITYPE* collected_counts, __global ITYPE* memory_ptrs, __global ITYPE* collected_indices, float scale, int4 dim)
{
    int kix = get_global_id(0);
    int kiy = get_global_id(1);
    int kiz = get_global_id(2);

    if(kix >= dim.x || kiy >= dim.y || kiz >= dim.z)
        return;

    int index = IDX(kix,kiy,kiz);

    ITYPE my_count = collected_counts[index];
    ITYPE my_memory_start = memory_ptrs[index];

    ///e == p == ph
    float vadm_p = 0;

    for(ITYPE i=0; i < my_count; i++)
    {
        ITYPE gidx = i + my_memory_start;

        ITYPE geodesic_idx = collected_indices[gidx];

        float mass = masses[geodesic_idx];

        if(mass == 0)
            continue;

        float3 world_pos = {positions[GET_IDX(geodesic_idx, 0)], positions[GET_IDX(geodesic_idx, 1)], positions[GET_IDX(geodesic_idx, 2)]};

        float3 cell_wp = voxel_to_world_unrounded((float3)(kix, kiy, kiz), dim, scale);

        float3 voxel_pos = world_to_voxel(world_pos, dim, scale);

        float base_radius = get_particle_radius(scale);

        float to_centre_distance = fast_length(cell_wp - world_pos);

        float f_sp = dirac_disc(to_centre_distance, base_radius);

        if(f_sp == 0)
            continue;

        float lorentz = lorentz_in[geodesic_idx];

        vadm_p += mass * lorentz * f_sp;
    }

    adm_p[index] = vadm_p;
}

#endif // INITIAL_PARTICLES
