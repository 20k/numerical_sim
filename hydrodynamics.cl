#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#include "common.cl"
#include "evolution_common.cl"
#include "transform_position.cl"

#define MIN_P_STAR 1e-6f

#define HYDRO_ORDER 2

///this is incorrect due to intermediates needing to be 0 ?
__kernel
void calculate_hydro_evolved(__global const ushort4* points, int point_count,
                             STANDARD_CONST_ARGS(),
                             float scale, int4 dim, __global const ushort* order_ptr,
                             __global char* restrict should_evolve)
{
    int local_idx = get_global_id(0);

    if(local_idx >= point_count)
        return;

    int ix = points[local_idx].x;
    int iy = points[local_idx].y;
    int iz = points[local_idx].z;

    int index = IDX(ix, iy, iz);
    int order = order_ptr[index];

    if((order & D_FULL) == 0 && (order & D_LOW) == 0)
    {
        should_evolve[index] = false;
        return;
    }

    float P_count = 0;

    #pragma unroll
    for(int i=-HYDRO_ORDER; i <= HYDRO_ORDER; i++)
    {
        P_count += max(Dp_star[IDX(ix + i, iy, iz)], 0.f);
    }

    #pragma unroll
    for(int i=-HYDRO_ORDER; i <= HYDRO_ORDER; i++)
    {
        if(i == 0)
            continue;

        P_count += max(Dp_star[IDX(ix, iy + i, iz)], 0.f);
    }

    #pragma unroll
    for(int i=-HYDRO_ORDER; i <= HYDRO_ORDER; i++)
    {
        if(i == 0)
            continue;

        P_count += max(Dp_star[IDX(ix, iy, iz + i)], 0.f);
    }

    should_evolve[index] = P_count > 0;
}

///does not use any derivatives
__kernel
void calculate_hydro_intermediates(__global const ushort4* points, int point_count,
                                   STANDARD_CONST_ARGS(),
                                   __global float* restrict pressure,
                                   __global float* restrict hW,
                                   float scale, int4 dim, __global const ushort* order_ptr, __global const char* restrict should_evolve)
{
    int local_idx = get_global_id(0);

    if(local_idx >= point_count)
        return;

    int ix = points[local_idx].x;
    int iy = points[local_idx].y;
    int iz = points[local_idx].z;

    int index = IDX(ix, iy, iz);
    int order = order_ptr[index];

    int any_valid = 0;

    #pragma unroll
    for(int i=-HYDRO_ORDER; i <= HYDRO_ORDER; i++)
    {
        any_valid += should_evolve[IDX(ix + i, iy, iz)];
    }

    #pragma unroll
    for(int i=-HYDRO_ORDER; i <= HYDRO_ORDER; i++)
    {
        if(i == 0)
            continue;

        any_valid += should_evolve[IDX(ix, iy + i, iz)];
    }

    #pragma unroll
    for(int i=-HYDRO_ORDER; i <= HYDRO_ORDER; i++)
    {
        if(i == 0)
            continue;

        any_valid += should_evolve[IDX(ix, iy, iz + i)];
    }

    if(any_valid == 0)
        return;

    if(Dp_star[index] < MIN_P_STAR)
    {
        pressure[index] = 0;
        hW[index] = 0;
        return;
    }

    float TEMPORARIEShydrointermediates;

    float cpress = init_pressure;
    float W_var = init_W;

    pressure[index] = cpress;
    hW[index] = W_var;

    NANCHECK(pressure);
    NANCHECK(hW);
}

__kernel
void add_hydro_artificial_viscosity(__global const ushort4* points, int point_count,
                                    STANDARD_CONST_ARGS(),
                                    __global float* restrict pressure,
                                    __global const float* restrict hW,
                                    float scale, int4 dim, __global const ushort* order_ptr, __global const char* restrict should_evolve)
{

    int local_idx = get_global_id(0);

    if(local_idx >= point_count)
        return;

    int ix = points[local_idx].x;
    int iy = points[local_idx].y;
    int iz = points[local_idx].z;

    int index = IDX(ix, iy, iz);
    int order = order_ptr[index];

    if(((order & D_FULL) == 0 && ((order & D_LOW) == 0)) || should_evolve[index] == 0)
        return;

    float TEMPORARIEShydroviscosity;

    float added = init_artificial_viscosity;

    if(added == 0)
        return;

    pressure[IDX(ix,iy,iz)] += added;
}

///does use derivatives
__kernel
void evolve_hydro_all(__global const ushort4* points, int point_count,
                      STANDARD_CONST_ARGS(),
                      STANDARD_ARGS(o),
                      STANDARD_CONST_ARGS(base_),
                      __global const float* restrict pressure,
                      __global const float* restrict hW,
                      float scale, int4 dim, __global const ushort* order_ptr, __global const char* restrict should_evolve, float timestep)
{
    int local_idx = get_global_id(0);

    if(local_idx >= point_count)
        return;

    int ix = points[local_idx].x;
    int iy = points[local_idx].y;
    int iz = points[local_idx].z;

    int index = IDX(ix, iy, iz);
    int order = order_ptr[index];

    ///we're copying over base. Is that correct? Because sommerfeld
    ///for hydro this is likely a significant overhead
    if(((order & D_FULL) == 0 && ((order & D_LOW) == 0)) || should_evolve[index] == 0)
    {
        oDp_star[index] = Dp_star[index];
        oDe_star[index] = De_star[index];

        oDcS0[index] = DcS0[index];
        oDcS1[index] = DcS1[index];
        oDcS2[index] = DcS2[index];
        return;
    }

    float f_dtp_star = init_dtp_star;

    float base_p_star =  base_Dp_star[index];
    float fin_p_star = f_dtp_star * timestep + base_p_star;

    LNANCHECK(base_p_star);

    float TEMPORARIEShydrofinal;

    if(fin_p_star <= MIN_P_STAR)
    {
        oDp_star[index] = 0;
        oDe_star[index] = 0;
        oDcS0[index] = 0;
        oDcS1[index] = 0;
        oDcS2[index] = 0;

        return;
    }

    float f_dte_star = init_dte_star;
    float f_dtSk0 = init_dtSk0;
    float f_dtSk1 = init_dtSk1;
    float f_dtSk2 = init_dtSk2;

    float base_e_star = base_De_star[index];
    float base_cS0 = base_DcS0[index];
    float base_cS1 = base_DcS1[index];
    float base_cS2 = base_DcS2[index];

    LNANCHECK(base_e_star);
    LNANCHECK(base_cS0);
    LNANCHECK(base_cS1);
    LNANCHECK(base_cS2);

    float fin_e_star = f_dte_star * timestep + base_e_star;
    float fin_cS0 = f_dtSk0 * timestep + base_cS0;
    float fin_cS1 = f_dtSk1 * timestep + base_cS1;
    float fin_cS2 = f_dtSk2 * timestep + base_cS2;

    ///clamping to 0.05 this fixes some issues
    ///this makes a big difference to stability around collisions
    fin_cS0 = clamp(fin_cS0, -0.05f, 0.05f);
    fin_cS1 = clamp(fin_cS1, -0.05f, 0.05f);
    fin_cS2 = clamp(fin_cS2, -0.05f, 0.05f);

    /*fin_cS0 = clamp(fin_cS0, -1.f, 1.f);
    fin_cS1 = clamp(fin_cS1, -1.f, 1.f);
    fin_cS2 = clamp(fin_cS2, -1.f, 1.f);*/

    ///?
    if(fin_p_star < 1e-5 * p_star_max)
    {
        fin_e_star = min(fin_e_star, 10 * fin_p_star);
    }

    /*if(fin_p_star > 1)
    {
        fin_p_star = 1;
        fin_e_star = min(fin_e_star, 10 * fin_p_star);
    }*/

    ///this *does* seem to help
    /*if(X[index] < 0.1)
    {
        fin_p_star = 0;
        fin_e_star = 0;

        fin_cS0 = 0;
        fin_cS1 = 0;
        fin_cS2 = 0;
    }*/

    float area_half_width = scale * max(max(dim.x, dim.y), dim.z) / 2.f;

    float3 offset = transform_position(ix, iy, iz, dim, scale);

    float my_radius = fast_length(offset);

    float diss = 5;

    ///either interior to the black hole, or near the border. The latter is kind of hacky
    if(GET_GA < 0.15f || my_radius >= area_half_width * 0.85f)
    {
        fin_p_star += (0 - fin_p_star) * timestep * diss;
        fin_e_star += (0 - fin_e_star) * timestep * diss;

        fin_cS0 += (0 - fin_cS0) * timestep * diss;
        fin_cS1 += (0 - fin_cS1) * timestep * diss;
        fin_cS2 += (0 - fin_cS2) * timestep * diss;
    }

    fin_p_star = max(fin_p_star, 0.f);
    fin_e_star = max(fin_e_star, 0.f);

    oDp_star[index] = fin_p_star;
    oDe_star[index] = fin_e_star;

    oDcS0[index] = fin_cS0;
    oDcS1[index] = fin_cS1;
    oDcS2[index] = fin_cS2;

    NANCHECK(oDp_star);
    NANCHECK(oDe_star);
    NANCHECK(oDcS0);
    NANCHECK(oDcS1);
    NANCHECK(oDcS2);

    //if(ix == 97 && iy == 124 && iz == 124)
    /*if(ix == 94 && iy == 123 && iz == 125)
    {
        printf("McSigh p* %f e* %f cS0 %f cS1 %f cS2 %f lhs %f rhs %f fulldt %f\n", oDp_star[index], oDe_star[index], oDcS0[index], oDcS1[index], oDcS2[index], lhs_dtsk0, rhs_dtsk0, f_dtSk0);
    }*/


    /*if(ix == 98 && iy == 125 && iz == 125)
    {
        printf("Base? %f\n", base_e_star);

        printf("dg1 %.24f %f %f\n", DINTERIOR, DP1, DP2);

        printf("AVAL %f pq %f irhs %f p0eps %f rhsdte %f lhs %f estar %f %f %f\n", DBG_A, DBG_PQVIS, DBG_IRHS, DBG_p0eps, DBG_RHS_DTESTAR, DBG_LHS_DTESTAR, DBG_ESTARVI0, DBG_ESTARVI1, DBG_ESTARVI2);
    }*/
}

__kernel
void hydro_advect(__global const ushort4* points, int point_count,
                  STANDARD_CONST_ARGS(),
                  __global const float* restrict hW,
                  __global const float* quantity_base,
                  __global const float* quantity_in,
                  __global float* quantity_out,
                  float scale, int4 dim, __global const ushort* order_ptr, __global const char* restrict should_evolve, float timestep)
{
    int local_idx = get_global_id(0);

    if(local_idx >= point_count)
        return;

    int ix = points[local_idx].x;
    int iy = points[local_idx].y;
    int iz = points[local_idx].z;

    int index = IDX(ix, iy, iz);
    int order = order_ptr[index];

    if(((order & D_FULL) == 0 && ((order & D_LOW) == 0)) || should_evolve[index] == 0)
    {
        quantity_out[IDX(ix,iy,iz)] = quantity_in[IDX(ix,iy,iz)];
        return;
    }

    float TEMPORARIEShydroadvect;

    float f_quantity = HYDRO_ADVECT;

    float fin = f_quantity * timestep + quantity_base[index];

    //fin = clamp(fin, 0.f, 1.f);

    if(fin < 0)
        fin = 0;

    if(Dp_star[index] < MIN_P_STAR)
        fin = 0;

    quantity_out[index] = fin;
}

__kernel
void calculate_hydrodynamic_initial_conditions(STANDARD_ARGS(),
                                               __global const float* restrict pressure_in,
                                               __global const float* restrict rho_in,
                                               __global const float* restrict rhoH_in,
                                               __global const float* restrict p0_in,
                                               __global const float* restrict Si0_in,
                                               __global const float* restrict Si1_in,
                                               __global const float* restrict Si2_in,
                                               __global const float* restrict colour0_in,
                                               __global const float* restrict colour1_in,
                                               __global const float* restrict colour2_in,
                                               __global const float* restrict tov_phi,
                                               float scale, int4 dim,
                                               int use_colour)
{
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);

    if(ix >= dim.x || iy >= dim.y || iz >= dim.z)
        return;

    int index = IDX(ix,iy,iz);

    float3 offset = transform_position(ix, iy, iz, dim, scale);

    float ox = offset.x;
    float oy = offset.y;
    float oz = offset.z;

    float TEMPORARIEShydroconvert;

    float dp_val = build_p_star;
    float de_val = build_e_star;

    float cS0 = build_sk0;
    float cS1 = build_sk1;
    float cS2 = build_sk2;

    //printf("w2 %f %i %i %i\n", debug_w2, ix, iy,iz);

    ///dp_val and cS are both regular
    if(dp_val < MIN_P_STAR)
    {
        dp_val = 0;
        de_val = 0;
        cS0 = 0;
        cS1 = 0;
        cS2 = 0;
    }

    dp_val = max(dp_val, 0.f);
    de_val = max(de_val, 0.f);

    /*LNANCHECK(D_eps_p0);
    LNANCHECK(D_p0);
    LNANCHECK(D_h);
    LNANCHECK(D_pressure);
    LNANCHECK(D_gA_u0);

    if(D_eps_p0 < 0)
    {
        printf("ep0 %f h %f p0 %f press %f p* %f rho %f rhoH %f conf press %f p0 %f phi %f W2 %f littlee %f eps %f\n", D_eps_p0, D_h, D_p0, D_pressure, D_p_star, D_rho, D_rhoH, D_conformal_pressure, D_conformal_rest_mass, D_phi, D_W2, D_littlee, D_eps);
    }*/

    //printf("%f %f\n", D_enthalpy, D_gA_u0);

    /*if(ix == 107 && iy == 125 && iz == 125)
    {
        printf("Dtc %f %f %f %f %f %f %f %f\n", D_p0, D_gA, D_u0, D_chip, D_X, D_phi, D_u, D_DYN);
    }*/


    /*if((ix == 98 || ix == 99) && iy == 125 && iz == 125)
    {
        printf("Debugging yay %f %f %f\n", p0D, pD, phiasdf);
    }*/

    Dp_star[index] = dp_val;
    De_star[index] = de_val;

    DcS0[index] = cS0;
    DcS1[index] = cS1;
    DcS2[index] = cS2;

    NANCHECK(Dp_star);
    NANCHECK(De_star);
    NANCHECK(DcS0);
    NANCHECK(DcS1);
    NANCHECK(DcS2);

    #ifdef HAS_COLOUR
    if(use_colour)
    {
        dRed[index] = (build_cR) * dp_val;
        dGreen[index] = (build_cG) * dp_val;
        dBlue[index] = (build_cB) * dp_val;
    }
    #endif

    ///89.000000 106.000000 106.000000
    /*if(ix == 87 && iy == 106 && iz == 106)
    {
        printf("Si %f %f %f p* %f\n", cS0, cS1, cS2, dp_val);
    }*/

    /*Dp_star[index] = 0;
    De_star[index] = 0;
    DcS0[index] = 0;
    DcS1[index] = 0;
    DcS2[index] = 0;*/
}
