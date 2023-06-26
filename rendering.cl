#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#include "transform_position.cl"
#include "common.cl"
#include "evolution_common.cl"
#include "evolve_points.cl"

///because we need to cutoff slightly before the real edge due to various factors
#define RENDERING_CUTOFF_MULT 0.95f

float srgb_to_lin_single(float in)
{
    if(in < 0.04045f)
        return in / 12.92f;
    else
        return pow((in + 0.055f) / 1.055f, 2.4f);
}

float3 srgb_to_lin(float3 in)
{
    return (float3)(srgb_to_lin_single(in.x), srgb_to_lin_single(in.y), srgb_to_lin_single(in.z));
}

struct lightray_simple
{
    float lp1;
    float lp2;
    float lp3;

    float V0;
    float V1;
    float V2;

    int x, y;

    float iter_frac;
    int hit_type;
    //float density;

    float R, G, B;
    float ku_uobsu;
};

enum ds_result
{
    DS_NONE,
    DS_SKIP,
    DS_RETURN,
};

int calculate_ds_error(float err, float current_ds, float3 next_acceleration, float* next_ds_out)
{
    #define MIN_STEP 0.5f
    #define MAX_STEP 2.f

    float next_ds = err * 1/fast_length(next_acceleration);

    ///produces strictly worse results for kerr
    //next_ds = 0.99f * current_ds * clamp(next_ds / current_ds, 0.1f, 4.f);

    next_ds = clamp(next_ds, MIN_STEP, MAX_STEP);

    *next_ds_out = next_ds;

    //if(next_ds == MIN_STEP)
    //    return DS_RETURN;

    if(next_ds < current_ds/1.2f)
        return DS_SKIP;

    return DS_NONE;
}

///https://www.ccs.neu.edu/home/fell/CS4300/Lectures/Ray-TracingFormulas.pdf
float3 fix_ray_position(float3 cartesian_pos, float3 cartesian_velocity, float sphere_radius)
{
    cartesian_velocity = fast_normalize(cartesian_velocity);

    float3 C = (float3){0,0,0};

    float a = 1;
    float b = 2 * dot(cartesian_velocity, (cartesian_pos - C));
    float c = dot(C, C) + dot(cartesian_pos, cartesian_pos) - 2 * (dot(cartesian_pos, C)) - sphere_radius * sphere_radius;

    float discrim = b*b - 4 * a * c;

    if(discrim < 0)
        return cartesian_pos;

    float t0 = (-b - native_sqrt(discrim)) / (2 * a);
    float t1 = (-b + native_sqrt(discrim)) / (2 * a);

    float my_t = 0;

    if(fabs(t0) < fabs(t1))
        my_t = t0;
    else
        my_t = t1;

    return cartesian_pos + my_t * cartesian_velocity;
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

struct render_ray_info
{
    float X, Y, Z;
    float dX, dY, dZ;

    int hit_type;

    float R, G, B;
    float background_power;

    int x, y;
    float zp1;
};


///takes a linear colour
float3 redshift(float3 v, float z)
{
    z = max(z, -0.999f);

    ///1 + z = gtt(recv) / gtt(src)
    ///1 + z = lnow / lthen
    ///1 + z = wsrc / wobs

    float radiant_energy = v.x*0.2125f + v.y*0.7154f + v.z*0.0721f;

    float3 red = (float3){1/0.2125f, 0.f, 0.f};
    float3 blue = (float3){0.f, 0.f, 1/0.0721};

    float3 result;

    if(z > 0)
    {
        result = mix(v, radiant_energy * red, tanh(z));
    }
    else
    {
        float iv1pz = (1/(1 + z)) - 1;

        result = mix(v, radiant_energy * blue, tanh(iv1pz));
    }

    result = clamp(result, 0.f, 1.f);

    return result;
}

float3 redshift_with_intensity(float3 lin_result, float z_shift)
{
    z_shift = max(z_shift, -0.999f);

    ///linf / le = z + 1
    ///le =  linf / (z + 1)

    ///So, this is an incredibly, incredibly gross approximation
    ///there are several problems here
    ///1. Fundamentally I do not have a spectrographic map of the surrounding universe, which means any data is very approximate
    ///EG blueshifting of infrared into visible light is therefore impossible
    ///2. Converting sRGB information into wavelengths is possible, but also unphysical
    ///This might be a worthwhile approximation as it might correctly bunch frequencies together
    ///3. Its not possible to correctly render red/blueshifting, so it maps the range [-1, +inf] to [red, blue], mixing the colours with parameter [x <= 0 -> abs(x), x > 0 -> tanh(x)]]
    ///this means that even if I did all the above correctly, its still a mess

    ///This estimates luminance from the rgb value, which should be pretty ok at least!
    float real_sol = 299792458;

    ///Pick an arbitrary wavelength, the peak of human vision
    float test_wavelength = 555 / real_sol;

    float local_wavelength = test_wavelength / (z_shift + 1);

    ///this is relative luminance instead of absolute specific intensity, but relative_luminance / wavelength^3 should still be lorenz invariant (?)
    float relative_luminance = 0.2126f * lin_result.x + 0.7152f * lin_result.y + 0.0722f * lin_result.z;

    ///Iv = I1 / v1^3, where Iv is lorenz invariant
    ///Iv = I2 / v2^3 in our new frame of reference
    ///therefore we can calculate the new intensity in our new frame of reference as...
    ///I1/v1^3 = I2 / v2^3
    ///I2 = v2^3 * I1/v1^3

    float new_relative_luminance = pow(local_wavelength, 3) * relative_luminance / pow(test_wavelength, 3);

    new_relative_luminance = clamp(new_relative_luminance, 0.f, 1.f);

    if(relative_luminance > 0.00001)
    {
        lin_result = (new_relative_luminance / relative_luminance) * lin_result;

        lin_result = clamp(lin_result, 0.f, 1.f);
    }

    lin_result = redshift(lin_result, z_shift);

    lin_result = clamp(lin_result, 0.f, 1.f);

    return lin_result;
}


__kernel
void calculate_adm_texture_coordinates(__global struct render_ray_info* finished_rays, __global float2* texture_coordinates, int width, int height,
                                       float scale, int4 dim)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x >= width || y >= height)
        return;

    __global struct render_ray_info* ray = &finished_rays[y * width + x];

    if(ray->hit_type == 1)
    {
        texture_coordinates[y * width + x] = (float2){0,0};
        return;
    }

    float3 cpos = {ray->X, ray->Y, ray->Z};
    float3 cvel = {ray->dX, ray->dY, ray->dZ};

    float uni_size = universe_size;

    cpos = fix_ray_position(cpos, cvel, uni_size * RENDERING_CUTOFF_MULT);

    float fr = fast_length(cpos);
    float theta = acos(cpos.z / fr);
    float phi = atan2(cpos.y, cpos.x);

    float3 npolar = (float3)(fr, theta, phi);

    float thetaf = fmod(npolar.y, 2 * M_PI);
    float phif = npolar.z;

    if(thetaf >= M_PI)
    {
        phif += M_PI;
        thetaf -= M_PI;
    }

    phif = fmod(phif, 2 * M_PI);

    float sxf = (phif) / (2 * M_PI);
    float syf = thetaf / M_PI;

    sxf += 0.5f;

    texture_coordinates[y * width + x] = (float2)(sxf, syf);
}

struct lightray4
{
    float4 pos;
    float4 vel;
    float ku_uobsu;
};

__kernel
void init_rays4(__global struct lightray4* rays,
               STANDARD_ARGS(),
               float scale, float3 camera_pos, float4 camera_quat,
               int4 dim, int width, int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x >= width)
        return;

    if(y >= height)
        return;

    float pX;
    float pY;
    float pZ;
    float pW;

    float dX;
    float dY;
    float dZ;
    float dW;

    float ku_uobsu;

    {
        float3 world_pos = camera_pos;

        float3 voxel_pos = world_to_voxel(world_pos, dim, scale);

        float fx = voxel_pos.x;
        float fy = voxel_pos.y;
        float fz = voxel_pos.z;

        float TEMPORARIES5;

        pX = lp0_d;
        pY = lp1_d;
        pZ = lp2_d;
        pW = lp3_d;

        dX = lv0_d;
        dY = lv1_d;
        dZ = lv2_d;
        dW = lv3_d;

        ku_uobsu = GET_KU_UOBSU;
    }

    struct lightray4 ray = {};
    ray.pos = (float4){pX, pY, pZ, pW};
    ray.vel = (float4){dX, dY, dZ, dW};
    ray.ku_uobsu = ku_uobsu;

    rays[y * width + x] = ray;
}

__kernel
void init_rays(__global struct lightray_simple* rays,
                STANDARD_ARGS(),
                float scale, float3 camera_pos, float4 camera_quat,
                int4 dim, int width, int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x >= width)
        return;

    if(y >= height)
        return;

    float lp0;
    float lp1;
    float lp2;
    float lp3;

    float V0;
    float V1;
    float V2;

    float ku_uobsu;

    ///https://arxiv.org/pdf/1207.4234.pdf (12 for ku_uobsu)
    {
        float3 world_pos = camera_pos;

        float3 voxel_pos = world_to_voxel(world_pos, dim, scale);

        float fx = voxel_pos.x;
        float fy = voxel_pos.y;
        float fz = voxel_pos.z;

        float TEMPORARIES5;

        lp0 = lp0_d;
        lp1 = lp1_d;
        lp2 = lp2_d;
        lp3 = lp3_d;

        V0 = V0_d;
        V1 = V1_d;
        V2 = V2_d;

        ku_uobsu = GET_KU_UOBSU;
    }

    struct lightray_simple out;
    out.lp1 = lp1;
    out.lp2 = lp2;
    out.lp3 = lp3;

    out.V0 = V0;
    out.V1 = V1;
    out.V2 = V2;

    out.x = x;
    out.y = y;
    out.iter_frac = 0;
    out.hit_type = 0;
    out.ku_uobsu = ku_uobsu;

    rays[y * width + x] = out;
}

float length_sq(float3 in)
{
    return dot(in, in);
}

float get_static_verlet_ds(float3 Xpos, __global float* X, float scale, int4 dim)
{
    float X_far = 0.9f;
    float X_near = 0.6f;

    float3 voxel_pos = world_to_voxel(Xpos, dim, scale);
    voxel_pos = clamp(voxel_pos, (float3)(BORDER_WIDTH,BORDER_WIDTH,BORDER_WIDTH), (float3)(dim.x, dim.y, dim.z) - BORDER_WIDTH - 1);

    float fx = voxel_pos.x;
    float fy = voxel_pos.y;
    float fz = voxel_pos.z;

    float BH_X = GET_X_DS;

    float my_fraction = (clamp(BH_X, X_near, X_far) - X_near) / (X_far - X_near);

    my_fraction = clamp(my_fraction, 0.f, 1.f);

    //#ifdef TRACE_MATTER_P
    #ifdef USE_REDSHIFT
    return scale * 0.25f;
    #endif
    //#endif // TRACE_MATTER_P

    #if defined(RENDER_MATTER) || defined(TRACE_MATTER_P)
    return mix(0.4f, 4.f, my_fraction) * 0.1f;
    #else
    return mix(0.4f, 4.f, my_fraction);
    #endif
}

#define SOLID_DENSITY 0.1

float4 get_accel4(float4 pos, float4 vel, float scale, int4 dim, STANDARD_ARGS())
{
    float3 voxel_pos = world_to_voxel(pos.yzw, dim, scale);
    voxel_pos = clamp(voxel_pos, (float3)(BORDER_WIDTH,BORDER_WIDTH,BORDER_WIDTH), (float3)(dim.x, dim.y, dim.z) - BORDER_WIDTH - 1);

    float fx = voxel_pos.x;
    float fy = voxel_pos.y;
    float fz = voxel_pos.z;

    float TEMPORARIESgeo4;

    float4 out;
    out.x = ACCEL40;
    out.y = ACCEL41;
    out.z = ACCEL42;
    out.w = ACCEL43;

    return out;
}

float4 lower4(float3 Xpos, float4 upper, float scale, int4 dim, STANDARD_ARGS())
{
    float3 voxel_pos = world_to_voxel(Xpos, dim, scale);
    voxel_pos = clamp(voxel_pos, (float3)(BORDER_WIDTH,BORDER_WIDTH,BORDER_WIDTH), (float3)(dim.x, dim.y, dim.z) - BORDER_WIDTH - 1);

    float fx = voxel_pos.x;
    float fy = voxel_pos.y;
    float fz = voxel_pos.z;

    float TEMPORARIESredshift;

    return (float4){LOWER40, LOWER41, LOWER42, LOWER43};
}

float3 raise3(float3 Xpos, float3 lower, float scale, int4 dim, STANDARD_ARGS())
{
    float3 voxel_pos = world_to_voxel(Xpos, dim, scale);
    voxel_pos = clamp(voxel_pos, (float3)(BORDER_WIDTH,BORDER_WIDTH,BORDER_WIDTH), (float3)(dim.x, dim.y, dim.z) - BORDER_WIDTH - 1);

    float fx = voxel_pos.x;
    float fy = voxel_pos.y;
    float fz = voxel_pos.z;

    float TEMPORARIESredshift;

    return (float3){RAISE30, RAISE31, RAISE32};
}

float4 adm_3velocity_to_full(float3 Xpos, float3 upper, float scale, int4 dim, STANDARD_ARGS())
{
    float3 voxel_pos = world_to_voxel(Xpos, dim, scale);
    voxel_pos = clamp(voxel_pos, (float3)(BORDER_WIDTH,BORDER_WIDTH,BORDER_WIDTH), (float3)(dim.x, dim.y, dim.z) - BORDER_WIDTH - 1);

    float fx = voxel_pos.x;
    float fy = voxel_pos.y;
    float fz = voxel_pos.z;

    float TEMPORARIESredshift;

    return (float4){ADMFULL0, ADMFULL1, ADMFULL2, ADMFULL3};
}

float3 get_3vel_upper(float3 Xpos, float scale, int4 dim, STANDARD_ARGS(), STANDARD_UTILITY())
{
    float3 voxel_pos = world_to_voxel(Xpos, dim, scale);
    voxel_pos = clamp(voxel_pos, (float3)(BORDER_WIDTH,BORDER_WIDTH,BORDER_WIDTH), (float3)(dim.x, dim.y, dim.z) - BORDER_WIDTH - 1);

    float fx = voxel_pos.x;
    float fy = voxel_pos.y;
    float fz = voxel_pos.z;

    float TEMPORARIESgetmatter;

    return (float3){GET_3VEL_UPPER0, GET_3VEL_UPPER1, GET_3VEL_UPPER2};
}

float get_matter_p(float3 Xpos, float scale, int4 dim, STANDARD_ARGS(), STANDARD_UTILITY())
{
    float3 voxel_pos = world_to_voxel(Xpos, dim, scale);
    voxel_pos = clamp(voxel_pos, (float3)(BORDER_WIDTH,BORDER_WIDTH,BORDER_WIDTH), (float3)(dim.x, dim.y, dim.z) - BORDER_WIDTH - 1);

    float fx = voxel_pos.x;
    float fy = voxel_pos.y;
    float fz = voxel_pos.z;

    float TEMPORARIESgetmatter;

    return GET_ADM_P;
}

__kernel
void trace_rays4(__global struct lightray4* rays_in, __global struct render_ray_info* rays_terminated,
                STANDARD_ARGS(),
                STANDARD_UTILITY(),
                int use_colour,
                float scale, int4 dim, int width, int height, float err_in)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x >= width || y >= height)
        return;

    struct lightray4 ray_in = rays_in[y * width + x];

    float u_sq = (universe_size * RENDERING_CUTOFF_MULT) * (universe_size * RENDERING_CUTOFF_MULT);

    float4 pos = ray_in.pos;
    float4 vel = ray_in.vel;

    float4 last_pos = pos;

    float accum_R = 0;
    float accum_G = 0;
    float accum_B = 0;

    int hit_type = 1;

    int max_iterations = 2048;

    float camera_ku = ray_in.ku_uobsu;

    ///av is the absorption coefficient
    ///jv is the emission coefficient
    float integration_Tv = 0;
    float integration_intensity = 0;

    //float emitter_ku = dot(lower4(pos.yzw, vel, scale, dim, GET_STANDARD_ARGS()), (float4)(1, 0, 0, 0));

    float background_power = 1;

    for(int iteration=0; iteration < max_iterations; iteration++)
    {
        ///next iteration
        float ds = get_static_verlet_ds(pos.yzw, X, scale, dim);

        float4 accel = get_accel4(pos, vel, scale, dim, GET_STANDARD_ARGS());

        last_pos = pos;

        pos += vel * ds;
        vel += accel * ds;

        float3 Xpos = pos.yzw;

        if(length_sq(Xpos) >= u_sq)
        {
            hit_type = 0;
            break;
        }

        if(length_sq(vel.yzw) < 0.2f * 0.2f)
        {
            hit_type = 1;
            break;
        }

        if(fabs(pos.x) > 100)
        {
            hit_type = 1;
            break;
        }

        if(fabs(vel.x) > 10)
        {
            hit_type = 1;
            break;
        }

        #if defined(TRACE_MATTER_P) || defined(RENDER_MATTER)
        {
            float3 voxel_pos = world_to_voxel(Xpos, dim, scale);
            voxel_pos = clamp(voxel_pos, (float3)(BORDER_WIDTH,BORDER_WIDTH,BORDER_WIDTH), (float3)(dim.x, dim.y, dim.z) - BORDER_WIDTH - 1);

            ///> 0 except at singularity where there is matter
            float matter_p = get_matter_p(pos.yzw, scale, dim, GET_STANDARD_ARGS(), GET_STANDARD_UTILITY());

            float p_val = fabs(matter_p);

            ///https://arxiv.org/pdf/1207.4234.pdf
            float absorption = 0;
            //float emission = 2.f * (p_val * PARTICLE_BRIGHTNESS)/MINIMUM_MASS;

            float next_R = 0;
            float next_G = 0;
            float next_B = 0;

            #ifdef TRACE_MATTER_P
            float emission = 2.f * fabs(buffer_read_linear(adm_p, voxel_pos, dim)) * PARTICLE_BRIGHTNESS/MINIMUM_MASS;
            absorption += (p_val * PARTICLE_BRIGHTNESS)/MINIMUM_MASS;

            next_R += emission;
            next_G += emission;
            next_B += emission;
            #endif

            #ifdef RENDER_MATTER
            float pstar_val = buffer_read_linear(Dp_star, voxel_pos, dim);

            absorption += pstar_val * 50.f;

            if(!use_colour)
            {
                next_R += pstar_val * 100;
                next_G += pstar_val * 100;
                next_B += pstar_val * 100;
            }
            else
            {
                #ifdef HAS_COLOUR
                next_R += buffer_read_linear(dRed, voxel_pos, dim) * 100;
                next_G += buffer_read_linear(dGreen, voxel_pos, dim) * 100;
                next_B += buffer_read_linear(dBlue, voxel_pos, dim) * 100;
                #endif
            }
            #endif

            //if(matter_p != 0)

            if(fabs(matter_p) >= 1e-5f)
            {
                float3 u_matter_upper = get_3vel_upper(pos.yzw, scale, dim, GET_STANDARD_ARGS(), GET_STANDARD_UTILITY());

                float4 full_matter_upper = adm_3velocity_to_full(pos.yzw, u_matter_upper, scale, dim, GET_STANDARD_ARGS());

                float4 current_vel_lower = lower4(pos.yzw, vel, scale, dim, GET_STANDARD_ARGS());

                float current_ku = dot(current_vel_lower, full_matter_upper);

                ///ilorentz
                float zp1 = current_ku / camera_ku;

                float3 intensity_colour = redshift_with_intensity((float3)(next_R, next_G, next_B), zp1 - 1);

                float dt_ds = zp1 * absorption * ds;
                float di_ds_unshifted = exp(-integration_Tv) * ds;
                float di_ds = zp1 * di_ds_unshifted;

                integration_Tv += dt_ds;

                accum_R += di_ds_unshifted * intensity_colour.x * exp(-integration_Tv) * 1.f;
                accum_G += di_ds_unshifted * intensity_colour.y * exp(-integration_Tv) * 1.f;
                accum_B += di_ds_unshifted * intensity_colour.z * exp(-integration_Tv) * 1.f;
                //accum_A += di_ds * emission * exp(-integration_Tv);
                //accum_A = integration_Tv;

                background_power = exp(-integration_Tv);
            }
        }
        #endif

        if(fabs(background_power) < 0.001f)
            break;
    }

    float4 final_observer_lowered = lower4(last_pos.yzw, (float4)(1, 0, 0, 0), scale, dim, GET_STANDARD_ARGS());

    float final_dot = dot(vel, final_observer_lowered);

    struct render_ray_info ray_out;
    ray_out.x = x;
    ray_out.y = y;

    ray_out.X = last_pos.y;
    ray_out.Y = last_pos.z;
    ray_out.Z = last_pos.w;

    ray_out.dX = vel.y;
    ray_out.dY = vel.z;
    ray_out.dZ = vel.w;

    ray_out.hit_type = hit_type;
    ray_out.R = accum_R;
    ray_out.G = accum_G;
    ray_out.B = accum_B;
    ray_out.background_power = clamp(background_power, 0.f, 1.f);

    ///float z_shift = (velocity.x / -ray->ku_uobsu) - 1;

    //ray_out.zp1 = vel.x / -emitter_ku;
    ray_out.zp1 = final_dot / camera_ku;

    rays_terminated[y * width + x] = ray_out;
}

///so, Y = dt/dT, or dt/dlambda
///so 1/y is dlambda/dt
///if we multiply an equation by 1/Y, we get the coordinate time parameterisation instead
float get_dtL(float3 Xpos, float3 vel, float in_L, float scale, int4 dim, STANDARD_ARGS())
{
    float3 voxel_pos = world_to_voxel(Xpos, dim, scale);
    voxel_pos = clamp(voxel_pos, (float3)(BORDER_WIDTH,BORDER_WIDTH,BORDER_WIDTH), (float3)(dim.x, dim.y, dim.z) - BORDER_WIDTH - 1);

    float fx = voxel_pos.x;
    float fy = voxel_pos.y;
    float fz = voxel_pos.z;

    float dt_L = LDiff;

    return dt_L;
}

float4 get_adm_full_geodesic_velocity(float3 Xpos, float3 vel, float in_L, float scale, int4 dim, STANDARD_ARGS())
{
    float3 voxel_pos = world_to_voxel(Xpos, dim, scale);
    voxel_pos = clamp(voxel_pos, (float3)(BORDER_WIDTH,BORDER_WIDTH,BORDER_WIDTH), (float3)(dim.x, dim.y, dim.z) - BORDER_WIDTH - 1);

    float fx = voxel_pos.x;
    float fy = voxel_pos.y;
    float fz = voxel_pos.z;

    float4 result = {GETADMFULL0, GETADMFULL1, GETADMFULL2, GETADMFULL3};

    return result;
}

__kernel
void trace_rays(__global struct lightray_simple* rays_in, __global struct render_ray_info* rays_terminated,
                STANDARD_ARGS(),
                STANDARD_UTILITY(),
                int use_colour,
                float scale, int4 dim, int width, int height, float err_in)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x >= width || y >= height)
        return;

    struct lightray_simple ray_in = rays_in[y * width + x];

    float3 Xpos = {ray_in.lp1, ray_in.lp2, ray_in.lp3};
    float3 vel = {ray_in.V0, ray_in.V1, ray_in.V2};
    float3 Xpos_last = Xpos;

    int hit_type = 1;

    float u_sq = (universe_size * RENDERING_CUTOFF_MULT) * (universe_size * RENDERING_CUTOFF_MULT);

    float3 VHalf = (float3)(0,0,0);
    float3 VFull_approx = (float3)(0,0,0);

    float ds = 0;

    ///so: this performs the first two iterations of verlet early
    ///this means that the main verlet loop does not contain separate memory reads, resulting in a 40ms -> 28ms speedup due to
    ///optimisation
    #define VERLET_2
    #ifdef VERLET_2
    {
        ds = get_static_verlet_ds(Xpos, X, scale, dim);

        float3 ABase;
        calculate_V_derivatives(&ABase, Xpos, vel, scale, dim, GET_STANDARD_ARGS());

        VHalf = vel + 0.5f * ABase * ds;

        VFull_approx = vel + ABase * ds;

        float3 XDiff;
        velocity_to_XDiff(&XDiff, Xpos, VHalf, scale, dim, GET_STANDARD_ARGS());

        float3 XFull = Xpos + XDiff * ds;

        Xpos = XFull;
    }
    #endif // VERLET_2

    float accum_R = 0;
    float accum_G = 0;
    float accum_B = 0;

    int max_iterations = 512;

    //#define NO_HORIZON_DETECTION
    #ifdef NO_HORIZON_DETECTION
    max_iterations = 4096;
    #endif // NO_HORIZON_DETECTION

    float integration_Tv = 0;
    float background_power = 1;
    float camera_ku = ray_in.ku_uobsu;
    float L = camera_ku;

    //#pragma unroll(16)
    for(int iteration=0; iteration < max_iterations; iteration++)
    {
        #ifdef VERLET_2
        ///finish previous iteration
        {
            float3 AFull_approx;
            calculate_V_derivatives(&AFull_approx, Xpos, VFull_approx, scale, dim, GET_STANDARD_ARGS());

            float3 VFull = VHalf + 0.5f * AFull_approx * ds;

            vel = VFull;
        }

        ///only used in the matter case
        {
            L += ds * get_dtL(Xpos, vel, L, scale, dim, GET_STANDARD_ARGS());
        }

        ///next iteration
        ds = get_static_verlet_ds(Xpos, X, scale, dim);

        float3 XDiff;

        {
            float3 ABase;
            calculate_V_derivatives(&ABase, Xpos, vel, scale, dim, GET_STANDARD_ARGS());

            VHalf = vel + 0.5f * ABase * ds;

            VFull_approx = vel + ABase * ds;

            velocity_to_XDiff(&XDiff, Xpos, VHalf, scale, dim, GET_STANDARD_ARGS());

            float3 XFull = Xpos + XDiff * ds;

            Xpos_last = Xpos;
            Xpos = XFull;
        }

        if(length_sq(Xpos) >= u_sq)
        {
            hit_type = 0;
            break;
        }

        #if defined(TRACE_MATTER_P) || defined(RENDER_MATTER)
        {
            float3 voxel_pos = world_to_voxel(Xpos, dim, scale);
            voxel_pos = clamp(voxel_pos, (float3)(BORDER_WIDTH,BORDER_WIDTH,BORDER_WIDTH), (float3)(dim.x, dim.y, dim.z) - BORDER_WIDTH - 1);

            ///> 0 except at singularity where there is matter
            float matter_p = get_matter_p(Xpos, scale, dim, GET_STANDARD_ARGS(), GET_STANDARD_UTILITY());

            float p_val = fabs(matter_p);

            ///https://arxiv.org/pdf/1207.4234.pdf
            float absorption = 0;
            //float emission = 2.f * (p_val * PARTICLE_BRIGHTNESS)/MINIMUM_MASS;

            float next_R = 0;
            float next_G = 0;
            float next_B = 0;

            #ifdef TRACE_MATTER_P
            //float emission = 20.f * fabs(buffer_read_linear(adm_p, voxel_pos, dim)) * PARTICLE_BRIGHTNESS/MINIMUM_MASS;

            float emission = 20.f * p_val * PARTICLE_BRIGHTNESS/MINIMUM_MASS;
            absorption += (p_val * PARTICLE_BRIGHTNESS)/MINIMUM_MASS;

            next_R += emission;
            next_G += emission;
            next_B += emission;
            #endif

            #ifdef RENDER_MATTER
            float pstar_val = buffer_read_linear(Dp_star, voxel_pos, dim);

            absorption += pstar_val * 50.f;

            if(!use_colour)
            {
                next_R += pstar_val * 100;
                next_G += pstar_val * 100;
                next_B += pstar_val * 100;
            }
            else
            {
                #ifdef HAS_COLOUR
                next_R += buffer_read_linear(dRed, voxel_pos, dim) * 100;
                next_G += buffer_read_linear(dGreen, voxel_pos, dim) * 100;
                next_B += buffer_read_linear(dBlue, voxel_pos, dim) * 100;
                #endif
            }
            #endif

            if(fabs(matter_p) > 0.f)
            {
                float lapse = buffer_read_linear(gA, voxel_pos, dim);

                float3 u_matter_upper = get_3vel_upper(Xpos, scale, dim, GET_STANDARD_ARGS(), GET_STANDARD_UTILITY());

                float4 full_matter_upper = adm_3velocity_to_full(Xpos, u_matter_upper, scale, dim, GET_STANDARD_ARGS());
                float4 full_geodesic_upper = get_adm_full_geodesic_velocity(Xpos, vel, L, scale, dim, GET_STANDARD_ARGS());

                float4 current_vel_lower = lower4(Xpos, full_geodesic_upper, scale, dim, GET_STANDARD_ARGS());

                float current_ku = dot(current_vel_lower, full_matter_upper);

                ///ilorentz
                float zp1 = current_ku / camera_ku;

                ///u = L(nu + Vu)
                ///u0 = Lnu = L/a

                ///a/L reparameterises by coordinate time
                float dt_ds = zp1 * absorption * ds * lapse / L;
                float di_ds_unshifted = zp1 * exp(-integration_Tv) * ds * lapse / L;

                integration_Tv += fabs(dt_ds);

                accum_R += fabs(di_ds_unshifted * next_R);
                accum_G += fabs(di_ds_unshifted * next_G);
                accum_B += fabs(di_ds_unshifted * next_B);

                background_power = exp(-integration_Tv);
            }
        }

        if(fabs(background_power) < 0.001f)
            break;
        #endif

        #if 0
        #ifdef TRACE_MATTER_P
        {
            float3 voxel_pos = world_to_voxel(Xpos, dim, scale);
            voxel_pos = clamp(voxel_pos, (float3)(BORDER_WIDTH,BORDER_WIDTH,BORDER_WIDTH), (float3)(dim.x, dim.y, dim.z) - BORDER_WIDTH - 1);

            float p_val = fabs(buffer_read_linear(adm_p, voxel_pos, dim));

            float voxels_intersected = fast_length(XDiff) * ds;

            accum_R += p_val * PARTICLE_BRIGHTNESS * voxels_intersected/MINIMUM_MASS;
            accum_G += p_val * PARTICLE_BRIGHTNESS * voxels_intersected/MINIMUM_MASS;
            accum_B += p_val * PARTICLE_BRIGHTNESS * voxels_intersected/MINIMUM_MASS;

            if(accum_R > 1 && accum_G > 1 && accum_G > 1)
                break;
        }
        #endif

        #ifdef RENDER_MATTER
        {
            float3 voxel_pos = world_to_voxel(Xpos, dim, scale);
            voxel_pos = clamp(voxel_pos, (float3)(BORDER_WIDTH,BORDER_WIDTH,BORDER_WIDTH), (float3)(dim.x, dim.y, dim.z) - BORDER_WIDTH - 1);

            float pstar_val = buffer_read_linear(Dp_star, voxel_pos, dim);

            /*if(pstar_val > 0.001f)
            {
                Xpos_last = Xpos;
                hit_type = 2;
                break;
            }*/

            if(!use_colour)
            {
                accum_R += pstar_val;
                accum_G += pstar_val;
                accum_B += pstar_val;
            }
            else
            {
                #ifdef HAS_COLOUR
                accum_R += buffer_read_linear(dRed, voxel_pos, dim) * 1;
                accum_G += buffer_read_linear(dGreen, voxel_pos, dim) * 1;
                accum_B += buffer_read_linear(dBlue, voxel_pos, dim) * 1;
                #endif
            }

            /*if(density > SOLID_DENSITY)
            {
                Xpos_last = Xpos;
                hit_type = 2;
                break;
            }*/
        }
        #endif // RENDER_MATTER
        #endif // 0
        #endif // VERLET_2

        //#define EULER
        #ifdef EULER
        float ds = mix(0.1f, 2.f, my_fraction);

        float3 accel;
        calculate_V_derivatives(&accel, Xpos, vel, scale, dim, GET_STANDARD_ARGS());

        ///uncomment the accel*ds to get symplectic euler
        float3 XDiff;
        velocity_to_XDiff(&XDiff, Xpos, vel /*+ accel * ds*/, scale, dim, GET_STANDARD_ARGS());

        Xpos += XDiff * ds;

        if(length_sq(Xpos) >= u_sq)
        {
            break;
        }

        vel += accel * ds;
        #endif // EULER

        /*if(x == (int)width/2 && y == (int)height/2)
        {
            printf("%f %f %f  %f %f %f\n", V0, V1, V2, lp1, lp2, lp3);
        }*/

        #ifndef NO_HORIZON_DETECTION
        if(length_sq(XDiff) < 0.2f * 0.2f)
        {
            hit_type = 1;
            break;
        }
        #endif
    }

    struct render_ray_info ray_out;
    ray_out.x = x;
    ray_out.y = y;

    ray_out.X = Xpos_last.x;
    ray_out.Y = Xpos_last.y;
    ray_out.Z = Xpos_last.z;

    float3 XDiff;
    velocity_to_XDiff(&XDiff, Xpos_last, vel, scale, dim, GET_STANDARD_ARGS());

    ray_out.dX = XDiff.x;
    ray_out.dY = XDiff.y;
    ray_out.dZ = XDiff.z;

    ray_out.hit_type = hit_type;
    ray_out.R = accum_R;
    ray_out.G = accum_G;
    ray_out.B = accum_B;
    ray_out.background_power = clamp(background_power, 0.f, 1.f);
    ray_out.zp1 = 1;

    rays_terminated[y * width + x] = ray_out;
}

float4 read_mipmap(image2d_t mipmap1, sampler_t sam, float2 pos, float lod)
{
    return read_imagef(mipmap1, sam, pos, lod);
}

float circular_diff(float f1, float f2)
{
    float a1 = f1 * M_PI * 2;
    float a2 = f2 * M_PI * 2;

    float2 v1 = {cos(a1), sin(a1)};
    float2 v2 = {cos(a2), sin(a2)};

    return atan2(v1.x * v2.y - v1.y * v2.x, v1.x * v2.x + v1.y * v2.y) / (2 * M_PI);
}

float2 circular_diff2(float2 f1, float2 f2)
{
    return (float2)(circular_diff(f1.x, f2.x), circular_diff(f1.y, f2.y));
}

#define MIPMAP_CONDITIONAL(x) (x(mip_background))

__kernel void render_rays(__global struct render_ray_info* rays_in, __write_only image2d_t screen,
                          float scale, int4 dim, int width, int height,
                          __read_only image2d_t mip_background,
                          __global float2* texture_coordinates, sampler_t sam)
{
    int idx = get_global_id(0);

    if(idx >= width * height)
        return;

    struct render_ray_info ray_in = rays_in[idx];

    float3 density_col = {ray_in.R, ray_in.G, ray_in.B};

    int x = ray_in.x;
    int y = ray_in.y;

    if(ray_in.hit_type == 1)
    {
        write_imagef(screen, (int2){x, y}, (float4)(density_col.xyz,1));
        return;
    }

    float lp1 = ray_in.X;
    float lp2 = ray_in.Y;
    float lp3 = ray_in.Z;

    float V0 = ray_in.dX;
    float V1 = ray_in.dY;
    float V2 = ray_in.dZ;

    float3 cpos = {lp1, lp2, lp3};
    float3 cvel = {V0, V1, V2};

    /*float density_frac = clamp(ray_in.density / SOLID_DENSITY, 0.f, 1.f);

    float3 density_col = (float3)(1,1,1) * density_frac;*/


    /*#ifndef TRACE_MATTER_P
    if(any(density_col > 1))
    {
        density_col /= max(density_col.x, max(density_col.y, density_col.z));
    }
    #else
    density_col = clamp(density_col, 0.f, 1.f);
    #endif*/

    if(any(density_col > 1))
    {
        density_col /= max(density_col.x, max(density_col.y, density_col.z));
    }

    float uni_size = universe_size;

    if(ray_in.hit_type == 0)
    {
        cpos = fix_ray_position(cpos, cvel, uni_size * RENDERING_CUTOFF_MULT);

        float sxf = texture_coordinates[y * width + x].x;
        float syf = texture_coordinates[y * width + x].y;

        #if 0
        float fr = fast_length(cpos);
        float theta = acos(cpos.z / fr);
        float phi = atan2(cpos.y, cpos.x);

        float sxf = (phi + M_PI) / (2 * M_PI);
        float syf = theta / M_PI;

        float4 val = (float4)(0,0,0,1);

        int x_half = fabs(fmod((sxf + 1) * 10.f, 1.f)) > 0.5 ? 1 : 0;
        int y_half = fabs(fmod((syf + 1) * 10.f, 1.f)) > 0.5 ? 1 : 0;

        val.x = x_half;
        val.y = y_half;

        if(syf < 0.1 || syf >= 0.9)
        {
            val.x = 0;
            val.y = 0;
            val.z = 1;
        }

        //val.xyz = clamp(ray_in.iter_frac, 0.f, 1.f);

        write_imagef(screen, (int2){x, y}, val);
        #endif // 0

        #define MIPMAPPING
        #ifdef MIPMAPPING
        int dx = 1;
        int dy = 1;

        if(x == width-1)
            dx = -1;

        if(y == height-1)
            dy = -1;

        float2 tl = texture_coordinates[y * width + x];
        float2 tr = texture_coordinates[y * width + x + dx];
        float2 bl = texture_coordinates[(y + dy) * width + x];

        ///higher = sharper
        float bias_frac = 1.3;

        //TL x 0.435143 TR 0.434950 TD -0.000149, aka (tr.x - tl.x) / 1.3
        float2 dx_vtc = circular_diff2(tl, tr) / bias_frac;
        float2 dy_vtc = circular_diff2(tl, bl) / bias_frac;

        if(dx == -1)
        {
            dx_vtc = -dx_vtc;
        }

        if(dy == -1)
        {
            dy_vtc = -dy_vtc;
        }

        //#define TRILINEAR
        #ifdef TRILINEAR
        dx_vtc.x *= MIPMAP_CONDITIONAL(get_image_width);
        dy_vtc.x *= MIPMAP_CONDITIONAL(get_image_width);

        dx_vtc.y *= MIPMAP_CONDITIONAL(get_image_height);
        dy_vtc.y *= MIPMAP_CONDITIONAL(get_image_height);

        //dx_vtc.x /= 10.f;
        //dy_vtc.x /= 10.f;

        dx_vtc /= 2.f;
        dy_vtc /= 2.f;

        float delta_max_sqr = max(dot(dx_vtc, dx_vtc), dot(dy_vtc, dy_vtc));

        float mip_level = 0.5 * log2(delta_max_sqr);

        //mip_level -= 0.5;

        float mip_clamped = clamp(mip_level, 0.f, 5.f);

        float4 end_result = MIPMAP_CONDITIONAL_READ(read_imagef, sam, ((float2){sxf, syf}), mip_clamped);
        #else

        dx_vtc.x *= MIPMAP_CONDITIONAL(get_image_width);
        dy_vtc.x *= MIPMAP_CONDITIONAL(get_image_width);

        dx_vtc.y *= MIPMAP_CONDITIONAL(get_image_height);
        dy_vtc.y *= MIPMAP_CONDITIONAL(get_image_height);

        ///http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1002.1336&rep=rep1&type=pdf
        float dv_dx = dx_vtc.y;
        float dv_dy = dy_vtc.y;

        float du_dx = dx_vtc.x;
        float du_dy = dy_vtc.x;

        float Ann = dv_dx * dv_dx + dv_dy * dv_dy;
        float Bnn = -2 * (du_dx * dv_dx + du_dy * dv_dy);
        float Cnn = du_dx * du_dx + du_dy * du_dy; ///only tells lies

        ///hecc
        #define HECKBERT
        #ifdef HECKBERT
        Ann = dv_dx * dv_dx + dv_dy * dv_dy + 1;
        Cnn = du_dx * du_dx + du_dy * du_dy + 1;
        #endif // HECKBERT

        float F = Ann * Cnn - Bnn * Bnn / 4;
        float A = Ann / F;
        float B = Bnn / F;
        float C = Cnn / F;

        float root = sqrt((A - C) * (A - C) + B*B);
        float a_prime = (A + C - root) / 2;
        float c_prime = (A + C + root) / 2;

        float majorRadius = native_rsqrt(a_prime);
        float minorRadius = native_rsqrt(c_prime);

        float theta = atan2(B, (A - C)/2);

        majorRadius = max(majorRadius, 1.f);
        minorRadius = max(minorRadius, 1.f);

        majorRadius = max(majorRadius, minorRadius);

        float fProbes = 2 * (majorRadius / minorRadius) - 1;
        int iProbes = floor(fProbes + 0.5f);

        int maxProbes = 8;

        iProbes = min(iProbes, maxProbes);

        if(iProbes < fProbes)
            minorRadius = 2 * majorRadius / (iProbes + 1);

        float levelofdetail = log2(minorRadius);

        int maxLod = MIPMAP_CONDITIONAL(get_image_num_mip_levels) - 1;

        if(levelofdetail > maxLod)
        {
            levelofdetail = maxLod;
            iProbes = 1;
        }

        float4 end_result = 0;

        if(iProbes == 1 || iProbes <= 1)
        {
            if(iProbes < 1)
                levelofdetail = maxLod;

            end_result = read_mipmap(mip_background, sam, (float2){sxf, syf}, levelofdetail);
        }
        else
        {
            float lineLength = 2 * (majorRadius - minorRadius);
            float du = cos(theta) * lineLength / (iProbes - 1);
            float dv = sin(theta) * lineLength / (iProbes - 1);

            float4 totalWeight = 0;
            float accumulatedProbes = 0;

            int startN = 0;

            ///odd probes
            if((iProbes % 2) == 1)
            {
                int probeArm = (iProbes - 1) / 2;

                startN = -2 * probeArm;
            }
            else
            {
                int probeArm = (iProbes / 2);

                startN = -2 * probeArm - 1;
            }

            int currentN = startN;
            float alpha = 2;

            float sU = du / MIPMAP_CONDITIONAL(get_image_width);
            float sV = dv / MIPMAP_CONDITIONAL(get_image_height);

            for(int cnt = 0; cnt < iProbes; cnt++)
            {
                float d_2 = (currentN * currentN / 4.f) * (du * du + dv * dv) / (majorRadius * majorRadius);

                ///not a performance issue
                float relativeWeight = native_exp(-alpha * d_2);

                float centreu = sxf;
                float centrev = syf;

                float cu = centreu + (currentN / 2.f) * sU;
                float cv = centrev + (currentN / 2.f) * sV;

                float4 fval = read_mipmap(mip_background, sam, (float2){cu, cv}, levelofdetail);

                totalWeight += relativeWeight * fval;
                accumulatedProbes += relativeWeight;

                currentN += 2;
            }

            end_result = totalWeight / accumulatedProbes;
        }

        #endif // TRILINEAR
        #endif // MIPMAPPING

        float3 linear_col = srgb_to_lin(end_result.xyz);

        linear_col = redshift_with_intensity(linear_col, ray_in.zp1 - 1);

        float3 with_density = linear_col * ray_in.background_power + density_col;

        with_density = clamp(with_density, 0.f, 1.f);

        //float3 with_density = clamp(mix(linear_col, density_col, ray_in.A), 0.f, 1.f);

        write_imagef(screen, (int2){x, y}, (float4)(with_density, 1.f));
    }
}

#if 0
struct lightray
{
    float4 pos;
    float4 vel;
    int x, y;
};

///todo: unify this with the above
///the memory overhead is extremely minimal for a huge performance boost
__kernel
void init_accurate_rays(STANDARD_ARGS(),
                        float scale, float3 camera_pos, float4 camera_quat,
                        int4 dim, __write_only image2d_t screen,
                        __global struct lightray* ray)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x >= get_image_width(screen))
        return;

    if(y >= get_image_height(screen))
        return;

    float width = get_image_width(screen);
    float height = get_image_height(screen);

    ///ray location

    float3 pos = camera_pos;

    pos = clamp(pos, (float3)(BORDER_WIDTH,BORDER_WIDTH,BORDER_WIDTH), (float3)(dim.x, dim.y, dim.z) - BORDER_WIDTH - 1);

    ///temporary while i don't do interpolation
    float lp0;
    float lp1;
    float lp2;
    float lp3;

    float V0;
    float V1;
    float V2;

    {
        float3 world_pos = camera_pos;

        float3 voxel_pos = world_to_voxel(world_pos, dim, scale);

        float fx = voxel_pos.x;
        float fy = voxel_pos.y;
        float fz = voxel_pos.z;

        float TEMPORARIES5;

        lp0 = lp0_d;
        lp1 = lp1_d;
        lp2 = lp2_d;
        lp3 = lp3_d;

        V0 = V0_d;
        V1 = V1_d;
        V2 = V2_d;
    }

    int ray_idx = y * (int)width + x;

    ray[ray_idx].pos = (float4){lp0, lp1, lp2, lp3};
    ray[ray_idx].vel = (float4){V0, V1, V2, 0};
    ray[ray_idx].x = x;
    ray[ray_idx].y = y;
}

__kernel
void step_accurate_rays(STANDARD_ARGS(),
                        float scale, float3 camera_pos, float4 camera_quat,
                        int4 dim, __write_only image2d_t screen,
                        __global struct lightray* ray, float timestep)
{
    float width = get_image_width(screen);
    float height = get_image_height(screen);

    int ray_idx = get_global_id(0);
    int ray_count = (int)width * (int)height;

    if(ray_idx >= ray_count)
        return;

    float lp1 = ray[ray_idx].pos.y;
    float lp2 = ray[ray_idx].pos.z;
    float lp3 = ray[ray_idx].pos.w;

    int x = ray[ray_idx].x;
    int y = ray[ray_idx].y;

    float3 cpos = {lp1, lp2, lp3};

    float3 voxel_pos = world_to_voxel(cpos, dim, scale);

    voxel_pos = clamp(voxel_pos, (float3)(BORDER_WIDTH,BORDER_WIDTH,BORDER_WIDTH), (float3)(dim.x, dim.y, dim.z) - BORDER_WIDTH - 1);

    float fx = voxel_pos.x;
    float fy = voxel_pos.y;
    float fz = voxel_pos.z;

    float TEMPORARIES6;

    float terminate_length = fast_length(cpos);

    if(terminate_length >= universe_size / 5.01f)
    {
        float fr = fast_length(cpos);
        float theta = acos(cpos.z / fr);
        float phi = atan2(cpos.y, cpos.x);

        float sxf = (phi + M_PI) / (2 * M_PI);
        float syf = theta / M_PI;

        float4 val = (float4)(0,0,0,1);

        int x_half = fabs(fmod((sxf + 1) * 10.f, 1.f)) > 0.5 ? 1 : 0;
        int y_half = fabs(fmod((syf + 1) * 10.f, 1.f)) > 0.5 ? 1 : 0;

        val.x = x_half;
        val.y = y_half;

        if(syf < 0.1 || syf >= 0.9)
        {
            val.x = 0;
            val.y = 0;
            val.z = 1;
        }

        write_imagef(screen, (int2){x, y}, val);
        return;
    }

    float V0 = ray[ray_idx].vel.x;
    float V1 = ray[ray_idx].vel.y;
    float V2 = ray[ray_idx].vel.z;

    float dX0 = X0Diff;
    float dX1 = X1Diff;
    float dX2 = X2Diff;

    float dV0 = V0Diff;
    float dV1 = V1Diff;
    float dV2 = V2Diff;

    V0 += dV0 * timestep;
    V1 += dV1 * timestep;
    V2 += dV2 * timestep;

    lp1 += dX0 * timestep;
    lp2 += dX1 * timestep;
    lp3 += dX2 * timestep;

    //if(x == (int)width/2 && y == (int)height/2)
    //printf("Pos %f %f %f\n", lp1, lp2, lp3);

    ray[ray_idx].vel.xyz = (float3)(V0, V1, V2);
    ray[ray_idx].pos.yzw = (float3)(lp1, lp2, lp3);

    //if(fast_length((float3){dX0, dX1, dX2}) < 0.01f)
    {
        write_imagef(screen, (int2){x, y}, (float4)(0, 0, 0, 1));
    }
}
#endif // 0

float3 rot_quat(const float3 point, float4 quat)
{
    quat = fast_normalize(quat);

    float3 t = 2.f * cross(quat.xyz, point);

    return point + quat.w * t + cross(quat.xyz, t);
}

__kernel
void trace_metric(STANDARD_ARGS(),
                  float scale, float3 camera_pos, float4 camera_quat,
                  int4 dim, __write_only image2d_t screen)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x >= get_image_width(screen))
        return;

    if(y >= get_image_height(screen))
        return;

    float width = get_image_width(screen);
    float height = get_image_height(screen);

    ///ray location
    float3 pos = world_to_voxel(camera_pos, dim, scale);

    pos = clamp(pos, (float3)(BORDER_WIDTH,BORDER_WIDTH,BORDER_WIDTH), (float3)(dim.x, dim.y, dim.z) - BORDER_WIDTH - 1);

    ///temporary while i don't do interpolation
    float p0 = pos.x;
    float p1 = pos.y;
    float p2 = pos.z;

    float FOV = 90;

    float fov_rad = (FOV / 360.f) * 2 * M_PI;

    float nonphysical_plane_half_width = width/2;
    float nonphysical_f_stop = nonphysical_plane_half_width / tan(fov_rad/2);

    float3 pixel_direction = {x - width/2, y - height/2, nonphysical_f_stop};

    pixel_direction = rot_quat(normalize(pixel_direction), camera_quat);

    float max_scalar = 0;

    for(int iteration=0; iteration < 8000; iteration++)
    {
        if(p0 < BORDER_WIDTH || p0 >= dim.x - BORDER_WIDTH - 1 || p1 < BORDER_WIDTH || p1 >= dim.y - BORDER_WIDTH - 1 || p2 < BORDER_WIDTH || p2 >= dim.z - BORDER_WIDTH - 1)
            break;

        #define TRACE_CONFORMAL
        #ifdef TRACE_CONFORMAL
        float Yxx = buffer_read_linear(cY0, (float3)(p0,p1,p2), dim);
        float Yxy = buffer_read_linear(cY1, (float3)(p0,p1,p2), dim);
        float Yxz = buffer_read_linear(cY2, (float3)(p0,p1,p2), dim);
        float Yyy = buffer_read_linear(cY3, (float3)(p0,p1,p2), dim);
        float Yyz = buffer_read_linear(cY4, (float3)(p0,p1,p2), dim);
        float cX = buffer_read_linear(X, (float3)(p0,p1,p2), dim);

        float Yzz = (1 + Yyy * Yxz * Yxz - 2 * Yxy * Yyz * Yxz + Yxx * Yyz * Yyz) / (Yxx * Yyy - Yxy * Yxy);

        float curvature = fabs(Yxx / cX) +
                          fabs(Yxy / cX) +
                          fabs(Yxz / cX) +
                          fabs(Yyy / cX) +
                          fabs(Yyz / cX) +
                          fabs(Yzz / cX);

        float ascalar = curvature / 1000.f;
        #endif // TRACE_CONFORMAL

        //#define TRACE_EXTRINSIC
        #ifdef TRACE_EXTRINSIC
        float cAxx = buffer_read_linear(cA0, (float3)(p0,p1,p2), dim);
        float cAxy = buffer_read_linear(cA1, (float3)(p0,p1,p2), dim);
        float cAxz = buffer_read_linear(cA2, (float3)(p0,p1,p2), dim);
        float cAyy = buffer_read_linear(cA3, (float3)(p0,p1,p2), dim);
        float cAyz = buffer_read_linear(cA4, (float3)(p0,p1,p2), dim);
        float cAzz = buffer_read_linear(cA5, (float3)(p0,p1,p2), dim);

        float curvature = fabs(cAxx) +
                          fabs(cAxy) +
                          fabs(cAxz) +
                          fabs(cAyy) +
                          fabs(cAyz) +
                          fabs(cAzz);

        float ascalar = curvature / 100.f;
        #endif // TRACE_EXTRINSIC

        //#define TRACE_K
        #ifdef TRACE_K
        float lK = buffer_read_linear(K,  (float3)(p0,p1,p2), dim);

        float ascalar = fabs(lK / 2.f);
        #endif // TRACE_K

        max_scalar = max(ascalar, max_scalar);

        p0 += pixel_direction.x;
        p1 += pixel_direction.y;
        p2 += pixel_direction.z;
    }

    max_scalar = max_scalar * 40;

    max_scalar = clamp(max_scalar, 0.f, 1.f);

    write_imagef(screen, (int2)(x, y), (float4)(max_scalar, max_scalar, max_scalar, 1));
}
