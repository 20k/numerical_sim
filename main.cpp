#include <iostream>
#include <toolkit/render_window.hpp>
#include <toolkit/texture.hpp>
#include <vec/vec.hpp>
#include <GLFW/glfw3.h>
#include <SFML/Graphics.hpp>
#include <CL/cl_ext.h>
#include <geodesic/dual.hpp>
#include <geodesic/dual_value.hpp>
#include <geodesic/numerical.hpp>

///all conformal variables are explicitly labelled
struct bssnok_data
{
    /**
    conformal
    [0, 1, 2,
     X, 3, 4,
     X, X, 5]
    */
    cl_float cY0, cY1, cY2, cY3, cY4, cY5;

    /**
    conformal
    [0, 1, 2,
     X, 3, 4,
     X, X, 5]
    */
    cl_float cA0, cA1, cA2, cA3, cA4, cA5;

    cl_float cGi0, cGi1, cGi2;

    cl_float K;
    cl_float X;

    cl_float gA;
    cl_float gB0;
    cl_float gB1;
    cl_float gB2;
};

bssnok_data get_conditions(vec3f pos, vec3f centre, float scale)
{
    tensor<float, 3, 3> kronecker;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            if(i == j)
                kronecker.idx(i, j) = 1;
            else
                kronecker.idx(i, j) = 0;
        }
    }

    ///I could fix this by improving the dual library to allow for algebraic substitution
    float BL_conformal = 1;

    float r = (pos - centre).length() * scale;

    if(r < 0.01)
        r = 0.01;

    //value vr("r");

    std::vector<float> black_hole_r{0};
    std::vector<float> black_hole_m{1};

    ///3.57 https://scholarworks.rit.edu/cgi/viewcontent.cgi?article=11286&context=theses
    for(int i=0; i < (int)black_hole_r.size(); i++)
    {
        float Mi = black_hole_m[i];
        float ri = black_hole_r[i];

        BL_conformal = BL_conformal + Mi / (2 * fabs(r - ri));
    }

    tensor<float, 3, 3> yij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            yij.idx(i, j) = pow(BL_conformal, 4) * kronecker.idx(i, j);
        }
    }

    ///https://arxiv.org/pdf/gr-qc/9810065.pdf, 11
    float Y = yij.det();

    ///https://scholarworks.rit.edu/cgi/viewcontent.cgi?article=11286&context=theses pre 3.65
    float conformal_factor = (1/12.f) * log(Y);

    tensor<float, 3, 3> cyij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            cyij.idx(i, j) = exp(-4 * conformal_factor) * yij.idx(i, j);
        }
    }

    ///determinant of cyij is 1
    ///
    {
        float cY = cyij.det();
        //float real_value = cY.get_constant();
        float real_value = cY;

        //std::cout << "REAL " << real_value << std::endl;
    }

    float real_conformal = conformal_factor;

    ///so, via 3.47, Kij = Aij + (1/3) Yij K
    ///via 3.55, Kij = 0 for the initial conditions
    ///therefore Aij = 0
    ///cAij = exp(-4 phi) Aij
    ///therefore cAij = 0?
    tensor<float, 3, 3> cAij;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            cAij.idx(i, j) = 0;
        }
    }

    float X = exp(-4 * real_conformal);

    std::string v1 = "x";
    std::string v2 = "y";
    std::string v3 = "z";

    /*tensor<value, 3, 3, 3> christoff = christoffel_symbols_2(cyij, vec<3, std::string>{v1, v2, v3});

    ///3.59 says the christoffel symbols are 0 in cartesian
    {
        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                for(int k=0; k < 3; k++)
                {
                    assert(christoff.idx(i, j, k).get_constant() == 0);

                    //std::cout << "CIJK " << type_to_string(christoff.idx(i, j, k)) << std::endl;
                }
            }
        }
    }*/

    tensor<float, 3, 3, 3> christoff;

    for(int i=0; i < 3; i++)
    {
        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                christoff.idx(i, j, k) = 0;
            }
        }
    }

    tensor<float, 3, 3> inverse_cYij = cyij.invert();
    vec3f cGi = {0,0,0};

    ///https://arxiv.org/pdf/gr-qc/9810065.pdf (21)
    ///aka cy^jk cGijk

    for(int i=0; i < 3; i++)
    {
        float sum = 0;

        for(int j=0; j < 3; j++)
        {
            for(int k=0; k < 3; k++)
            {
                sum += inverse_cYij.idx(j, k) * christoff.idx(i, j, k);
            }
        }

        cGi[i] = sum;
    }

    for(int i=0; i < 3; i++)
    {
        if(cGi[i] != 0)
        {
            std::cout << cGi[i] << std::endl;
            std::cout << "y " << Y << std::endl;
        }

        assert(cGi[i] == 0);
    }

    /*auto iv_cYij = cyij.invert();

    tensor<float, 4> cGi;

    for(int i=0; i < 4; i++)
    {
        cGi = iv_cYij.differentiate()
    }*/

    ///Kij = Aij + (1/3) yij K, where K is trace
    ///in the initial data, Kij = 0
    ///Which means K = 0
    ///which means that Aij.. is 0?

    bssnok_data ret;

    ret.cA0 = cAij.idx(0, 0);
    ret.cA1 = cAij.idx(0, 1);
    ret.cA2 = cAij.idx(0, 2);
    ret.cA3 = cAij.idx(1, 1);
    ret.cA4 = cAij.idx(1, 2);
    ret.cA5 = cAij.idx(2, 2);

    ret.cY0 = cyij.idx(0, 0);
    ret.cY1 = cyij.idx(0, 1);
    ret.cY2 = cyij.idx(0, 2);
    ret.cY3 = cyij.idx(1, 1);
    ret.cY4 = cyij.idx(1, 2);
    ret.cY5 = cyij.idx(2, 2);

    ret.cGi0 = cGi[0];
    ret.cGi1 = cGi[1];
    ret.cGi2 = cGi[2];

    ret.K = 0;
    ret.X = X;

    ///https://arxiv.org/pdf/1404.6523.pdf section A, initial data
    ret.gA = 1/BL_conformal;
    ret.gB0 = 1/BL_conformal;
    ret.gB1 = 1/BL_conformal;
    ret.gB2 = 1/BL_conformal;

    return ret;
}

int main()
{
    int width = 1422;
    int height = 800;

    render_settings sett;
    sett.width = width;
    sett.height = height;
    sett.opencl = true;
    sett.no_double_buffer = true;

    render_window win(sett, "Geodesics");

    assert(win.clctx);

    opencl_context& clctx = *win.clctx;

    std::string argument_string = "-O3 -cl-std=CL2.2 ";

    cl::program prog(clctx.ctx, "cl.cl");
    prog.build(clctx.ctx, argument_string);

    texture_settings tsett;
    tsett.width = width;
    tsett.height = height;
    tsett.is_srgb = false;

    std::array<texture, 2> tex;
    tex[0].load_from_memory(tsett, nullptr);
    tex[1].load_from_memory(tsett, nullptr);

    std::array<cl::gl_rendertexture, 2> rtex{clctx.ctx, clctx.ctx};
    rtex[0].create_from_texture(tex[0].handle);
    rtex[1].create_from_texture(tex[1].handle);

    std::array<cl::buffer, 2> bssnok_datas{clctx.ctx, clctx.ctx};

    vec3i size = {100, 100, 100};

    bssnok_datas[0].alloc(size.x() * size.y() * size.z() * sizeof(bssnok_data));
    bssnok_datas[1].alloc(size.x() * size.y() * size.z() * sizeof(bssnok_data));

    float c_at_max = 10;
    std::vector<bssnok_data> cpu_data;

    for(int z=0; z < size.z(); z++)
    {
        for(int y=0; y < size.y(); y++)
        {
            for(int x=0; x < size.x(); x++)
            {
                vec3f pos = {x, y, z};
                vec3f centre = {size.x()/2, size.y()/2, size.z()/2};

                float scale = c_at_max / size.largest_elem();

                cpu_data.push_back(get_conditions(pos, centre, scale));
            }
        }
    }

    bssnok_datas[0].write(clctx.cqueue, cpu_data);

    int which_buffer = 0;

    get_conditions({0, 1, 0}, {0, 0, 0}, 1);

    while(!win.should_close())
    {
        win.poll();

        auto buffer_size = rtex[which_buffer].size<2>();

        if((vec2i){buffer_size.x(), buffer_size.y()} != win.get_window_size())
        {
            width = buffer_size.x();
            height = buffer_size.y();

            texture_settings new_sett;
            new_sett.width = width;
            new_sett.height = height;
            new_sett.is_srgb = false;

            tex[0].load_from_memory(new_sett, nullptr);
            tex[1].load_from_memory(new_sett, nullptr);

            rtex[0].create_from_texture(tex[0].handle);
            rtex[1].create_from_texture(tex[1].handle);
        }

        glFinish();

        rtex[which_buffer].acquire(clctx.cqueue);

        clctx.cqueue.flush();

        rtex[which_buffer].unacquire(clctx.cqueue);

        which_buffer = (which_buffer + 1) % 2;

        clctx.cqueue.block();

        {
            ImDrawList* lst = ImGui::GetBackgroundDrawList();

            ImVec2 screen_pos = ImGui::GetMainViewport()->Pos;

            ImVec2 tl = {0,0};
            ImVec2 br = {win.get_window_size().x(),win.get_window_size().y()};

            if(win.get_render_settings().viewports)
            {
                tl.x += screen_pos.x;
                tl.y += screen_pos.y;

                br.x += screen_pos.x;
                br.y += screen_pos.y;
            }

            lst->AddImage((void*)rtex[which_buffer].texture_id, tl, br, ImVec2(0, 0), ImVec2(1.f, 1.f));
        }

        win.display();
    }
}
