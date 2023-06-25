#ifndef SINGLE_SOURCE_KERNEL_CONTEXT_HPP_INCLUDED
#define SINGLE_SOURCE_KERNEL_CONTEXT_HPP_INCLUDED

#include <vector>

struct equation_context;

namespace single_source
{
    namespace impl
    {
        struct input;

        struct kernel_context
        {
            bool is_func = false;
            std::vector<input> inputs;
            std::vector<input> ret;
        };

        struct input
        {
            std::string type;
            bool pointer = false;
            std::string name;

            std::string format()
            {
                if(pointer)
                {
                    return "__global " + type + "* __restrict__ " + name;
                }
                else
                {
                    return type + " " + name;
                }
            }
        };

        template<typename R, typename... Args>
        void setup_kernel(kernel_context& kctx, equation_context& ectx, R(*func)(equation_context&, Args...));
    }
}

#endif // SINGLE_SOURCE_KERNEL_CONTEXT_HPP_INCLUDED
