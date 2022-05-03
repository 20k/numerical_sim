#ifndef REF_COUNTED_HPP_INCLUDED
#define REF_COUNTED_HPP_INCLUDED

#include <toolkit/opencl.hpp>

struct ref_counted
{
    int* data = nullptr;

    ref_counted(){data = new int(1);}

    ref_counted(const ref_counted& other)
    {
        data = other.data;

        if(data)
        {
            (*data)++;
        }
    }

    void consume(int* block)
    {
        if(data)
        {
            (*data)--;
        }

        data = block;
    }

    ref_counted& operator=(const ref_counted& other)
    {
        if(this == &other)
            return *this;

        if(data)
        {
            (*data)--;
        }

        data = other.data;

        if(data)
        {
            (*data)++;
        }

        return *this;
    }

    int ref_count()
    {
        if(data == nullptr)
            throw std::runtime_error("No data");

        return *data;
    }

    ~ref_counted()
    {
        if(data)
        {
            (*data)--;
        }
    }
};

struct ref_counted_buffer : cl::buffer, ref_counted
{

};

#endif // REF_COUNTED_HPP_INCLUDED
