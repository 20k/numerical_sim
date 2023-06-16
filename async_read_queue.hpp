#ifndef ASYNC_READ_QUEUE_HPP_INCLUDED
#define ASYNC_READ_QUEUE_HPP_INCLUDED

#include <toolkit/opencl.hpp>

template<typename T>
struct async_read_queue
{
    std::vector<std::pair<cl::event, cl_float2*>> gpu_data_in_flight;
    std::vector<T*> pending_unprocessed_data;
    std::vector<cl::buffer> buffers;
    int element_count = 0;
    uint32_t next_buffer = 0;

    cl::command_queue read_queue;

    async_read_queue(cl::context& ctx) : read_queue(ctx){}

    void start(cl::context& ctx, cl::command_queue& _read_queue, int _element_count)
    {
        read_queue = _read_queue;
        element_count = _element_count;

        int N = 8;

        for(int i=0; i < N; i++)
        {
            cl::buffer& buf = buffers.emplace_back(ctx);
            buf.alloc(sizeof(T) * element_count);
        }
    }

    cl::buffer fetch_next_buffer()
    {
        next_buffer++;

        return buffers.at(next_buffer % buffers.size());
    }

    void issue(cl::buffer next_buffer, cl::event depends_on)
    {
        T* next = new T[element_count];

        cl::event event = next_buffer.read_async(read_queue, (char*)next, element_count * sizeof(T), {depends_on});
        read_queue.flush();

        gpu_data_in_flight.push_back({event, next});
    }

    std::vector<std::vector<T>> process()
    {
        if(gpu_data_in_flight.size() >= buffers.size())
        {
            printf("Super bad error in process()\n");
        }

        std::vector<std::vector<T>> ret;

        while(gpu_data_in_flight.size() > 0 && gpu_data_in_flight.front().first.is_finished())
        {
            printf("hi\n");

            T* ptr = gpu_data_in_flight.front().second;

            std::vector<T> vals;

            for(int i=0; i < element_count; i++)
            {
                vals.push_back(ptr[i]);
            }

            ret.push_back(vals);

            delete [] ptr;

            gpu_data_in_flight.erase(gpu_data_in_flight.begin());

            printf("hi3\n");
        }

        return ret;
    }
};
#endif // ASYNC_READ_QUEUE_HPP_INCLUDED
