// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCPRIM_DEVICE_DEVICE_FIND_END_HPP_
#define ROCPRIM_DEVICE_DEVICE_FIND_END_HPP_

#include "../detail/temp_storage.hpp"

#include "../config.hpp"

#include "../block/block_reduce.hpp"
#include "../intrinsics.hpp"
#include "../iterator/reverse_iterator.hpp"
#include "config_types.hpp"
#include "device_find_end_config.hpp"
#include "device_transform.hpp"

#include <iostream>
#include <iterator>

#include <cstddef>
#include <cstdio>

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

#define ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR(name, size, start)                           \
    do                                                                                           \
    {                                                                                            \
        hipError_t _error = hipGetLastError();                                                   \
        if(_error != hipSuccess)                                                                 \
            return _error;                                                                       \
        if(debug_synchronous)                                                                    \
        {                                                                                        \
            std::cout << name << "(" << size << ")";                                             \
            hipError_t __error = hipStreamSynchronize(stream);                                   \
            if(__error != hipSuccess)                                                            \
                return __error;                                                                  \
            auto _end = std::chrono::steady_clock::now();                                        \
            auto _d   = std::chrono::duration_cast<std::chrono::duration<double>>(_end - start); \
            std::cout << " " << _d.count() * 1000 << " ms" << '\n';                              \
        }                                                                                        \
    }                                                                                            \
    while(0)

#define RETURN_ON_ERROR(...)              \
    do                                    \
    {                                     \
        hipError_t error = (__VA_ARGS__); \
        if(error != hipSuccess)           \
        {                                 \
            return error;                 \
        }                                 \
    }                                     \
    while(0)

template<class Config,
         class InputIterator1,
         class InputIterator2,
         class OutputType,
         class BinaryFunction>
ROCPRIM_KERNEL
__launch_bounds__(device_params<Config>().kernel_config.block_size)
void search_kernel(InputIterator1 input,
                   InputIterator2 keys,
                   OutputType*    output,
                   size_t         size,
                   size_t         keys_size,
                   BinaryFunction compare_function)
{
    constexpr find_end_config_params params = device_params<Config>();

    constexpr unsigned int block_size       = params.kernel_config.block_size;
    constexpr unsigned int items_per_thread = params.kernel_config.items_per_thread;
    constexpr unsigned int items_per_block  = block_size * items_per_thread;

    const unsigned int flat_id       = rocprim::detail::block_thread_id<0>();
    const unsigned int flat_block_id = rocprim::detail::block_id<0>();

    const OutputType offset       = flat_id * items_per_thread + flat_block_id * items_per_block;
    bool             find_pattern = false;

    if(offset + keys_size > size || offset > atomic_load(output))
    {
        return;
    }

    OutputType index = 0;
    for(OutputType id = offset; id < offset + items_per_thread; id++)
    {
        OutputType i          = 0;
        OutputType current_id = id;
        for(; i < keys_size - 1 && current_id < size; i++, current_id++)
        {
            if(!compare_function(input[current_id], keys[i]))
            {
                break;
            }
        }
        if(current_id < size && i == (keys_size - 1)
           && compare_function(input[current_id], keys[i]))
        {
            index        = id;
            find_pattern = true;
            break;
        }
    }

    // Construct a mask of threads in this wave which have the same digit.
    rocprim::lane_mask_type peer_mask = ::rocprim::match_any<2>(find_pattern);

    rocprim::wave_barrier();

    // The number of threads in the warp that have the same digit AND whose lane id is lower
    // than the current thread's.
    const unsigned int peer_digit_prefix = rocprim::masked_bit_count(peer_mask);

    if(find_pattern && (peer_digit_prefix == 0))
    {
        atomic_min(output, index);
    }
}

template<class Config,
         unsigned int SharedMemSize,
         class InputIterator1,
         class InputIterator2,
         class OutputType,
         class BinaryFunction>
ROCPRIM_KERNEL
__launch_bounds__(device_params<Config>().kernel_config.block_size)
void search_kernel_shared(InputIterator1 input,
                          InputIterator2 keys,
                          OutputType*    output,
                          size_t         size,
                          size_t         keys_size,
                          BinaryFunction compare_function)
{
    using value_type = typename std::iterator_traits<InputIterator1>::value_type;
    using key_type   = typename std::iterator_traits<InputIterator2>::value_type;

    constexpr find_end_config_params params = device_params<Config>();

    constexpr unsigned int block_size       = params.kernel_config.block_size;
    constexpr unsigned int items_per_thread = params.kernel_config.items_per_thread;
    constexpr unsigned int items_per_block  = block_size * items_per_thread;

    const unsigned int flat_id       = rocprim::detail::block_thread_id<0>();
    const unsigned int flat_block_id = rocprim::detail::block_id<0>();

    const OutputType block_offset = flat_block_id * items_per_block;
    const OutputType offset       = flat_id * items_per_thread;
    bool find_pattern = false;

    ROCPRIM_SHARED_MEMORY uninitialized_array<key_type, SharedMemSize> local_keys_;
    ROCPRIM_SHARED_MEMORY uninitialized_array<value_type, items_per_block> local_input_;

    if(block_offset > atomic_load(output))
    {
        return;
    }

    const size_t batch_size = ceiling_div(keys_size, block_size);
    for(size_t i = 0; i < batch_size; i++)
    {
        const size_t index = flat_id * batch_size + i;
        if(index < keys_size)
        {
            local_keys_.emplace(index, keys[index]);
        }
    }

    using block_load_input = block_load<value_type, items_per_thread, items_per_thread>;

    value_type elements[items_per_thread];

    const bool is_complete_block = block_offset + items_per_block <= size;

    if(is_complete_block)
    {
        block_load_input().load(input + block_offset, elements);
        for(size_t i = 0; i < items_per_thread; i++)
        {
            const size_t index = flat_id * items_per_thread + i;
            local_input_.emplace(index, elements[i]);
        }
    }
    else
    {
        block_load_input().load(input + block_offset, elements, size - block_offset);
        for(size_t i = 0; i < items_per_thread; i++)
        {
            const size_t index       = flat_id * items_per_thread + i;
            const size_t index_value = block_offset + index;
            if(index_value < size)
            {
                local_input_.emplace(index, elements[i]);
            }
        }
    }

    const key_type*   local_keys  = local_keys_.get_unsafe_array();
    const value_type* local_input = local_input_.get_unsafe_array();

    syncthreads();

    if(offset + block_offset + keys_size > size || offset > atomic_load(output))
    {
        return;
    }

    OutputType       index      = 0;
    const OutputType check      = size - block_offset;
    const OutputType check_both = rocprim::min(check, OutputType(items_per_block));
    for(OutputType id = offset; id < offset + items_per_thread; id++)
    {
        OutputType i          = 0;
        OutputType current_id = id;
        for(; i < keys_size - 1 && current_id < check_both; i++, current_id++)
        {
            if(!compare_function(local_input[current_id], local_keys[i]))
            {
                break;
            }
        }
        for(; i < keys_size - 1 && current_id < check; i++, current_id++)
        {
            if(!compare_function(input[current_id + block_offset], local_keys[i]))
            {
                break;
            }
        }

        if(current_id + block_offset < size && i == (keys_size - 1)
           && compare_function(current_id < items_per_block ? local_input[current_id]
                                                            : input[current_id + block_offset],
                               local_keys[i]))
        {
            index        = id + block_offset;
            find_pattern = true;
            break;
        }
    }

    // Construct a mask of threads in this wave which have the same digit.
    rocprim::lane_mask_type peer_mask = ::rocprim::match_any<2>(find_pattern);

    rocprim::wave_barrier();

    // The number of threads in the warp that have the same digit AND whose lane id is lower
    // than the current thread's.
    const unsigned int peer_digit_prefix = rocprim::masked_bit_count(peer_mask);

    if(find_pattern && (peer_digit_prefix == 0))
    {
        atomic_min(output, index);
    }
}

template<class OutputType>
ROCPRIM_KERNEL
void set_output_kernel(OutputType* output, size_t size)
{
    *output = static_cast<OutputType>(size);
}

template<class OutputType>
ROCPRIM_KERNEL
void reverse_index_kernel(OutputType* output, size_t size, size_t keys_size)
{
    if(*output < size)
    {
        *output = static_cast<OutputType>(size - keys_size) - *output;
    }
}

template<class Config,
         class InputIterator1,
         class InputIterator2,
         class OutputIterator,
         class BinaryFunction>
ROCPRIM_INLINE
hipError_t find_end_impl(void*          temporary_storage,
                         size_t&        storage_size,
                         InputIterator1 input,
                         InputIterator2 keys,
                         OutputIterator output,
                         size_t         size,
                         size_t         keys_size,
                         BinaryFunction compare_function,
                         hipStream_t    stream,
                         bool           debug_synchronous)
{
    using output_type = typename std::iterator_traits<OutputIterator>::value_type;
    using input_type  = typename std::iterator_traits<InputIterator1>::value_type;

    static_assert(rocprim::is_integral<output_type>::value
                      && rocprim::is_unsigned<output_type>::value,
                  "Output type should be an unsigned integral type");

    using config = wrapped_find_end_config<Config, input_type>;

    target_arch target_arch;
    RETURN_ON_ERROR(host_target_arch(stream, target_arch));

    const find_end_config_params params = dispatch_target_arch<config>(target_arch);

    const unsigned int block_size       = params.kernel_config.block_size;
    const unsigned int items_per_thread = params.kernel_config.items_per_thread;
    const unsigned int items_per_block  = block_size * items_per_thread;

    constexpr unsigned int shared_mem_size = 256;

    // Start point for time measurements
    std::chrono::steady_clock::time_point start;

    const auto start_timer = [&start, debug_synchronous]()
    {
        if(debug_synchronous)
        {
            start = std::chrono::steady_clock::now();
        }
    };

    if(temporary_storage == nullptr)
    {
        storage_size = sizeof(output_type);
        return hipSuccess;
    }

    if(keys_size > size)
    {
        return hipErrorInvalidValue;
    }

    auto input_iterator = make_reverse_iterator(input + size);
    auto keys_iterator  = make_reverse_iterator(keys + keys_size);

    output_type* tmp_output = reinterpret_cast<output_type*>(temporary_storage);

    start_timer();
    set_output_kernel<<<1, 1, 0, stream>>>(tmp_output, size);
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("set_output_kernel", 1, start);

    if(size > 0 && keys_size > 0)
    {
        const unsigned int num_blocks = ceiling_div(size, items_per_block);
        if(keys_size < shared_mem_size)
        {
            start_timer();
            search_kernel_shared<config, shared_mem_size>
                <<<num_blocks, block_size, 0, stream>>>(input_iterator,
                                                        keys_iterator,
                                                        tmp_output,
                                                        size,
                                                        keys_size,
                                                        compare_function);
            ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("search_kernel_shared", size, start);
        }
        else
        {
            start_timer();
            search_kernel<config><<<num_blocks, block_size, 0, stream>>>(input_iterator,
                                                                         keys_iterator,
                                                                         tmp_output,
                                                                         size,
                                                                         keys_size,
                                                                         compare_function);
            ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("search_kernel", size, start);
        }

        start_timer();
        reverse_index_kernel<<<1, 1, 0, stream>>>(tmp_output, size, keys_size);
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("reverse_index_kernel", 1, start);
    }

    RETURN_ON_ERROR(transform(tmp_output,
                              output,
                              1,
                              rocprim::identity<output_type>(),
                              stream,
                              debug_synchronous));

    return hipSuccess;
}

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR
#undef RETURN_ON_ERROR

} // namespace detail

/// \addtogroup devicemodule
/// @{

template<class Config = default_config,
         class InputIterator1,
         class InputIterator2,
         class OutputIterator,
         class BinaryFunction
         = rocprim::equal_to<typename std::iterator_traits<InputIterator1>::value_type>>
ROCPRIM_INLINE
hipError_t find_end(void*          temporary_storage,
                    size_t&        storage_size,
                    InputIterator1 input,
                    InputIterator2 keys,
                    OutputIterator output,
                    size_t         size,
                    size_t         keys_size,
                    BinaryFunction compare_function  = BinaryFunction(),
                    hipStream_t    stream            = 0,
                    bool           debug_synchronous = false)
{
    return detail::find_end_impl<Config>(temporary_storage,
                                         storage_size,
                                         input,
                                         keys,
                                         output,
                                         size,
                                         keys_size,
                                         compare_function,
                                         stream,
                                         debug_synchronous);
}

/// @}
// end of group devicemodule

END_ROCPRIM_NAMESPACE

#endif
