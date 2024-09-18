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

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_SEARCH_N_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_SEARCH_N_HPP_

#include "../../common.hpp"
#include "../../config.hpp"
#include "../../detail/temp_storage.hpp"
#include "../../intrinsics.hpp"
#include "../../iterator/reverse_iterator.hpp"
#include "../config_types.hpp"
#include "../device_search_n_config.hpp"
#include "../device_transform.hpp"

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <iterator>

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

ROCPRIM_KERNEL
void set_search_n_kernel(size_t* output, size_t target)
{
    *output = target;
}

template<class Config, class InputIterator, class BinaryFunction>
ROCPRIM_DEVICE
void search_n_kernel_impl(InputIterator                                                   input,
                          size_t*                                                         output,
                          size_t                                                          size,
                          size_t                                                          count,
                          typename std::iterator_traits<InputIterator>::value_type const* value,
                          BinaryFunction compare_function)
{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;

    constexpr auto params = device_params<Config>();

    constexpr unsigned int block_size       = params.kernel_config.block_size;
    constexpr unsigned int items_per_thread = params.kernel_config.items_per_thread;
    constexpr unsigned int items_per_block  = block_size * items_per_thread;
    const unsigned int     b_id             = blockIdx.x;
    const unsigned int     t_id             = threadIdx.x;

    size_t cur_start_idx = (b_id * items_per_block) + (items_per_thread * t_id);
    size_t tar_count     = count;

    for(size_t i = cur_start_idx; i < cur_start_idx + items_per_thread && i < size; i++)
    {
    inside_loop:
        size_t started_from = i - (count - tar_count);
        if(started_from < atomic_load(output) && i + tar_count <= size)
        {
            if(compare_function(*(input + i), *value))
            {
                tar_count--;
                if(tar_count == 0)
                {
                    atomic_min(output, started_from);
                    break;
                }
                else
                {
                    i++;
                    goto inside_loop;
                }
            }
            else
            {
                tar_count = count;
            }
        }
        else
        {
            break;
        }
    }
}

template<class Config, class InputIterator, class BinaryFunction>
ROCPRIM_KERNEL __launch_bounds__(device_params<Config>().kernel_config.block_size)
void search_n_kernel(InputIterator                                                   input,
                     size_t*                                                         output,
                     size_t                                                          size,
                     size_t                                                          count,
                     typename std::iterator_traits<InputIterator>::value_type const* value,
                     BinaryFunction compare_function)
{
    search_n_kernel_impl<Config>(input, output, size, count, value, compare_function);
}

template<class Config, class InputIterator, class OutputIterator, class BinaryFunction>
ROCPRIM_INLINE
hipError_t search_n_impl(void*          temporary_storage,
                         size_t&        storage_size,
                         InputIterator  input,
                         OutputIterator output,
                         size_t         size,
                         size_t         count,
                         typename std::iterator_traits<InputIterator>::value_type const* value,
                         BinaryFunction compare_function,
                         hipStream_t    stream,
                         bool           debug_synchronous)
{
    using input_type  = typename std::iterator_traits<InputIterator>::value_type;
    using output_type = typename std::iterator_traits<OutputIterator>::value_type;
    using config      = wrapped_search_n_config<Config, input_type>;

    target_arch target_arch;
    RETURN_ON_ERROR(host_target_arch(stream, target_arch));

    const auto params = dispatch_target_arch<config>(target_arch);

    const unsigned int block_size       = params.kernel_config.block_size;
    const unsigned int items_per_thread = params.kernel_config.items_per_thread;
    const unsigned int items_per_block  = block_size * items_per_thread;
    size_t*            tmp_output       = reinterpret_cast<size_t*>(temporary_storage);
    const unsigned int num_blocks       = ceiling_div(size, items_per_block);

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
        storage_size = sizeof(size_t);
        return hipSuccess;
    }

    if(count > size)
    {
        return hipErrorInvalidValue;
    }

    if(size == 0 || count == 0)
    {
        // return end
        start_timer();
        set_search_n_kernel<<<1, 1, 0, stream>>>(tmp_output, count == 0 ? 0 : size);
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("set_search_n_kernel", 1, start);
        RETURN_ON_ERROR(transform(tmp_output,
                                  output,
                                  1,
                                  rocprim::identity<output_type>(),
                                  stream,
                                  debug_synchronous));
        return hipSuccess;
    }

    start_timer();
    set_search_n_kernel<<<1, 1, 0, stream>>>(tmp_output, size);
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("set_search_n_kernel", 1, start);

    search_n_kernel<config><<<num_blocks, block_size, 0, stream>>>(input,
                                                                   tmp_output,
                                                                   size,
                                                                   count,
                                                                   value,
                                                                   compare_function);
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("search_n_kernel", size, start);

    RETURN_ON_ERROR(transform(tmp_output,
                              output,
                              1,
                              rocprim::identity<output_type>(),
                              stream,
                              debug_synchronous));

    return hipSuccess;
}

} // namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_SEARCH_N_HPP_
