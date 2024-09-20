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

#define CHECK(...) printf(#__VA_ARGS__ " = %d\n", (__VA_ARGS__));
namespace detail
{

ROCPRIM_KERNEL
void set_search_n_kernel(size_t* output, size_t target)
{
    *output = target;
}

template<class Config, class InputIterator, class BinaryPredicate>
ROCPRIM_KERNEL __launch_bounds__(device_params<Config>().kernel_config.block_size)
void search_n_kernel(InputIterator                                                   input,
                     size_t*                                                         output,
                     size_t                                                          size,
                     size_t                                                          count,
                     typename std::iterator_traits<InputIterator>::value_type const* value,
                     BinaryPredicate binary_predicate)
{
    constexpr auto params           = device_params<Config>();
    constexpr auto block_size       = params.kernel_config.block_size;
    constexpr auto items_per_thread = params.kernel_config.items_per_thread;
    constexpr auto items_per_block  = block_size * items_per_thread;

    const auto t_id = block_thread_id<0>();
    const auto b_id = block_id<0>();

    const size_t this_thread_start_idx = (b_id * items_per_block) + (items_per_thread * t_id);

    if(size < count + this_thread_start_idx)
    { // not able to find a sequence equal to or longer than count
        return;
    }

    size_t remaining_count    = count;
    size_t sequence_start_idx = this_thread_start_idx;

#define __LOCAL_SEARCH_N_LOOP_BODY__                \
    if(binary_predicate(*(input + i), *value))      \
    {                                               \
        if(--remaining_count == 0)                  \
        {                                           \
            atomic_min(output, sequence_start_idx); \
            return;                                 \
        }                                           \
    }                                               \
    else                                            \
    {                                               \
        remaining_count    = count;                 \
        sequence_start_idx = i + 1;                 \
    }

    if(size - this_thread_start_idx < items_per_thread)
    { // last thread
        const size_t num_valid_in_last_thread = size - this_thread_start_idx;
        for(size_t i = this_thread_start_idx;
            sequence_start_idx - this_thread_start_idx < num_valid_in_last_thread
            && i + remaining_count <= size;
            ++i)
        {
            __LOCAL_SEARCH_N_LOOP_BODY__
        }
        return;
    }
    // complete block
    for(size_t i = this_thread_start_idx;
        sequence_start_idx - this_thread_start_idx < items_per_thread
        && i + remaining_count <= size;
        ++i)
    {
        __LOCAL_SEARCH_N_LOOP_BODY__
    }

#undef __LOCAL_SEARCH_N_LOOP_BODY__
}

template<class Config, class InputIterator, class OutputIterator, class BinaryPredicate>
ROCPRIM_INLINE
hipError_t search_n_impl(void*          temporary_storage,
                         size_t&        storage_size,
                         InputIterator  input,
                         OutputIterator output,
                         size_t         size,
                         size_t         count,
                         typename std::iterator_traits<InputIterator>::value_type const* value,
                         BinaryPredicate binary_predicate,
                         hipStream_t     stream,
                         bool            debug_synchronous)
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

    if(size == 0 || count <= 0)
    {
        // return end
        start_timer();
        set_search_n_kernel<<<1, 1, 0, stream>>>(output, count <= 0 ? 0 : size);
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
                                                                   binary_predicate);
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
