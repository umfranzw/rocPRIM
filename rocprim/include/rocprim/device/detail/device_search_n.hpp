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
#include <iterator>

BEGIN_ROCPRIM_NAMESPACE
namespace detail
{

inline void search_n_start_timer(std::chrono::steady_clock::time_point& start,
                                 const bool                             debug_synchronous)
{
    if(debug_synchronous)
    {
        start = std::chrono::steady_clock::now();
    }
}

ROCPRIM_KERNEL __launch_bounds__(1)
void init_search_n_kernel(size_t* output, const size_t target)
{
    *output = target;
}

/// \brief Supports all forms of search_n operations,
/// but the efficiency is insufficient when `items_per_block is` too large.
template<class Config, class InputIterator, class BinaryPredicate>
ROCPRIM_KERNEL __launch_bounds__(device_params<Config>().kernel_config.block_size)
void search_n_kernel(InputIterator                                                   input,
                     size_t*                                                         output,
                     const size_t                                                    size,
                     const size_t                                                    count,
                     const typename std::iterator_traits<InputIterator>::value_type* value,
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

    const size_t items_this_thread
        = std::min<size_t>(size - this_thread_start_idx, items_per_thread);
    for(size_t i = this_thread_start_idx;
        sequence_start_idx - this_thread_start_idx < items_this_thread
        && i + remaining_count <= size;
        ++i)
    {
        if(binary_predicate(*(input + i), *value))
        {
            if(--remaining_count == 0)
            {
                atomic_min(output, sequence_start_idx);
                return;
            }
        }
        else
        {
            remaining_count    = count;
            sequence_start_idx = i + 1;
        }
    }
}

template<class Config, class InputIterator, class BinaryPredicate>
ROCPRIM_KERNEL __launch_bounds__(device_params<Config>().kernel_config.block_size)
void search_n_find_head_kernel(
    InputIterator                                                   input,
    const size_t                                                    size,
    const size_t                                                    count,
    const typename std::iterator_traits<InputIterator>::value_type* value,
    const BinaryPredicate                                           binary_predicate,
    size_t*                                                         head_of_each_group,
    const size_t                                                    num_groups,
    const size_t                                                    blocks_per_group)
{
    constexpr auto params           = device_params<Config>();
    constexpr auto block_size       = params.kernel_config.block_size;
    constexpr auto items_per_thread = params.kernel_config.items_per_thread;
    constexpr auto items_per_block  = block_size * items_per_thread;

    // only one block
    const auto b_id = block_id<0>();
    const auto t_id = block_thread_id<0>();
    const auto g_id = b_id / blocks_per_group; // group id

    const size_t this_thread_start_idx = (b_id * items_per_block) + (items_per_thread * t_id);
    const size_t items_this_thread
        = std::min<size_t>(this_thread_start_idx < size ? size - this_thread_start_idx : 0,
                           items_per_thread);

    for(size_t i = this_thread_start_idx; i < this_thread_start_idx + items_this_thread; i++)
    {
        if(binary_predicate(input[i], *value))
        {
            if(i == 0)
            { // is head
                atomic_min(head_of_each_group + g_id, size - i - 1);
            }
            else
            {
                if(!binary_predicate(input[i - 1], *value))
                { // is head
                    atomic_min(head_of_each_group + g_id, size - i - 1);
                }
            }
        }
    }
}

template<class Config>
ROCPRIM_KERNEL __launch_bounds__(device_params<Config>().kernel_config.block_size)
void search_n_heads_filter_kernel(const size_t  size,
                                  const size_t  count,
                                  const size_t* heads,
                                  const size_t  heads_size,
                                  size_t*       filtered_heads,
                                  size_t*       filtered_heads_size)
{
    constexpr auto params           = device_params<Config>();
    constexpr auto block_size       = params.kernel_config.block_size;
    constexpr auto items_per_thread = params.kernel_config.items_per_thread;
    constexpr auto items_per_block  = block_size * items_per_thread;

    const auto b_id = block_id<0>();
    const auto t_id = block_thread_id<0>();

    const size_t this_thread_start_idx = (b_id * items_per_block) + (t_id * items_per_thread);
    for(size_t i = 0; i < items_per_thread; ++i)
    {
        size_t idx = this_thread_start_idx + i;
        if(idx >= heads_size)
        {
            break;
        }
        // do filter
        if(heads[idx] != (size_t)-1)
        {
            size_t this_head = size - heads[idx] - 1;
            if(idx + 1 < heads_size)
            {
                if(heads[idx + 1] != (size_t)-1)
                {
                    size_t next_head = size - heads[idx + 1] - 1;
                    if(next_head - this_head - 1 >= count)
                    {
                        filtered_heads[atomic_add(filtered_heads_size, 1)] = this_head;
                    }
                }
                else
                {
                    filtered_heads[atomic_add(filtered_heads_size, 1)] = this_head;
                }
            }
            else
            {
                filtered_heads[atomic_add(filtered_heads_size, 1)] = this_head;
            }
        }
    }
}

template<class Config, class InputIterator, class BinaryPredicate>
ROCPRIM_KERNEL __launch_bounds__(device_params<Config>().kernel_config.block_size)
void search_n_reduce_kernel(InputIterator                                                   input,
                            const size_t                                                    size,
                            const size_t                                                    count,
                            const typename std::iterator_traits<InputIterator>::value_type* value,
                            const BinaryPredicate binary_predicate,
                            size_t*               heads,
                            const size_t          blocks_per_group)
{
    constexpr auto params           = device_params<Config>();
    constexpr auto block_size       = params.kernel_config.block_size;
    constexpr auto items_per_thread = params.kernel_config.items_per_thread;
    constexpr auto items_per_block  = block_size * items_per_thread;

    const auto b_id            = block_id<0>();
    const auto t_id            = block_thread_id<0>();
    const auto g_id            = b_id / blocks_per_group; // group id
    const auto g_b_id          = b_id % blocks_per_group;
    const auto items_per_group = items_per_block * blocks_per_group;

    const size_t check_head  = heads[g_id] + 1;
    const size_t check_count = count - 1;

    const size_t this_thread_start_idx = (g_b_id * items_per_block) + (t_id * items_per_thread);
    for(size_t i = 0; i < items_per_block; i++)
    {
        size_t idx = check_head + this_thread_start_idx + i;
        if(idx >= size || idx >= (check_head + check_count))
        {
            break;
        }
        if(!binary_predicate(input[idx], *value))
        {
            heads[g_id] = size;
            return;
        }
    }
}

template<class Config>
ROCPRIM_KERNEL __launch_bounds__(device_params<Config>().kernel_config.block_size)
void search_n_min_kernel(const size_t* heads, const size_t heads_size, size_t* output)
{
    constexpr auto params           = device_params<Config>();
    constexpr auto block_size       = params.kernel_config.block_size;
    constexpr auto items_per_thread = params.kernel_config.items_per_thread;
    constexpr auto items_per_block  = block_size * items_per_thread;

    const auto b_id = block_id<0>();
    const auto t_id = block_thread_id<0>();

    const size_t this_thread_start_idx = (b_id * items_per_block) + (t_id * items_per_thread);
    for(size_t i = 0; i < items_per_thread; ++i)
    {
        size_t idx = this_thread_start_idx + i;
        if(idx >= heads_size)
        {
            break;
        }
        atomic_min(output, heads[idx]);
    }
}

template<class Config,
         bool calculate_storage_size,
         class InputIterator,
         class OutputIterator,
         class BinaryPredicate>
    ROCPRIM_INLINE
hipError_t search_n_impl(void*          temporary_storage,
                         size_t&        storage_size,
                         InputIterator  input,
                         OutputIterator output,
                         const size_t   size,
                         const size_t   count,
                         const typename std::iterator_traits<InputIterator>::value_type* value,
                         const BinaryPredicate binary_predicate,
                         const hipStream_t     stream,
                         const bool            debug_synchronous)
{
    using input_type  = typename std::iterator_traits<InputIterator>::value_type;
    using output_type = typename std::iterator_traits<OutputIterator>::value_type;
    using config      = wrapped_search_n_config<Config, input_type>;

    target_arch target_arch;
    RETURN_ON_ERROR(host_target_arch(stream, target_arch));

    const auto         params           = dispatch_target_arch<config>(target_arch);
    const unsigned int block_size       = params.kernel_config.block_size;
    const unsigned int items_per_thread = params.kernel_config.items_per_thread;
    const unsigned int items_per_block  = block_size * items_per_thread;
    const unsigned int num_blocks       = ceiling_div(size, items_per_block);

    const size_t blocks_per_group = count / items_per_block;

    std::chrono::steady_clock::time_point start;

    size_t*      tmp_output = reinterpret_cast<size_t*>(temporary_storage);
    const size_t num_groups = blocks_per_group ? ceiling_div(num_blocks, blocks_per_group) : 0;

    if(tmp_output == nullptr)
    {
        storage_size = blocks_per_group == 0 ?: sizeof(size_t) + (sizeof(size_t) * num_groups * 2);
        return hipSuccess;
    }

    if(count > size)
    {
        return hipErrorInvalidValue;
    }

    if(size == 0 || count <= 0)
    {
        // return end or begin
        search_n_start_timer(start, debug_synchronous);
        init_search_n_kernel<<<1, 1, 0, stream>>>(tmp_output, count <= 0 ? 0 : size);
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("init_search_n_kernel", 1, start);

        RETURN_ON_ERROR(transform(tmp_output,
                                  output,
                                  1,
                                  rocprim::identity<output_type>(),
                                  stream,
                                  debug_synchronous));
        return hipSuccess;
    }

    if(blocks_per_group == 0)
    {
        // normal search_n
        search_n_start_timer(start, debug_synchronous);
        init_search_n_kernel<<<1, 1, 0, stream>>>(tmp_output, size);
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("init_search_n_kernel", 1, start);

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
    }
    else
    {
        size_t* head_of_each_group = tmp_output + 1;
        size_t* filtered_heads     = head_of_each_group + num_groups;

        search_n_start_timer(start, debug_synchronous);
        // initialization
        HIP_CHECK(hipMemsetAsync(tmp_output, 0, sizeof(size_t), stream));
        HIP_CHECK(hipMemsetAsync(head_of_each_group, -1, sizeof(size_t) * num_groups * 2, stream));

        // find head
        search_n_find_head_kernel<config><<<num_blocks, block_size, 0, stream>>>(input,
                                                                                 size,
                                                                                 count,
                                                                                 value,
                                                                                 binary_predicate,
                                                                                 head_of_each_group,
                                                                                 num_groups,
                                                                                 blocks_per_group);
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("search_n_find_head_kernel", size, start);

        // do filter for heads
        const size_t num_blocks_for_heads = ceiling_div(num_groups, items_per_block);
        search_n_heads_filter_kernel<config>
            <<<num_blocks_for_heads, block_size, 0, stream>>>(size,
                                                              count,
                                                              head_of_each_group,
                                                              num_groups,
                                                              filtered_heads,
                                                              tmp_output);
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("search_n_find_head_kernel", num_groups, start);

        size_t h_filtered_heads_size;
        HIP_CHECK(hipMemcpyAsync(&h_filtered_heads_size,
                                 tmp_output,
                                 sizeof(size_t),
                                 hipMemcpyDeviceToHost,
                                 stream));
        HIP_CHECK(hipStreamSynchronize(stream));

        init_search_n_kernel<<<1, 1, 0, stream>>>(tmp_output, size);
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("init_search_n_kernel", 1, start);

        if(h_filtered_heads_size == 0)
        {
            RETURN_ON_ERROR(transform(tmp_output,
                                      output,
                                      1,
                                      rocprim::identity<output_type>(),
                                      stream,
                                      debug_synchronous));
            return hipSuccess;
        }

        const size_t num_blocks_for_reduce = h_filtered_heads_size * (blocks_per_group + 1);
        search_n_reduce_kernel<config>
            <<<num_blocks_for_reduce, block_size, 0, stream>>>(input,
                                                               size,
                                                               count,
                                                               value,
                                                               binary_predicate,
                                                               filtered_heads,
                                                               blocks_per_group + 1);
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("search_n_reduce_kernel",
                                                    h_filtered_heads_size,
                                                    start);

        const size_t num_blocks_for_min = ceiling_div(h_filtered_heads_size, items_per_block);
        search_n_min_kernel<config>
            <<<num_blocks_for_min, block_size, 0, stream>>>(filtered_heads,
                                                            h_filtered_heads_size,
                                                            tmp_output);
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("search_n_min_head",
                                                    h_filtered_heads_size,
                                                    start);

        RETURN_ON_ERROR(transform(tmp_output,
                                  output,
                                  1,
                                  rocprim::identity<output_type>(),
                                  stream,
                                  debug_synchronous));
    }
    return hipSuccess;
}

} // namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_SEARCH_N_HPP_
