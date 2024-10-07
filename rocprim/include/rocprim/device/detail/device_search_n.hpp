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

/// \brief This kernel will mark blocks whose elements satisfy the \p binary_predicate.
/// The marks will be stored in d_full_blocks_flags.
template<class Config, class InputIterator, class BinaryPredicate>
ROCPRIM_KERNEL __launch_bounds__(device_params<Config>().kernel_config.block_size)
void search_n_full_blocks_kernel(
    InputIterator                                                   input,
    const size_t                                                    size,
    unsigned char*                                                  d_full_blocks_flags,
    unsigned char*                                                  d_standard_flag_value,
    const typename std::iterator_traits<InputIterator>::value_type* value,
    BinaryPredicate                                                 binary_predicate)
{
    constexpr auto params           = device_params<Config>();
    constexpr auto block_size       = params.kernel_config.block_size;
    constexpr auto items_per_thread = params.kernel_config.items_per_thread;
    constexpr auto items_per_block  = block_size * items_per_thread;

    const auto b_id = block_id<0>();
    const auto t_id = block_thread_id<0>();

    // set standard value which will be use later
    if(b_id == 0 && t_id == 0)
    {
        *d_standard_flag_value = 1;
    }

    if(b_id == (size / items_per_block)) // is_incomplete_block
    {
        if(t_id == 0)
        {
            d_full_blocks_flags[b_id] = 0;
        }
    }
    else
    {
        if(t_id == 0)
        {
            d_full_blocks_flags[b_id] = 1;
        }

        rocprim::syncthreads();

        const size_t this_thread_start_idx = (b_id * items_per_block) + (items_per_thread * t_id);
        ROCPRIM_UNROLL
        for(size_t i = 0; i < items_per_thread; i++)
        {
            if(d_full_blocks_flags[b_id] != 0
               && !binary_predicate(*(input + i + this_thread_start_idx), *value))
            {
                d_full_blocks_flags[b_id] = 0;
            }
        }
    }
}

template<class Config, class InputIterator, class BinaryPredicate>
ROCPRIM_KERNEL __launch_bounds__(device_params<Config>().kernel_config.block_size)
void search_n_bound_detect_kernel(
    const size_t                                                    full_blocks_start,
    const size_t                                                    num_blocks,
    const size_t                                                    full_blocks_required,
    size_t*                                                         output,
    InputIterator                                                   input,
    const size_t                                                    input_size,
    const size_t                                                    input_count,
    const typename std::iterator_traits<InputIterator>::value_type* value,
    BinaryPredicate                                                 binary_predicate)
{
    constexpr auto params           = device_params<Config>();
    constexpr auto block_size       = params.kernel_config.block_size;
    constexpr auto items_per_thread = params.kernel_config.items_per_thread;
    constexpr auto items_per_block  = block_size * items_per_thread;

    // only one block
    const auto b_id = block_id<0>();
    const auto t_id = block_thread_id<0>();

    // const size_t full_blocks_start      = *p_full_blocks_start;
    size_t items_need_tobe_checked = input_count - (full_blocks_required * items_per_block);

    if(full_blocks_start == num_blocks)
    {
        if(t_id == 0)
        {
            *output = input_size;
        }
    }
    else if(items_need_tobe_checked == 0)
    {
        if(t_id == 0)
        {
            *output = full_blocks_start * items_per_block;
        }
    }
    else
    {
        // further check
        const size_t this_thread_start_idx = (b_id * items_per_block) + (items_per_thread * t_id);

        // detect left bound
        size_t left_bound
            = full_blocks_start * items_per_block; // also indicate how many items in left sequence
        size_t check_left_size
            = std::min<size_t>(std::min<size_t>(left_bound, items_need_tobe_checked),
                               items_per_block);
        size_t check_left_start = left_bound - check_left_size;

        if(t_id == 0)
        {
            *output = check_left_size;
        }
        rocprim::syncthreads();

        if(this_thread_start_idx < check_left_size)
        {
            const size_t items_this_thread
                = std::min<size_t>(check_left_size - this_thread_start_idx, items_per_thread);

            for(size_t i = this_thread_start_idx; i < this_thread_start_idx + items_this_thread;
                i++)
            {
                if(!binary_predicate(*(input + check_left_start + i), *value))
                {
                    atomic_min(output, check_left_size - i - 1);
                }
            }
        }

        rocprim::syncthreads();
        left_bound -= *output; // read only no racing
        items_need_tobe_checked -= *output;
        // detect right bound
        size_t right_bound = (full_blocks_start + full_blocks_required) * items_per_block
                             + items_need_tobe_checked;
        if(right_bound > input_size)
        {
            if(t_id == 0)
            {
                *output = input_size;
            }
        }
        else
        {
            // verify the right count
            size_t check_right_size = items_need_tobe_checked;
            size_t check_right_start = (full_blocks_start + full_blocks_required) * items_per_block;

            // detect right bound

            if(t_id == 0)
            {
                *output = check_right_size;
            }
            rocprim::syncthreads();

            if(this_thread_start_idx < check_right_size)
            {
                const size_t items_this_thread
                    = std::min<size_t>(check_right_size - this_thread_start_idx, items_per_thread);

                for(size_t i = this_thread_start_idx; i < this_thread_start_idx + items_this_thread;
                    i++)
                {
                    if(!binary_predicate(*(input + check_right_start + i), *value))
                    {
                        atomic_min(output, i);
                    }
                    if(check_right_start + i + items_per_block < right_bound)
                    {
                        if(!binary_predicate(*(input + check_right_start + i + items_per_block),
                                             *value))
                        {
                            atomic_min(output, i + items_per_block);
                        }
                    }
                }
            }

            rocprim::syncthreads();
            if(t_id == 0)
            {
                *output = *output == check_right_size ? left_bound : input_size;
            }
        }
    }
    return;
}

template<class Config,
         bool calculate_storage_size,
         class InputIterator,
         class OutputIterator,
         class BinaryPredicate>
ROCPRIM_INLINE
hipError_t search_n_impl(void*          temporary_storage,
                         size_t&        storage_size,
                         size_t         storage_offset,
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

    const unsigned int num_blocks = ceiling_div(size, items_per_block);

    // We need to find `expected_full_blocks_count` number of blocks
    size_t expected_full_blocks_count = count / items_per_block;
    expected_full_blocks_count
        = expected_full_blocks_count > 0 ? expected_full_blocks_count - 1 : 0;

    // Start point for time measurements
    std::chrono::steady_clock::time_point start;

    // calculate temprory_size
    if ROCPRIM_IF_CONSTEXPR(calculate_storage_size)
    {
        size_t       size_needed = sizeof(size_t) + sizeof(unsigned char);
        size_t       next_count  = expected_full_blocks_count;
        unsigned int next_size   = num_blocks;
        while(next_count > 1)
        {
            size_needed += sizeof(unsigned char) * next_size;
            next_size  = ceiling_div(next_size, items_per_block);
            next_count = next_count / items_per_block;
            next_count = next_count > 0 ? next_count - 1 : 0;
        }

        if(temporary_storage == nullptr || storage_size != size_needed)
        {
            storage_size = size_needed;
            return hipSuccess;
        }
    }

    size_t* tmp_output = reinterpret_cast<size_t*>(temporary_storage);

    if(count > size)
    {
        return hipErrorInvalidValue;
    }

    if(size == 0 || count <= 0)
    {
        // return end
        search_n_start_timer(start, debug_synchronous);
        init_search_n_kernel<<<1, 1, 0, stream>>>(tmp_output, count <= 0 ? 0 : size);
        RETURN_ON_ERROR(transform(tmp_output,
                                  output,
                                  1,
                                  rocprim::identity<output_type>(),
                                  stream,
                                  debug_synchronous));
        return hipSuccess;
    }

    if(expected_full_blocks_count <= 1)
    { // In this case
        // normal search_n
        // printf("1\n");
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

        return hipSuccess;
    }
    else
    {
        // search adjacent full_blocks & search_n
        unsigned char* d_standard_flag_value
            = reinterpret_cast<unsigned char*>(temporary_storage) + sizeof(*tmp_output);
        unsigned char* d_full_blocks_flags
            = d_standard_flag_value + sizeof(*d_standard_flag_value) + storage_offset;

        // Mark blocks as `equal` or `not equal`
        // Equal means each item in this block satisfy the `binary_predicate`
        search_n_start_timer(start, debug_synchronous);
        search_n_full_blocks_kernel<config>
            <<<num_blocks, block_size, 0, stream>>>(input,
                                                    size,
                                                    d_full_blocks_flags,
                                                    d_standard_flag_value,
                                                    value,
                                                    binary_predicate);
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("search_n_full_blocks_kernel", size, start);

        size_t full_flags_size         = num_blocks;
        size_t full_flags_start_offset = 0;
        size_t h_full_block_start;
        size_t h_tmp_output;

        while(1)
        {
            // Find the adjacent sequence of `full_blocks` (with false `debug_synchronous`)
            RETURN_ON_ERROR(search_n_impl<Config, false>(
                temporary_storage,
                storage_size,
                num_blocks, // the next temp storage offset of d_full_blocks_flags
                d_full_blocks_flags, // this is the input
                tmp_output, // write full blocks start index into temp_output
                full_flags_size, // this is the input size
                expected_full_blocks_count, // expected items count in the sequence
                d_standard_flag_value, // assigned to be 1 by `search_n_full_blocks_kernel`
                rocprim::equal_to<unsigned int>{},
                stream,
                false));

            HIP_CHECK(hipMemcpyAsync(&h_full_block_start,
                                     tmp_output,
                                     sizeof(*tmp_output),
                                     hipMemcpyDeviceToHost,
                                     stream));
            h_full_block_start += full_flags_start_offset;

            search_n_bound_detect_kernel<config><<<1, block_size, 0, stream>>>(
                h_full_block_start, // now this pointer stores `full blocks start index`
                num_blocks,
                expected_full_blocks_count,
                tmp_output, // use same buffer to store the output
                input,
                size,
                count,
                value,
                binary_predicate);
            ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("search_n_bound_detect_kernel",
                                                        size,
                                                        start);

            HIP_CHECK(hipMemcpyAsync(&h_tmp_output,
                                     tmp_output,
                                     sizeof(*tmp_output),
                                     hipMemcpyDeviceToHost,
                                     stream));

            if(h_tmp_output == size && h_full_block_start + expected_full_blocks_count < num_blocks)
            {
                full_flags_size -= h_full_block_start + expected_full_blocks_count;
                d_full_blocks_flags += h_full_block_start + expected_full_blocks_count;
                full_flags_start_offset += h_full_block_start + expected_full_blocks_count;
            }
            else
            {
                break;
            }
        };

        RETURN_ON_ERROR(transform(tmp_output,
                                  output,
                                  1,
                                  rocprim::identity<output_type>(),
                                  stream,
                                  debug_synchronous));
        return hipSuccess;
    }
}

} // namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_SEARCH_N_HPP_
