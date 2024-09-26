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
void set_search_n_kernel(size_t* output, size_t target)
{
    *output = target;
}

/// \brief Supports all forms of search_n operations,
/// but the efficiency is insufficient when `items_per_block is` too large.
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


/// \brief This kernel will return count of adjacent items that satisfies the \p binary_predicate
/// and adjacent to the edge of the \p input sequence e.g. one of them must have the index 0 or size - 1
/// This kernel can search from forward or backward
///
/// The kernel only processes one block.
///
/// \tparam forward Used to indicate whether the detection goes from back to front or vice versa
///
/// \param [in] size This kernel will process the elements in [input + 0, input + size).
/// \param [in,out] output if all elements satisfie the \p binary_predicate, \p size will be written into \p output
///
template<class Config, bool forward, class InputIterator, class BinaryPredicate>
ROCPRIM_KERNEL
void search_n_detect_edge_kernel(
    InputIterator                                                   input,
    size_t*                                                         output,
    size_t                                                          size,
    typename std::iterator_traits<InputIterator>::value_type const* value,
    BinaryPredicate                                                 binary_predicate)
{

    constexpr auto params           = device_params<Config>();
    constexpr auto block_size       = params.kernel_config.block_size;
    constexpr auto items_per_thread = params.kernel_config.items_per_thread;
    constexpr auto items_per_block  = block_size * items_per_thread;

    const auto b_id = block_id<0>();
    const auto t_id = block_thread_id<0>();

    const size_t this_thread_start_idx = (b_id * items_per_block) + (items_per_thread * t_id);

    if(b_id == 0 && t_id == 0)
    {
        *output = size;
    }

    if(this_thread_start_idx >= size)
    {
        return;
    }

    rocprim::syncthreads();

    const size_t items_this_thread = size - this_thread_start_idx < items_per_thread
                                         ? size - this_thread_start_idx
                                         : items_per_thread;

    if ROCPRIM_IF_CONSTEXPR(forward)
    {
        for(size_t i = this_thread_start_idx; i < this_thread_start_idx + items_this_thread; i++)
        {
            if(!binary_predicate(*(input + i), *value))
            {
                atomic_min(output, size - i - 1);
            }
        }
    }
    else
    {
        for(size_t i = this_thread_start_idx; i < this_thread_start_idx + items_this_thread; i++)
        {
            // find smallest i which not satisfy the binary_predicate
            if(!binary_predicate(*(input + i), *value))
            {
                atomic_min(output, i);
            }
        }
    }
}

/// \brief This kernel will mark blocks whose elements satisfy the \p binary_predicate.
/// The marks will be stored in d_equal_flags.
template<class Config, bool check_incomplete_block, class InputIterator, class BinaryPredicate>
ROCPRIM_KERNEL __launch_bounds__(device_params<Config>().kernel_config.block_size)
void search_n_equal_blocks_kernel(
    InputIterator                                                   input,
    size_t                                                          size,
    unsigned char*                                                  d_equal_flags,
    typename std::iterator_traits<InputIterator>::value_type const* value,
    BinaryPredicate                                                 binary_predicate)
{
    constexpr auto params           = device_params<Config>();
    constexpr auto block_size       = params.kernel_config.block_size;
    constexpr auto items_per_thread = params.kernel_config.items_per_thread;
    constexpr auto items_per_block  = block_size * items_per_thread;

    const auto b_id                = block_id<0>();
    const auto t_id                = block_thread_id<0>();
    const bool is_incomplete_block = b_id == (size / items_per_block);

    if(is_incomplete_block)
    {
        if ROCPRIM_IF_CONSTEXPR(check_incomplete_block)
        {
            if(t_id == 0)
            {
                d_equal_flags[b_id] = 1;
            }

            rocprim::syncthreads();

            const size_t this_thread_start_idx
                = (b_id * items_per_block) + (items_per_thread * t_id);
            for(size_t i = this_thread_start_idx;
                i < this_thread_start_idx + items_per_thread && i < size;
                i++)
            {
                if(d_equal_flags[b_id] != 0 && !binary_predicate(*(input + i), *value))
                {
                    d_equal_flags[b_id] = 0; // atomic is not necessary
                }
            }
        }
        else
        {
            if(t_id == 0)
            {
                d_equal_flags[b_id] = 0;
            }
        }
    }
    else
    {
        if(t_id == 0)
        {
            d_equal_flags[b_id] = 1;
        }

        rocprim::syncthreads();

        const size_t this_thread_start_idx = (b_id * items_per_block) + (items_per_thread * t_id);
        for(size_t i = this_thread_start_idx; i < this_thread_start_idx + items_per_thread; i++)
        {
            if(d_equal_flags[b_id] != 0 && !binary_predicate(*(input + i), *value))
            {
                d_equal_flags[b_id] = 0; // atomic is not necessary
            }
        }
    }
}

template<class Config, class InputIterator, class OutputIterator, class BinaryPredicate>
ROCPRIM_INLINE
hipError_t search_n_impl_2(void*          temporary_storage,
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

    const unsigned int num_blocks = ceiling_div(size, items_per_block);

    // We need to find `equal_blocks_count` number of blocks
    size_t equal_blocks_count = count / items_per_block;
    equal_blocks_count        = equal_blocks_count > 0 ? equal_blocks_count - 1 : 0;

    size_t* tmp_output              = reinterpret_cast<size_t*>(temporary_storage);
    size_t* d_equal_flags_start_pos = tmp_output + 1;
    // Start point for time measurements
    std::chrono::steady_clock::time_point start;

    if(temporary_storage == nullptr)
    {
        storage_size = sizeof(size_t) * 2;
        return hipSuccess;
    }

    if(count > size)
    {
        return hipErrorInvalidValue;
    }

    if(size == 0 || count <= 0)
    {
        // return end
        search_n_start_timer(start, debug_synchronous);
        set_search_n_kernel<<<1, 1, 0, stream>>>(output, count <= 0 ? 0 : size);
        return hipSuccess;
    }

    if(equal_blocks_count <= 1)
    { // In this case
        // normal search_n
        search_n_start_timer(start, debug_synchronous);
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
    else
    {
        // search adjacent equal_blocks & search_n
        unsigned char* d_equal_flags;
        unsigned char* d_standard_flag;

        unsigned char h_temp_flag;
        size_t        h_temp_output;
        size_t        h_equal_flags_start_pos;
        size_t        h_items_need_tobe_checked;

        // TODO: move all these device vars into temporary_storage
        RETURN_ON_ERROR(hipMallocAsync(&d_equal_flags, num_blocks * sizeof(unsigned char), stream));
        RETURN_ON_ERROR(hipMallocAsync(&d_standard_flag, sizeof(unsigned char), stream));

        RETURN_ON_ERROR(
            hipMemsetAsync(d_equal_flags, 0, num_blocks * sizeof(unsigned char), stream));
        RETURN_ON_ERROR(hipMemsetAsync(d_standard_flag, 1, sizeof(unsigned char), stream));

        // Mark blocks as `equal` or `not equal`
        // Equal means each item in this block satisfy the `binary_predicate`
        search_n_start_timer(start, debug_synchronous);
        search_n_equal_blocks_kernel<config, false>
            <<<num_blocks, block_size, 0, stream>>>(input,
                                                    size,
                                                    d_equal_flags,
                                                    value,
                                                    binary_predicate);
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("search_n_equal_blocks_kernel", size, start);

        // Find the adjacent sequence of `equal_blocks` (with false `debug_synchronous`)
        RETURN_ON_ERROR(search_n_impl_2<Config>(temporary_storage,
                                                storage_size,
                                                d_equal_flags,
                                                d_equal_flags_start_pos,
                                                num_blocks,
                                                equal_blocks_count,
                                                d_standard_flag,
                                                rocprim::equal_to<unsigned int>{},
                                                stream,
                                                false));
        // Read the result from device
        RETURN_ON_ERROR(hipMemcpyAsync(&h_equal_flags_start_pos,
                                       d_equal_flags_start_pos,
                                       sizeof(h_equal_flags_start_pos),
                                       hipMemcpyDeviceToHost,
                                       stream));

        h_items_need_tobe_checked = count - (equal_blocks_count * items_per_block);

        if(h_equal_flags_start_pos == num_blocks)
        { // `equal_blocks` is not enough -- return `end`
            set_search_n_kernel<<<1, 1, 0, stream>>>(output, size);
            ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("set_search_n_kernel", size, start);
        }
        else if(h_items_need_tobe_checked == 0)
        { // enough equal_blocks, no further check needed, return `h_equal_flags_start_pos * items_per_block`
            set_search_n_kernel<<<1, 1, 0, stream>>>(output,
                                                     h_equal_flags_start_pos * items_per_block);
            ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("set_search_n_kernel", size, start);
        }
        else
        { // further check
            // determine the left bound
            size_t left_bound       = h_equal_flags_start_pos * items_per_block;
            size_t check_left_size  = min(left_bound, h_items_need_tobe_checked);
            size_t check_left_start = left_bound - check_left_size;
            search_n_detect_edge_kernel<config, true>
                <<<1, block_size, 0, stream>>>(input + check_left_start,
                                               tmp_output,
                                               check_left_size,
                                               value,
                                               binary_predicate);
            ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("search_n_detect_edge_kernel", size, start);
            RETURN_ON_ERROR(hipMemcpyAsync(&h_temp_output,
                                           tmp_output,
                                           sizeof(h_temp_output),
                                           hipMemcpyDeviceToHost,
                                           stream));
            left_bound -= h_temp_output;
            h_items_need_tobe_checked -= h_temp_output;

            size_t right_bound = (h_equal_flags_start_pos + equal_blocks_count) * items_per_block
                                 + h_items_need_tobe_checked;
            if(right_bound > size)
            { // return end
                set_search_n_kernel<<<1, 1, 0, stream>>>(output, size);
                ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("set_search_n_kernel", size, start);
            }
            else
            { // verify the right count
                size_t check_right_size = h_items_need_tobe_checked;
                size_t check_right_start
                    = (h_equal_flags_start_pos + equal_blocks_count) * items_per_block;
                search_n_detect_edge_kernel<config, false>
                    <<<1, block_size, 0, stream>>>(input + check_right_start,
                                                   tmp_output,
                                                   check_right_size,
                                                   value,
                                                   binary_predicate);
                ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("search_n_detect_edge_kernel",
                                                            size,
                                                            start);
                RETURN_ON_ERROR(hipMemcpyAsync(&h_temp_output,
                                               tmp_output,
                                               sizeof(h_temp_output),
                                               hipMemcpyDeviceToHost,
                                               stream));
                h_items_need_tobe_checked -= h_temp_output;
                set_search_n_kernel<<<1, 1, 0, stream>>>(output,
                                                         h_items_need_tobe_checked == 0 ? left_bound
                                                                                        : size);
                ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("set_search_n_kernel", size, start);
            }
        }

        // TODO: use temporary_storage
        RETURN_ON_ERROR(hipFree(d_equal_flags));
        RETURN_ON_ERROR(hipFree(d_standard_flag));

        return hipSuccess;
    }
}

template<class Config, class InputIterator, class OutputIterator, class BinaryPredicate>
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
