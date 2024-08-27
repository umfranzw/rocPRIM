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

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_ADJACENT_FIND_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_ADJACENT_FIND_HPP_

#include "device_config_helper.hpp"
#include "ordered_block_id.hpp"

#include "../../block/block_load.hpp"
#include "../../block/block_reduce.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{
namespace adjacent_find
{
template<typename Config,
         typename TransformedInputIterator,
         typename ReduceIndexIterator,
         typename BinaryPred>
ROCPRIM_KERNEL __launch_bounds__(device_params<Config>().kernel_config.block_size)
void block_reduce_kernel(TransformedInputIterator transformed_input,
                         ReduceIndexIterator      reduce_output,
                         const std::size_t        size,
                         BinaryPred               op)
{
    static constexpr adjacent_find_config_params params     = device_params<Config>();
    static constexpr unsigned int                block_size = params.kernel_config.block_size;
    static constexpr unsigned int items_per_thread          = params.kernel_config.items_per_thread;
    static constexpr unsigned int items_per_block           = block_size * items_per_thread;

    using transformed_input_type =
        typename std::iterator_traits<TransformedInputIterator>::value_type;
    using block_reduce_type = ::rocprim::block_reduce<
        transformed_input_type,
        block_size,
        block_reduce_algorithm::raking_reduce>;

    ROCPRIM_SHARED_MEMORY union
    {
        std::size_t global_reduce_output;
    } storage;

    const unsigned int bid        = blockIdx.x;
    const unsigned int tid        = threadIdx.x;
    const unsigned int grid_items = ::rocprim::detail::grid_size<0>() * items_per_block;

    const std::size_t block_offset = bid * items_per_block;

    for(std::size_t tile_offset = block_offset; tile_offset < size; tile_offset += grid_items)
    {

        // First thread of each block loads the latest global adjacent index found
        if(tid == 0)
        {
            storage.global_reduce_output = atomic_load(reduce_output);
        }
        syncthreads();

        // Early exit if a previous block found an adjacent pair
        if(storage.global_reduce_output < tile_offset)
        {
            return;
        }

        // Do block reduction
        transformed_input_type transformed_input_values[items_per_thread];
        transformed_input_type output_value;

        if(tile_offset + items_per_block > size) /* Last incomplete processing */
        {
            const std::size_t valid_in_last_iteration = size - tile_offset;
            block_load_direct_striped<block_size>(tid,
                                                  transformed_input + tile_offset,
                                                  transformed_input_values,
                                                  valid_in_last_iteration);

            // Thread reductions with boundary check
            output_value = transformed_input_values[0];
            ROCPRIM_UNROLL
            for(unsigned int i = 1; i < items_per_thread; i++)
            {
                if(tid + i * block_size < valid_in_last_iteration)
                {
                    output_value = op(output_value, transformed_input_values[i]);
                }
            }
            // Reduce thread reductions
            block_reduce_type().reduce(output_value, // input
                                       output_value, // output
                                       std::min(valid_in_last_iteration, std::size_t{block_size}),
                                       op);
        }
        else /* Complete processings */
        {
            block_load_direct_striped<block_size>(tid,
                                                  transformed_input + tile_offset,
                                                  transformed_input_values);
            block_reduce_type().reduce(transformed_input_values, // input
                                       output_value, // output
                                       op);
        }

        // Save reduction's index into output if an adjacent pair is found
        if(tid == 0)
        {
            if(::rocprim::get<0>(output_value) != 0)
            {
                atomic_min(reduce_output, ::rocprim::get<1>(output_value));
            }
        }
    }
}
} // namespace adjacent_find
} // namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_ADJACENT_FIND_HPP_
