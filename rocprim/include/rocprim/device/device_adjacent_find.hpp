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

#ifndef ROCPRIM_DEVICE_DEVICE_ADJACENT_FIND_HPP_
#define ROCPRIM_DEVICE_DEVICE_ADJACENT_FIND_HPP_

#include "detail/device_adjacent_find.hpp"
#include "detail/device_config_helper.hpp"
#include "device_adjacent_find_config.hpp"
#include "device_reduce.hpp"
#include "device_transform.hpp"

#include "../functional.hpp"
#include "../iterator/counting_iterator.hpp"
#include "../iterator/transform_iterator.hpp"
#include "../iterator/zip_iterator.hpp"
#include "../types/tuple.hpp"

#include <cstring>

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{
#define ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR(name, size, start)                           \
    {                                                                                            \
        auto _error = hipGetLastError();                                                         \
        if(_error != hipSuccess)                                                                 \
            return _error;                                                                       \
        if(debug_synchronous)                                                                    \
        {                                                                                        \
            std::cout << name << "(" << size << ")";                                             \
            auto __error = hipStreamSynchronize(stream);                                         \
            if(__error != hipSuccess)                                                            \
                return __error;                                                                  \
            auto _end = std::chrono::high_resolution_clock::now();                               \
            auto _d   = std::chrono::duration_cast<std::chrono::duration<double>>(_end - start); \
            std::cout << " " << _d.count() * 1000 << " ms" << '\n';                              \
        }                                                                                        \
    }

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

template<class T, class IdxT>
struct reduce_op
{
    // The reduction logic should be as follows:
    // If both have 1's first, return the one with the smallest index
    // Else:
    //  * [opt 1] If only one of them has a 1 first, return the greater tuple (the one with
    //            the 1 in the first place)
    //  * [opt 2] If no 1's first, we can return any of the two tuples
    // But if instead of 1s we use (-1)s, we can perform the reduction operation much faster by always
    // taking the "lesser" tuple because:
    //  * If both have (-1)s first, we still return the one with the smallest index
    //  * If both have 0s first, we can return any of the two tuples
    //  * If only one of them has a (-1) first, we return the lesser tuple (the one with
    //    the (-1) in the first place)
    ROCPRIM_DEVICE
    inline constexpr ::rocprim::tuple<T, IdxT>
        operator()(const ::rocprim::tuple<T, IdxT>& lhs, const ::rocprim::tuple<T, IdxT>& rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }
};

template<typename Config = default_config,
         typename InputIterator,
         typename OutputIterator,
         typename BinaryPred>
ROCPRIM_INLINE
hipError_t adjacent_find_impl(void* const       temporary_storage,
                              std::size_t&      storage_size,
                              InputIterator     input,
                              OutputIterator    output,
                              const std::size_t size,
                              BinaryPred        op,
                              const hipStream_t stream,
                              const bool        debug_synchronous)
{
    // Data types
    using input_type             = typename std::iterator_traits<InputIterator>::value_type;
    using op_result_type         = int; // use signed type to store (-1)s instead of 1s
    using index_type             = std::size_t;
    using wrapped_input_type     = ::rocprim::tuple<input_type, input_type, index_type>;
    using transformed_input_type = ::rocprim::tuple<op_result_type, index_type>;

    // Operations types
    using reduce_op_type = reduce_op<op_result_type, index_type>;

    // Use dynamic tile id
    using ordered_tile_id_type = detail::ordered_block_id<unsigned long long>;

    // Kernel launch config
    using config = wrapped_adjacent_find_config<Config, input_type>;

    // Calculate required temporary storage
    ordered_tile_id_type::id_type* ordered_tile_id_storage;
    index_type*                    reduce_output = nullptr;

    hipError_t result = detail::temp_storage::partition(
        temporary_storage,
        storage_size,
        detail::temp_storage::make_linear_partition(
            detail::temp_storage::make_partition(&ordered_tile_id_storage,
                                                 ordered_tile_id_type::get_temp_storage_layout()),
            detail::temp_storage::ptr_aligned_array(&reduce_output, sizeof(*reduce_output))));

    if(result != hipSuccess || temporary_storage == nullptr)
    {
        return result;
    }

    RETURN_ON_ERROR(hipMemcpyAsync(reduce_output,
                                   &size,
                                   sizeof(*reduce_output),
                                   hipMemcpyHostToDevice,
                                   stream));

    if(size > 1)
    {
        auto ordered_tile_id = ordered_tile_id_type::create(ordered_tile_id_storage);
        adjacent_find::init_ordered_tile_id<<<1, 1, 0, stream>>>(ordered_tile_id);

        // Wrap adjacent input in zip iterator with idx values
        auto iota = ::rocprim::make_counting_iterator<index_type>(0);
        auto wrapped_input
            = ::rocprim::make_zip_iterator(::rocprim::make_tuple(input, input + 1, iota));

        // Transform input
        auto wrapped_equal_op = [op](const wrapped_input_type& a) -> transformed_input_type
        {
            return ::rocprim::make_tuple(
                -op_result_type(op(::rocprim::get<0>(a), ::rocprim::get<1>(a))),
                ::rocprim::get<2>(a));
        };
        auto transformed_input
            = ::rocprim::make_transform_iterator(wrapped_input, wrapped_equal_op);

        auto adjacent_find_block_reduce_kernel
            = adjacent_find::block_reduce_kernel<config,
                                                 decltype(transformed_input),
                                                 decltype(reduce_output),
                                                 reduce_op_type,
                                                 decltype(ordered_tile_id)>;
        target_arch target_arch;
        RETURN_ON_ERROR(host_target_arch(stream, target_arch));
        const adjacent_find_config_params params     = dispatch_target_arch<config>(target_arch);
        const unsigned int                block_size = params.kernel_config.block_size;
        const unsigned int                items_per_thread = params.kernel_config.items_per_thread;
        const unsigned int                items_per_block  = block_size * items_per_thread;
        const unsigned int grid_size        = (size + items_per_block - 1) / items_per_block;
        const unsigned int shared_mem_bytes = 0; /*no dynamic shared mem*/

        // Get grid size for maximum occupancy, as we may not be able to schedule all the blocks
        // at the same time
        int min_grid_size      = 0;
        int optimal_block_size = 0;
        RETURN_ON_ERROR(hipOccupancyMaxPotentialBlockSize(&min_grid_size,
                                                          &optimal_block_size,
                                                          adjacent_find_block_reduce_kernel,
                                                          shared_mem_bytes,
                                                          int(block_size)));
        min_grid_size = std::min(static_cast<unsigned int>(min_grid_size), grid_size);

        // Launch adjacent_find::block_reduce_kernel
        std::chrono::time_point<std::chrono::high_resolution_clock> start;
        if(debug_synchronous)
        {
            start = std::chrono::high_resolution_clock::now();
        }

        // Launch adjacent_find::block_reduce_kernel
        adjacent_find_block_reduce_kernel<<<min_grid_size, block_size, shared_mem_bytes, stream>>>(
            transformed_input,
            reduce_output,
            std::size_t{size - 1},
            reduce_op_type{},
            ordered_tile_id);
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR(
            "rocprim::detail::adjacent_find::block_reduce_kernel",
            size,
            start);
    }

    RETURN_ON_ERROR(hipMemcpyAsync(output,
                                   reduce_output,
                                   sizeof(*reduce_output),
                                   hipMemcpyDeviceToDevice,
                                   stream));

    return hipSuccess;
}

} // namespace detail

template<typename Config = default_config,
         typename InputIterator,
         typename OutputIterator,
         typename BinaryPred
         = ::rocprim::equal_to<typename std::iterator_traits<InputIterator>::value_type>>
ROCPRIM_INLINE
hipError_t adjacent_find(void* const       temporary_storage,
                         std::size_t&      storage_size,
                         InputIterator     input,
                         OutputIterator    output,
                         const std::size_t size,
                         BinaryPred        op                = BinaryPred{},
                         const hipStream_t stream            = 0,
                         const bool        debug_synchronous = false)
{
    return detail::adjacent_find_impl<Config>(temporary_storage,
                                              storage_size,
                                              input,
                                              output,
                                              size,
                                              op,
                                              stream,
                                              debug_synchronous);
}

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_ADJACENT_FIND_HPP_