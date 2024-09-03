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

template<class Config, class InputIterator1, class InputIterator2, class BinaryFunction>
ROCPRIM_KERNEL
__launch_bounds__(device_params<Config>().kernel_config.block_size)
void search_kernel(InputIterator1 input,
                   InputIterator2 keys,
                   size_t*        output,
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

    const size_t offset       = flat_id * items_per_thread + flat_block_id * items_per_block;
    bool         find_pattern = false;

    // Check if it can have fit a key and a key has not yet be found with a lower index.
    if(offset + keys_size > size || offset > atomic_load(output))
    {
        return;
    }

    size_t index = 0;
    for(size_t id = offset; id < offset + items_per_thread; id++)
    {
        size_t i          = 0;
        size_t current_id = id;
        for(; i < keys_size - 1 && current_id < size; i++, current_id++)
        {
            if(!compare_function(input[current_id], keys[i]))
            {
                break;
            }
        }

        // If the i is the last value for the key and the compare is also true,
        // the pattern is found.
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

template<class Config, class InputIterator1, class InputIterator2, class BinaryFunction>
ROCPRIM_KERNEL
__launch_bounds__(device_params<Config>().kernel_config.block_size)
void search_kernel_shared(InputIterator1 input,
                          InputIterator2 keys,
                          size_t*        output,
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
    constexpr unsigned int max_shared_key   = params.max_shared_key_bytes / sizeof(key_type);

    const unsigned int flat_id       = rocprim::detail::block_thread_id<0>();
    const unsigned int flat_block_id = rocprim::detail::block_id<0>();

    const size_t block_offset = flat_block_id * items_per_block;
    const size_t offset       = flat_id * items_per_thread;
    bool find_pattern = false;

    ROCPRIM_SHARED_MEMORY uninitialized_array<key_type, max_shared_key> local_keys_;
    ROCPRIM_SHARED_MEMORY uninitialized_array<value_type, items_per_block> local_input_;

    // Check if a key was already found in a place before this block
    if(block_offset > atomic_load(output))
    {
        return;
    }

    // Load in key in shared memory
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

    // Load in all the input values that are guaranteed to be loaded.
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

    // Check if it can have fit a key and a key has not yet be found with a lower index.
    if(offset + block_offset + keys_size > size || offset > atomic_load(output))
    {
        return;
    }

    size_t       index      = 0;
    const size_t check      = size - block_offset;
    const size_t check_both = rocprim::min(check, size_t(items_per_block));
    for(size_t id = offset; id < offset + items_per_thread; id++)
    {
        size_t i          = 0;
        size_t current_id = id;
        // Values till the items_per_block are in shared_memory
        for(; i < keys_size - 1 && current_id < check_both; i++, current_id++)
        {
            if(!compare_function(local_input[current_id], local_keys[i]))
            {
                break;
            }
        }
        // Compare values that are not in the shared memory
        for(; current_id >= items_per_block && i < keys_size - 1 && current_id < check;
            i++, current_id++)
        {
            if(!compare_function(input[current_id + block_offset], local_keys[i]))
            {
                break;
            }
        }

        // If the i is the last value for the key and the compare is also true,
        // the pattern is found.
        if(current_id + block_offset < size && i == (keys_size - 1)
           && compare_function(current_id < items_per_block ? local_input[current_id]
                                                            : input[current_id + block_offset],
                               local_keys[i]))
        {
            index        = id + block_offset;
            find_pattern = true;
            // Want to find the first occurance, do not need to search further.
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

ROCPRIM_KERNEL
void set_output_kernel(size_t* output, size_t value)
{
    *output = static_cast<size_t>(value);
}

ROCPRIM_KERNEL
void reverse_index_kernel(size_t* output, size_t size, size_t keys_size)
{
    // Return the reverse index as long as the index is lower than the size.
    if(*output < size)
    {
        *output = static_cast<size_t>(size - keys_size) - *output;
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
    using input_type  = typename std::iterator_traits<InputIterator1>::value_type;
    using key_type    = typename std::iterator_traits<InputIterator2>::value_type;
    using output_type = typename std::iterator_traits<OutputIterator>::value_type;

    using config = wrapped_find_end_config<Config, input_type>;

    target_arch target_arch;
    RETURN_ON_ERROR(host_target_arch(stream, target_arch));

    const find_end_config_params params = dispatch_target_arch<config>(target_arch);

    const unsigned int block_size       = params.kernel_config.block_size;
    const unsigned int items_per_thread = params.kernel_config.items_per_thread;
    const unsigned int items_per_block  = block_size * items_per_thread;

    const unsigned int shared_key_mem_size_bytes = params.max_shared_key_bytes;
    const unsigned int key_size_bytes            = keys_size * sizeof(key_type);

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

    if(keys_size > size)
    {
        return hipErrorInvalidValue;
    }

    // The search kernel gives the first occurance, find_end wants the last.
    auto input_iterator = make_reverse_iterator(input + size);
    auto keys_iterator  = make_reverse_iterator(keys + keys_size);

    size_t* tmp_output = reinterpret_cast<size_t*>(temporary_storage);

    start_timer();
    set_output_kernel<<<1, 1, 0, stream>>>(tmp_output, size);
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("set_output_kernel", 1, start);

    if(size > 0 && keys_size > 0)
    {
        const unsigned int num_blocks = ceiling_div(size, items_per_block);
        if(key_size_bytes < shared_key_mem_size_bytes)
        {
            start_timer();
            search_kernel_shared<config><<<num_blocks, block_size, 0, stream>>>(input_iterator,
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

/// \brief Searches for the last occurrence of the sequence.
///
/// Searches the input for the last sequence where the the comparison
///   function returns true with the key sequence. Then outputs the index
///   of the start of the last sequence or if the sequence is not in the
///   input it returns the size.
///
/// \par Overview
/// * The contents of the inputs are not altered by the function.
/// * Returns the required size of `temporary_storage` in `storage_size`
/// if `temporary_storage` is a null pointer.
/// * Accepts custom compare_functions for find_end across the device.
/// * Streams in graph capture mode are supported
///
/// \tparam Config [optional] configuration of the primitive. It has to be `find_end_config`.
/// \tparam InputIterator1 [inferred] random-access iterator type of the input range. Must meet the
///   requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam InputIterator2 [inferred] random-access iterator type of the input range. Must meet the
///   requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam OutputIterator [inferred] random-access iterator type of the input range. Must meet the
///   requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam CompareFunction [inferred] Type of binary function that accepts a argument of the
///   type `InputIterator1` and of the type `InputIterator1` returns a value convertible to bool.
///   Default type is `rocprim::less<>.`
///
/// \param [in] temporary_storage pointer to a device-accessible temporary storage. When
///   a null pointer is passed, the required allocation size (in bytes) is written to
/// `storage_size` and function returns without performing the find_end.
/// \param [in,out] storage_size reference to a size (in bytes) of `temporary_storage`.
/// \param [in] input iterator to the input range.
/// \param [in] keys iterator to the key range.
/// \param [out] output iterator to the output range. The output is one element.
/// \param [in] size number of element in the input range.
/// \param [in] key_size number of element in the key range.
/// \param [in] compare_function binary operation function object that will be used for comparison.
///   The signature of the function should be equivalent to the following:
///   <tt>bool f(const T &a, const T &b);</tt>. The signature does not need to have
///   <tt>const &</tt>, but function object must not modify the objects passed to it.
///   The comparator must meet the C++ named requirement Compare.
///   The default value is `BinaryFunction()`.
/// \param [in] stream [optional] HIP stream object. Default is `0` (default stream).
/// \param [in] debug_synchronous [optional] If true, synchronization after every kernel
///   launch is forced in order to check for errors. Default value is `false`.
///
/// \returns `hipSuccess` (`0`) after successful rearrangement; otherwise a HIP runtime error of
///   type `hipError_t`.
///
/// \par Example
/// \parblock
/// In this example a device-level find_end is performed where input values are
///   represented by an array of unsigned integers and the key is also an array
///   of unsigned integers.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t size;           // e.g., 10
/// size_t key_size;       // e.g., 3
/// unsigned int * input;  // e.g., [ 6, 3, 5, 4, 1, 8, 2, 5, 4, 1 ]
/// unsigned int * key;    // e.g., [ 5, 4, 1 ]
/// unsigned int * output; // e.g., empty array of size 1
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::find_end(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, key, output, size, key_size
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform find_end
/// rocprim::find_end(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, key, output, size, key_size
/// );
/// // output:   [ 7 ]
/// \endcode
/// \endparblock
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
