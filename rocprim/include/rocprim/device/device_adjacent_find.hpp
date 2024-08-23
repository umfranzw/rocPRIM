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

#include "device_reduce.hpp"
#include "device_transform.hpp"

#include "../functional.hpp"
#include "../iterator/counting_iterator.hpp"
#include "../iterator/transform_iterator.hpp"
#include "../iterator/zip_iterator.hpp"
#include "../types/tuple.hpp"

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

template<class InputT, class IdxT, class OpT, class OpResultT>
struct bounded_equal_op
{
    IdxT size;
    OpT  op;

    // If the third element of the tuple (index) is within [0, size - 2], apply op to the two
    // first elements of the tuple and return the results (i.e. return <op(tuple.first, tuple.second), idx>).
    // Else, return <0, idx> (so the last element of the range has a 0 as result of the op and we do not
    // access out-of-bounds memory).
    ROCPRIM_HOST_DEVICE
    inline constexpr ::rocprim::tuple<OpResultT, IdxT>
        operator()(const ::rocprim::tuple<InputT, InputT, IdxT>& a) const
    {
        const IdxT idx = ::rocprim::get<2>(a);
        return (idx < size - 1) ? ::rocprim::make_tuple(
                   -OpResultT(op(::rocprim::get<0>(a), ::rocprim::get<1>(a))),
                   idx) /*idx <= size - 2 */
                                : ::rocprim::make_tuple(OpResultT(0), idx); /*idx == size - 1 */
    }
};

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

template<class T, class IdxT>
struct select_adjacent_or_end_op
{
    IdxT size;

    ROCPRIM_DEVICE
    inline constexpr IdxT
        operator()(const ::rocprim::tuple<T, IdxT>& a) const
    {
        return (::rocprim::get<0>(a) != 0) ? ::rocprim::get<1>(a) : size;
    }
};

template<typename Config = default_config,
         typename InputIteratorType,
         typename OutputIteratorType,
         typename BinaryPred>
ROCPRIM_INLINE
hipError_t adjacent_find_impl(void* const        temporary_storage,
                              std::size_t&       storage_size,
                              InputIteratorType  input,
                              OutputIteratorType output,
                              const std::size_t  size,
                              BinaryPred         op,
                              const hipStream_t  stream,
                              const bool         debug_synchronous)
{
    // Data types
    using input_type             = typename std::iterator_traits<InputIteratorType>::value_type;
    using op_result_type         = int; // use signed type to store (-1)s instead of 1s
    using index_type             = std::size_t;
    using wrapped_input_type     = ::rocprim::tuple<input_type, input_type, index_type>;
    using transformed_input_type = ::rocprim::tuple<op_result_type, index_type>;

    // Operations types
    using reduce_op_type          = reduce_op<op_result_type, index_type>;
    using final_transform_op_type = select_adjacent_or_end_op<op_result_type, index_type>;

    // Wrap adjacent input in zip iterator with idx values
    auto iota = ::rocprim::make_counting_iterator<index_type>(0);
    auto wrapped_input
        = ::rocprim::make_zip_iterator(::rocprim::make_tuple(input, input + 1, iota));

    // Transform input
    auto wrapped_equal_op
        = [op](const wrapped_input_type& a) ROCPRIM_HOST_DEVICE -> transformed_input_type
    {
        return ::rocprim::make_tuple(-op_result_type(op(::rocprim::get<0>(a), ::rocprim::get<1>(a))),
                                     ::rocprim::get<2>(a));
    };
    auto transformed_input = ::rocprim::make_transform_iterator(wrapped_input, wrapped_equal_op);

    hipError_t              result;
    const reduce_op_type    reduce_op{};
    transformed_input_type* reduce_output = nullptr;

    // Calculate size of temporary storage for reduce operation
    std::size_t reduce_bytes;
    result = ::rocprim::reduce<Config>(nullptr,
                                       reduce_bytes,
                                       transformed_input,
                                       reduce_output,
                                       std::size_t{size - 1},
                                       reduce_op,
                                       stream,
                                       debug_synchronous);
    if(result != hipSuccess)
    {
        return result;
    }
    reduce_bytes = ::rocprim::detail::align_size(reduce_bytes);

    // Calculate size of reduction
    result = detail::temp_storage::partition(
        temporary_storage,
        storage_size,
        detail::temp_storage::make_linear_partition(
            detail::temp_storage::ptr_aligned_array(&reduce_output,
                                                    sizeof(transformed_input_type))));

    if(temporary_storage == nullptr)
    {
        storage_size += reduce_bytes;
        return result;
    }
    if(result != hipSuccess)
    {
        return result;
    }

    index_type* index_output = reinterpret_cast<index_type*>(temporary_storage);
    result
        = hipMemcpyAsync(index_output, &size, sizeof(*index_output), hipMemcpyHostToDevice, stream);
    if(result != hipSuccess)
    {
        return result;
    }

    if(size > 1)
    {
        // Launch reduction
        std::chrono::time_point<std::chrono::high_resolution_clock> start;
        if(debug_synchronous)
            start = std::chrono::high_resolution_clock::now();
        result = ::rocprim::reduce<Config>(temporary_storage,
                                           reduce_bytes,
                                           transformed_input,
                                           reduce_output,
                                           std::size_t{size - 1},
                                           reduce_op,
                                           stream,
                                           debug_synchronous);
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("rocprim::reduce", size, start)

        // Get final result. If an adjacent equal pair was found (reduce_output.first == 1) return the
        // distance to the first element of the pair. Else, return distance to the end.
        result = ::rocprim::transform<Config>(reduce_output,
                                              index_output,
                                              1,
                                              final_transform_op_type{size},
                                              stream,
                                              debug_synchronous);
    }

    result = ::rocprim::transform<Config>(index_output,
                                          output,
                                          1,
                                          ::rocprim::identity<void>(),
                                          stream,
                                          debug_synchronous);

    return result;
}

} // namespace detail

template<typename Config = default_config,
         typename InputIteratorType,
         typename OutputIteratorType,
         typename BinaryPred
         = ::rocprim::equal_to<typename std::iterator_traits<InputIteratorType>::value_type>>
ROCPRIM_INLINE
hipError_t adjacent_find(void* const        temporary_storage,
                         std::size_t&       storage_size,
                         InputIteratorType  input,
                         OutputIteratorType output,
                         const std::size_t  size,
                         BinaryPred         op                = BinaryPred{},
                         const hipStream_t  stream            = 0,
                         const bool         debug_synchronous = false)
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