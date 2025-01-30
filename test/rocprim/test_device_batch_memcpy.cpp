// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "common_test_header.hpp"
#include "indirect_iterator.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_custom_test_types.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_types.hpp"

#include "rocprim/detail/various.hpp"
#include "rocprim/device/device_copy.hpp"
#include "rocprim/device/device_memcpy.hpp"
#include "rocprim/intrinsics/thread.hpp"
#include "rocprim/iterator.hpp"

#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <random>
#include <type_traits>

#include <cstddef>
#include <cstring>

#include <stdint.h>

template<class ContainerMemCpy,
         class ContainerCopy,
         class ptr,
         class OffsetContainer,
         class SizesContainer,
         class byte_offset_type>
void check_result(ContainerMemCpy& h_input_for_memcpy,
                  ContainerCopy& /*h_input_for_copy*/,
                  ptr              d_output,
                  byte_offset_type total_num_bytes,
                  byte_offset_type /*total_num_elements*/,
                  int              num_buffers,
                  OffsetContainer& src_offsets,
                  OffsetContainer& dst_offsets,
                  SizesContainer&  h_buffer_num_bytes)
{
    using value_type                    = typename ContainerCopy::value_type;
    std::vector<unsigned char> h_output = std::vector<unsigned char>(total_num_bytes);

    // ** Hang happens here: **
    HIP_CHECK(hipMemcpy(h_output.data(), d_output, total_num_bytes, hipMemcpyDeviceToHost));

    for(int i = 0; i < num_buffers; ++i)
    {
        ASSERT_EQ(std::memcmp(h_input_for_memcpy.data() + src_offsets[i] * sizeof(value_type),
                              h_output.data() + dst_offsets[i] * sizeof(value_type),
                              h_buffer_num_bytes[i]),
                  0)
            << "with index = " << i;
    }
}

TEST(RocprimDeviceBatchMemcpyTests, SizeAndTypeVariation)
{
    using value_type         = uint8_t;
    using buffer_size_type   = unsigned int;
    using buffer_offset_type = unsigned int;
    using byte_offset_type   = size_t;

    constexpr int  num_buffers           = 1024 * 1024;
    constexpr int  max_size              = 256;
    constexpr bool shuffled              = false;
    constexpr bool use_indirect_iterator = false;

    constexpr int wlev_min_size = rocprim::batch_memcpy_config<>::wlev_size_threshold;
    constexpr int blev_min_size = rocprim::batch_memcpy_config<>::blev_size_threshold;

    constexpr int wlev_min_elems = rocprim::detail::ceiling_div(wlev_min_size, sizeof(value_type));
    constexpr int blev_min_elems = rocprim::detail::ceiling_div(blev_min_size, sizeof(value_type));
    constexpr int max_elems      = max_size / sizeof(value_type);

    constexpr int enabled_size_categories
        = (blev_min_elems <= max_elems) + (wlev_min_elems <= max_elems) + 1;

    constexpr int num_blev
        = blev_min_elems <= max_elems ? num_buffers / enabled_size_categories : 0;
    constexpr int num_wlev
        = wlev_min_elems <= max_elems ? num_buffers / enabled_size_categories : 0;
    constexpr int num_tlev = num_buffers - num_blev - num_wlev;

    seed_type seed_value = 0;
    SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);
    std::mt19937_64 rng{seed_value};

    std::vector<buffer_size_type> h_buffer_num_elements(num_buffers);

    auto iter = h_buffer_num_elements.begin();

    for (unsigned int i = 0; i < num_tlev; i++)
    {
        *iter = (1 + i) % (wlev_min_elems - 1);
        iter++;
    }

    for (unsigned int i = 0; i < num_wlev; i++)
    {
        *iter = (wlev_min_elems + i) % (blev_min_elems - 1);
        iter++;
    }

    for (unsigned int i = 0; i < num_blev; i++)
    {
        *iter = (blev_min_elems + i) % (max_elems);
        iter++;
    }

    const byte_offset_type total_num_elements = std::accumulate(h_buffer_num_elements.begin(),
                                                                h_buffer_num_elements.end(),
                                                                byte_offset_type{0});

    // Shuffle the sizes so that size classes aren't clustered
    std::shuffle(h_buffer_num_elements.begin(), h_buffer_num_elements.end(), rng);

    // And the total byte size
    const byte_offset_type total_num_bytes = total_num_elements * sizeof(value_type);

    // Device pointers
    value_type*       d_input        = nullptr;
    value_type*       d_output       = nullptr;
    value_type**      d_buffer_srcs  = nullptr;
    value_type**      d_buffer_dsts  = nullptr;
    buffer_size_type* d_buffer_sizes = nullptr;

    // Calculate temporary storage
    size_t temp_storage_bytes = 0;
    HIP_CHECK(rocprim::batch_memcpy(nullptr,
                                    temp_storage_bytes,
                                    d_buffer_srcs,
                                    d_buffer_dsts,
                                    d_buffer_sizes,
                                    num_buffers,
                                    hipStreamDefault));

    void* d_temp_storage = nullptr;

    // Allocate memory.
    HIP_CHECK(hipMalloc(&d_input, total_num_bytes));
    HIP_CHECK(hipMalloc(&d_output, total_num_bytes));

    HIP_CHECK(hipMalloc(&d_buffer_srcs, num_buffers * sizeof(*d_buffer_srcs)));
    HIP_CHECK(hipMalloc(&d_buffer_dsts, num_buffers * sizeof(*d_buffer_dsts)));
    HIP_CHECK(hipMalloc(&d_buffer_sizes, num_buffers * sizeof(*d_buffer_sizes)));

    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));

    // Generate data.
    std::vector<unsigned char> h_input_for_memcpy;
    std::vector<value_type>    h_input_for_copy;

    const size_t num_ints = rocprim::detail::ceiling_div(total_num_bytes, sizeof(uint64_t));
    h_input_for_memcpy    = std::vector<unsigned char>(num_ints * sizeof(uint64_t));
    for (int i = 0; i < h_input_for_memcpy.size(); i++)
    {
        h_input_for_memcpy[i] = i % 256;
    }

    // Generate the source and shuffled destination offsets.
    std::vector<buffer_offset_type> src_offsets;
    std::vector<buffer_offset_type> dst_offsets;

    src_offsets = std::vector<buffer_offset_type>(num_buffers);
    dst_offsets = std::vector<buffer_offset_type>(num_buffers);

    // Consecutive offsets (no shuffling).
    // src/dst offsets first element is 0, so skip that!
    std::partial_sum(h_buffer_num_elements.begin(),
                     h_buffer_num_elements.end() - 1,
                     src_offsets.begin() + 1);
    std::partial_sum(h_buffer_num_elements.begin(),
                     h_buffer_num_elements.end() - 1,
                     dst_offsets.begin() + 1);

    // Get the byte size of each buffer
    std::vector<buffer_size_type> h_buffer_num_bytes(num_buffers);
    for(size_t i = 0; i < num_buffers; ++i)
    {
        h_buffer_num_bytes[i] = h_buffer_num_elements[i] * sizeof(value_type);
    }

    // Generate the source and destination pointers.
    std::vector<value_type*> h_buffer_srcs(num_buffers);
    std::vector<value_type*> h_buffer_dsts(num_buffers);

    for(int i = 0; i < num_buffers; ++i)
    {
        h_buffer_srcs[i] = d_input + src_offsets[i];
        h_buffer_dsts[i] = d_output + dst_offsets[i];
    }

    // Prepare the batch memcpy.
    HIP_CHECK(hipMemcpy(d_input,
                        h_input_for_memcpy.data(),
                        total_num_bytes,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_buffer_sizes,
                        h_buffer_num_bytes.data(),
                        h_buffer_num_bytes.size() * sizeof(*d_buffer_sizes),
                        hipMemcpyHostToDevice));

    HIP_CHECK(hipMemcpy(d_buffer_srcs,
                        h_buffer_srcs.data(),
                        h_buffer_srcs.size() * sizeof(*d_buffer_srcs),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_buffer_dsts,
                        h_buffer_dsts.data(),
                        h_buffer_dsts.size() * sizeof(*d_buffer_dsts),
                        hipMemcpyHostToDevice));

    const auto input_src_it
        = test_utils::wrap_in_indirect_iterator<use_indirect_iterator>(d_buffer_srcs);
    const auto output_src_it
        = test_utils::wrap_in_indirect_iterator<use_indirect_iterator>(d_buffer_dsts);

    HIP_CHECK(rocprim::batch_memcpy(d_temp_storage,
                                    temp_storage_bytes,
                                    input_src_it,
                                    output_src_it,
                                    d_buffer_sizes,
                                    num_buffers,
                                    hipStreamDefault));

    // Verify results.
    check_result(h_input_for_memcpy,
                           h_input_for_copy,
                           d_output,
                           total_num_bytes,
                           total_num_elements,
                           num_buffers,
                           src_offsets,
                           dst_offsets,
                           h_buffer_num_bytes);

    HIP_CHECK(hipFree(d_temp_storage));
    HIP_CHECK(hipFree(d_buffer_sizes));
    HIP_CHECK(hipFree(d_buffer_dsts));
    HIP_CHECK(hipFree(d_buffer_srcs));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_input));
}
