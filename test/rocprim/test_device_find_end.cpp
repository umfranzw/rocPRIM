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

// required test headers
#include "indirect_iterator.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_custom_float_type.hpp"
#include "test_utils_custom_test_types.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_types.hpp"

#include "../common_test_header.hpp"

#include "rocprim/device/device_find_end.hpp"

#include <cstddef>
#include <vector>

// Params for tests
template<class ValueType,
         class KeyType            = ValueType,
         class IndexType          = size_t,
         class CompareFunction    = rocprim::equal_to<KeyType>,
         class Config             = rocprim::default_config,
         bool UseGraphs           = false,
         bool UseIndirectIterator = false>
struct DeviceFindEndParams
{
    using value_type                            = ValueType;
    using key_type                              = KeyType;
    using index_type                            = IndexType;
    using compare_function                      = CompareFunction;
    using config                                = Config;
    static constexpr bool use_graphs            = UseGraphs;
    static constexpr bool use_indirect_iterator = UseIndirectIterator;
};

template<class Params>
class RocprimDeviceFindEndTests : public ::testing::Test
{
public:
    using value_type                            = typename Params::value_type;
    using key_type                              = typename Params::key_type;
    using index_type                            = typename Params::index_type;
    using compare_function                      = typename Params::compare_function;
    using config                                = typename Params::config;
    const bool            debug_synchronous     = false;
    static constexpr bool use_graphs            = Params::use_graphs;
    static constexpr bool use_indirect_iterator = Params::use_indirect_iterator;
};

using RocprimDeviceFindEndTestsParams = ::testing::Types<
    DeviceFindEndParams<unsigned short>,
    DeviceFindEndParams<signed char>,
    DeviceFindEndParams<int, int, unsigned int>,
    DeviceFindEndParams<int, int, int>,
    DeviceFindEndParams<test_utils::custom_test_type<int>>,
    DeviceFindEndParams<unsigned long>,
    DeviceFindEndParams<long long>,
    DeviceFindEndParams<float>,
    DeviceFindEndParams<int8_t>,
    DeviceFindEndParams<uint8_t>,
    DeviceFindEndParams<rocprim::half, rocprim::half, size_t, rocprim::equal_to<rocprim::half>>,
    DeviceFindEndParams<rocprim::bfloat16,
                        rocprim::bfloat16,
                        size_t,
                        rocprim::equal_to<rocprim::bfloat16>>,
    DeviceFindEndParams<short>,
    DeviceFindEndParams<double>,
    DeviceFindEndParams<test_utils::custom_test_type<float>>,
    DeviceFindEndParams<test_utils::custom_float_type>,
    DeviceFindEndParams<test_utils::custom_test_array_type<int, 4>>,
    DeviceFindEndParams<int, int, size_t, rocprim::equal_to<int>, rocprim::default_config, true>,
    DeviceFindEndParams<int, int, size_t, rocprim::greater_equal<int>, rocprim::default_config>,
    DeviceFindEndParams<int, int, size_t, rocprim::greater<int>, rocprim::default_config>,
    DeviceFindEndParams<int,
                        int,
                        size_t,
                        rocprim::equal_to<int>,
                        rocprim::default_config,
                        false,
                        true>,
    DeviceFindEndParams<int,
                        int,
                        size_t,
                        rocprim::equal_to<int>,
                        rocprim::search_config<64, 16, 1024>,
                        false,
                        false>>;

TYPED_TEST_SUITE(RocprimDeviceFindEndTests, RocprimDeviceFindEndTestsParams);

TYPED_TEST(RocprimDeviceFindEndTests, FindEnd)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using value_type                     = typename TestFixture::value_type;
    using key_type                       = typename TestFixture::key_type;
    using index_type                     = typename TestFixture::index_type;
    using compare_function               = typename TestFixture::compare_function;
    using config                         = typename TestFixture::config;
    const bool     debug_synchronous     = TestFixture::debug_synchronous;
    constexpr bool use_indirect_iterator = TestFixture::use_indirect_iterator;

    std::vector<size_t> key_sizes = {0, 1, 10, 1000, 10000};

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            hipStream_t stream = 0; // default

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            for(size_t key_size : key_sizes)
            {
                SCOPED_TRACE(testing::Message() << "with key size = " << key_size);

                if(TestFixture::use_graphs)
                {
                    // Default stream does not support hipGraph stream capture, so create one
                    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
                }

                size_t pattern = 0;
                if(size > 0)
                {
                    pattern = test_utils::get_random_value<size_t>(0, size - 1, seed_value);
                }

                SCOPED_TRACE(testing::Message() << "with index = " << pattern);

                // Generate data
                std::vector<value_type> input;
                if(rocprim::is_floating_point<value_type>::value)
                {
                    input = test_utils::get_random_data<value_type>(size, -1000, 1000, seed_value);
                }
                else
                {
                    input = test_utils::get_random_data<value_type>(
                        size,
                        test_utils::numeric_limits<value_type>::min(),
                        test_utils::numeric_limits<value_type>::max(),
                        seed_value);
                }

                std::vector<key_type> keys(key_size);
                if(pattern + key_size < size)
                {
                    keys.assign(input.begin() + pattern, input.begin() + pattern + key_size);
                }
                else
                {
                    keys.assign(input.begin() + pattern, input.end());
                }

                value_type* d_input;
                key_type*   d_keys;
                index_type* d_output;
                HIP_CHECK(
                    test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(*d_input)));

                HIP_CHECK(
                    test_common_utils::hipMallocHelper(&d_keys, keys.size() * sizeof(*d_keys)));

                HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, sizeof(*d_output)));

                HIP_CHECK(hipMemcpy(d_input,
                                    input.data(),
                                    input.size() * sizeof(*d_input),
                                    hipMemcpyHostToDevice));

                HIP_CHECK(hipMemcpy(d_keys,
                                    keys.data(),
                                    keys.size() * sizeof(*d_keys),
                                    hipMemcpyHostToDevice));

                const auto input_it
                    = test_utils::wrap_in_indirect_iterator<use_indirect_iterator>(d_input);
                const auto keys_it
                    = test_utils::wrap_in_indirect_iterator<use_indirect_iterator>(d_keys);
                const auto output_keys
                    = test_utils::wrap_in_indirect_iterator<use_indirect_iterator>(d_output);

                // compare function
                compare_function compare_op;

                // temp storage
                size_t temp_storage_size_bytes;
                void*  d_temp_storage = nullptr;
                // Get size of d_temp_storage
                HIP_CHECK(rocprim::find_end<config>(nullptr,
                                                    temp_storage_size_bytes,
                                                    input_it,
                                                    keys_it,
                                                    output_keys,
                                                    input.size(),
                                                    keys.size(),
                                                    compare_op,
                                                    stream,
                                                    debug_synchronous));

                // temp_storage_size_bytes must be >0
                ASSERT_GT(temp_storage_size_bytes, 0);

                // allocate temporary storage
                HIP_CHECK(
                    test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));

                hipGraph_t graph;
                if(TestFixture::use_graphs)
                {
                    graph = test_utils::createGraphHelper(stream);
                }

                // Run
                HIP_CHECK(rocprim::find_end<config>(d_temp_storage,
                                                    temp_storage_size_bytes,
                                                    input_it,
                                                    keys_it,
                                                    output_keys,
                                                    input.size(),
                                                    keys.size(),
                                                    compare_op,
                                                    stream,
                                                    debug_synchronous));

                hipGraphExec_t graph_instance;
                if(TestFixture::use_graphs)
                {
                    graph_instance = test_utils::endCaptureGraphHelper(graph, stream, true, true);
                }

                HIP_CHECK(hipGetLastError());
                HIP_CHECK(hipDeviceSynchronize());

                index_type output;
                // Copy output to host
                HIP_CHECK(hipMemcpy(&output, d_output, sizeof(*d_output), hipMemcpyDeviceToHost));

                index_type expected = std::find_end(input.begin(),
                                                    input.end(),
                                                    keys.begin(),
                                                    keys.end(),
                                                    compare_op)
                                      - input.begin();

                ASSERT_EQ(output, expected);

                HIP_CHECK(hipFree(d_input));
                HIP_CHECK(hipFree(d_keys));
                HIP_CHECK(hipFree(d_output));
                HIP_CHECK(hipFree(d_temp_storage));

                if(TestFixture::use_graphs)
                {
                    test_utils::cleanupGraphHelper(graph, graph_instance);
                    HIP_CHECK(hipStreamDestroy(stream));
                }
            }
        }
    }
}

TYPED_TEST(RocprimDeviceFindEndTests, FindEndRepetition)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using value_type                     = typename TestFixture::value_type;
    using key_type                       = typename TestFixture::key_type;
    using index_type                     = typename TestFixture::index_type;
    using compare_function               = typename TestFixture::compare_function;
    using config                         = typename TestFixture::config;
    const bool     debug_synchronous     = TestFixture::debug_synchronous;
    constexpr bool use_indirect_iterator = TestFixture::use_indirect_iterator;

    size_t key_size = 10;

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(size_t size : test_utils::get_sizes(seed_value))
        {
            hipStream_t stream = 0; // default

            if(size < key_size)
            {
                continue;
            }

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            if(TestFixture::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }

            // Generate data
            std::vector<key_type> keys;
            if(rocprim::is_floating_point<value_type>::value)
            {
                keys = test_utils::get_random_data<key_type>(key_size, -1000, 1000, seed_value);
            }
            else
            {
                keys = test_utils::get_random_data<key_type>(
                    key_size,
                    test_utils::numeric_limits<key_type>::min(),
                    test_utils::numeric_limits<key_type>::max(),
                    seed_value);
            }

            std::vector<value_type> input(size);
            for(size_t i = 0; i < size / key_size; i++)
            {
                std::copy(keys.begin(), keys.end(), input.begin() + i * key_size);
            }

            value_type* d_input;
            key_type*   d_keys;
            index_type* d_output;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(*d_input)));

            HIP_CHECK(test_common_utils::hipMallocHelper(&d_keys, keys.size() * sizeof(*d_keys)));

            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, sizeof(*d_output)));

            HIP_CHECK(hipMemcpy(d_input,
                                input.data(),
                                input.size() * sizeof(*d_input),
                                hipMemcpyHostToDevice));

            HIP_CHECK(hipMemcpy(d_keys,
                                keys.data(),
                                keys.size() * sizeof(*d_keys),
                                hipMemcpyHostToDevice));

            const auto input_it
                = test_utils::wrap_in_indirect_iterator<use_indirect_iterator>(d_input);

            // compare function
            compare_function compare_op;

            // temp storage
            size_t temp_storage_size_bytes;
            void*  d_temp_storage = nullptr;
            // Get size of d_temp_storage
            HIP_CHECK(rocprim::find_end<config>(nullptr,
                                                temp_storage_size_bytes,
                                                input_it,
                                                d_keys,
                                                d_output,
                                                input.size(),
                                                keys.size(),
                                                compare_op,
                                                stream,
                                                debug_synchronous));

            // temp_storage_size_bytes must be >0
            ASSERT_GT(temp_storage_size_bytes, 0);

            // allocate temporary storage
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));

            hipGraph_t graph;
            if(TestFixture::use_graphs)
            {
                graph = test_utils::createGraphHelper(stream);
            }

            // Run
            HIP_CHECK(rocprim::find_end<config>(d_temp_storage,
                                                temp_storage_size_bytes,
                                                input_it,
                                                d_keys,
                                                d_output,
                                                input.size(),
                                                keys.size(),
                                                compare_op,
                                                stream,
                                                debug_synchronous));

            hipGraphExec_t graph_instance;
            if(TestFixture::use_graphs)
            {
                graph_instance = test_utils::endCaptureGraphHelper(graph, stream, true, true);
            }

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            index_type output;
            // Copy output to host
            HIP_CHECK(hipMemcpy(&output, d_output, sizeof(*d_output), hipMemcpyDeviceToHost));

            index_type expected
                = std::find_end(input.begin(), input.end(), keys.begin(), keys.end(), compare_op)
                  - input.begin();

            ASSERT_EQ(output, expected);

            HIP_CHECK(hipFree(d_input));
            HIP_CHECK(hipFree(d_keys));
            HIP_CHECK(hipFree(d_output));
            HIP_CHECK(hipFree(d_temp_storage));

            if(TestFixture::use_graphs)
            {
                test_utils::cleanupGraphHelper(graph, graph_instance);
                HIP_CHECK(hipStreamDestroy(stream));
            }
        }
    }
}
