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

#include "../common_test_header.hpp"

#include "test_utils_custom_test_types.hpp"
#include "test_utils_device_ptr.hpp"
#include "test_utils_types.hpp"

#include <rocprim/device/device_search_n.hpp>

#include <time.h>

template<class T>
using limit_type = test_utils::numeric_limits<T>;

template<class InputIterator,
         class OutputIterator     = size_t,
         class BinaryPredicate    = rocprim::equal_to<InputIterator>,
         class Config             = rocprim::default_config,
         bool UseGraphs           = false,
         bool UseIndirectIterator = false>
struct DeviceSearchNParams
{
    using input_type                            = InputIterator;
    using output_type                           = OutputIterator;
    using op_type                               = BinaryPredicate;
    using config                                = Config;
    static constexpr bool use_graphs            = UseGraphs;
    static constexpr bool use_indirect_iterator = UseIndirectIterator;
};

template<class Params>
class RocprimDeviceSearchNTests : public ::testing::Test
{
public:
    using input_type                            = typename Params::input_type;
    using output_type                           = typename Params::output_type;
    using op_type                               = typename Params::op_type;
    using config                                = typename Params::config;
    static constexpr bool use_graphs            = Params::use_graphs;
    static constexpr bool use_indirect_iterator = Params::use_indirect_iterator;
    static constexpr bool debug_synchronous     = false;
};

// Custom types
using custom_int2        = test_utils::custom_test_type<int>;
using custom_double2     = test_utils::custom_test_type<double>;
using custom_int64_array = test_utils::custom_test_array_type<std::int64_t, 4>;

// Custom configs
using custom_config_0 = rocprim::search_n_config<256, 4>;

using RocprimDeviceSearchNTestsParams = ::testing::Types<
    // Tests with default configuration
    DeviceSearchNParams<int8_t>,
    DeviceSearchNParams<int>,
    DeviceSearchNParams<rocprim::half>,
    DeviceSearchNParams<rocprim::bfloat16>,
    DeviceSearchNParams<float>,
    DeviceSearchNParams<double>,
    // Tests for custom types
    DeviceSearchNParams<custom_int2>,
    DeviceSearchNParams<custom_double2>,
    DeviceSearchNParams<custom_int64_array>,
    // Tests for supported config structs
    DeviceSearchNParams<rocprim::bfloat16,
                        size_t,
                        rocprim::equal_to<rocprim::bfloat16>,
                        custom_config_0>,
    // Tests for hipGraph support
    DeviceSearchNParams<unsigned int,
                        size_t,
                        rocprim::equal_to<unsigned int>,
                        rocprim::default_config,
                        true>,
    // Tests for when output's value_type is void
    DeviceSearchNParams<int, size_t, rocprim::equal_to<int>, rocprim::default_config, false, true>>;

TYPED_TEST_SUITE(RocprimDeviceSearchNTests, RocprimDeviceSearchNTestsParams);

// #define SINGLE_TEST
#ifndef SINGLE_TEST

TYPED_TEST(RocprimDeviceSearchNTests, RandomTest)
{
    int device_id = test_common_utils::obtain_device_from_ctest();

    HIP_CHECK(hipSetDevice(device_id));

    using input_type  = typename TestFixture::input_type;
    using output_type = typename TestFixture::output_type;
    using op_type     = typename TestFixture::op_type;
    using config      = typename TestFixture::config;

    constexpr bool use_indirect_iterator = TestFixture::use_indirect_iterator;
    constexpr bool debug_synchronous     = TestFixture::debug_synchronous;
    op_type        op{};

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];

        for(const auto size : test_utils::get_sizes(seed_value))
        {

            hipStream_t    stream = 0; // default
            size_t         count  = test_utils::get_random_value<size_t>(0, size, ++seed_value);
            hipGraph_t     graph;
            hipGraphExec_t graph_instance;
            size_t         temp_storage_size = sizeof(size_t);
            input_type     h_value
                = test_utils::get_random_value<input_type>(0,
                                                           limit_type<input_type>::max(),
                                                           ++seed_value);
            std::vector<input_type> h_input
                = test_utils::get_random_data<input_type>(size,
                                                          0,
                                                          limit_type<input_type>::max(),
                                                          ++seed_value);
            output_type                         h_output;
            test_utils::device_ptr<input_type>  d_input(h_input);
            test_utils::device_ptr<input_type>  d_value(&h_value, 1);
            test_utils::device_ptr<output_type> d_output(1);
            test_utils::device_ptr<void>        d_temp_storage(temp_storage_size);

            SCOPED_TRACE(testing::Message() << "with size = " << h_input.size());
            SCOPED_TRACE(testing::Message() << "with count = " << count);
            SCOPED_TRACE(testing::Message() << "with value = " << h_value);

            if ROCPRIM_IF_CONSTEXPR(TestFixture::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }

            HIP_CHECK(rocprim::search_n<config>(d_temp_storage.get(),
                                                temp_storage_size,
                                                d_input.get(),
                                                d_output.get(),
                                                h_input.size(),
                                                count,
                                                d_value.get(),
                                                op,
                                                stream,
                                                debug_synchronous));

            if ROCPRIM_IF_CONSTEXPR(TestFixture::use_graphs)
            {
                HIP_CHECK(hipStreamSynchronize(stream));
            }

            const auto expected
                = std::search_n(h_input.cbegin(), h_input.cend(), count, h_value, op)
                  - h_input.cbegin();

            h_output = d_output.load()[0];

            ASSERT_EQ(h_output, expected);

            if ROCPRIM_IF_CONSTEXPR(TestFixture::use_graphs)
            {
                HIP_CHECK(hipStreamDestroy(stream));
            }

            (void)graph;
            (void)graph_instance;
            (void)use_indirect_iterator;
        }
    }
}

TYPED_TEST(RocprimDeviceSearchNTests, EqualSequence)
{
    int device_id = test_common_utils::obtain_device_from_ctest();

    HIP_CHECK(hipSetDevice(device_id));

    using input_type  = typename TestFixture::input_type;
    using output_type = typename TestFixture::output_type;
    using op_type     = typename TestFixture::op_type;
    using config      = typename TestFixture::config;

    constexpr bool use_indirect_iterator = TestFixture::use_indirect_iterator;
    constexpr bool debug_synchronous     = TestFixture::debug_synchronous;
    op_type        op{};

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];

        for(const auto size : test_utils::get_sizes(seed_value))
        {

            hipStream_t    stream = 0; // default
            size_t         count  = test_utils::get_random_value<size_t>(0, size, ++seed_value);
            hipGraph_t     graph;
            hipGraphExec_t graph_instance;
            size_t         temp_storage_size = sizeof(size_t);
            input_type     h_value
                = test_utils::get_random_value<input_type>(0,
                                                           limit_type<input_type>::max(),
                                                           ++seed_value);
            std::vector<input_type>             h_input(size, h_value);
            output_type                         h_output;
            test_utils::device_ptr<input_type>  d_input(h_input);
            test_utils::device_ptr<input_type>  d_value(&h_value, 1);
            test_utils::device_ptr<output_type> d_output(1);
            test_utils::device_ptr<void>        d_temp_storage(temp_storage_size);

            SCOPED_TRACE(testing::Message() << "with size = " << h_input.size());
            SCOPED_TRACE(testing::Message() << "with count = " << count);
            SCOPED_TRACE(testing::Message() << "with value = " << h_value);

            if ROCPRIM_IF_CONSTEXPR(TestFixture::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }

            HIP_CHECK(rocprim::search_n<config>(d_temp_storage.get(),
                                                temp_storage_size,
                                                d_input.get(),
                                                d_output.get(),
                                                h_input.size(),
                                                count,
                                                d_value.get(),
                                                op,
                                                stream,
                                                debug_synchronous));

            if ROCPRIM_IF_CONSTEXPR(TestFixture::use_graphs)
            {
                HIP_CHECK(hipStreamSynchronize(stream));
            }

            const auto expected
                = std::search_n(h_input.cbegin(), h_input.cend(), count, h_value, op)
                  - h_input.cbegin();

            h_output = d_output.load()[0];

            ASSERT_EQ(h_output, expected);

            if ROCPRIM_IF_CONSTEXPR(TestFixture::use_graphs)
            {
                HIP_CHECK(hipStreamDestroy(stream));
            }

            (void)graph;
            (void)graph_instance;
            (void)use_indirect_iterator;
        }
    }
}

TYPED_TEST(RocprimDeviceSearchNTests, MaxCount)
{
    int device_id = test_common_utils::obtain_device_from_ctest();

    HIP_CHECK(hipSetDevice(device_id));

    using input_type  = typename TestFixture::input_type;
    using output_type = typename TestFixture::output_type;
    using op_type     = typename TestFixture::op_type;
    using config      = typename TestFixture::config;

    constexpr bool use_indirect_iterator = TestFixture::use_indirect_iterator;
    constexpr bool debug_synchronous     = TestFixture::debug_synchronous;
    op_type        op{};

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];

        for(const auto size : test_utils::get_sizes(seed_value))
        {

            hipStream_t    stream = 0; // default
            size_t         count  = size;
            hipGraph_t     graph;
            hipGraphExec_t graph_instance;
            size_t         temp_storage_size = sizeof(size_t);
            input_type     h_value
                = test_utils::get_random_value<input_type>(0,
                                                           limit_type<input_type>::max(),
                                                           ++seed_value);
            std::vector<input_type>             h_input(size, h_value);
            output_type                         h_output;
            test_utils::device_ptr<input_type>  d_input(h_input);
            test_utils::device_ptr<input_type>  d_value(&h_value, 1);
            test_utils::device_ptr<output_type> d_output(1);
            test_utils::device_ptr<void>        d_temp_storage(temp_storage_size);

            SCOPED_TRACE(testing::Message() << "with size = " << h_input.size());
            SCOPED_TRACE(testing::Message() << "with count = " << count);
            SCOPED_TRACE(testing::Message() << "with value = " << h_value);

            if ROCPRIM_IF_CONSTEXPR(TestFixture::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }

            HIP_CHECK(rocprim::search_n<config>(d_temp_storage.get(),
                                                temp_storage_size,
                                                d_input.get(),
                                                d_output.get(),
                                                h_input.size(),
                                                count,
                                                d_value.get(),
                                                op,
                                                stream,
                                                debug_synchronous));

            if ROCPRIM_IF_CONSTEXPR(TestFixture::use_graphs)
            {
                HIP_CHECK(hipStreamSynchronize(stream));
            }

            const auto expected
                = std::search_n(h_input.cbegin(), h_input.cend(), count, h_value, op)
                  - h_input.cbegin();

            h_output = d_output.load()[0];

            ASSERT_EQ(h_output, expected);

            if ROCPRIM_IF_CONSTEXPR(TestFixture::use_graphs)
            {
                HIP_CHECK(hipStreamDestroy(stream));
            }

            (void)graph;
            (void)graph_instance;
            (void)use_indirect_iterator;
        }
    }
}

TYPED_TEST(RocprimDeviceSearchNTests, MinCount)
{
    int device_id = test_common_utils::obtain_device_from_ctest();

    HIP_CHECK(hipSetDevice(device_id));

    using input_type  = typename TestFixture::input_type;
    using output_type = typename TestFixture::output_type;
    using op_type     = typename TestFixture::op_type;
    using config      = typename TestFixture::config;

    constexpr bool use_indirect_iterator = TestFixture::use_indirect_iterator;
    constexpr bool debug_synchronous     = TestFixture::debug_synchronous;
    op_type        op{};

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];

        for(const auto size : test_utils::get_sizes(seed_value))
        {

            hipStream_t    stream = 0; // default
            size_t         count  = 0;
            hipGraph_t     graph;
            hipGraphExec_t graph_instance;
            size_t         temp_storage_size = sizeof(size_t);
            input_type     h_value
                = test_utils::get_random_value<input_type>(0,
                                                           limit_type<input_type>::max(),
                                                           ++seed_value);
            std::vector<input_type>             h_input(size, h_value);
            output_type                         h_output;
            test_utils::device_ptr<input_type>  d_input(h_input);
            test_utils::device_ptr<input_type>  d_value(&h_value, 1);
            test_utils::device_ptr<output_type> d_output(1);
            test_utils::device_ptr<void>        d_temp_storage(temp_storage_size);

            SCOPED_TRACE(testing::Message() << "with size = " << h_input.size());
            SCOPED_TRACE(testing::Message() << "with count = " << count);
            SCOPED_TRACE(testing::Message() << "with value = " << h_value);

            if ROCPRIM_IF_CONSTEXPR(TestFixture::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }

            HIP_CHECK(rocprim::search_n<config>(d_temp_storage.get(),
                                                temp_storage_size,
                                                d_input.get(),
                                                d_output.get(),
                                                h_input.size(),
                                                count,
                                                d_value.get(),
                                                op,
                                                stream,
                                                debug_synchronous));

            if ROCPRIM_IF_CONSTEXPR(TestFixture::use_graphs)
            {
                HIP_CHECK(hipStreamSynchronize(stream));
            }

            const auto expected
                = std::search_n(h_input.cbegin(), h_input.cend(), count, h_value, op)
                  - h_input.cbegin();

            h_output = d_output.load()[0];

            ASSERT_EQ(h_output, expected);

            if ROCPRIM_IF_CONSTEXPR(TestFixture::use_graphs)
            {
                HIP_CHECK(hipStreamDestroy(stream));
            }

            (void)graph;
            (void)graph_instance;
            (void)use_indirect_iterator;
        }
    }
}

TYPED_TEST(RocprimDeviceSearchNTests, StartFromBegin)
{
    int device_id = test_common_utils::obtain_device_from_ctest();

    HIP_CHECK(hipSetDevice(device_id));

    using input_type  = typename TestFixture::input_type;
    using output_type = typename TestFixture::output_type;
    using op_type     = typename TestFixture::op_type;
    using config      = typename TestFixture::config;

    constexpr bool use_indirect_iterator = TestFixture::use_indirect_iterator;
    constexpr bool debug_synchronous     = TestFixture::debug_synchronous;
    op_type        op{};

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];

        for(const auto size : test_utils::get_sizes(seed_value))
        {

            hipStream_t             stream = 0; // default
            size_t                  count  = size / 2;
            hipGraph_t              graph;
            hipGraphExec_t          graph_instance;
            size_t                  temp_storage_size = sizeof(size_t);
            input_type              h_value{1};
            std::vector<input_type> h_input(size);
            std::fill(h_input.begin(), h_input.begin() + (size - count), h_value);
            std::fill(h_input.begin() + count, h_input.end(), 0);
            output_type                         h_output;
            test_utils::device_ptr<input_type>  d_input(h_input);
            test_utils::device_ptr<input_type>  d_value(&h_value, 1);
            test_utils::device_ptr<output_type> d_output(1);
            test_utils::device_ptr<void>        d_temp_storage(temp_storage_size);

            SCOPED_TRACE(testing::Message() << "with size = " << h_input.size());
            SCOPED_TRACE(testing::Message() << "with count = " << count);
            SCOPED_TRACE(testing::Message() << "with value = " << h_value);

            if ROCPRIM_IF_CONSTEXPR(TestFixture::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }

            HIP_CHECK(rocprim::search_n<config>(d_temp_storage.get(),
                                                temp_storage_size,
                                                d_input.get(),
                                                d_output.get(),
                                                h_input.size(),
                                                count,
                                                d_value.get(),
                                                op,
                                                stream,
                                                debug_synchronous));

            if ROCPRIM_IF_CONSTEXPR(TestFixture::use_graphs)
            {
                HIP_CHECK(hipStreamSynchronize(stream));
            }

            const auto expected
                = std::search_n(h_input.cbegin(), h_input.cend(), count, h_value, op)
                  - h_input.cbegin();

            h_output = d_output.load()[0];

            ASSERT_EQ(h_output, expected);

            if ROCPRIM_IF_CONSTEXPR(TestFixture::use_graphs)
            {
                HIP_CHECK(hipStreamDestroy(stream));
            }

            (void)graph;
            (void)graph_instance;
            (void)use_indirect_iterator;
        }
    }
}

TYPED_TEST(RocprimDeviceSearchNTests, StartFromMiddle)
{
    int device_id = test_common_utils::obtain_device_from_ctest();

    HIP_CHECK(hipSetDevice(device_id));

    using input_type  = typename TestFixture::input_type;
    using output_type = typename TestFixture::output_type;
    using op_type     = typename TestFixture::op_type;
    using config      = typename TestFixture::config;

    constexpr bool use_indirect_iterator = TestFixture::use_indirect_iterator;
    constexpr bool debug_synchronous     = TestFixture::debug_synchronous;
    op_type        op{};

    for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];

        for(const auto size : test_utils::get_sizes(seed_value))
        {

            hipStream_t             stream = 0; // default
            size_t                  count  = size / 2;
            hipGraph_t              graph;
            hipGraphExec_t          graph_instance;
            size_t                  temp_storage_size = sizeof(size_t);
            input_type              h_value{1};
            std::vector<input_type> h_input(size);
            std::fill(h_input.begin(), h_input.begin() + (size - count), 0);
            std::fill(h_input.begin() + count, h_input.end(), h_value);
            output_type                         h_output;
            test_utils::device_ptr<input_type>  d_input(h_input);
            test_utils::device_ptr<input_type>  d_value(&h_value, 1);
            test_utils::device_ptr<output_type> d_output(1);
            test_utils::device_ptr<void>        d_temp_storage(temp_storage_size);

            SCOPED_TRACE(testing::Message() << "with size = " << h_input.size());
            SCOPED_TRACE(testing::Message() << "with count = " << count);
            SCOPED_TRACE(testing::Message() << "with value = " << h_value);

            if ROCPRIM_IF_CONSTEXPR(TestFixture::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }

            HIP_CHECK(rocprim::search_n<config>(d_temp_storage.get(),
                                                temp_storage_size,
                                                d_input.get(),
                                                d_output.get(),
                                                h_input.size(),
                                                count,
                                                d_value.get(),
                                                op,
                                                stream,
                                                debug_synchronous));

            if ROCPRIM_IF_CONSTEXPR(TestFixture::use_graphs)
            {
                HIP_CHECK(hipStreamSynchronize(stream));
            }

            const auto expected
                = std::search_n(h_input.cbegin(), h_input.cend(), count, h_value, op)
                  - h_input.cbegin();

            h_output = d_output.load()[0];

            ASSERT_EQ(h_output, expected);

            if ROCPRIM_IF_CONSTEXPR(TestFixture::use_graphs)
            {
                HIP_CHECK(hipStreamDestroy(stream));
            }

            (void)graph;
            (void)graph_instance;
            (void)use_indirect_iterator;
        }
    }
}
#else

// Single test
int main(int argc, char* argv[])
{

    size_t size      = 100000;
    size_t count     = 3538;
    size_t start_pos = 0;

    // for(size_t start_pos = 0; start_pos < 100; start_pos+=3)
    {

        int device_id = test_common_utils::obtain_device_from_ctest();

        HIP_CHECK(hipSetDevice(device_id));

        using input_type  = int;
        using output_type = size_t;
        using op_type     = rocprim::equal_to<input_type>;
        using config      = rocprim::default_config;

        constexpr bool use_indirect_iterator = false;
        constexpr bool debug_synchronous     = false;

        op_type     op{};
        hipStream_t stream = 0; // default

        hipGraph_t              graph;
        hipGraphExec_t          graph_instance;
        size_t                  temp_storage_size = sizeof(size_t);
        input_type              h_value{1};
        std::vector<input_type> h_input(size);
        if(start_pos + count <= size)
        {
            std::fill(h_input.begin(), h_input.begin() + start_pos, 0);
            std::fill(h_input.begin() + start_pos, h_input.begin() + start_pos + count, h_value);
            std::fill(h_input.begin() + start_pos + count, h_input.end(), 0);
        }
        else
        {
            std::fill(h_input.begin(), h_input.end(), 0);
        }

        output_type                         h_output;
        test_utils::device_ptr<input_type>  d_input(h_input);
        test_utils::device_ptr<input_type>  d_value(&h_value, 1);
        test_utils::device_ptr<output_type> d_output(1);
        test_utils::device_ptr<void>        d_temp_storage(temp_storage_size);

        HIP_CHECK(rocprim::search_n<config>(d_temp_storage.get(),
                                            temp_storage_size,
                                            d_input.get(),
                                            d_output.get(),
                                            h_input.size(),
                                            count + 1,
                                            d_value.get(),
                                            op,
                                            stream,
                                            debug_synchronous));

        const auto expected
            = std::search_n(h_input.cbegin(), h_input.cend(), count + 1, h_value, op)
              - h_input.cbegin();

        h_output = d_output.load()[0];

        if(h_output == expected)
        {
            printf("equal %td: exp:%td\n", h_output, expected);
            // return 0;
        }
        else
        {
            printf("!!! not equal %td: exp:%td\n", h_output, expected);
            // return -1;
        }
    }
}

#endif
