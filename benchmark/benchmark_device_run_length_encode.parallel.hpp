// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BENCHMARK_DEVICE_RUN_LENGTH_ENCODE_PARALLEL_HPP_
#define ROCPRIM_BENCHMARK_DEVICE_RUN_LENGTH_ENCODE_PARALLEL_HPP_

#include "benchmark_utils.hpp"

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/device/detail/device_config_helper.hpp>
#include <rocprim/device/device_run_length_encode.hpp>

#include <algorithm>
#include <string>
#include <type_traits>
#include <vector>
#ifdef BENCHMARK_CONFIG_TUNING
    #include <memory>
#endif

template<typename Config>
std::string run_length_encode_config_name()
{
    const rocprim::detail::reduce_by_key_config_params config = Config();
    return "{bs:" + std::to_string(config.kernel_config.block_size)
           + ",ipt:" + std::to_string(config.kernel_config.items_per_thread) + "}";
}

template<>
inline std::string run_length_encode_config_name<rocprim::default_config>()
{
    return "default_config";
}

template<typename T, size_t MaxLength, typename Config = rocprim::default_config>
struct device_run_length_encode_benchmark : public config_autotune_interface
{
    std::string name() const override
    {
        return bench_naming::format_name("{lvl:device,algo:run_length_encode,key_type:"
                                         + std::string(Traits<T>::name())
                                         + ",keys_max_length:" + std::to_string(MaxLength)
                                         + ",cfg:" + run_length_encode_config_name<Config>() + "}");
    }
    void run(benchmark::State&   state,
             size_t              bytes,
             const managed_seed& seed,
             hipStream_t         stream) const override
    {
        using key_type   = T;
        using count_type = unsigned int;

        const size_t size = bytes / sizeof(T);

        // Generate data
        std::vector<key_type> input(size);

        unsigned int        runs_count   = 0;
        const auto          random_range = limit_random_range<size_t>(1, MaxLength);
        std::vector<size_t> key_counts   = get_random_data<size_t>(100000,
                                                                 random_range.first,
                                                                 random_range.second,
                                                                 seed.get_0());
        size_t              offset       = 0;
        while(offset < size)
        {
            const size_t key_count = key_counts[runs_count % key_counts.size()];
            const size_t end       = std::min(size, offset + key_count);
            for(size_t i = offset; i < end; ++i)
            {
                input[i] = runs_count;
            }

            ++runs_count;
            offset += key_count;
        }

        key_type* d_input;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_input), size * sizeof(key_type)));
        HIP_CHECK(hipMemcpy(d_input, input.data(), size * sizeof(key_type), hipMemcpyHostToDevice));

        key_type*   d_unique_output;
        count_type* d_counts_output;
        count_type* d_runs_count_output;
        HIP_CHECK(
            hipMalloc(reinterpret_cast<void**>(&d_unique_output), runs_count * sizeof(key_type)));
        HIP_CHECK(
            hipMalloc(reinterpret_cast<void**>(&d_counts_output), runs_count * sizeof(count_type)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_runs_count_output), sizeof(count_type)));

        void*  d_temporary_storage     = nullptr;
        size_t temporary_storage_bytes = 0;

        HIP_CHECK(rocprim::run_length_encode<Config>(nullptr,
                                                     temporary_storage_bytes,
                                                     d_input,
                                                     size,
                                                     d_unique_output,
                                                     d_counts_output,
                                                     d_runs_count_output,
                                                     stream,
                                                     false));

        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        // Warm-up
        for(size_t i = 0; i < 10; ++i)
        {
            HIP_CHECK(rocprim::run_length_encode<Config>(d_temporary_storage,
                                                         temporary_storage_bytes,
                                                         d_input,
                                                         size,
                                                         d_unique_output,
                                                         d_counts_output,
                                                         d_runs_count_output,
                                                         stream,
                                                         false));
        }
        HIP_CHECK(hipDeviceSynchronize());

        // HIP events creation
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        const unsigned int batch_size = 10;
        for(auto _ : state)
        {
            // Record start event
            HIP_CHECK(hipEventRecord(start, stream));

            for(size_t i = 0; i < batch_size; ++i)
            {
                HIP_CHECK(rocprim::run_length_encode<Config>(d_temporary_storage,
                                                             temporary_storage_bytes,
                                                             d_input,
                                                             size,
                                                             d_unique_output,
                                                             d_counts_output,
                                                             d_runs_count_output,
                                                             stream,
                                                             false));
            }

            // Record stop event and wait until it completes
            HIP_CHECK(hipEventRecord(stop, stream));
            HIP_CHECK(hipEventSynchronize(stop));

            float elapsed_mseconds;
            HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
            state.SetIterationTime(elapsed_mseconds / 1000);
        }

        // Destroy HIP events
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));

        state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(key_type));
        state.SetItemsProcessed(state.iterations() * batch_size * size);

        HIP_CHECK(hipFree(d_temporary_storage));
        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_unique_output));
        HIP_CHECK(hipFree(d_counts_output));
        HIP_CHECK(hipFree(d_runs_count_output));
    }
};

#ifdef BENCHMARK_CONFIG_TUNING

template<typename T, unsigned int BlockSize>
struct device_run_length_encode_benchmark_generator
{
    template<unsigned int ItemsPerThread>
    struct create_ipt
    {
        void operator()(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
        {
            using config
                = rocprim::reduce_by_key_config<BlockSize,
                                                ItemsPerThread,
                                                rocprim::block_load_method::block_load_transpose,
                                                rocprim::block_load_method::block_load_transpose,
                                                rocprim::block_scan_algorithm::using_warp_scan>;

            storage.emplace_back(
                std::make_unique<device_run_length_encode_benchmark<T, 10, config>>());
            storage.emplace_back(
                std::make_unique<device_run_length_encode_benchmark<T, 1000, config>>());
        }
    };

    static void create(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
    {
        static_for_each<make_index_range<unsigned int, 4u, 15u>, create_ipt>(storage);
    }
};

#endif // BENCHMARK_CONFIG_TUNING

#endif // ROCPRIM_BENCHMARK_DEVICE_RUN_LENGTH_ENCODE_PARALLEL_HPP_
