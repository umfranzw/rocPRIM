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

#include "benchmark_utils.hpp"
#include "cmdparser.hpp"

// gbench
#include <benchmark/benchmark.h>

// HIP
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/detail/various.hpp>
#include <rocprim/device/device_search_n.hpp>

// C++ Standard Library
#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using custom_int2            = custom_type<int>;
using custom_double2         = custom_type<double>;
using custom_longlong_double = custom_type<long long, double>;

template<class InputType, class OutputType = size_t>
void run_search_n_benchmark(benchmark::State&   state,
                            size_t              count,
                            size_t              bytes,
                            const managed_seed& seed,
                            hipStream_t         stream)
{
    using input_type  = InputType;
    using output_type = OutputType;

    const size_t            size              = bytes / sizeof(input_type);
    const size_t            warmup_size       = 10;
    const size_t            batch_size        = 10;
    size_t                  temp_storage_size = sizeof(size_t);
    std::vector<input_type> h_input;
    input_type              h_value;
    void*                   d_temp_storage = nullptr;
    input_type*             d_input;
    output_type*            d_output;
    input_type*             d_value;

    // Generate data
    h_value = get_random_value<input_type>(0, generate_limits<input_type>::max(), seed.get_0());

    h_input = get_random_data<input_type>(size,
                                          generate_limits<input_type>::min(),
                                          generate_limits<input_type>::max(),
                                          seed.get_1());

    HIP_CHECK(hipMalloc(&d_value, sizeof(input_type)));
    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size));
    HIP_CHECK(hipMalloc(&d_input, sizeof(input_type) * h_input.size()));
    HIP_CHECK(hipMalloc(&d_output, sizeof(output_type)));
    HIP_CHECK(hipMemcpy(d_value, &h_value, sizeof(input_type), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_input,
                        h_input.data(),
                        sizeof(input_type) * h_input.size(),
                        hipMemcpyHostToDevice));

    auto launch_search_n = [&]()
    {
        ::rocprim::search_n(d_temp_storage,
                            temp_storage_size,
                            d_input,
                            d_output,
                            size,
                            count,
                            d_value,
                            rocprim::equal_to<input_type>{},
                            stream,
                            false);
    };

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        launch_search_n();
    }
    HIP_CHECK(hipDeviceSynchronize());

    // HIP events creation
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    // Run
    for(auto _ : state)
    {
        // Record start event
        HIP_CHECK(hipEventRecord(start, stream));

        for(size_t i = 0; i < batch_size; i++)
        {
            launch_search_n();
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

    state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(*d_input));
    state.SetItemsProcessed(state.iterations() * batch_size * size);

    HIP_CHECK(hipFree(d_value));
    HIP_CHECK(hipFree(d_temp_storage));
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    return;
}

#define CREATE_SEARCH_N_BENCHMARK(T)                                                   \
    benchmark::RegisterBenchmark(                                                      \
        bench_naming::format_name("{lvl:device,algo:search_n,input_type:" #T ",count:" \
                                  + std::to_string(count) + ",cfg:default_config}")    \
            .c_str(),                                                                  \
        run_search_n_benchmark<T>,                                                     \
        count,                                                                         \
        bytes,                                                                         \
        seed,                                                                          \
        stream)

void add_search_n_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                             size_t                                        count,
                             size_t                                        bytes,
                             const managed_seed&                           seed,
                             hipStream_t                                   stream)
{
    std::vector<benchmark::internal::Benchmark*> bs = {
        // Custom types
        CREATE_SEARCH_N_BENCHMARK(custom_int2),
        CREATE_SEARCH_N_BENCHMARK(custom_longlong_double),

        // Tuned types
        CREATE_SEARCH_N_BENCHMARK(int8_t),
        CREATE_SEARCH_N_BENCHMARK(int16_t),
        CREATE_SEARCH_N_BENCHMARK(int32_t),
        CREATE_SEARCH_N_BENCHMARK(int64_t),
        CREATE_SEARCH_N_BENCHMARK(rocprim::half),
        CREATE_SEARCH_N_BENCHMARK(float),
        CREATE_SEARCH_N_BENCHMARK(double),

    };

    benchmarks.insert(benchmarks.end(), bs.begin(), bs.end());

    return;
}

int main(int argc, char* argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", size_t{2} << 30, "number of input bytes");
    parser.set_optional<int>("trials", "trials", -1, "number of iterations");
    parser.set_optional<std::string>("name_format",
                                     "name_format",
                                     "human",
                                     "either: json,human,txt");
    parser.set_optional<std::string>("seed", "seed", "random", get_seed_message());
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const size_t size   = parser.get<size_t>("size");
    const int    trials = parser.get<int>("trials");
    bench_naming::set_format(parser.get<std::string>("name_format"));
    const std::string  seed_type = parser.get<std::string>("seed");
    const managed_seed seed(seed_type);

    // HIP
    hipStream_t stream = 0; // default

    // Benchmark info
    add_common_benchmark_info();
    benchmark::AddCustomContext("size", std::to_string(size));
    benchmark::AddCustomContext("seed", seed_type);

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks{};
    add_search_n_benchmarks(benchmarks, size * 0.1, size, seed, stream);
    add_search_n_benchmarks(benchmarks, size * 0.5, size, seed, stream);
    add_search_n_benchmarks(benchmarks, size * 0.9, size, seed, stream);

    // Use manual timing
    for(auto& b : benchmarks)
    {
        b->UseManualTime();
        b->Unit(benchmark::kMillisecond);
    }

    // Force number of iterations
    if(trials > 0)
    {
        for(auto& b : benchmarks)
        {
            b->Iterations(trials);
        }
    }

    // Run benchmarks
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}
