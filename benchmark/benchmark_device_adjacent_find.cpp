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
#include <rocprim/device/device_adjacent_find.hpp>

// C++ Standard Library
#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef DEFAULT_N
const size_t DEFAULT_BYTES = size_t{2} << 30; // 2 GiB
#endif

template<class InputT, class OutputT = std::size_t>
void run_adjacent_find_benchmark(benchmark::State&   state,
                                 double              first_adj_pos,
                                 size_t              bytes,
                                 const managed_seed& seed,
                                 hipStream_t         stream)
{
    using input_type  = InputT;
    using output_type = OutputT;

    const size_t size        = bytes / sizeof(input_type);
    const size_t warmup_size = 5;
    const size_t batch_size  = 10;

    // Generate data ensuring there is no adjacent pair before first_adj_index
    std::vector<input_type> input(size);
    std::iota(input.begin(), input.end(), 0);

    // Insert first adjacent pair
    std::size_t first_adj_index = static_cast<std::size_t>(size * first_adj_pos);
    if(first_adj_index >= size - 1)
    {
        first_adj_index = size - 2;
    }

    input[first_adj_index] = input[first_adj_index + 1];

    input_type*  d_input;
    output_type* d_output;
    HIP_CHECK(hipMalloc(&d_input, size * sizeof(*d_input)));
    HIP_CHECK(hipMalloc(&d_output, sizeof(*d_output)));
    HIP_CHECK(
        hipMemcpy(d_input, input.data(), input.size() * sizeof(*d_input), hipMemcpyHostToDevice));

    std::size_t tmp_storage_size;
    void*       d_tmp_storage        = nullptr;
    auto        launch_adjacent_find = [&]()
    {
        HIP_CHECK(::rocprim::adjacent_find(d_tmp_storage,
                                           tmp_storage_size,
                                           d_input,
                                           d_output,
                                           size,
                                           rocprim::equal_to<input_type>{},
                                           stream,
                                           false));
    };

    // Get size of tmporary storage
    launch_adjacent_find();
    HIP_CHECK(hipMalloc(&d_tmp_storage, tmp_storage_size));

    // Warm-up
    for(size_t i = 0; i < warmup_size; i++)
    {
        launch_adjacent_find();
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
            launch_adjacent_find();
        }

        // Record stop event and wait until it completes
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        float elapsed_mseconds;
        HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
        state.SetIterationTime(elapsed_mseconds / 1000);
    }

    // Destroy HIP events
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    state.SetBytesProcessed(state.iterations() * batch_size * first_adj_index * sizeof(*d_input));
    state.SetItemsProcessed(state.iterations() * batch_size * first_adj_index);

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_tmp_storage));
}

#define CREATE_BENCHMARK(T, P)                                                         \
    benchmark::RegisterBenchmark(                                                      \
        bench_naming::format_name(                                                     \
            "{lvl:device,algo:adjacent_find,input_type:" #T ",size:"                   \
            + std::to_string(std::size_t{bytes / sizeof(T)}) + ",first_adjacent_pair:" \
            + std::to_string(static_cast<std::size_t>(bytes / sizeof(T) * P))          \
            + ",cfg:default_config}")                                                  \
            .c_str(),                                                                  \
        run_adjacent_find_benchmark<T>,                                                \
        P,                                                                             \
        bytes,                                                                         \
        seed,                                                                          \
        stream)

#define CREATE_ADJACENT_FIND_BENCHMARK(T)                                          \
    CREATE_BENCHMARK(T, 0.01), CREATE_BENCHMARK(T, 0.2), CREATE_BENCHMARK(T, 0.5), \
        CREATE_BENCHMARK(T, 0.9)

void add_adjacent_find_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                                  size_t                                        bytes,
                                  const managed_seed&                           seed,
                                  hipStream_t                                   stream)
{
    using custom_float2          = custom_type<float, float>;
    using custom_double2         = custom_type<double, double>;
    using custom_int2            = custom_type<int, int>;
    using custom_char_double     = custom_type<char, double>;
    using custom_longlong_double = custom_type<long long, double>;

    // Tuned types
    std::vector<benchmark::internal::Benchmark*> bs
        = {// Tuned types
           CREATE_ADJACENT_FIND_BENCHMARK(int8_t),
           CREATE_ADJACENT_FIND_BENCHMARK(int16_t),
           CREATE_ADJACENT_FIND_BENCHMARK(int32_t),
           CREATE_ADJACENT_FIND_BENCHMARK(int64_t),
           CREATE_ADJACENT_FIND_BENCHMARK(rocprim::half),
           CREATE_ADJACENT_FIND_BENCHMARK(float),
           CREATE_ADJACENT_FIND_BENCHMARK(double),
           // Custom types
           CREATE_ADJACENT_FIND_BENCHMARK(custom_float2),
           CREATE_ADJACENT_FIND_BENCHMARK(custom_double2),
           CREATE_ADJACENT_FIND_BENCHMARK(custom_int2),
           CREATE_ADJACENT_FIND_BENCHMARK(custom_char_double),
           CREATE_ADJACENT_FIND_BENCHMARK(custom_longlong_double)};
}

int main(int argc, char* argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", DEFAULT_BYTES, "number of input bytes");
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
    add_adjacent_find_benchmarks(benchmarks, size, seed, stream);

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
