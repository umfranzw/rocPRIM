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

namespace
{
template<typename First, typename... Types>
struct type_arr
{
    using type = First;
    using next = type_arr<Types...>;
};
template<typename First>
struct type_arr<First>
{
    using type = First;
};
template<typename T, typename = void>
constexpr bool is_type_arr_end = true;
template<typename T>
constexpr bool is_type_arr_end<T, std::__void_t<typename T::next>> = false;

} // namespace

template<class InputType, class OutputType = size_t>
class benchmark_search_n
{
public:
    const managed_seed     seed;
    const hipStream_t      stream;
    const size_t           size_byte;
    const size_t           count_byte;
    InputType              value;
    std::vector<InputType> input;

private:
    size_t       size;
    size_t       count;
    const size_t warmup_size       = 10;
    const size_t batch_size        = 10;
    size_t       temp_storage_size = sizeof(size_t);

    hipEvent_t start;
    hipEvent_t stop;

    void*       d_temp_storage = nullptr;
    InputType*  d_input;
    OutputType* d_output;
    InputType*  d_value;

    void create() noexcept
    {
        HIP_CHECK(hipMalloc(&d_value, sizeof(InputType)));
        HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_size));
        HIP_CHECK(hipMalloc(&d_input, sizeof(InputType) * input.size()));
        HIP_CHECK(hipMalloc(&d_output, sizeof(OutputType)));
        HIP_CHECK(hipMemcpy(d_value, &value, sizeof(InputType), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_input,
                            input.data(),
                            sizeof(InputType) * input.size(),
                            hipMemcpyHostToDevice));

        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
    }

    void release() noexcept
    {
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        HIP_CHECK(hipFree(d_value));
        HIP_CHECK(hipFree(d_temp_storage));
        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
    }

    static void run(benchmark::State& state, benchmark_search_n const& _self)
    {
        auto& self = const_cast<benchmark_search_n&>(_self);
        self.create();
        auto launch_search_n = [&]()
        {
            ::rocprim::search_n(self.d_temp_storage,
                                self.temp_storage_size,
                                self.d_input,
                                self.d_output,
                                self.size,
                                self.count,
                                self.d_value,
                                rocprim::equal_to<InputType>{},
                                self.stream,
                                false);
        };

        // Warm-up
        for(size_t i = 0; i < self.warmup_size; i++)
        {
            launch_search_n();
        }
        HIP_CHECK(hipDeviceSynchronize());
        // Run
        for(auto _ : state)
        {
            // Record start event
            HIP_CHECK(hipEventRecord(self.start, self.stream));

            for(size_t i = 0; i < self.batch_size; i++)
            {
                launch_search_n();
            }

            // Record stop event and wait until it completes
            HIP_CHECK(hipEventRecord(self.stop, self.stream));
            HIP_CHECK(hipEventSynchronize(self.stop));

            float elapsed_mseconds;
            HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, self.start, self.stop));
            state.SetIterationTime(elapsed_mseconds / 1000);
        }

        // Destroy HIP events

        state.SetBytesProcessed(state.iterations() * self.batch_size * self.size
                                * sizeof(*(self.d_input)));
        state.SetItemsProcessed(state.iterations() * self.batch_size * self.size);

        self.release();
    }

public:
    benchmark_search_n(const managed_seed       _seed,
                       const hipStream_t        _stream,
                       const size_t             _size_byte,
                       const size_t             _count_byte,
                       InputType                _value,
                       std::vector<InputType>&& _input) noexcept
        : seed(_seed)
        , stream(_stream)
        , size_byte(_size_byte)
        , count_byte(_count_byte)
        , value(_value)
        , input(_input)
    {
        size  = size_byte / sizeof(InputType);
        count = _count_byte / sizeof(InputType);
    }

    benchmark::internal::Benchmark* bench_register() const noexcept
    {
        return benchmark::RegisterBenchmark(
            bench_naming::format_name("{lvl:device,algo:search_n,input_type:"
                                      + std::string(typeid(InputType).name())
                                      + ",count:" + std::to_string(count)
                                      + ",size:" + std::to_string(size) + ",cfg:default_config}")
                .c_str(),
            run,
            *this);
    }
};

template<class T>
inline void add_one_benchmark_search_n(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                                       const managed_seed                            _seed,
                                       const hipStream_t                             _stream,
                                       const size_t                                  _size_byte)
{
    auto           size       = _size_byte / sizeof(T);
    std::vector<T> random_vec = get_random_data<T>(size,
                                                   generate_limits<T>::min(),
                                                   generate_limits<T>::max(),
                                                   _seed.get_1());
    std::vector<T> start_from_middle_vec(size);
    auto           start_from_middle_count = size / 2;
    std::fill(start_from_middle_vec.begin(),
              start_from_middle_vec.begin() + (size - start_from_middle_count),
              0);
    std::fill(start_from_middle_vec.begin() + start_from_middle_count,
              start_from_middle_vec.end(),
              1);

    benchmark_search_n<T> bench_random(
        _seed,
        _stream,
        _size_byte,
        sizeof(T) * 2,
        get_random_value<T>(0, generate_limits<T>::max(), _seed.get_0()),
        std::move(random_vec));

    benchmark_search_n<T> bench_equal_sequence(_seed,
                                               _stream,
                                               _size_byte,
                                               sizeof(T) * 2,
                                               1,
                                               std::vector<T>(_size_byte / sizeof(T), 1));
    benchmark_search_n<T> start_from_begin(_seed,
                                           _stream,
                                           _size_byte,
                                           _size_byte / 2,
                                           1,
                                           std::vector<T>(_size_byte / sizeof(T), 1));
    benchmark_search_n<T> start_from_middle(_seed,
                                            _stream,
                                            _size_byte,
                                            start_from_middle_count * sizeof(T),
                                            1,
                                            std::move(start_from_middle_vec));

    std::vector<benchmark::internal::Benchmark*> bs = {bench_random.bench_register(),
                                                       bench_equal_sequence.bench_register(),
                                                       start_from_begin.bench_register(),
                                                       start_from_middle.bench_register()};

    benchmarks.insert(benchmarks.end(), bs.begin(), bs.end());
}

template<typename T, std::enable_if_t<!is_type_arr_end<T>, bool> = true>
inline void add_benchmark_search_n(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                                   const managed_seed                            _seed,
                                   const hipStream_t                             _stream,
                                   const size_t                                  _size_byte)
{
    add_one_benchmark_search_n<typename T::type>(benchmarks, _seed, _stream, _size_byte);
    add_benchmark_search_n<typename T::next>(benchmarks, _seed, _stream, _size_byte);
}
template<typename T, std::enable_if_t<is_type_arr_end<T>, bool> = true>
inline void add_benchmark_search_n(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                                   const managed_seed                            _seed,
                                   const hipStream_t                             _stream,
                                   const size_t                                  _size_byte)
{
    add_one_benchmark_search_n<typename T::type>(benchmarks, _seed, _stream, _size_byte);
}

typedef type_arr<custom_int2,
                 custom_longlong_double,
                 int8_t,
                 int16_t,
                 int32_t,
                 int64_t,
                 rocprim::half,
                 float,
                 double>
    benchmark_search_n_types;

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
    add_benchmark_search_n<benchmark_search_n_types>(benchmarks, seed, stream, size);

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
