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

#ifndef ROCPRIM_BENCHMARK_DEVICE_SEARCH_N_PARALLEL_HPP_
#define ROCPRIM_BENCHMARK_DEVICE_SEARCH_N_PARALLEL_HPP_

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
template<typename...>
using void_type = void;
template<typename T, typename = void>
constexpr bool is_type_arr_end = true;
template<typename T>
constexpr bool is_type_arr_end<T, void_type<typename T::next>> = false;

template<class Config, class InputType>
unsigned int search_n_get_item_per_block()
{
    using input_type     = InputType;
    using config         = Config;
    using wrapped_config = rocprim::detail::wrapped_search_n_config<config, input_type>;

    hipStream_t                  stream = 0; // default
    rocprim::detail::target_arch target_arch;
    HIP_CHECK(rocprim::detail::host_target_arch(stream, target_arch));
    const auto         params = rocprim::detail::dispatch_target_arch<wrapped_config>(target_arch);
    const unsigned int block_size       = params.kernel_config.block_size;
    const unsigned int items_per_thread = params.kernel_config.items_per_thread;
    const unsigned int items_per_block  = block_size * items_per_thread;
    return items_per_block;
}

} // namespace

enum class benchmark_search_n_mode
{
    NORMAL = 0,
    NOISE  = 1,
};

inline std::string to_string(benchmark_search_n_mode e) noexcept
{
    switch(e)
    {
        case benchmark_search_n_mode::NORMAL: return "NORMAL";
        case benchmark_search_n_mode::NOISE: return "NOISE";
        default: return "UNKNOWN";
    }
}

template<class InputType, class OutputType, benchmark_search_n_mode mode>
class benchmark_search_n
{
public:
    const managed_seed     seed;
    const hipStream_t      stream;
    size_t                 size_byte;
    size_t                 count_byte;
    size_t                 start_pos_byte;
    InputType              value;
    std::vector<InputType> input;

private:
    size_t       size;
    size_t       count;
    size_t       start_pos;
    const size_t warmup_size       = 10;
    const size_t batch_size        = 10;
    size_t       temp_storage_size = 0;
    size_t       noise_sequence    = 0;
    bool         create_noise      = false;

    hipEvent_t start;
    hipEvent_t stop;

    void*       d_temp_storage = nullptr;
    InputType*  d_input;
    OutputType* d_output;
    InputType*  d_value;

    void create() noexcept
    {
        switch(mode)
        {
            case benchmark_search_n_mode::NORMAL:
                {
                    input.resize(size);
                    if(start_pos + count < size)
                    {
                        std::fill(input.begin(), input.begin() + start_pos, 0);
                        std::fill(input.begin() + start_pos,
                                  input.begin() + count + start_pos,
                                  value);
                        std::fill(input.begin() + count + start_pos, input.end(), 0);
                    }
                    else
                    {
                        std::fill(input.begin(), input.end(), 0);
                    }
                    break;
                }
            case benchmark_search_n_mode::NOISE:
                {
                    InputType h_noise{0};
                    input = std::vector<InputType>(size, value);

                    if(create_noise)
                    {
                        size_t cur_tile  = 0;
                        size_t last_tile = size / count - 1;
                        while(cur_tile != last_tile)
                        {
                            input[cur_tile * count + count - 1] = h_noise;
                            ++cur_tile;
                        }
                    }
                    break;
                }
            default:
                {
                    break;
                }
        }

        HIP_CHECK(hipMalloc(&d_value, sizeof(InputType)));
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

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
    }

    static void run(benchmark::State& state, benchmark_search_n const& _self)
    {
        auto& self = const_cast<benchmark_search_n&>(_self);
        self.create();
        auto launch_search_n = [&]()
        {
            HIP_CHECK(::rocprim::search_n(self.d_temp_storage,
                                          self.temp_storage_size,
                                          self.d_input,
                                          self.d_output,
                                          self.size,
                                          self.count,
                                          self.d_value,
                                          rocprim::equal_to<InputType>{},
                                          self.stream,
                                          false));

            HIP_CHECK(hipMalloc(&self.d_temp_storage, self.temp_storage_size));

            HIP_CHECK(::rocprim::search_n(self.d_temp_storage,
                                          self.temp_storage_size,
                                          self.d_input,
                                          self.d_output,
                                          self.size,
                                          self.count,
                                          self.d_value,
                                          rocprim::equal_to<InputType>{},
                                          self.stream,
                                          false));

            HIP_CHECK(hipFree(self.d_temp_storage));
            self.temp_storage_size = 0;
            self.d_temp_storage    = nullptr;
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
    benchmark_search_n(const managed_seed _seed,
                       const hipStream_t  _stream,
                       const size_t       _2,
                       const size_t       _3, // for NOISE benchmarks, this is the multiple of count
                       const size_t       _4) noexcept
        : seed(_seed)
        , stream(_stream)
        , size_byte(0)
        , count_byte(0)
        , start_pos_byte(0)
        , value{1}
        , input()
    {

        switch(mode)
        {
            case benchmark_search_n_mode::NORMAL:
                {
                    size_byte      = _2;
                    count_byte     = _3;
                    start_pos_byte = _4;

                    size      = size_byte / sizeof(InputType);
                    count     = count_byte / sizeof(InputType);
                    start_pos = start_pos_byte / sizeof(InputType);
                    break;
                }
            case benchmark_search_n_mode::NOISE:
                {
                    size_byte  = _2;
                    count_byte = _3;

                    size  = size_byte / sizeof(InputType);
                    count = count_byte;
                    noise_sequence
                        = _4 == (size_t)-1
                              ? search_n_get_item_per_block<rocprim::default_config, InputType>()
                              : _4;

                    if(size > noise_sequence * count)
                    {
                        count        = noise_sequence * count;
                        create_noise = true;
                    }
                    break;
                }
        }
    }

    benchmark::internal::Benchmark* bench_register() const noexcept
    {
        return benchmark::RegisterBenchmark(
            bench_naming::format_name(
                "{lvl:device,algo:search_n,input_type:" + std::string(typeid(InputType).name())
                + ",size:" + std::to_string(size) + ",count:" + std::to_string(count)
                + ",mode:" + to_string(mode) + ",cfg:default_config}")
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
    // small count test
    benchmark_search_n<T, size_t, benchmark_search_n_mode::NOISE> small_count6(_seed,
                                                                               _stream,
                                                                               _size_byte,
                                                                               1, // count times
                                                                               6);
    benchmark_search_n<T, size_t, benchmark_search_n_mode::NOISE> small_count10(_seed,
                                                                                _stream,
                                                                                _size_byte,
                                                                                1, // count times
                                                                                10);
    benchmark_search_n<T, size_t, benchmark_search_n_mode::NOISE> small_count256(_seed,
                                                                                 _stream,
                                                                                 _size_byte,
                                                                                 1, // count times
                                                                                 256);
    benchmark_search_n<T, size_t, benchmark_search_n_mode::NOISE> small_count512(_seed,
                                                                                 _stream,
                                                                                 _size_byte,
                                                                                 1, // count times
                                                                                 512);
    benchmark_search_n<T, size_t, benchmark_search_n_mode::NOISE> small_count1023(_seed,
                                                                                  _stream,
                                                                                  _size_byte,
                                                                                  1, // count times
                                                                                  1023);
    // mid count test
    benchmark_search_n<T, size_t, benchmark_search_n_mode::NOISE> mid_count1024(_seed,
                                                                                _stream,
                                                                                _size_byte,
                                                                                1, // count times
                                                                                1024);
    benchmark_search_n<T, size_t, benchmark_search_n_mode::NOISE> mid_count1536(_seed,
                                                                                _stream,
                                                                                _size_byte,
                                                                                1, // count times
                                                                                1536);
    benchmark_search_n<T, size_t, benchmark_search_n_mode::NOISE> mid_count2047(_seed,
                                                                                _stream,
                                                                                _size_byte,
                                                                                1, // count times
                                                                                2047);
    benchmark_search_n<T, size_t, benchmark_search_n_mode::NOISE> mid_count2560(_seed,
                                                                                _stream,
                                                                                _size_byte,
                                                                                1, // count times
                                                                                2560);
    benchmark_search_n<T, size_t, benchmark_search_n_mode::NOISE> mid_count4095(_seed,
                                                                                _stream,
                                                                                _size_byte,
                                                                                1, // count times
                                                                                4095);
    std::vector<benchmark::internal::Benchmark*> bs = {small_count6.bench_register(),
                                                       small_count10.bench_register(),
                                                       small_count256.bench_register(),
                                                       small_count512.bench_register(),
                                                       small_count1023.bench_register(),
                                                       mid_count1024.bench_register(),
                                                       mid_count1536.bench_register(),
                                                       mid_count2047.bench_register(),
                                                       mid_count2560.bench_register(),
                                                       mid_count4095.bench_register()};

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

template<typename InputT, unsigned int BlockSize>
struct device_search_n_benchmark_generator
{
    // TODO: add implementation
    struct create_search_n_algorithm
    {};
    // TODO: add implementation
    static void create(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
    {
        (void)storage;
    }
};

#endif // ROCPRIM_BENCHMARK_DEVICE_SEARCH_N_PARALLEL_HPP_
