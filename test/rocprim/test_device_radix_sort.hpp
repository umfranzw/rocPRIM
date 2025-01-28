// MIT License
//
// Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef TEST_DEVICE_RADIX_SORT_HPP_
#define TEST_DEVICE_RADIX_SORT_HPP_

#include <iostream>
#include <vector>
#include <numeric>

// Google Test
#include <gtest/gtest.h>

// HIP API
#include <hip/hip_runtime.h>

#ifndef HIP_CHECK
    #define HIP_CHECK(condition)                                                            \
        {                                                                                   \
            hipError_t error = condition;                                                   \
            if(error != hipSuccess)                                                         \
            {                                                                               \
                [error]()                                                                   \
                { FAIL() << "HIP error " << error << ": " << hipGetErrorString(error); }(); \
                exit(error);                                                                \
            }                                                                               \
        }
#endif

#define HIP_CHECK_MEMORY(condition)                                                         \
    {                                                                                       \
        hipError_t error = condition;                                                       \
        if(error == hipErrorOutOfMemory)                                                    \
        {                                                                                   \
            std::cout << "Out of memory. Skipping size = " << size << std::endl;            \
            break;                                                                          \
        }                                                                                   \
        if(error != hipSuccess)                                                             \
        {                                                                                   \
            std::cout << "HIP error: " << hipGetErrorString(error) << " line: " << __LINE__ \
                      << std::endl;                                                         \
            exit(error);                                                                    \
        }                                                                                   \
    }

inline void sort_keys_large_sizes()
{
    using key_type                   = uint8_t;

    const std::vector<size_t> sizes = {(size_t{1} << 35) - 1};
    for (const size_t size : sizes)
    {
        // Generate data
        std::vector<key_type> keys_input(size);
        std::iota(keys_input.begin(), keys_input.end(), 0);

        key_type* d_keys;
        std::cout << "Attempting to allocate d_keys, size: " << size * sizeof(key_type) << std::endl;
        HIP_CHECK_MEMORY(hipMalloc(&d_keys, size * sizeof(key_type)));
        std::cout << "Done" << std::endl;

        void*  d_temporary_storage     = nullptr;
        size_t temporary_storage_bytes = 34628177919;
        std::cout << "Attempting to allocate d_temporary_storage, size: " << temporary_storage_bytes << std::endl;
        HIP_CHECK_MEMORY(
            hipMalloc(&d_temporary_storage, temporary_storage_bytes));
        std::cout << "Done" << std::endl;

        HIP_CHECK(hipFree(d_temporary_storage));
        HIP_CHECK(hipFree(d_keys));
    }
}

#endif // TEST_DEVICE_RADIX_SORT_HPP_
