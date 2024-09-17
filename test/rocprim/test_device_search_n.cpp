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
#include "test_utils_types.hpp"

#include <rocprim/device/device_search_n.hpp>

#include <time.h>

int main()
{
    using InputType  = int;
    using OutputType = int;

    size_t                 count   = 3;
    InputType              h_value = 7;
    InputType*             d_value;
    std::vector<InputType> h_input;
    InputType*             d_input           = nullptr;
    OutputType*            d_output          = nullptr;
    size_t                 temp_storage_size = sizeof(size_t);
    void*                  d_temp_storage    = nullptr;

    std::srand(time(0));
    h_input = {1,2,3,4,5,7,7,7,9,9,9};

    for(auto i : h_input)
    {
        printf("%d ", i);
    }
    printf("\n");

    // auto func = rocprim::greater<InputType>();

    HIP_CHECK(test_common_utils::hipMallocHelper(&d_value, sizeof(InputType)));
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size));
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, sizeof(InputType) * h_input.size()));
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, sizeof(OutputType)));
    HIP_CHECK(hipMemcpy(d_value, &h_value, sizeof(InputType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_input,
                        h_input.data(),
                        sizeof(InputType) * h_input.size(),
                        hipMemcpyHostToDevice));

    auto err = rocprim::search_n<rocprim::search_n_config<256, 4>>(d_temp_storage,
                                                                   temp_storage_size,
                                                                   d_input,
                                                                   d_output,
                                                                   h_input.size(),
                                                                   count,
                                                                   d_value);
    
    OutputType out;
    HIP_CHECK(hipMemcpy(&out, d_output, sizeof(OutputType), hipMemcpyDeviceToHost));
    printf("the result is:%d\n",out);
    
    HIP_CHECK(err);
    HIP_CHECK(hipFree(d_value));
    HIP_CHECK(hipFree(d_temp_storage));
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    return 0;
}