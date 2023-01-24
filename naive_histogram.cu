// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "naive_histogram.h"

namespace naive_histogram {

using namespace ::dali;

__global__ void naive_histogram_kernel(
        const uint8_t *input, const int input_size, const float one_over_values_per_bin,
        int32_t *output) {
  for (int i = 0; i < input_size; i++) {
    output[static_cast<int>(input[i] * one_over_values_per_bin)]++;
  }
}


template<>
void NaiveHistogram<GPUBackend>::RunImpl(Workspace &ws) {
  const auto &input = ws.Input<GPUBackend>(0);
  const auto &shape = input.shape();
  auto &output = ws.Output<GPUBackend>(0);
  for (int sample_idx = 0; sample_idx < shape.num_samples(); sample_idx++) {  // Iterating over all samples in a batch.
    naive_histogram_kernel<<<1, 1, 0, ws.stream()>>>(
            input[sample_idx].data<uint8_t>(),
            volume(input.tensor_shape(sample_idx)),
            n_histogram_bins_ / 255.f,
            output[sample_idx].mutable_data<int32_t>()
    );
  }
}

}  // namespace naive_histogram