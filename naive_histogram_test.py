#  Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import nvidia.dali as dali
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
from nvidia.dali.types import DALIImageType

# Load the Custom Operator
import nvidia.dali.plugin_manager as plugin_manager
plugin_manager.load_library('./build/libnaivehistogram.so')

# List test files
test_file_list = [
    "/home/mszolucha/workspace/DALI_extra/db/single/jpeg/100/swan-3584559_640.jpg",
    "/home/mszolucha/workspace/DALI_extra/db/single/jpeg/113/snail-4368154_1280.jpg",
    "/home/mszolucha/workspace/DALI_extra/db/single/jpeg/100/swan-3584559_640.jpg",
    "/home/mszolucha/workspace/DALI_extra/db/single/jpeg/113/snail-4368154_1280.jpg",
    "/home/mszolucha/workspace/DALI_extra/db/single/jpeg/100/swan-3584559_640.jpg",
    "/home/mszolucha/workspace/DALI_extra/db/single/jpeg/113/snail-4368154_1280.jpg",
]


# DALI pipeline definition
@pipeline_def
def naive_hist_pipe():
    img, _ = fn.readers.file(files=test_file_list)
    img = fn.decoders.image(img, device='mixed', output_type=DALIImageType.GRAY)
    img = img.gpu()
    img = fn.naive_histogram(img)
    return img


pipe = naive_hist_pipe(batch_size=2, num_threads=1, device_id=0)
pipe.build()
out = pipe.run()
print(out[0].as_cpu().as_array())
