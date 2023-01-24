# DALI custom operator example

This is an example/template of a DALI Custom operator.
The implemented operation is a (very) naive histogram on a grayscale image.
The NaiveHistogram operator is a GPU-only operator, which uses a badly written
CUDA kernel to calculate the histogram. The purpose here is not to show-case
how the write CUDA kernels, but to present how to use CUDA kernels within
DALI operators.

# Running the example

To run the example, please follow the steps:

1. Clone or download the repo
1. Install DALI wheel, for example using [these instructions](https://docs.nvidia.com/deeplearning/dali/main-user-guide/docs/installation.html#pip-official-releases)
1. Build the `libnaivehistogram.so`:
```bash
$ cd naive_histogram
$ mkdir build
$ cd build
$ cmake ..
$ make -j
```

4. Run the provided example (your output should look like below):
```bash
$ cd naive_histogram
$ python naive_histogram_test.py
[[10076  7742  4664  5350  8022 12316 15216 14263 19850 23855 30523 39412
  22499 10582  7308  4341  4404  4593  3274  3597  3205  1830  1238  1965]
 [ 1235   445   918 29833 82856 71328 74338 54495 57659 44534 40799 49006
  49242 46561 52237 43778 44008 41435 38017 51877 72865 64621 62300 17461]]
```

# More information

More information about DALI Custom Operator you can find in [DALI Documentation](https://docs.nvidia.com/deeplearning/dali/main-user-guide/docs/examples/custom_operations/custom_operator/create_a_custom_operator.html)

