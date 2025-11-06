# Image Filtering using NVIDIA NPP with CUDA

## Overview

This project demonstrates the use of NVIDIA Performance Primitives (NPP) library with CUDA to perform image filtering. The goal is to utilize GPU acceleration to efficiently filter a given image by a specified filter, leveraging the computational power of modern GPUs.

## Prerequisites

Download and install the [CUDA Toolkit 12.4](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.

## Build and Run

### Windows
The Windows program is built using the Visual Studio 2022 IDE.

### Linux
The Linux program is built using makefiles. To use the makefiles, just run make:
```
$ make
```

After building the project, you can run the program using the following command:

```bash
make run
```

This command will execute the compiled binary, filtering the input image (Lena.png) by a 5x5 box filter with replicate border, and save the result as Lena_box_replicate.png in the data/ directory.

If you wish to run the binary directly with custom input/output files, you can use:

```bash
./bin/npp-filters --input data/Lena.png --filter box --border replicate --output data/Lena_rotated.png
```

You can run all filters/borders combinaison:

```bash
./run.sh
```

## Program options

| Options | Description | Values |
|--------|-------------|--------|
|\-\-input| Input filename | data/Lena.png(Default) |
|\-\-output| Output filename | |
|\-\-filter| Select filter type | box(Default), sobel_h, sobel_v, roberts_up, roberts_down, laplace, gauss, highpass, lowpass, sharpen, wiener |
|\-\-border| Select border type | none, replicate(Default) |

| Filter | Description |
|--------|-------------|
|box|[Computes the average pixel values of the pixels under a rectangular mask](https://docs.nvidia.com/cuda/npp/image_filtering_functions.html#image-filter-box)|
|sobel_h|[Filters the image using a horizontal Sobel filter kernel](https://docs.nvidia.com/cuda/npp/image_filtering_functions.html#image-filter-sobel)|
|sobel_v|[Filters the image using a vertical Sobel filter kernel](https://docs.nvidia.com/cuda/npp/image_filtering_functions.html#image-filter-sobel)|
|roberts_down|[Filters the image using a horizontal Roberts filter kernel](https://docs.nvidia.com/cuda/npp/image_filtering_functions.html#image-filter-roberts)|
|roberts_up|[Filters the image using a vertical Roberts filter kernel](https://docs.nvidia.com/cuda/npp/image_filtering_functions.html#image-filter-roberts)|
|laplace|[Filters the image using a Laplacian filter kernel](https://docs.nvidia.com/cuda/npp/image_filtering_functions.html#image-filter-laplace)|
|gauss|[Filters the image using a Gaussian filter kernel](https://docs.nvidia.com/cuda/npp/image_filtering_functions.html#image-filter-gauss)|
|highpass|[Filters the image using a high-pass filter kernel](https://docs.nvidia.com/cuda/npp/image_filtering_functions.html#image-filter-high-pass)|
|lowpass|[Filters the image using a low-pass filter kernel](https://docs.nvidia.com/cuda/npp/image_filtering_functions.html#image-filter-low-pass)|
|sharpen|[Filters the image using a sharpening filter kernel](https://docs.nvidia.com/cuda/npp/image_filtering_functions.html#image-filter-sharpen)|
|wiener|[Noise removal filtering of an image using an adaptive Wiener filter with border control](https://docs.nvidia.com/cuda/npp/image_filtering_functions.html#image-filter-wiener-border)|

## Output Sample

```bash
./bin/npp-filters --filter sobel_v --border none
```
```bash
bin/npp-filters Starting...

GPU Device 0: "Ampere" with compute capability 8.6

NPP Library Version 12.3.1
  CUDA Driver  Version: 12.6
  CUDA Runtime Version: 12.6
  Device 0: <          Ampere >, Compute SM 8.6 detected
npp-filters opened: <./data/Lena.png> successfully!
Saved image: ./data/Lena_filter_sobel_v_none.png
```
<img width="1666" height="1250" alt="image" src="https://github.com/user-attachments/assets/b7aa0b9d-dcd1-4e0d-abfc-c1ce13b84c59" />

