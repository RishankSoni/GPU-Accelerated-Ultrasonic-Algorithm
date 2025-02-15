# GPU-Accelerated-Ultrasonic-Algorithm

This project implements GPU-optimized ultrasonic beamforming algorithms for faster image processing. The implementation uses CUDA for GPU acceleration through MEX files in MATLAB.

## Description

The project contains:
- CUDA implementation of Delay-and-Sum (DAS) beamforming algorithm
- MATLAB code for ultrasonic image generation
- GPU-accelerated processing using MEX files

## Prerequisites

- MATLAB with Parallel Computing Toolbox
- NVIDIA CUDA Toolkit 
- Compatible NVIDIA GPU

## Project Structure

- `DAS-Algorithm/` - Standard Delay-and-Sum implementation
  - `BF_das_rephaseSignal.cu` - CUDA source code
  - `Image-generation_code.m` - MATLAB image generation script

- `pth-Root-Algorithm/` - P-th root DAS implementation
  - `BF_das_rephaseSignalpth.cu` - CUDA source code
  - `Image-generation_code.m` - MATLAB image generation script

## Usage

1. Ensure all prerequisites are installed:
   - MATLAB with Parallel Computing Toolbox
   - NVIDIA CUDA Toolkit
   - Compatible NVIDIA GPU

2. Run the MATLAB script directly:
   ```matlab
   Image-generation_code.m
   ```

Note: The MEX files are pre-compiled and included in the repository. You don't need to compile them again unless you modify the CUDA source code.

If you need to recompile the MEX files, use:
```matlab
mexcuda BF_das_rephaseSignal.cu
mexcuda BF_das_rephaseSignalpth.cu
```
