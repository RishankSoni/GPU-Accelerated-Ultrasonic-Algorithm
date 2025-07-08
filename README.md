# GPU-Accelerated-Ultrasonic-Algorithm

This project implements GPU-optimized ultrasonic beamforming algorithms for faster image processing. The implementation uses CUDA for GPU acceleration through MEX files in MATLAB.

## Authors  
- **Arjun Anand Mallya**  
- **Rishank Soni**  

### Under the guidance of:  
**Professor Himanshu Shekhar**  

---

## Description

The project contains:
- CUDA implementation of Delay-and-Sum (DAS) Beamforming and Non Linear pth Root Beamforming Algorithms
- MATLAB code for ultrasonic image generation
- GPU-accelerated processing using MEX files

## Prerequisites

- MATLAB with Parallel Computing Toolbox
- NVIDIA CUDA Toolkit 
- Compatible NVIDIA GPU

## Project Structure

- `DAS-Algorithm/`  
  - `BF_das_rephaseSignal.cu` - CUDA source code for DAS
  - `Image-generation_code.m` - MATLAB script for DAS image generation

- `pth-Root-Algorithm/`  
  - `BF_das_rephaseSignalpth.cu` - CUDA source code for p-th root DAS
  - `Image-generation_code.m` - MATLAB script for p-th root DAS image generation

- `RCB/`  
  - `comcudas.cu` - CUDA source code for Robust Capon Beamforming (RCB)
  - `PCI_RBC_onefile.m` - MATLAB script for RCB processing

- `GPU-Accelerated-Ultrasonic-Algorithm/`  
  - `PCI_RBC_onefile.m` - Top-level MATLAB script for PCI RBC workflow

## Usage

1. Ensure all prerequisites are installed:
   - MATLAB with Parallel Computing Toolbox
   - NVIDIA CUDA Toolkit
   - Compatible NVIDIA GPU

2. Run the MATLAB scripts directly from their respective folders:
   ```matlab
   % For DAS or p-th root DAS:
   Image-generation_code

   % For Robust Capon Beamforming:
   PCI_RBC_onefile
   ```

**Note:**  
The MEX files are pre-compiled and included in the repository. You don't need to compile them again unless you modify the CUDA source code.

If you need to recompile the MEX files, use:
```matlab
% For DAS:
mexcuda BF_das_rephaseSignal.cu

% For p-th root DAS:
mexcuda BF_das_rephaseSignalpth.cu

% For Robust Capon Beamforming:
mexcuda comcudas.cu -lcufft -lcublas -lcusolver