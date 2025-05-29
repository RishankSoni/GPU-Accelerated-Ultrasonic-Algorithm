#include "mex.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define CUDA_CHECK(err) { cudaError_t err_ = (err); if (err_ != cudaSuccess) { mexPrintf("CUDA error %d at %s:%d: %s\n", err_, __FILE__, __LINE__, cudaGetErrorString(err_)); mexErrMsgIdAndTxt("PCIimagingSparseCUDA:cudaError", cudaGetErrorString(err_)); } }
#define KERNEL_CHECK() { CUDA_CHECK(cudaGetLastError()); }
#define CUFFT_CHECK(err) { cufftResult err_ = (err); if (err_ != CUFFT_SUCCESS) { mexPrintf("CUFFT error %d at %s:%d\n", err_, __FILE__, __LINE__); mexErrMsgIdAndTxt("PCIimagingSparseCUDA:cufftError", "CUFFT error"); } }

// --- CUDA Kernels ---

__global__ void computeFftBinKernel(float* fftBin, int nfft, int binStart) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nfft) {
        int shift = (nfft % 2 == 0) ? nfft / 2 : (nfft - 1) / 2;
        int srcIndex = (idx + shift) % nfft;
        float value = (float)srcIndex - (float)binStart;
        fftBin[idx] = (2.0f * M_PI * value) / (float)nfft;
    }
}

__global__ void realToComplexKernel(const float* realInput, cufftComplex* complexOutput, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        complexOutput[idx].x = realInput[idx];
        complexOutput[idx].y = 0.0f;
    }
}

__device__ cufftComplex cmul(cufftComplex a, cufftComplex b) {
    cufftComplex res;
    res.x = a.x * b.x - a.y * b.y;
    res.y = a.x * b.y + a.y * b.x;
    return res;
}

// Main kernel: one thread per pixel in batch
__global__ void PCI_beamform_batch_kernel(
    const cufftComplex* __restrict__ d_RFFT, // [nfft x ncols]
    const float* __restrict__ d_fftBin,           // [nfft]
    const float* __restrict__ d_all_delays,       // [ncols x total_pixels]
    int nfft, int ncols, int total_pixels,
    int batch_start_idx, int batch_size,
    int FlowerIndex, int freq_band_len,
    float p_param, int check,
    float* __restrict__ d_output                  // [total_pixels]
) {
    int batch_pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_pixel >= batch_size) return;
    int pixel_idx = batch_start_idx + batch_pixel;
    if (pixel_idx >= total_pixels) return;

    const float* delays = d_all_delays + pixel_idx * ncols;
    float pixel_sum = 0.0f;

    for (int f = 0; f < freq_band_len; ++f) {
        int r = FlowerIndex + f;

        // --- Build var for this frequency: phase-shifted, all channels ---
        cufftComplex var[128]; // adjust 128 if needed for max ncols
        for (int c = 0; c < ncols; ++c) {
            float angle = -delays[c] * d_fftBin[r];
            cufftComplex factor = {cosf(angle), sinf(angle)};
            int rfft_idx = r + c * nfft;
            cufftComplex val = d_RFFT[rfft_idx];
            var[c].x = val.x * factor.x - val.y * factor.y;
            var[c].y = val.x * factor.y + val.y * factor.x;
        }

        if (check == 1) {
            // --- pCF branch ---
            float sum_real = 0.0f, sum_imag = 0.0f;
            for (int c = 0; c < ncols; ++c) {
                float mag = hypotf(var[c].x, var[c].y);
                float phase = atan2f(var[c].y, var[c].x);
                sum_real += mag * cosf(phase);
                sum_imag += mag * sinf(phase);
            }
            float pDAS = hypotf(sum_real, sum_imag);

            float sum_real_p = 0.0f, sum_imag_p = 0.0f, Dr = 0.0f;
            for (int c = 0; c < ncols; ++c) {
                float mag = hypotf(var[c].x, var[c].y);
                float phase = atan2f(var[c].y, var[c].x);
                float comp = powf(mag, 1.0f / p_param);
                sum_real_p += comp * cosf(phase);
                sum_imag_p += comp * sinf(phase);
                Dr += mag * mag;
            }
            float abs_sum_p = hypotf(sum_real_p, sum_imag_p);
            float beamformed = powf(abs_sum_p, p_param);
            float Nr = beamformed * beamformed;
            if (Dr == 0.0f) Dr = 1e-12f;
            float CF = (1.0f / ncols) * (Nr / Dr);

            float DCoffset = 0.0f;
            for (int c = 0; c < ncols; ++c) {
                float real_cf = var[c].x * CF;
                float imag_cf = var[c].y * CF;
                DCoffset += real_cf * real_cf + imag_cf * imag_cf;
            }

            float val = pDAS * CF;
            pixel_sum += (val * val) - DCoffset;
        } else {
            // --- pDAS branch ---
            float sum_real = 0.0f, sum_imag = 0.0f;
            for (int c = 0; c < ncols; ++c) {
                float mag = hypotf(var[c].x, var[c].y);
                float phase = atan2f(var[c].y, var[c].x);
                float comp = powf(mag, 1.0f / p_param);
                sum_real += comp * cosf(phase);
                sum_imag += comp * sinf(phase);
            }
            float pDAS = powf(hypotf(sum_real, sum_imag), p_param);

            float DCoffset = 0.0f;
            for (int c = 0; c < ncols; ++c) {
                DCoffset += var[c].x * var[c].x + var[c].y * var[c].y;
            }

            pixel_sum += (pDAS * pDAS) - DCoffset;
        }
    }
    d_output[pixel_idx] = pixel_sum;
}

// --- MEX Gateway Function ---
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // --- Input Validation ---
    if (nrhs != 11) mexErrMsgIdAndTxt("PCIimagingSparseCUDA:nrhs", "Eleven inputs required: RF_Arr, element_Pos_Array_um, speed_Of_Sound_umps, RF_Start_Time, sampling_Freq, image_Range_X_um, image_Range_Z_um, p, all_delays_matrix, range_frq, check");

    // --- Get Inputs & Dimensions ---
    const float* h_RF_Arr = (const float*)mxGetData(prhs[0]);
    size_t nfft = mxGetM(prhs[0]);
    size_t ncols = mxGetN(prhs[0]);
    size_t total_pixels = mxGetN(prhs[8]);
    const float* h_all_delays = (const float*)mxGetData(prhs[8]);
    float p_param = (float)mxGetScalar(prhs[7]);
    float* range_frq = (float*)mxGetData(prhs[9]);
    int check = (int)mxGetScalar(prhs[10]);
    float sampling_Freq = (float)mxGetScalar(prhs[4]);
    size_t numX = mxGetNumberOfElements(prhs[5]);
    size_t numZ = mxGetNumberOfElements(prhs[6]);

    // --- Frequency band indices ---
    std::vector<float> fk(nfft);
    for (size_t i = 0; i < nfft; ++i) fk[i] = ((float)i) * sampling_Freq / (float)nfft;
    size_t FlowerIndex = 0, FupperIndex = nfft-1;
    float minDiff = fabsf(fk[0] - range_frq[0]);
    for (size_t i = 1; i < nfft; ++i) if (fabsf(fk[i] - range_frq[0]) < minDiff) { FlowerIndex = i; minDiff = fabsf(fk[i] - range_frq[0]); }
    minDiff = fabsf(fk[0] - range_frq[1]);
    for (size_t i = 1; i < nfft; ++i) if (fabsf(fk[i] - range_frq[1]) < minDiff) { FupperIndex = i; minDiff = fabsf(fk[i] - range_frq[1]); }
    size_t freq_band_len = FupperIndex - FlowerIndex + 1;

    mexPrintf("Starting CUDA PCI Imaging (p=%.2f, check=%d, freq band [%zu:%zu])\n", p_param, check, FlowerIndex, FupperIndex);

    // --- Output Array ---
    plhs[0] = mxCreateNumericMatrix(numX, numZ, mxSINGLE_CLASS, mxREAL);
    float* h_beamformed_Image = (float*)mxGetData(plhs[0]);

    // --- Allocate Device Memory ---
    cufftComplex *d_RFFT;
    float *d_fftBin;
    float *d_beamformed_Image;
    float *d_all_delays;
    CUDA_CHECK(cudaMalloc(&d_RFFT, nfft * ncols * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_fftBin, nfft * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beamformed_Image, total_pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_all_delays, ncols * total_pixels * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_all_delays, h_all_delays, ncols * total_pixels * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_beamformed_Image, 0, total_pixels * sizeof(float)));

    // --- Precomputations ---
    int blockSize = 256;
    int gridSize = (nfft + blockSize - 1) / blockSize;
    int binStart = nfft / 2;
    computeFftBinKernel<<<gridSize, blockSize>>>(d_fftBin, nfft, binStart);
    KERNEL_CHECK();

    // Convert RF_Arr to complex and FFT -> d_RFFT
    cufftComplex* d_RData_complex_temp;
    float* d_RF_Arr_temp;
    size_t total_rf_elements = nfft * ncols;
    CUDA_CHECK(cudaMalloc(&d_RData_complex_temp, total_rf_elements * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_RF_Arr_temp, total_rf_elements * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_RF_Arr_temp, h_RF_Arr, total_rf_elements * sizeof(float), cudaMemcpyHostToDevice));
    int gridSize_rf = (total_rf_elements + blockSize - 1) / blockSize;
    realToComplexKernel<<<gridSize_rf, blockSize>>>(d_RF_Arr_temp, d_RData_complex_temp, total_rf_elements);
    KERNEL_CHECK();
    CUDA_CHECK(cudaFree(d_RF_Arr_temp));
    cufftHandle plan_forward;
    CUFFT_CHECK(cufftPlan1d(&plan_forward, nfft, CUFFT_C2C, ncols));
    CUFFT_CHECK(cufftExecC2C(plan_forward, d_RData_complex_temp, d_RFFT, CUFFT_FORWARD));
    CUFFT_CHECK(cufftDestroy(plan_forward));
    CUDA_CHECK(cudaFree(d_RData_complex_temp));

    // --- Launch Batching Kernel ---
    int batch_size = 8192; // Tune for your GPU memory and occupancy
    int threadsPerBlock = 128; // Tune for your GPU
    for (size_t batch_start_idx = 0; batch_start_idx < total_pixels; batch_start_idx += batch_size) {
        int current_batch_size = std::min(batch_size, (int)(total_pixels - batch_start_idx));
        int blocksPerGrid = (current_batch_size + threadsPerBlock - 1) / threadsPerBlock;
        PCI_beamform_batch_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_RFFT, d_fftBin, d_all_delays, nfft, ncols, total_pixels,
            batch_start_idx, current_batch_size,
            FlowerIndex, freq_band_len, p_param, check, d_beamformed_Image
        );
        KERNEL_CHECK();
    }

    // --- Copy Result Back to Host ---
    CUDA_CHECK(cudaMemcpy(h_beamformed_Image, d_beamformed_Image, total_pixels * sizeof(float), cudaMemcpyDeviceToHost));

    // --- Cleanup ---
    CUDA_CHECK(cudaFree(d_RFFT));
    CUDA_CHECK(cudaFree(d_fftBin));
    CUDA_CHECK(cudaFree(d_beamformed_Image));
    CUDA_CHECK(cudaFree(d_all_delays));

    mexPrintf("CUDA PCI Imaging complete.\n");
}yi