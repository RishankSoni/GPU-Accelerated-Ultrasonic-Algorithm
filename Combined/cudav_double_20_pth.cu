#include "mex.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_CHECK(err) { cudaError_t err_ = (err); if (err_ != cudaSuccess) { mexPrintf("CUDA error %d at %s:%d: %s\n", err_, __FILE__, __LINE__, cudaGetErrorString(err_)); mexErrMsgIdAndTxt("PCIimagingSparseCUDA:cudaError", cudaGetErrorString(err_)); } }
#define KERNEL_CHECK() { CUDA_CHECK(cudaGetLastError()); }
#define CUFFT_CHECK(err) { cufftResult err_ = (err); if (err_ != CUFFT_SUCCESS) { mexPrintf("CUFFT error %d at %s:%d\n", err_, __FILE__, __LINE__); mexErrMsgIdAndTxt("PCIimagingSparseCUDA:cufftError", "CUFFT error"); } }

// --- CUDA Kernels ---

__global__ void computeFftBinKernel(double* fftBin, int nfft, int binStart) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nfft) {
        int shift = (nfft % 2 == 0) ? nfft / 2 : (nfft - 1) / 2;
        int srcIndex = (idx + shift) % nfft;
        double value = static_cast<double>(srcIndex) - static_cast<double>(binStart);
        fftBin[idx] = (2.0 * M_PI * value) / static_cast<double>(nfft);
    }
}

__global__ void realToComplexKernel(const double* realInput, cufftDoubleComplex* complexOutput, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        complexOutput[idx].x = realInput[idx];
        complexOutput[idx].y = 0.0;
    }
}

__global__ void shiftDataBatchKernel(const cufftDoubleComplex* RFFT, const double* fftBin, const double* batch_delays, int nfft, int ncols, int current_batch_size, cufftDoubleComplex* ShiftDataBatch) {
    int idx_out = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = (size_t)nfft * ncols * current_batch_size;
    if (idx_out < total_elements) {
        int p = idx_out / (nfft * ncols);
        int rc_idx = idx_out % (nfft * ncols);
        int r = rc_idx % nfft;
        int c = rc_idx / nfft;
        double delay_val = batch_delays[c + p * ncols];
        double fftBin_val = fftBin[r];
        double angle = -delay_val * fftBin_val;
        double cosval = cos(angle);
        double sinval = sin(angle);
        cufftDoubleComplex factor = {cosval, sinval};
        int rfft_idx = r + c * nfft;
        cufftDoubleComplex val = RFFT[rfft_idx];
        cufftDoubleComplex res;
        res.x = val.x * factor.x - val.y * factor.y;
        res.y = val.x * factor.y + val.y * factor.x;
        ShiftDataBatch[idx_out] = res;
    }
}

// Extract frequency band [FlowerIndex:FupperIndex]
__global__ void extractFreqBandKernel(
    const cufftDoubleComplex* ShiftDataBatch, // [nfft x ncols x batch_size]
    int nfft, int ncols, int batch_size,
    int FlowerIndex, int freq_band_len,
    cufftDoubleComplex* FreqBandBatch // [freq_band_len x ncols x batch_size]
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)freq_band_len * ncols * batch_size;
    if (idx < total) {
        int p = idx / (freq_band_len * ncols);
        int rem = idx % (freq_band_len * ncols);
        int c = rem / freq_band_len;
        int f = rem % freq_band_len;
        int src_idx = (FlowerIndex + f) + c * nfft + p * nfft * ncols;
        int dst_idx = f + c * freq_band_len + p * freq_band_len * ncols;
        FreqBandBatch[dst_idx] = ShiftDataBatch[src_idx];
    }
}

// pthrootfreq: for each freq, sum over channels, abs, ^p
__global__ void pthrootfreqKernel(
    const cufftDoubleComplex* FreqBandBatch, // [freq_band_len x ncols x batch_size]
    double p, int freq_band_len, int ncols, int batch_size,
    double* pDASBatch // [freq_band_len x batch_size]
) {
    int f = blockIdx.x;
    int pidx = blockIdx.y;
    if (f >= freq_band_len || pidx >= batch_size) return;
    double sum_real = 0.0, sum_imag = 0.0;
    for (int c = 0; c < ncols; ++c) {
        int idx = f + c * freq_band_len + pidx * freq_band_len * ncols;
        double mag = hypot(FreqBandBatch[idx].x, FreqBandBatch[idx].y);
        double phase = atan2(FreqBandBatch[idx].y, FreqBandBatch[idx].x);
        double comp = pow(mag, 1.0/p);
        sum_real += comp * cos(phase);
        sum_imag += comp * sin(phase);
    }
    double abs_sum = hypot(sum_real, sum_imag);
    pDASBatch[f + pidx * freq_band_len] = pow(abs_sum, p);
}

// pthcoherencefactorfreq: as in MATLAB
__global__ void pthcoherencefactorfreqKernel(
    const cufftDoubleComplex* FreqBandBatch,
    double p, int freq_band_len, int ncols, int batch_size,
    double* CFBatch // [freq_band_len x batch_size]
) {
    int f = blockIdx.x;
    int pidx = blockIdx.y;
    if (f >= freq_band_len || pidx >= batch_size) return;
    double sum_real = 0.0, sum_imag = 0.0;
    double Dr = 0.0;
    for (int c = 0; c < ncols; ++c) {
        int idx = f + c * freq_band_len + pidx * freq_band_len * ncols;
        double mag = hypot(FreqBandBatch[idx].x, FreqBandBatch[idx].y);
        double phase = atan2(FreqBandBatch[idx].y, FreqBandBatch[idx].x);
        double comp = pow(mag, 1.0/p);
        sum_real += comp * cos(phase);
        sum_imag += comp * sin(phase);
        Dr += mag * mag;
    }
    double abs_sum = hypot(sum_real, sum_imag);
    double beamformed = pow(abs_sum, p);
    double Nr = beamformed * beamformed;
    if (Dr == 0.0) Dr = 1e-12;
    double CF = (1.0 / ncols) * (Nr / Dr);
    CFBatch[f + pidx * freq_band_len] = CF;
}

// DC offset: sum(abs(var).^2,2)
__global__ void DCoffsetKernel(
    const cufftDoubleComplex* FreqBandBatch,
    int freq_band_len, int ncols, int batch_size,
    double* DCoffsetBatch // [freq_band_len x batch_size]
) {
    int f = blockIdx.x;
    int pidx = blockIdx.y;
    if (f >= freq_band_len || pidx >= batch_size) return;
    double sum = 0.0;
    for (int c = 0; c < ncols; ++c) {
        int idx = f + c * freq_band_len + pidx * freq_band_len * ncols;
        double mag = hypot(FreqBandBatch[idx].x, FreqBandBatch[idx].y);
        sum += mag * mag;
    }
    DCoffsetBatch[f + pidx * freq_band_len] = sum;
}

// Final pixel value: sum over freq_band_len
__global__ void finalPixelValueKernel(
    const double* pDASBatch, const double* CFBatch, const double* DCoffsetBatch,
    int freq_band_len, int batch_size, int batch_start_idx, int numX, int check,
    double* d_beamformed_Image
) {
    int pidx = blockIdx.x;
    if (pidx >= batch_size) return;
    double pixel_sum = 0.0;
    for (int f = 0; f < freq_band_len; ++f) {
        int idx = f + pidx * freq_band_len;
        if (check == 1) {
            double val = pDASBatch[idx] * CFBatch[idx];
            pixel_sum += (val * val) - DCoffsetBatch[idx];
        } else {
            double val = pDASBatch[idx];
            pixel_sum += (val * val) - DCoffsetBatch[idx];
        }
    }
    int global_pixel_idx = batch_start_idx + pidx;
    d_beamformed_Image[global_pixel_idx] = pixel_sum;
}

__device__ cufftDoubleComplex cmul(cufftDoubleComplex a, cufftDoubleComplex b) {
    cufftDoubleComplex res;
    res.x = a.x * b.x - a.y * b.y;
    res.y = a.x * b.y + a.y * b.x;
    return res;
}

// Main kernel: one thread per pixel
__global__ void PCI_beamform_kernel(
    const cufftDoubleComplex* __restrict__ d_RFFT, // [nfft x ncols]
    const double* __restrict__ d_fftBin,           // [nfft]
    const double* __restrict__ d_all_delays,       // [ncols x total_pixels]
    int nfft, int ncols, int total_pixels,
    int FlowerIndex, int freq_band_len,
    double p_param, int check,
    double* __restrict__ d_output                  // [total_pixels]
) {
    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_idx >= total_pixels) return;

    // 1. Load delays for this pixel
    const double* delays = d_all_delays + pixel_idx * ncols;

    // 2. For each channel, apply phase shift in frequency domain
    //    and extract frequency band
    extern __shared__ cufftDoubleComplex freq_band[]; // [freq_band_len x ncols]
    for (int c = 0; c < ncols; ++c) {
        for (int f = 0; f < freq_band_len; ++f) {
            int r = FlowerIndex + f;
            double angle = -delays[c] * d_fftBin[r];
            cufftDoubleComplex factor = {cos(angle), sin(angle)};
            int rfft_idx = r + c * nfft;
            cufftDoubleComplex val = d_RFFT[rfft_idx];
            freq_band[f * ncols + c] = cmul(val, factor);
        }
    }
    // 3. For each frequency in band, sum over channels (pDAS/pCF logic)
    double pixel_sum = 0.0;
    for (int f = 0; f < freq_band_len; ++f) {
        // pthrootfreq
        double sum_real = 0.0, sum_imag = 0.0;
        double Dr = 0.0;
        for (int c = 0; c < ncols; ++c) {
            cufftDoubleComplex v = freq_band[f * ncols + c];
            double mag = hypot(v.x, v.y);
            double phase = atan2(v.y, v.x);
            double comp = pow(mag, 1.0 / (check ? 1.0 : p_param));
            sum_real += comp * cos(phase);
            sum_imag += comp * sin(phase);
            Dr += mag * mag;
        }
        double abs_sum = hypot(sum_real, sum_imag);
        double pDAS = pow(abs_sum, check ? 1.0 : p_param);
        double CF = 1.0;
        if (check) {
            // pthcoherencefactorfreq
            double beamformed = pow(abs_sum, p_param);
            double Nr = beamformed * beamformed;
            if (Dr == 0.0) Dr = 1e-12;
            CF = (1.0 / ncols) * (Nr / Dr);
        }
        double DCoffset = Dr;
        double val = check ? (pDAS * CF) : pDAS;
        pixel_sum += (val * val) - DCoffset;
    }
    d_output[pixel_idx] = pixel_sum;
}

// Main kernel: one thread per pixel in batch
__global__ void PCI_beamform_batch_kernel(
    const cufftDoubleComplex* __restrict__ d_RFFT, // [nfft x ncols]
    const double* __restrict__ d_fftBin,           // [nfft]
    const double* __restrict__ d_all_delays,       // [ncols x total_pixels]
    int nfft, int ncols, int total_pixels,
    int batch_start_idx, int batch_size,
    int FlowerIndex, int freq_band_len,
    double p_param, int check,
    double* __restrict__ d_output                  // [total_pixels]
) {
    int batch_pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_pixel >= batch_size) return;
    int pixel_idx = batch_start_idx + batch_pixel;
    if (pixel_idx >= total_pixels) return;

    const double* delays = d_all_delays + pixel_idx * ncols;
    double pixel_sum = 0.0;

    for (int f = 0; f < freq_band_len; ++f) {
        int r = FlowerIndex + f;

        // --- Build var for this frequency: phase-shifted, all channels ---
        cufftDoubleComplex var[128]; // adjust 128 if needed for max ncols
        for (int c = 0; c < ncols; ++c) {
            double angle = -delays[c] * d_fftBin[r];
            cufftDoubleComplex factor = {cos(angle), sin(angle)};
            int rfft_idx = r + c * nfft;
            cufftDoubleComplex val = d_RFFT[rfft_idx];
            var[c].x = val.x * factor.x - val.y * factor.y;
            var[c].y = val.x * factor.y + val.y * factor.x;
        }

        if (check == 1) {
            // --- pCF branch ---
            // pDAS = pthrootfreq(var, 1)
            double sum_real = 0.0, sum_imag = 0.0;
            for (int c = 0; c < ncols; ++c) {
                double mag = hypot(var[c].x, var[c].y);
                double phase = atan2(var[c].y, var[c].x);
                sum_real += mag * cos(phase);
                sum_imag += mag * sin(phase);
            }
            double pDAS = hypot(sum_real, sum_imag);

            // CF = pthcoherencefactorfreq(var, p_param)
            double sum_real_p = 0.0, sum_imag_p = 0.0, Dr = 0.0;
            for (int c = 0; c < ncols; ++c) {
                double mag = hypot(var[c].x, var[c].y);
                double phase = atan2(var[c].y, var[c].x);
                double comp = pow(mag, 1.0 / p_param);
                sum_real_p += comp * cos(phase);
                sum_imag_p += comp * sin(phase);
                Dr += mag * mag;
            }
            double abs_sum_p = hypot(sum_real_p, sum_imag_p);
            double beamformed = pow(abs_sum_p, p_param);
            double Nr = beamformed * beamformed;
            if (Dr == 0.0) Dr = 1e-12;
            double CF = (1.0 / ncols) * (Nr / Dr);

            // DCoffset = sum(abs(var .* CF).^2, 2)
            double DCoffset = 0.0;
            for (int c = 0; c < ncols; ++c) {
                double real_cf = var[c].x * CF;
                double imag_cf = var[c].y * CF;
                DCoffset += real_cf * real_cf + imag_cf * imag_cf;
            }

            double val = pDAS * CF;
            pixel_sum += (val * val) - DCoffset;
        } else {
            // --- pDAS branch ---
            // pDAS = pthrootfreq(var, p_param)
            double sum_real = 0.0, sum_imag = 0.0;
            for (int c = 0; c < ncols; ++c) {
                double mag = hypot(var[c].x, var[c].y);
                double phase = atan2(var[c].y, var[c].x);
                double comp = pow(mag, 1.0 / p_param);
                sum_real += comp * cos(phase);
                sum_imag += comp * sin(phase);
            }
            double pDAS = pow(hypot(sum_real, sum_imag), p_param);

            // DCoffset = sum(abs(var).^2, 2)
            double DCoffset = 0.0;
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
    if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0])) mexErrMsgIdAndTxt("PCIimagingSparseCUDA:input", "RF_Arr must be a real double matrix.");
    if (!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1])) mexErrMsgIdAndTxt("PCIimagingSparseCUDA:input", "element_Pos_Array_um must be a real double matrix.");
    if (!mxIsScalar(prhs[2]) || !mxIsDouble(prhs[2])) mexErrMsgIdAndTxt("PCIimagingSparseCUDA:input", "speed_Of_Sound_umps must be a real double scalar.");
    if (!mxIsScalar(prhs[4]) || !mxIsDouble(prhs[4])) mexErrMsgIdAndTxt("PCIimagingSparseCUDA:input", "sampling_Freq must be a real double scalar.");
    if (!mxIsDouble(prhs[5]) || mxIsComplex(prhs[5])) mexErrMsgIdAndTxt("PCIimagingSparseCUDA:input", "image_Range_X_um must be a real double vector.");
    if (!mxIsDouble(prhs[6]) || mxIsComplex(prhs[6])) mexErrMsgIdAndTxt("PCIimagingSparseCUDA:input", "image_Range_Z_um must be a real double vector.");
    if (!mxIsScalar(prhs[7]) || !mxIsDouble(prhs[7])) mexErrMsgIdAndTxt("PCIimagingSparseCUDA:input", "p must be a real double scalar.");
    if (!mxIsDouble(prhs[8]) || mxIsComplex(prhs[8])) mexErrMsgIdAndTxt("PCIimagingSparseCUDA:input", "all_delays_matrix must be a real double matrix.");
    if (!mxIsDouble(prhs[9]) || mxGetNumberOfElements(prhs[9]) != 2) mexErrMsgIdAndTxt("PCIimagingSparseCUDA:input", "range_frq must be a double vector of length 2.");
    if (!mxIsDouble(prhs[10]) || !mxIsScalar(prhs[10])) mexErrMsgIdAndTxt("PCIimagingSparseCUDA:input", "check must be a scalar double.");

    // --- Get Inputs & Dimensions ---
    const double* h_RF_Arr = mxGetPr(prhs[0]);
    size_t nfft = mxGetM(prhs[0]);
    size_t ncols = mxGetN(prhs[0]);
    size_t total_pixels = mxGetN(prhs[8]);
    const double* h_all_delays = mxGetPr(prhs[8]);
    double p_param = mxGetScalar(prhs[7]);
    double* range_frq = (double*)mxGetPr(prhs[9]);
    int check = (int)mxGetScalar(prhs[10]);
    double sampling_Freq = mxGetScalar(prhs[4]);
    const double* h_image_Range_X = mxGetPr(prhs[5]);
    size_t numX = mxGetNumberOfElements(prhs[5]);
    const double* h_image_Range_Z = mxGetPr(prhs[6]);
    size_t numZ = mxGetNumberOfElements(prhs[6]);

    // --- Frequency band indices ---
    std::vector<double> fk(nfft);
    for (size_t i = 0; i < nfft; ++i) fk[i] = ((double)i) * sampling_Freq / (double)nfft;
    size_t FlowerIndex = 0, FupperIndex = nfft-1;
    double minDiff = fabs(fk[0] - range_frq[0]);
    for (size_t i = 1; i < nfft; ++i) if (fabs(fk[i] - range_frq[0]) < minDiff) { FlowerIndex = i; minDiff = fabs(fk[i] - range_frq[0]); }
    minDiff = fabs(fk[0] - range_frq[1]);
    for (size_t i = 1; i < nfft; ++i) if (fabs(fk[i] - range_frq[1]) < minDiff) { FupperIndex = i; minDiff = fabs(fk[i] - range_frq[1]); }
    size_t freq_band_len = FupperIndex - FlowerIndex + 1;

    mexPrintf("Starting CUDA PCI Imaging (p=%.2f, check=%d, freq band [%zu:%zu])\n", p_param, check, FlowerIndex, FupperIndex);

    // --- Output Array ---
    plhs[0] = mxCreateDoubleMatrix(numX, numZ, mxREAL);
    double* h_beamformed_Image = mxGetPr(plhs[0]);

    // --- Allocate Device Memory ---
    cufftDoubleComplex *d_RFFT;
    double *d_fftBin;
    double *d_beamformed_Image;
    double *d_all_delays;
    CUDA_CHECK(cudaMalloc(&d_RFFT, nfft * ncols * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_fftBin, nfft * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_beamformed_Image, total_pixels * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_all_delays, ncols * total_pixels * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_all_delays, h_all_delays, ncols * total_pixels * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_beamformed_Image, 0, total_pixels * sizeof(double)));

    // --- Precomputations ---
    int blockSize = 256;
    int gridSize = (nfft + blockSize - 1) / blockSize;
    int binStart = nfft / 2;
    computeFftBinKernel<<<gridSize, blockSize>>>(d_fftBin, nfft, binStart);
    KERNEL_CHECK();

    // Convert RF_Arr to complex and FFT -> d_RFFT
    cufftDoubleComplex* d_RData_complex_temp;
    double* d_RF_Arr_temp;
    size_t total_rf_elements = nfft * ncols;
    CUDA_CHECK(cudaMalloc(&d_RData_complex_temp, total_rf_elements * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_RF_Arr_temp, total_rf_elements * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_RF_Arr_temp, h_RF_Arr, total_rf_elements * sizeof(double), cudaMemcpyHostToDevice));
    int gridSize_rf = (total_rf_elements + blockSize - 1) / blockSize;
    realToComplexKernel<<<gridSize_rf, blockSize>>>(d_RF_Arr_temp, d_RData_complex_temp, total_rf_elements);
    KERNEL_CHECK();
    CUDA_CHECK(cudaFree(d_RF_Arr_temp));
    cufftHandle plan_forward;
    CUFFT_CHECK(cufftPlan1d(&plan_forward, nfft, CUFFT_Z2Z, ncols));
    CUFFT_CHECK(cufftExecZ2Z(plan_forward, d_RData_complex_temp, d_RFFT, CUFFT_FORWARD));
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
    CUDA_CHECK(cudaMemcpy(h_beamformed_Image, d_beamformed_Image, total_pixels * sizeof(double), cudaMemcpyDeviceToHost));

    // --- Cleanup ---
    CUDA_CHECK(cudaFree(d_RFFT));
    CUDA_CHECK(cudaFree(d_fftBin));
    CUDA_CHECK(cudaFree(d_beamformed_Image));
    CUDA_CHECK(cudaFree(d_all_delays));

    mexPrintf("CUDA PCI Imaging complete.\n");
}