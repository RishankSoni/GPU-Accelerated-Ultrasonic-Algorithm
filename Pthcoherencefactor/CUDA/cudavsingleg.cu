#include "mex.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <cfloat> // For FLT_EPSILON or other float constants if needed

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif

#define CUDA_CHECK(err) { cudaError_t err_ = (err); if (err_ != cudaSuccess) { mexPrintf("CUDA error %d at %s:%d: %s\n", err_, __FILE__, __LINE__, cudaGetErrorString(err_)); mexErrMsgIdAndTxt("PCIimagingSparseCUDA:cudaError", cudaGetErrorString(err_)); } }
#define KERNEL_CHECK() { cudaError_t err_ = cudaGetLastError(); if (err_ != cudaSuccess) { mexPrintf("CUDA error %d at %s:%d (kernel launch): %s\n", err_, __FILE__, __LINE__, cudaGetErrorString(err_)); mexErrMsgIdAndTxt("PCIimagingSparseCUDA:cudaError", cudaGetErrorString(err_)); } }
#define CUFFT_CHECK(err) { cufftResult err_ = (err); if (err_ != CUFFT_SUCCESS) { mexPrintf("CUFFT error %d at %s:%d\n", err_, __FILE__, __LINE__); mexErrMsgIdAndTxt("PCIimagingSparseCUDA:cufftError", "CUFFT error"); } }

// --- CUDA Kernels ---

__global__ void computeFftBinKernel(float* fftBin, int nfft, int binStart) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nfft) {
        // Standard FFT shift logic: (idx + N/2) % N, then subtract N/2 to center frequencies around 0.
        // Example: nfft=4. shift=2. binStart=2.
        // idx=0: src=(0+2)%4=2. val=2-2=0. fftBin[0] = 0 (DC)
        // idx=1: src=(1+2)%4=3. val=3-2=1. fftBin[1] = 2*pi*1/4 (fs/4)
        // idx=2: src=(2+2)%4=0. val=0-2=-2. fftBin[2] = 2*pi*(-2)/4 (-fs/2, Nyquist for 0-idx)
        // idx=3: src=(3+2)%4=1. val=1-2=-1. fftBin[3] = 2*pi*(-1)/4 (-fs/4)
        // This makes fftBin[k] the correct angular frequency for the k-th component of an unshifted FFT.
        int shift = (nfft % 2 == 0) ? nfft / 2 : (nfft - 1) / 2;
        int srcIndex = (idx + shift) % nfft; // This is applying fftshift to indices
        float value = (float)srcIndex - (float)binStart; // binStart is nfft/2 or (nfft-1)/2
        fftBin[idx] = (2.0f * M_PIf * value) / (float)nfft;
    }
}

__global__ void realToComplexKernel(const float* realInput, cufftComplex* complexOutput, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        complexOutput[idx].x = realInput[idx];
        complexOutput[idx].y = 0.0f;
    }
}

// Main kernel: one thread per pixel in batch
__global__ void PCI_beamform_batch_kernel(
    const cufftComplex* __restrict__ d_RFFT, // [nfft x ncols] (accessed col-major like: r + c * nfft)
    const float* __restrict__ d_fftBin,           // [nfft]
    const float* __restrict__ d_all_delays,       // [ncols x total_pixels] (accessed col-major like: c + pixel_idx * ncols)
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

    // Delays for the current pixel: d_all_delays is effectively [ncols][total_pixels]
    // So &d_all_delays[pixel_idx * ncols] is the start of the column for this pixel.
    const float* delays_for_this_pixel = d_all_delays + (size_t)pixel_idx * ncols; 
    float pixel_sum = 0.0f;

    // var array holds phase-shifted complex values for all channels for a single frequency
    // Max ncols is 128, so this array is 128 * 8 bytes = 1KB. Stored in registers or local memory.
    cufftComplex var[128]; 

    for (int f = 0; f < freq_band_len; ++f) {
        int r = FlowerIndex + f; // Current frequency bin index

        // --- Build var for this frequency: phase-shifted, all channels ---
        for (int c = 0; c < ncols; ++c) {
            float angle = -delays_for_this_pixel[c] * d_fftBin[r];
            float s, c_val;
            __sincosf(angle, &s, &c_val); // Use __sincosf for combined sin and cos
            cufftComplex factor = {c_val, s};
            
            // d_RFFT is [nfft rows][ncols columns], stored column-major from cufft's perspective (batch of 1D FFTs)
            // Access: d_RFFT[element_in_transform + transform_idx * transform_length]
            // Here, r is element_in_transform, c is transform_idx, nfft is transform_length
            int rfft_idx = r + (size_t)c * nfft; 
            cufftComplex val_fft = d_RFFT[rfft_idx];
            
            // Complex multiplication: (a+ib)(c+id) = (ac-bd) + i(ad+bc)
            var[c].x = val_fft.x * factor.x - val_fft.y * factor.y;
            var[c].y = val_fft.x * factor.y + val_fft.y * factor.x;
        }

        // --- Perform calculations for this frequency bin ---
        float sum_var_x = 0.0f; // Sum of var[c].x
        float sum_var_y = 0.0f; // Sum of var[c].y
        float sum_var_x_p = 0.0f; // Sum for p-norm like terms (real part)
        float sum_var_y_p = 0.0f; // Sum for p-norm like terms (imaginary part)
        float sum_mag_sq = 0.0f;  // Sum of |var[c]|^2

        for (int c = 0; c < ncols; ++c) {
            float vx = var[c].x;
            float vy = var[c].y;

            if (check == 1) { // pCF branch specific sums
                sum_var_x += vx;
                sum_var_y += vy;
            }

            float mag_sq = vx * vx + vy * vy;
            sum_mag_sq += mag_sq;

            if (mag_sq < 1e-20f) { // Threshold for effectively zero magnitude to avoid issues with powf or sqrtf
                 // If mag_sq is zero, vx and vy are zero, comp_norm_vx and comp_norm_vy will be zero.
                 // No explicit addition needed for sum_var_x_p, sum_var_y_p.
            } else if (p_param == 1.0f) {
                sum_var_x_p += vx;
                sum_var_y_p += vy;
            } else {
                float mag = sqrtf(mag_sq);
                float comp_norm_factor; // This will be mag^((1/p)-1)
                if (p_param == 2.0f) { // p=2, exponent is -0.5
                    comp_norm_factor = rsqrtf(mag); // 1.0f / sqrtf(mag)
                } else {
                    comp_norm_factor = powf(mag, (1.0f / p_param) - 1.0f);
                }
                sum_var_x_p += comp_norm_factor * vx;
                sum_var_y_p += comp_norm_factor * vy;
            }
        }
        
        if (check == 1) { // --- pCF branch ---
            float pDAS_mag = hypotf(sum_var_x, sum_var_y); // This is |sum(var[c])|

            float abs_sum_p_val = hypotf(sum_var_x_p, sum_var_y_p);
            float beamformed_mag;
            if (p_param == 1.0f) {
                beamformed_mag = abs_sum_p_val;
            } else if (p_param == 2.0f) {
                beamformed_mag = abs_sum_p_val * abs_sum_p_val;
            } else {
                beamformed_mag = powf(abs_sum_p_val, p_param);
            }
            
            float Nr = beamformed_mag * beamformed_mag; // Numerator for CF part 2
            float Dr = sum_mag_sq;                      // Denominator for CF part 2 (sum of |var[c]|^2)
            if (Dr < 1e-20f) Dr = 1e-20f; // Avoid division by zero
            
            float CF = (1.0f / (float)ncols) * (Nr / Dr);

            // DCoffset = CF^2 * sum(|var[c]|^2) = CF^2 * Dr
            float DCoffset_val = CF * CF * Dr;
            
            float val_term = pDAS_mag * CF;
            pixel_sum += (val_term * val_term) - DCoffset_val;

        } else { // --- pDAS branch ---
            float abs_sum_p_val = hypotf(sum_var_x_p, sum_var_y_p);
            float pDAS_val;
             if (p_param == 1.0f) {
                pDAS_val = abs_sum_p_val;
            } else if (p_param == 2.0f) {
                pDAS_val = abs_sum_p_val * abs_sum_p_val;
            } else {
                pDAS_val = powf(abs_sum_p_val, p_param);
            }

            float DCoffset_val = sum_mag_sq; // Sum of |var[c]|^2
            pixel_sum += (pDAS_val * pDAS_val) - DCoffset_val;
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
    size_t nfft = mxGetM(prhs[0]);    // Number of time samples per RF signal (transform length)
    size_t ncols = mxGetN(prhs[0]);   // Number of RF signals / channels (batch size for FFT)

    if (ncols > 128) { // Check against hardcoded var[128] limit
        mexErrMsgIdAndTxt("PCIimagingSparseCUDA:ncolsTooLarge", "Number of columns (channels) exceeds hardcoded limit of 128.");
    }
    
    const float* h_all_delays = (const float*)mxGetData(prhs[8]);
    // mxGetM(prhs[8]) should be ncols
    // mxGetN(prhs[8]) should be total_pixels
    if (mxGetM(prhs[8]) != ncols) {
         mexErrMsgIdAndTxt("PCIimagingSparseCUDA:delayMatrixDimMismatch", "Number of rows in all_delays_matrix must match number of RF channels (ncols).");
    }
    size_t total_pixels = mxGetN(prhs[8]); 


    float p_param = (float)mxGetScalar(prhs[7]);
    float* range_frq_matlab = (float*)mxGetData(prhs[9]); // mxGetData returns column-major
    float range_frq[2] = {range_frq_matlab[0], range_frq_matlab[1]}; // ensure it's a 2-element array

    int check = (int)mxGetScalar(prhs[10]);
    float sampling_Freq = (float)mxGetScalar(prhs[4]);
    
    // Output dimensions numX, numZ are used for mxCreateNumericMatrix,
    // total_pixels must be numX * numZ.
    // mxGetNumberOfElements gives total elements, regardless of shape.
    size_t numX_elems = mxGetNumberOfElements(prhs[5]);
    size_t numZ_elems = mxGetNumberOfElements(prhs[6]);
    if (total_pixels != numX_elems * numZ_elems) {
        mexPrintf("Warning: total_pixels from all_delays_matrix (%zu) does not match numX (%zu) * numZ (%zu).\n",
                  total_pixels, numX_elems, numZ_elems);
        // Depending on desired strictness, this could be an error.
        // Assuming total_pixels from all_delays_matrix is authoritative for loop bounds,
        // and numX_elems, numZ_elems for output matrix shape.
    }


    // --- Frequency band indices ---
    std::vector<float> fk(nfft);
    for (size_t i = 0; i < nfft; ++i) fk[i] = ((float)i) * sampling_Freq / (float)nfft;
    
    size_t FlowerIndex = 0, FupperIndex = nfft-1; // Default to full band if not found
    float minDiff_low = fabsf(fk[0] - range_frq[0]);
    for (size_t i = 1; i < nfft; ++i) if (fabsf(fk[i] - range_frq[0]) < minDiff_low) { FlowerIndex = i; minDiff_low = fabsf(fk[i] - range_frq[0]); }
    
    float minDiff_high = fabsf(fk[0] - range_frq[1]);
    // Search up to nfft/2 for physical frequencies, or full nfft if appropriate for complex FFT interpretation
    // The original code searches full nfft, implies frequencies might wrap or represent negative side.
    for (size_t i = 1; i < nfft; ++i) if (fabsf(fk[i] - range_frq[1]) < minDiff_high) { FupperIndex = i; minDiff_high = fabsf(fk[i] - range_frq[1]); }
    
    if (FupperIndex < FlowerIndex) { // Ensure indices are ordered
        std::swap(FlowerIndex, FupperIndex);
    }
    size_t freq_band_len = (FupperIndex >= FlowerIndex) ? (FupperIndex - FlowerIndex + 1) : 0;
    if (freq_band_len == 0) {
        mexWarnMsgIdAndTxt("PCIimagingSparseCUDA:emptyFreqBand", "Frequency band is empty or invalid, processing no frequencies.");
    }

    mexPrintf("Starting CUDA PCI Imaging (p=%.2f, check=%d, freq band [%zu:%zu], len=%zu)\n", p_param, check, FlowerIndex, FupperIndex, freq_band_len);

    // --- Output Array ---
    // plhs[0] will be numX_elems x numZ_elems (column-major)
    plhs[0] = mxCreateNumericMatrix(numX_elems, numZ_elems, mxSINGLE_CLASS, mxREAL);
    float* h_beamformed_Image = (float*)mxGetData(plhs[0]);

    // --- Allocate Device Memory ---
    cufftComplex *d_RFFT;          // Holds FFT of RF data: ncols transforms, each of length nfft
    float *d_fftBin;               // Holds angular frequencies for each bin
    float *d_beamformed_Image;     // Output image on device
    float *d_all_delays_dev;       // Delays matrix on device

    size_t RFFT_size_bytes = nfft * ncols * sizeof(cufftComplex);
    size_t fftBin_size_bytes = nfft * sizeof(float);
    size_t beamformedImage_size_bytes = total_pixels * sizeof(float);
    // all_delays_matrix is ncols x total_pixels (MATLAB), so h_all_delays[c + p*ncols]
    // d_all_delays_dev will have same layout. Kernel accesses delays_for_this_pixel[c] where base is col_ptr.
    size_t allDelays_size_bytes = ncols * total_pixels * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_RFFT, RFFT_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_fftBin, fftBin_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_beamformed_Image, beamformedImage_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_all_delays_dev, allDelays_size_bytes)); // Renamed to avoid confusion
    
    CUDA_CHECK(cudaMemcpy(d_all_delays_dev, h_all_delays, allDelays_size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_beamformed_Image, 0, beamformedImage_size_bytes));

    // --- Precomputations ---
    // computeFftBinKernel
    int blockSize_fftBin = 256;
    int gridSize_fftBin = (nfft + blockSize_fftBin - 1) / blockSize_fftBin;
    int binStart = (nfft % 2 == 0) ? nfft / 2 : (nfft - 1) / 2; // Consistent with kernel's expectation
    computeFftBinKernel<<<gridSize_fftBin, blockSize_fftBin>>>(d_fftBin, nfft, binStart);
    KERNEL_CHECK();

    // Convert RF_Arr to complex and FFT -> d_RFFT
    cufftComplex* d_RData_complex_temp; // Temporary for conversion before FFT
    float* d_RF_Arr_temp;               // Temporary for host RF data on device
    size_t total_rf_elements = nfft * ncols;

    CUDA_CHECK(cudaMalloc(&d_RData_complex_temp, total_rf_elements * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_RF_Arr_temp, total_rf_elements * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_RF_Arr_temp, h_RF_Arr, total_rf_elements * sizeof(float), cudaMemcpyHostToDevice));
    
    int blockSize_realToCplx = 256;
    int gridSize_realToCplx = (total_rf_elements + blockSize_realToCplx - 1) / blockSize_realToCplx;
    realToComplexKernel<<<gridSize_realToCplx, blockSize_realToCplx>>>(d_RF_Arr_temp, d_RData_complex_temp, total_rf_elements);
    KERNEL_CHECK();
    CUDA_CHECK(cudaFree(d_RF_Arr_temp)); // Free temp RF array

    cufftHandle plan_forward;
    // Plan for 'ncols' 1D transforms, each of length 'nfft'.
    // Data is expected as [transform0_data, transform1_data, ...]
    // Each transform_data is nfft elements.
    CUFFT_CHECK(cufftPlan1d(&plan_forward, nfft, CUFFT_C2C, ncols));
    CUFFT_CHECK(cufftExecC2C(plan_forward, d_RData_complex_temp, d_RFFT, CUFFT_FORWARD));
    CUFFT_CHECK(cufftDestroy(plan_forward));
    CUDA_CHECK(cudaFree(d_RData_complex_temp)); // Free temp complex data

    // --- Launch Batching Kernel ---
    if (freq_band_len > 0) { // Only launch if there are frequencies to process
        int batch_pixels_max = 8192; // Max pixels per kernel launch (tune for GPU memory/occupancy)
        int threadsPerBlock = 128;   // Threads per block (tune for GPU)

        for (size_t current_pixel_offset = 0; current_pixel_offset < total_pixels; current_pixel_offset += batch_pixels_max) {
            int current_batch_actual_size = std::min((size_t)batch_pixels_max, total_pixels - current_pixel_offset);
            int blocksPerGrid = (current_batch_actual_size + threadsPerBlock - 1) / threadsPerBlock;
            
            PCI_beamform_batch_kernel<<<blocksPerGrid, threadsPerBlock>>>(
                d_RFFT, d_fftBin, d_all_delays_dev, 
                nfft, ncols, total_pixels,
                current_pixel_offset, current_batch_actual_size,
                FlowerIndex, freq_band_len, p_param, check, 
                d_beamformed_Image
            );
            KERNEL_CHECK();
        }
    }


    // --- Copy Result Back to Host ---
    CUDA_CHECK(cudaMemcpy(h_beamformed_Image, d_beamformed_Image, beamformedImage_size_bytes, cudaMemcpyDeviceToHost));

    // --- Cleanup ---
    CUDA_CHECK(cudaFree(d_RFFT));
    CUDA_CHECK(cudaFree(d_fftBin));
    CUDA_CHECK(cudaFree(d_beamformed_Image));
    CUDA_CHECK(cudaFree(d_all_delays_dev));

    mexPrintf("CUDA PCI Imaging complete.\n");
}