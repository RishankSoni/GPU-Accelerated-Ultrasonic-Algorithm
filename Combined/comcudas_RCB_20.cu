#include "mex.h"
#include "matrix.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cufft.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/complex.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>

#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define M_PIf ((float)M_PI)


#define CUDA_CHECK(call)                                            \
    do                                                              \
    {                                                               \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess)                                     \
        {                                                           \
            mexPrintf("CUDA Error in %s at line %d: %s\n",          \
                      __FILE__, __LINE__, cudaGetErrorString(err)); \
            mexErrMsgIdAndTxt("Combcuda:CUDAError", "CUDA error."); \
        }                                                           \
    } while (0)

#define CUBLAS_CHECK(call)                                              \
    do                                                                  \
    {                                                                   \
        cublasStatus_t status = call;                                   \
        if (status != CUBLAS_STATUS_SUCCESS)                            \
        {                                                               \
            mexPrintf("cuBLAS Error in %s at line %d: Status %d\n",     \
                      __FILE__, __LINE__, status);                      \
            mexErrMsgIdAndTxt("Combcuda:cuBLASError", "cuBLAS error."); \
        }                                                               \
    } while (0)

#define CUSOLVER_CHECK(call)                                                \
    do                                                                      \
    {                                                                       \
        cusolverStatus_t status = call;                                     \
        if (status != CUSOLVER_STATUS_SUCCESS)                              \
        {                                                                   \
            mexPrintf("cuSOLVER Error in %s at line %d: Status %d\n",       \
                      __FILE__, __LINE__, status);                          \
            mexErrMsgIdAndTxt("Combcuda:cuSOLVERError", "cuSOLVER error."); \
        }                                                                   \
    } while (0)

#define CUFFT_CHECK(call)                                             \
    do                                                                \
    {                                                                 \
        cufftResult status_cufft = call;                              \
        if (status_cufft != CUFFT_SUCCESS)                            \
        {                                                             \
            mexPrintf("cuFFT Error in %s at line %d: Status %d\n",    \
                      __FILE__, __LINE__, status_cufft);              \
            mexErrMsgIdAndTxt("Combcuda:cuFFTError", "cuFFT error."); \
        }                                                               \
    } while (0)

// --- CUDA Kernels ---
// computeFftBinKernel is unchanged

__global__ void computeFftBinKernel(float *fftBin, int nfft, int binStart)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nfft)
    {
        int shift = (nfft % 2 == 0) ? nfft / 2 : (nfft - 1) / 2;
        int srcIndex = (idx + shift) % nfft;
        float value = static_cast<float>(srcIndex) - static_cast<float>(binStart);
        fftBin[idx] = (2.0f * M_PIf * value) / static_cast<float>(nfft);
    }
}
// shiftDataKernel is unused in batched version, can be removed or kept for other uses

// --- CUDA kernel for batched phase shift (pixels in batch) ---
__global__ void shiftDataKernelBatch(
    const cufftComplex *RFFT, // [Nsamples x Nchannels]
    const float *fftBin,           // [Nsamples]
    const float *delay_batch,      // [Npixels x Nchannels]
    int Nsamples, int Nchannels, int Npixels_in_batch, // Renamed Npixels to Npixels_in_batch for clarity
    cufftComplex *ShiftData // [Npixels_in_batch x Nsamples x Nchannels] (Layout: Pixel1(Chan1(S1..SN), Chan2(S1..SN), ...), Pixel2(...) )
)
{
    int pixel_idx = blockIdx.z; // Index of the pixel in the current batch: 0 to Npixels_in_batch-1
    // idx iterates over one pixel's full data (all channels, all samples for that channel)
    // blockDim.x is for samples, blockDim.y for channels (if 2D grid) OR 1D grid for flat total_samples_channels
    // Current uses 1D grid for idx: blockIdx.x * blockDim.x + threadIdx.x
    int idx_in_pixel_allchannels_samples = blockIdx.x * blockDim.x + threadIdx.x; // Flat index over Nsamples * Nchannels
    int total_samples_channels_per_pixel = Nsamples * Nchannels;

    if (idx_in_pixel_allchannels_samples < total_samples_channels_per_pixel && pixel_idx < Npixels_in_batch)
    {
        // RFFT is [Nsamples x Nchannels] effectively, but passed as flat. Data for channel C then channel C+1.
        // RFFT itself is NOT batched by pixel. It's the global RF data FFT.
        // RFFT[c_channel * Nsamples + r_sample] or RFFT[r_sample + c_channel * Nsamples] depending on storage.
        // If RFFT is Nsamples rows, Nchannels cols (Matlab default), then flat is column major:
        // RFFT[r_sample + c_channel * Nsamples]

        int r_sample = idx_in_pixel_allchannels_samples % Nsamples; // Sample index (freq bin within a channel)
        int c_channel = idx_in_pixel_allchannels_samples / Nsamples; // Channel index

        float angle = -delay_batch[pixel_idx * Nchannels + c_channel] * fftBin[r_sample];
        float cosval = cosf(angle);
        float sinval = sinf(angle);
        
        cufftComplex factor;
        factor.x = cosval;
        factor.y = sinval;
        
        // RFFT is the original FFT of RF data, common for all pixels
        // Accessing RFFT data for current channel and sample:
        cufftComplex val = RFFT[c_channel * Nsamples + r_sample]; // Assuming RFFT is Nchannels blocks of Nsamples contiguous data
                                                                 // If RFFT is Nsamples blocks of Nchannels contiguous data (unlikely for FFT per channel):
                                                                 // cufftComplex val = RFFT[r_sample * Nchannels + c_channel];
                                                                 // Given cufftPlan1d with batch=Nchannels, RFFT is likely Chan1_data, Chan2_data,...
                                                                 // So RFFT[c_channel * Nsamples + r_sample] is correct if d_RFFT_cufft was organized that way.
                                                                 // Let's assume input hostRData_ptr (Nsamples x Nchannels) implies column major.
                                                                 // So RData_complex_h[i] where i = r + c*Nsamples. This is what gets copied.
                                                                 // cufftPlan1d (..., Nchannels) means Nchannels transforms.
                                                                 // idist/odist for this plan would be Nsamples. istride/ostride 1. This is implicit.
                                                                 // So, d_RFFT_cufft will also be laid out as col-major Nsamples x Nchannels.
                                                                 // RFFT[r_sample + c_channel * Nsamples] is the correct access for column-major.

        val = RFFT[r_sample + c_channel * Nsamples]; // Corrected access for column-major Nsamples x Nchannels
        
        cufftComplex res;
        res.x = val.x * factor.x - val.y * factor.y;
        res.y = val.x * factor.y + val.y * factor.x;
        
        // ShiftData is [Npixels_in_batch x Nsamples x Nchannels]
        // Output layout: Pixel1:[Chan1_S1..SN, Chan2_S1..SN, ...], Pixel2:[...], ...
        // This means for a given pixel `pixel_idx`, its data starts at `pixel_idx * total_samples_channels_per_pixel`.
        // Within that pixel's block, channel `c_channel` data starts at `c_channel * Nsamples`.
        // Sample `r_sample` within that channel is at offset `r_sample`.
        // So, index is: pixel_idx * total_samples_channels_per_pixel + c_channel * Nsamples + r_sample
        // This is identical to: pixel_idx * total_samples_channels_per_pixel + idx_in_pixel_allchannels_samples
        ShiftData[pixel_idx * total_samples_channels_per_pixel + idx_in_pixel_allchannels_samples] = res;
    }
}


// --- CUDA kernel for batched real extraction AND SCALING ---
__global__ void extractRealPartAndScaleKernelBatch(
    const cufftComplex *ShiftData_time_complex_unscaled,
    float *realOutput,
    int totalSize_per_pixel, // Nsamples * Nchannels
    int Npixels_in_batch,
    float scale_factor)     // Should be 1.0f / Nsamples
{
    int pixel_idx = blockIdx.z; 
    int idx_in_pixel_data = blockIdx.x * blockDim.x + threadIdx.x; 

    if (idx_in_pixel_data < totalSize_per_pixel && pixel_idx < Npixels_in_batch)
    {
        int global_idx = pixel_idx * totalSize_per_pixel + idx_in_pixel_data;
        realOutput[global_idx] = ShiftData_time_complex_unscaled[global_idx].x * scale_factor;
    }
}


// --- New Kernel for computing delays on device ---
// compute_delay_batch_kernel is unchanged

__global__ void compute_delay_batch_kernel(
    float *d_delay_batch, // Output: [this_batch x Nchannels]
    const float *d_image_Range_X_um_global,
    const float *d_image_Range_Z_um_global,
    const float *d_element_Pos_Array_um_X_flat, // Assumed [X0, Z0, X1, Z1, ...]
    int num_x_points_img,                        // For unflattening pixel_flat_offset
    int Nchannels_dev,
    float speed_Of_Sound_umps_dev,
    float sampling_Freq_dev,
    int pixel_start_offset_batch, // The starting flat pixel index for this batch
    int current_batch_size)
{
    int p_in_batch = blockIdx.x * blockDim.x + threadIdx.x; // Index within the current batch

    if (p_in_batch < current_batch_size)
    {
        int pixel_flat_global = pixel_start_offset_batch + p_in_batch;
        int zi_idx = pixel_flat_global / num_x_points_img;
        int xi_idx = pixel_flat_global % num_x_points_img;

        float current_image_Z_um = d_image_Range_Z_um_global[zi_idx];
        float current_image_X_um = d_image_Range_X_um_global[xi_idx];

        for (int ch = 0; ch < Nchannels_dev; ++ch)
        {
            float dx = current_image_X_um - d_element_Pos_Array_um_X_flat[ch * 2 + 0]; // X_elem
            float dz = current_image_Z_um - d_element_Pos_Array_um_X_flat[ch * 2 + 1]; // Z_elem
            float distance = sqrtf(dx * dx + dz * dz);
            float time_Pt_Along_RF = distance / speed_Of_Sound_umps_dev;
            d_delay_batch[p_in_batch * Nchannels_dev + ch] = -(time_Pt_Along_RF * sampling_Freq_dev);
        }
    }
}

// --- New Kernel for converting real covariance (lower from Ssyrk) to full complex Hermitian ---
// convert_real_lower_to_complex_hermitian_kernel is unchanged

__global__ void convert_real_lower_to_complex_hermitian_kernel(
    const float *R_syrk_lower_col_major,      // Input: Lower triangular part from Ssyrk (N x N)
    thrust::complex<float> *Rs_rcb_col_major, // Output: Full complex Hermitian matrix (N x N)
    int N_dim)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x; // column
    int r = blockIdx.y * blockDim.y + threadIdx.y; // row

    if (r < N_dim && c < N_dim)
    {
        int flat_idx_cm = r + c * N_dim;
        if (r >= c)
        { // Lower triangular part or diagonal
            Rs_rcb_col_major[flat_idx_cm] = thrust::complex<float>(R_syrk_lower_col_major[flat_idx_cm], 0.0f);
        }
        else
        { // Upper triangular part, R_uc = conj(R_lc_transpose) = R_lc_transpose for real
            Rs_rcb_col_major[flat_idx_cm] = thrust::complex<float>(R_syrk_lower_col_major[c + r * N_dim], 0.0f);
        }
    }
}

// --- Robust Capon Functors and Kernels (with minor tolerance adjustments) ---
struct fill_ones_functor_f
{
    __host__ __device__
        thrust::complex<float>
        operator()(const int &) const
    {
        return thrust::complex<float>(1.0f, 0.0f);
    }
};

struct g_transform_op_f
{
    const float x0_lambda; // This is the current lambda for root finding
    g_transform_op_f(float _x0) : x0_lambda(_x0) {}
    __host__ __device__ float operator()(const thrust::tuple<float, thrust::complex<float>> &t) const
    {
        float gamma_val = thrust::get<0>(t); // Eigenvalue gamma_k
        thrust::complex<float> z2_val = thrust::get<1>(t); // V_k^H * abar
        float abs_z2_sq = thrust::norm(z2_val); 
        float den_base = 1.0f + gamma_val * x0_lambda;
        if (fabsf(den_base) < 1e-6f) 
        {
            if (abs_z2_sq < 1e-9f) return 0.0f; // Effectively 0/0 case
            return HUGE_VALF; // abs_z2_sq / (small_num)^2 is HUGE_VALF
        }
        return abs_z2_sq / (den_base * den_base);
    }
};

struct gprime_transform_op_f
{
    const float current_lambda; // This is x0 in Newton iteration (the variable for differentiation)
    gprime_transform_op_f(float _current_lambda) : current_lambda(_current_lambda) {}
    __host__ __device__ float operator()(const thrust::tuple<float, thrust::complex<float>> &t) const
    {
        float gamma_k_val = thrust::get<0>(t); // Eigenvalue gamma_k
        thrust::complex<float> z2_k_val = thrust::get<1>(t); // V_k^H * abar
        float abs_z2_k_sq = thrust::norm(z2_k_val);
        float den_base = 1.0f + gamma_k_val * current_lambda;
        if (fabsf(den_base) < 1e-6f) 
            return 0.0f; // Derivative would blow up, return 0 to stabilize Newton? Or large signed?
                         // Returning 0 if den_base is small might make gprime_sum small and cause issues.
                         // A large value might be better:
                         // float sign = (den_base > 0) ? 1.0f : -1.0f;
                         // if (abs_z2_k_sq < 1e-9f || fabsf(gamma_k_val) < 1e-9f) return 0.0f;
                         // return (-2.0f * gamma_k_val * abs_z2_k_sq) * sign * HUGE_VALF; // simplified
                         // For now, let's keep it 0.0f if blowing up.
        return (-2.0f * gamma_k_val * abs_z2_k_sq) / (den_base * den_base * den_base);
    }
};

struct b_sum_transform_op_f
{
    __host__ __device__ float operator()(const thrust::tuple<float, thrust::complex<float>> &t) const
    {
        float b_diag_val = thrust::get<0>(t);
        thrust::complex<float> temp_vec3_val = thrust::get<1>(t); // (V^H * ahat)_k
        return b_diag_val * thrust::norm(temp_vec3_val); // b_kk * |(V^H*ahat)_k|^2 ; norm is |z|^2
    }
};

struct inv_term_times_z2_lambda_op_f
{
    const float lambda_val_member; // Final lambda
    inv_term_times_z2_lambda_op_f(float _lambda_val) : lambda_val_member(_lambda_val) {}
    __host__ __device__
        thrust::complex<float>
        operator()(const thrust::tuple<float, thrust::complex<float>> &t) const
    {
        float gamma_val = thrust::get<0>(t);
        thrust::complex<float> z2_val = thrust::get<1>(t);
        float den = 1.0f + lambda_val_member * gamma_val;
        if (fabsf(den) < 1e-6f) {
            if (thrust::norm(z2_val) < 1e-9f * thrust::norm(z2_val)) return thrust::complex<float>(0.0f,0.0f); // ~0/0
            // Return z2_val * large_number or (0,0) to prevent NaN propagation
            // This term is (1/(1+lambda*gamma_k)) * z2_k
            // If den is near zero, it means lambda ~ -1/gamma_k. This is problematic.
            return thrust::complex<float>(0.0f, 0.0f); // Heuristic to prevent Inf/NaN
        }
        return (1.0f / den) * z2_val;
    }
};

struct abar_minus_temp_vec_op_f
{
    __host__ __device__
        thrust::complex<float>
        operator()(const thrust::complex<float> &abar_val, const thrust::complex<float> &temp_val) const
    {
        return abar_val - temp_val;
    }
};

struct b_diag_op_f
{
    const float lambda_val; // Final lambda
    b_diag_op_f(float _lambda) : lambda_val(_lambda) {}
    __host__ __device__ float operator()(const float &gamma_val) const
    {
        if (fabsf(lambda_val) < 1e-7f) // If lambda is effectively zero
        {
            // b_k = gamma_k / (1/lambda + gamma_k)^2 -> lambda^2 * gamma_k / (1 + lambda*gamma_k)^2
            // If lambda -> 0, b_k -> 0.
            return 0.0f;
        }
        float lambda_inv = 1.0f / lambda_val;
        float den_term = lambda_inv + gamma_val;
        float den_sq = den_term * den_term;
        if (fabsf(den_sq) < 1e-7f) // (1/lambda + gamma_k)^2 is close to zero
        {
            // This case implies lambda ~ -1/gamma_k.
            // if gamma_val is also near zero, return 0. Else, blows up.
            if (fabsf(gamma_val) < 1e-9f) return 0.0f;
            return HUGE_VALF; // Or a large signed number depending on gamma_val
        }
        return gamma_val / den_sq;
    }
};

__global__ void flip_columns_kernel_f(const cufftComplex *A_in, cufftComplex *A_out, int N_rcb)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N_rcb && col < N_rcb)
    {
        A_out[row + col * N_rcb] = A_in[row + (N_rcb - 1 - col) * N_rcb];
    }
}

// --- RCB Workspace Structure ---
// RCB_Workspace_f is unchanged

struct RCB_Workspace_f
{
    thrust::device_vector<thrust::complex<float>> V_d;
    thrust::device_vector<float> gamma_d;
    thrust::device_vector<thrust::complex<float>> abar_d;
    thrust::device_vector<thrust::complex<float>> z2_d;
    thrust::device_vector<thrust::complex<float>> ahat_d;
    thrust::device_vector<thrust::complex<float>> Rs_copy_d;
    thrust::device_vector<thrust::complex<float>> work_d_eig; // Used to be cuDoubleComplex, now cuComplex
    thrust::device_vector<int> devInfo_d;
    thrust::device_vector<float> temp_terms_d_newton;
    thrust::device_vector<thrust::complex<float>> V_flipped_d;
    thrust::device_vector<thrust::complex<float>> temp_vec1_d;
    thrust::device_vector<thrust::complex<float>> temp_vec2_d;
    thrust::device_vector<float> b_diag_vals_d;
    thrust::device_vector<thrust::complex<float>> temp_vec3_d;
    int lwork_eig_val;

    RCB_Workspace_f(int n_capon, cusolverDnHandle_t cusolverH_rcb_init) : lwork_eig_val(0)
    {
        if (n_capon <= 0) {
             devInfo_d.resize(1); 
             work_d_eig.resize(1); // Minimal allocation
             V_d.resize(1); gamma_d.resize(1); abar_d.resize(1); z2_d.resize(1); ahat_d.resize(1);
             Rs_copy_d.resize(1); temp_terms_d_newton.resize(1); V_flipped_d.resize(1);
             temp_vec1_d.resize(1); temp_vec2_d.resize(1); b_diag_vals_d.resize(1); temp_vec3_d.resize(1);
             return;
        }

        size_t n_sq = static_cast<size_t>(n_capon) * n_capon;
        V_d.resize(n_sq);
        gamma_d.resize(n_capon);
        abar_d.resize(n_capon);
        z2_d.resize(n_capon);
        ahat_d.resize(n_capon);
        Rs_copy_d.resize(n_sq);
        devInfo_d.resize(1);
        temp_terms_d_newton.resize(n_capon);
        V_flipped_d.resize(n_sq);
        temp_vec1_d.resize(n_capon);
        temp_vec2_d.resize(n_capon);
        b_diag_vals_d.resize(n_capon);
        temp_vec3_d.resize(n_capon);
        
        CUSOLVER_CHECK(cusolverDnCheevd_bufferSize(cusolverH_rcb_init, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n_capon,
                                                   (cuComplex *)thrust::raw_pointer_cast(Rs_copy_d.data()), n_capon,
                                                   thrust::raw_pointer_cast(gamma_d.data()), 
                                                   &lwork_eig_val));
        work_d_eig.resize(lwork_eig_val > 0 ? lwork_eig_val : 1); // Ensure work_d_eig has at least size 1
    }
};

// --- Modified Robust Capon Beamforming Function (operates on device data) ---
void robustCaponBeamformingCUDA_onDevice_f(
    const thrust::complex<float> *d_Rs_in, 
    float epsilon, int n_capon,
    float *d_psi_out, 
    cublasHandle_t cublasH, cusolverDnHandle_t cusolverH,
    RCB_Workspace_f &ws) 
{
    if (n_capon <= 0)
    {
        float nan_val = std::numeric_limits<float>::quiet_NaN();
        // d_psi_out is a device pointer
        CUDA_CHECK(cudaMemcpy(d_psi_out, &nan_val, sizeof(float), cudaMemcpyHostToDevice));
        return;
    }

    CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(ws.Rs_copy_d.data()), d_Rs_in,
                          static_cast<size_t>(n_capon) * n_capon * sizeof(thrust::complex<float>),
                          cudaMemcpyDeviceToDevice));

    thrust::counting_iterator<int> first(0);
    thrust::counting_iterator<int> last = first + n_capon;
    thrust::transform(thrust::cuda::par, first, last, ws.abar_d.begin(), fill_ones_functor_f());

    float anorm = sqrtf(static_cast<float>(n_capon));
    if (epsilon < 0.0f) // Should be caught by MATLAB side too
        mexErrMsgIdAndTxt("RobustCaponInternal:EpsilonNegative", "Epsilon cannot be negative.");
    float sqrt_epsilon = sqrtf(epsilon); // If epsilon is very small, sqrt_epsilon can be denormalized or zero
    // Check if sqrt_epsilon is extremely small before division
    float bound_val;
    if (fabsf(sqrt_epsilon) < 1e-9f) { // If sqrt_epsilon is effectively zero
         // Constraint ||a-abar||^2 <= 0 means a=abar. Lambda problem changes.
         // Original code: (anorm > 0) ? HUGE_VALF : 0.0
         // This bound_val is related to lambda search range. Let's keep original logic.
         bound_val = (anorm > 1e-6f) ? HUGE_VALF : 0.0f; 
    } else {
        bound_val = (anorm - sqrt_epsilon) / sqrt_epsilon;
    }


    CUSOLVER_CHECK(cusolverDnCheevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n_capon,
                                    (cuComplex *)thrust::raw_pointer_cast(ws.Rs_copy_d.data()), n_capon,
                                    thrust::raw_pointer_cast(ws.gamma_d.data()), 
                                    (cuComplex *)thrust::raw_pointer_cast(ws.work_d_eig.data()), ws.lwork_eig_val,
                                    thrust::raw_pointer_cast(ws.devInfo_d.data())));
    
    CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(ws.V_d.data()), thrust::raw_pointer_cast(ws.Rs_copy_d.data()),
                          static_cast<size_t>(n_capon) * n_capon * sizeof(thrust::complex<float>),
                          cudaMemcpyDeviceToDevice));

    int devInfo_h;
    CUDA_CHECK(cudaMemcpy(&devInfo_h, thrust::raw_pointer_cast(ws.devInfo_d.data()), sizeof(int), cudaMemcpyDeviceToHost));
    if (devInfo_h != 0)
    {
        // mexPrintf might not be safe in a loop if it flushes stdout extensively. Consider a flag.
        // For now, this is okay for debugging.
        // mexPrintf("RCB cuSOLVER cheevd failed for a pixel: info = %d\n", devInfo_h);
        float nan_val = std::numeric_limits<float>::quiet_NaN();
        CUDA_CHECK(cudaMemcpy(d_psi_out, &nan_val, sizeof(float), cudaMemcpyHostToDevice));
        return;
    }

    thrust::reverse(thrust::cuda::par, ws.gamma_d.begin(), ws.gamma_d.end()); // Eigenvalues now largest to smallest

    if (n_capon > 0) {
        dim3 threadsPerBlock_fc(16, 16); // Standard block size
        dim3 numBlocks_fc((n_capon + threadsPerBlock_fc.x - 1) / threadsPerBlock_fc.x,
                          (n_capon + threadsPerBlock_fc.y - 1) / threadsPerBlock_fc.y);
        flip_columns_kernel_f<<<numBlocks_fc, threadsPerBlock_fc>>>(
            (const cufftComplex *)thrust::raw_pointer_cast(ws.V_d.data()), // V_d is output of cheevd (eigenvectors)
            (cufftComplex *)thrust::raw_pointer_cast(ws.V_flipped_d.data()), // Temporary buffer for flipped V
            n_capon);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize()); // Ensure kernel finishes before V_d is overwritten
        
        // Copy flipped eigenvectors back to V_d
        CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(ws.V_d.data()), 
                              thrust::raw_pointer_cast(ws.V_flipped_d.data()),
                              static_cast<size_t>(n_capon) * n_capon * sizeof(thrust::complex<float>),
                              cudaMemcpyDeviceToDevice));
    }


    cuComplex cu_alpha_blas = {1.0f, 0.0f};
    cuComplex cu_beta_blas = {0.0f, 0.0f};

    if (n_capon > 0) {
        CUBLAS_CHECK(cublasCgemv(cublasH, CUBLAS_OP_C, n_capon, n_capon, // z2 = V^H * abar
                                 &cu_alpha_blas,
                                 (const cuComplex *)thrust::raw_pointer_cast(ws.V_d.data()), n_capon,
                                 (const cuComplex *)thrust::raw_pointer_cast(ws.abar_d.data()), 1,
                                 &cu_beta_blas,
                                 (cuComplex *)thrust::raw_pointer_cast(ws.z2_d.data()), 1));
    }


    float min_gamma_val_h = 0.0f, max_gamma_val_h = 0.0f; // Host copies
    if (n_capon > 0) {
        // gamma_d is reversed: gamma_d[0] is largest, gamma_d[n_capon-1] is smallest
        CUDA_CHECK(cudaMemcpy(&max_gamma_val_h, thrust::raw_pointer_cast(ws.gamma_d.data()), sizeof(float), cudaMemcpyDeviceToHost)); 
        CUDA_CHECK(cudaMemcpy(&min_gamma_val_h, thrust::raw_pointer_cast(ws.gamma_d.data() + (n_capon - 1)), sizeof(float), cudaMemcpyDeviceToHost));
    }


    float lower_bound, upper_bound;
    // fabsf(epsilon) compared to a small number, e.g. 1e-7f times average eigenvalue, or similar relative check
    // For now, using absolute check as in original
    if (fabsf(epsilon) < 1e-7f) 
    {
        // Lambda is non-positive. Max lambda is 0. Min lambda is -1/gamma_max (if gamma_max > 0)
        lower_bound = (fabsf(max_gamma_val_h) < 1e-7f) ? 0.0f : (-1.0f / max_gamma_val_h);
        upper_bound = 0.0f;
    }
    else
    {
        // bound_val = (||abar|| - sqrt(eps)) / sqrt(eps)
        // lambda_extrema = bound_val / gamma_extrema
        lower_bound = (fabsf(max_gamma_val_h) < 1e-7f) ? ((bound_val > 0) ? HUGE_VALF : ((bound_val < 0) ? -HUGE_VALF : 0.0f)) : (bound_val / max_gamma_val_h);
        upper_bound = (fabsf(min_gamma_val_h) < 1e-7f) ? ((bound_val > 0) ? HUGE_VALF : ((bound_val < 0) ? -HUGE_VALF : 0.0f)) : (bound_val / min_gamma_val_h);
    }

    if (lower_bound > upper_bound) // Ensure lower_bound <= upper_bound
        std::swap(lower_bound, upper_bound);
    
    // Constraint: 1 + lambda * gamma_k > 0 for all k. So lambda > -1/gamma_k.
    // Strongest constraint: lambda > -1/gamma_max (since gamma_max is largest positive eigenvalue)
    if (max_gamma_val_h > 1e-7f) 
    {
        // Ensure lambda is slightly greater than -1/gamma_max
        float min_lambda_from_constraint = (-1.0f / max_gamma_val_h);
        lower_bound = std::max(lower_bound, min_lambda_from_constraint + fabsf(min_lambda_from_constraint)*1e-5f + 1e-7f ); // Small positive offset
    }

    float x0 = lower_bound; // Initial guess for lambda
    if (!std::isfinite(x0)) x0 = 0.0f; // Handle Inf/NaN bounds

    // Ensure initial x0 satisfies 1 + lambda*gamma_max > some_epsilon
    if (n_capon > 0 && max_gamma_val_h > 1e-7f && (1.0f + max_gamma_val_h * x0 <= 1e-5f) ) 
    {
        float min_lambda_from_constraint = (-1.0f / max_gamma_val_h);
        x0 = min_lambda_from_constraint + fabsf(min_lambda_from_constraint)*1e-4f + 1e-6f;
    }
    float x1 = x0; // x1 will be the updated lambda

    int iter = 0;
    const int MAX_ITER_NEWTON = 50; // Reduced max iterations for float
    const float TOL_NEWTON_F = 1e-4f; // Relative tolerance for g(lambda)

    if (n_capon > 0 && fabsf(epsilon) > 1e-7f) // Only run Newton if epsilon is meaningfully non-zero
    {
        for (iter = 0; iter < MAX_ITER_NEWTON; ++iter)
        {
            thrust::transform(thrust::cuda::par,
                              thrust::make_zip_iterator(thrust::make_tuple(ws.gamma_d.begin(), ws.z2_d.begin())),
                              thrust::make_zip_iterator(thrust::make_tuple(ws.gamma_d.end(), ws.z2_d.end())),
                              ws.temp_terms_d_newton.begin(), g_transform_op_f(x0));
            float g_sum = thrust::reduce(thrust::cuda::par, ws.temp_terms_d_newton.begin(), ws.temp_terms_d_newton.end(), 0.0f, thrust::plus<float>());
            float g_val = g_sum - epsilon; // We want g_val = 0

            if (fabsf(g_val) < TOL_NEWTON_F * epsilon + 1e-7f) // Converged if g_val is small enough
                break;

            thrust::transform(thrust::cuda::par,
                              thrust::make_zip_iterator(thrust::make_tuple(ws.gamma_d.begin(), ws.z2_d.begin())),
                              thrust::make_zip_iterator(thrust::make_tuple(ws.gamma_d.end(), ws.z2_d.end())),
                              ws.temp_terms_d_newton.begin(), gprime_transform_op_f(x0));
            float gprime_sum = thrust::reduce(thrust::cuda::par, ws.temp_terms_d_newton.begin(), ws.temp_terms_d_newton.end(), 0.0f, thrust::plus<float>());

            if (fabsf(gprime_sum) < 1e-7f) { // If derivative is too small, Newton step is unstable
                // mexPrintf("Warning: RCB Newton gprime_sum too small (%g) at iter %d, lambda_try %g. Stopping.\n", gprime_sum, iter, x0);
                break; 
            }
            x1 = x0 - (g_val / gprime_sum);

            // Ensure x1 stays within valid range (1 + lambda*gamma_max > some_epsilon)
            if (max_gamma_val_h > 1e-7f && (1.0f + max_gamma_val_h * x1 <= 1e-5f))
            {
                // If x1 invalid, try halfway between x0 and the boundary
                float min_lambda_from_constraint = (-1.0f / max_gamma_val_h);
                x1 = x0 * 0.5f + (min_lambda_from_constraint + fabsf(min_lambda_from_constraint)*1e-4f + 1e-6f) * 0.5f;
            }
            
            if (fabsf(x1 - x0) < TOL_NEWTON_F * (1e-5f + fabsf(x1))) // Converged if step size is small
                break;
            x0 = x1;
        }
        // if (iter == MAX_ITER_NEWTON)
            // mexPrintf("Warning: RCB Newton did not converge for a pixel. Lambda = %g.\n", x1);
    }
    else if (n_capon > 0) // Epsilon is effectively zero
    {
        // Check g(0). If g(0) <= epsilon (i.e. g(0) <= ~0), then lambda=0 is the solution.
        thrust::transform(thrust::cuda::par,
                          thrust::make_zip_iterator(thrust::make_tuple(ws.gamma_d.begin(), ws.z2_d.begin())),
                          thrust::make_zip_iterator(thrust::make_tuple(ws.gamma_d.end(), ws.z2_d.end())),
                          ws.temp_terms_d_newton.begin(), g_transform_op_f(0.0f));
        float g_at_zero = thrust::reduce(thrust::cuda::par, ws.temp_terms_d_newton.begin(), ws.temp_terms_d_newton.end(), 0.0f, thrust::plus<float>());
        
        if (g_at_zero <= epsilon + 1e-7f) 
            x1 = 0.0f;
        else // g(0) > epsilon. Lambda must be negative. Smallest possible lambda (closest to -1/gamma_max)
        {
            if (max_gamma_val_h > 1e-7f) {
                 float min_lambda_from_constraint = (-1.0f / max_gamma_val_h);
                 x1 = min_lambda_from_constraint + fabsf(min_lambda_from_constraint)*1e-4f + 1e-6f;
            } else { // max_gamma_val_h is zero (e.g. zero matrix Rs)
                 x1 = 0.0f; // No constraint from eigenvalues.
            }
        }
    } else { // n_capon == 0 (already handled, but defensive)
        x1 = 0.0f;
    }

    float final_lambda_val = x1;
    if (!std::isfinite(final_lambda_val))
    {
        // mexPrintf("Warning: RCB final lambda is not finite (%g). Defaulting to 0.\n", final_lambda_val);
        final_lambda_val = 0.0f; // Default to a safe value
    }
    // Final check on lambda validity
    if (n_capon > 0 && max_gamma_val_h > 1e-7f && (1.0f + max_gamma_val_h * final_lambda_val <= 1e-5f))
    {
        // mexPrintf("Warning: RCB final lambda %g is invalid. Adjusting.\n", final_lambda_val);
        float min_lambda_from_constraint = (-1.0f / max_gamma_val_h);
        final_lambda_val = min_lambda_from_constraint + fabsf(min_lambda_from_constraint)*1e-4f + 1e-6f;
    }


    float host_psi_val_calc;
    if (n_capon > 0) {
        // temp_vec1 = (I + lambda*Gamma)^-1 * z2
        thrust::transform(thrust::cuda::par,
                          thrust::make_zip_iterator(thrust::make_tuple(ws.gamma_d.begin(), ws.z2_d.begin())),
                          thrust::make_zip_iterator(thrust::make_tuple(ws.gamma_d.end(), ws.z2_d.end())),
                          ws.temp_vec1_d.begin(), inv_term_times_z2_lambda_op_f(final_lambda_val));

        // temp_vec2 = V * temp_vec1 = V * (I + lambda*Gamma)^-1 * V^H * abar
        CUBLAS_CHECK(cublasCgemv(cublasH, CUBLAS_OP_N, n_capon, n_capon,
                                 &cu_alpha_blas, (const cuComplex *)thrust::raw_pointer_cast(ws.V_d.data()), n_capon,
                                 (const cuComplex *)thrust::raw_pointer_cast(ws.temp_vec1_d.data()), 1,
                                 &cu_beta_blas, (cuComplex *)thrust::raw_pointer_cast(ws.temp_vec2_d.data()), 1));
        
        // ahat = abar - lambda * temp_vec2 = abar - lambda * V*(I+lambda*Gamma)^-1*V^H*abar
        // The formula for ahat is just abar - temp_vec2 if temp_vec2 = lambda * V * ...
        // From paper, a_hat = (R + lambda*I)^-1 * abar (if abar is steering vector).
        // Or using eigendecomposition: a_hat = V * (Gamma + lambda*I)^-1 * V^H * abar
        // Here, constraint is on ||a-abar||, derivation leads to a_hat = abar - lambda*R_inv*abar (approx if R used)
        // The specific RCB variant (e.g. Gu, Stoica) matters for exact a_hat formula.
        // The provided code seems to implement: a_hat = abar - V * diag(lambda/(1+lambda*gamma_k)) * V^H * abar
        // OR a_hat = V * diag(1/(1+lambda*gamma_k)) * V^H * abar
        // Given `ahat_d = abar_d - temp_vec2_d` and `temp_vec2_d = V * (diag(1/(1+lambda*gamma_k))) * z2_d`
        // So `ahat_d = abar_d - V * diag(1/(1+lambda*gamma_k)) * V^H * abar_d`. This is one form.

        // For power: P = 1 / (ahat^H * R_inv * ahat) if ahat is steering vec.
        // Or P = ahat^H * R * ahat if ahat is weight vector.
        // The reference material for this specific RCB psi calculation would be needed to verify the exact formulas for ahat and b_scalar.
        // Assuming the formulas from the double version are what's intended:
        // ahat = abar - temp_vec2
        thrust::transform(thrust::cuda::par, ws.abar_d.begin(), ws.abar_d.end(), ws.temp_vec2_d.begin(), ws.ahat_d.begin(), abar_minus_temp_vec_op_f());
        
        // b_diag_vals_d are B_kk = gamma_k / (1/lambda + gamma_k)^2
        thrust::transform(thrust::cuda::par, ws.gamma_d.begin(), ws.gamma_d.end(), ws.b_diag_vals_d.begin(), b_diag_op_f(final_lambda_val));

        // temp_vec3 = V^H * ahat
        CUBLAS_CHECK(cublasCgemv(cublasH, CUBLAS_OP_C, n_capon, n_capon,
                                 &cu_alpha_blas, (const cuComplex *)thrust::raw_pointer_cast(ws.V_d.data()), n_capon,
                                 (const cuComplex *)thrust::raw_pointer_cast(ws.ahat_d.data()), 1,
                                 &cu_beta_blas, (cuComplex *)thrust::raw_pointer_cast(ws.temp_vec3_d.data()), 1));
        
        // b_scalar = sum (B_kk * |(V^H*ahat)_k|^2) = ahat^H * V * diag(B_kk) * V^H * ahat
        // Note: b_sum_transform_op_f used thrust::norm which is |z|^2. This is correct.
        thrust::transform(thrust::cuda::par,
                          thrust::make_zip_iterator(thrust::make_tuple(ws.b_diag_vals_d.begin(), ws.temp_vec3_d.begin())),
                          thrust::make_zip_iterator(thrust::make_tuple(ws.b_diag_vals_d.end(), ws.temp_vec3_d.end())),
                          ws.temp_terms_d_newton.begin(), b_sum_transform_op_f()); // Using temp_terms_d_newton as scratchpad
        float b_scalar = thrust::reduce(thrust::cuda::par, ws.temp_terms_d_newton.begin(), ws.temp_terms_d_newton.end(), 0.0f, thrust::plus<float>());

        // Psi = N^2 / b_scalar (or similar, depends on RCB definition)
        if (fabsf(b_scalar) < 1e-9f) // Adjusted from 1e-8f, then 1e-12f
        {
            // If b_scalar is extremely small, result is Inf. Otherwise, large finite.
            host_psi_val_calc = (fabsf(b_scalar) < 1e-25f) ? std::numeric_limits<float>::infinity() : (static_cast<float>(n_capon) * static_cast<float>(n_capon)) / b_scalar;
        }
        else
        {
            host_psi_val_calc = (static_cast<float>(n_capon) * static_cast<float>(n_capon)) / b_scalar;
        }

        if (!std::isfinite(host_psi_val_calc)) { // If Inf or NaN
            // mexPrintf("Warning: RCB psi is Inf/NaN. b_scalar=%g, lambda=%g\n", b_scalar, final_lambda_val);
            // Potentially set to a large value or specific NaN marker if needed.
            // If it's Inf, it might be correct. If NaN, something went wrong.
            if (std::isinf(host_psi_val_calc) && host_psi_val_calc < 0) { // Negative infinity
                 host_psi_val_calc = 0.0f; // Cap negative infinity, often an artifact
            }
        } else if (host_psi_val_calc < 0) {
            // mexPrintf("Warning: RCB psi is negative (%g). Clamping to 0. b_scalar=%g, lambda=%g\n", host_psi_val_calc, b_scalar, final_lambda_val);
            host_psi_val_calc = 0.0f; // Power should be non-negative
        }


    } else { // n_capon == 0
        host_psi_val_calc = std::numeric_limits<float>::quiet_NaN();
    }
    
    CUDA_CHECK(cudaMemcpy(d_psi_out, &host_psi_val_calc, sizeof(float), cudaMemcpyHostToDevice));
}


// --- Main CUDA MEX Entry Point: Batched PCI Imaging ---
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Inputs:
    // 0: RF_Arr (Nsamples x Nchannels, real single)
    // 1: element_Pos_Array_um_X (2 x Nchannels, real single)
    // 2: speed_Of_Sound_umps (scalar, single)
    // 3: RF_Start_Time (scalar, single, unused)
    // 4: sampling_Freq (scalar, single)
    // 5: image_Range_X_um (vector, single)
    // 6: image_Range_Z_um (vector, single)
    // 7: p_epsilon (scalar, single)
    // 8: n_elements (scalar, int/single)

    if (nrhs != 9)
    {
        mexErrMsgIdAndTxt("Combcuda:nrhs", "Nine inputs required: RF_Arr, element_Pos_Array_um_X, speed_Of_Sound_umps, RF_Start_Time, sampling_Freq, image_Range_X_um, image_Range_Z_um, p_epsilon, n_elements.");
    }
    if (nlhs > 1)
    {
        mexErrMsgIdAndTxt("Combcuda:nlhs", "One output argument (beamformed_Image_out) allowed.");
    }

    // --- Parse Inputs ---
    if (!mxIsSingle(prhs[0]) || mxIsComplex(prhs[0]))
        mexErrMsgIdAndTxt("Combcuda:RF_ArrNotRealSingle", "Input RF_Arr must be a real single matrix.");
    mwSize Nsamples_mw = mxGetM(prhs[0]);
    mwSize Nchannels_mw = mxGetN(prhs[0]);
    int Nsamples = static_cast<int>(Nsamples_mw);
    int Nchannels = static_cast<int>(Nchannels_mw);
    const float *hostRData_ptr = (const float*)mxGetData(prhs[0]);

    if (!mxIsSingle(prhs[1]) || mxIsComplex(prhs[1]) || mxGetM(prhs[1]) != 2 || mxGetN(prhs[1]) != Nchannels)
        mexErrMsgIdAndTxt("Combcuda:element_Pos_Array_um_X", "element_Pos_Array_um_X must be 2 x Nchannels real single matrix.");
    const float *element_Pos_Array_um_X = (const float*)mxGetData(prhs[1]);

    float speed_Of_Sound_umps = static_cast<float>(mxGetScalar(prhs[2]));
    if (speed_Of_Sound_umps <= 0.0f)
        mexErrMsgIdAndTxt("Combcuda:SoSInvalid", "speed_Of_Sound_umps must be positive.");

    // prhs[3] RF_Start_Time is unused

    float sampling_Freq = static_cast<float>(mxGetScalar(prhs[4]));
    if (sampling_Freq <= 0.0f)
        mexErrMsgIdAndTxt("Combcuda:fsInvalid", "sampling_Freq must be positive.");

    if (!mxIsSingle(prhs[5]) || mxIsComplex(prhs[5]) || mxGetM(prhs[5]) * mxGetN(prhs[5]) < 1)
        mexErrMsgIdAndTxt("Combcuda:image_Range_X_um", "image_Range_X_um must be a non-empty real single vector.");
    int num_x_points = static_cast<int>(mxGetNumberOfElements(prhs[5]));
    const float *image_Range_X_um = (const float*)mxGetData(prhs[5]);

    if (!mxIsSingle(prhs[6]) || mxIsComplex(prhs[6]) || mxGetM(prhs[6]) * mxGetN(prhs[6]) < 1)
        mexErrMsgIdAndTxt("Combcuda:image_Range_Z_um", "image_Range_Z_um must be a non-empty real single vector.");
    int num_z_points = static_cast<int>(mxGetNumberOfElements(prhs[6]));
    const float *image_Range_Z_um = (const float*)mxGetData(prhs[6]);

    float p_epsilon = static_cast<float>(mxGetScalar(prhs[7]));
    // Epsilon check moved to RCB function for per-pixel handling if needed, or keep global check here.
    // if (p_epsilon < 0.0f) mexErrMsgIdAndTxt("Combcuda:p_epsilon", "p_epsilon must be non-negative.");


    int n_elements = static_cast<int>(mxGetScalar(prhs[8])); // n_elements is number of channels for Capon
    if (n_elements != Nchannels)
        mexErrMsgIdAndTxt("Combcuda:n_elements", "n_elements must match number of RF_Arr columns (Nchannels).");
     if (n_elements <= 0) // This implies Nchannels <= 0
        mexErrMsgIdAndTxt("Combcuda:n_elements_positive", "n_elements (and Nchannels) must be positive.");


    // --- Output Allocation ---
    mwSize dims[2] = {(mwSize)num_z_points, (mwSize)num_x_points}; // Output image ZxX
    plhs[0] = mxCreateNumericArray(2, dims, mxSINGLE_CLASS, mxREAL);
    float *beamformed_Image_out = (float*)mxGetData(plhs[0]);
    int Npixels = num_x_points * num_z_points;

    if (Nsamples == 0 || Npixels == 0 || Nchannels == 0) { 
        for (int i = 0; i < Npixels * num_z_points; ++i) beamformed_Image_out[i] = std::numeric_limits<float>::quiet_NaN();
        // If Npixels is product of num_x and num_z, then just Npixels for loop.
        // Image is num_z_points * num_x_points.
        size_t totalOutputElements = static_cast<size_t>(num_z_points) * num_x_points;
        for (size_t i = 0; i < totalOutputElements; ++i) beamformed_Image_out[i] = std::numeric_limits<float>::quiet_NaN();
        // Short-circuit if no work to do.
        // Free any resources if any were allocated before this check. (None yet)
        return;
    }

    // --- Precompute FFT bin (shared for all pixels) ---
    std::vector<float> fftBin_h(Nsamples);
    { // Nsamples > 0 here
        int binStart = Nsamples / 2; // Center bin for fftshift
        for (int idx = 0; idx < Nsamples; ++idx) {
            int shift = (Nsamples % 2 == 0) ? Nsamples / 2 : (Nsamples - 1) / 2;
            int srcIndex = (idx + shift) % Nsamples; // Apply fftshift logic
            float value = static_cast<float>(srcIndex) - static_cast<float>(binStart);
            fftBin_h[idx] = (2.0f * M_PIf * value) / static_cast<float>(Nsamples);
        }
    }
    float *d_fftBin;
    CUDA_CHECK(cudaMalloc((void **)&d_fftBin, Nsamples * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_fftBin, fftBin_h.data(), Nsamples * sizeof(float), cudaMemcpyHostToDevice));

    // Convert real input RF data to complex (zero imaginary part)
    std::vector<cufftComplex> RData_complex_h(static_cast<size_t>(Nsamples) * Nchannels);
    for (size_t r = 0; r < Nsamples; ++r) {
        for (size_t c = 0; c < Nchannels; ++c) {
            // Input hostRData_ptr is column-major (Nsamples x Nchannels)
            RData_complex_h[c * Nsamples + r].x = hostRData_ptr[r + c * Nsamples_mw]; // Correct for col-major
            RData_complex_h[c * Nsamples + r].y = 0.0f;
        }
    }
    // The above loop for RData_complex_h populates it as Nchannels blocks of Nsamples (Chan1_S1..SN, Chan2_S1..SN, ...).
    // This is suitable for cufftPlan1d with batch=Nchannels and dist=Nsamples.

    cufftComplex *d_RData_cufft; // Holds RF data on device, complexified
    CUDA_CHECK(cudaMalloc((void **)&d_RData_cufft, sizeof(cufftComplex) * Nsamples * Nchannels));
    CUDA_CHECK(cudaMemcpy(d_RData_cufft, RData_complex_h.data(), sizeof(cufftComplex) * Nsamples * Nchannels, cudaMemcpyHostToDevice));

    // FFT of RF data (Nchannels individual 1D FFTs of length Nsamples)
    cufftHandle plan_fft_rf;
    // Plan for Nchannels transforms, each of length Nsamples.
    // Data for each transform is contiguous (istride=ostride=1).
    // Distance between start of Nchannels transforms is Nsamples.
    CUFFT_CHECK(cufftPlanMany(&plan_fft_rf, 1, &Nsamples, 
                              nullptr, 1, Nsamples, // input embed, istride, idist
                              nullptr, 1, Nsamples, // output embed, ostride, odist
                              CUFFT_C2C, Nchannels)); // type, batch
    
    cufftComplex *d_RFFT_cufft; // Holds FFT of RF data on device
    CUDA_CHECK(cudaMalloc((void **)&d_RFFT_cufft, sizeof(cufftComplex) * Nsamples * Nchannels));
    CUFFT_CHECK(cufftExecC2C(plan_fft_rf, d_RData_cufft, d_RFFT_cufft, CUFFT_FORWARD));
    CUFFT_CHECK(cufftDestroy(plan_fft_rf));

    // cuBLAS and cuSOLVER handles
    cublasHandle_t cublasH_sdelay, cublasH_rcb; // sdelay for Ssyrk, rcb for RCB Cgemv
    cusolverDnHandle_t cusolverH_rcb;
    CUBLAS_CHECK(cublasCreate(&cublasH_sdelay));
    CUBLAS_CHECK(cublasCreate(&cublasH_rcb));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH_rcb));

    // RCB Workspace (n_elements is Nchannels)
    RCB_Workspace_f rcb_workspace(n_elements, cusolverH_rcb); 

    // Device copies of imaging parameters
    float *d_element_Pos_Array_um_X_flat_gpu, *d_image_Range_X_um_gpu, *d_image_Range_Z_um_gpu;
    CUDA_CHECK(cudaMalloc((void **)&d_element_Pos_Array_um_X_flat_gpu, 2 * Nchannels * sizeof(float))); // 2 for X,Z coords
    CUDA_CHECK(cudaMemcpy(d_element_Pos_Array_um_X_flat_gpu, element_Pos_Array_um_X, 2 * Nchannels * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void **)&d_image_Range_X_um_gpu, num_x_points * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_image_Range_X_um_gpu, image_Range_X_um, num_x_points * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void **)&d_image_Range_Z_um_gpu, num_z_points * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_image_Range_Z_um_gpu, image_Range_Z_um, num_z_points * sizeof(float), cudaMemcpyHostToDevice));

    // Batch processing setup
    int batch_size_pixels = 128; // Number of pixels to process in one GPU batch
    if (Npixels < batch_size_pixels) batch_size_pixels = Npixels; // Adjust if Npixels is small
    int total_batches = (Npixels + batch_size_pixels - 1) / batch_size_pixels;

    int totalElements_per_pixel_sdelay = Nsamples * Nchannels; // Samples x Channels for one pixel's shifted data
    
    // Device buffers for batch processing
    cufftComplex *d_ShiftData_cufft_batch; // Holds frequency-shifted data for a batch of pixels
    cufftComplex *d_ShiftData_time_cufft_batch; // Holds IFFT of above (time-domain shifted data)
    float *d_realShiftDataTime_sdelay_batch; // Real part of IFFT output, scaled
    float *d_delay_batch; // Delays for current batch of pixels [batch_size_pixels x Nchannels]

    CUDA_CHECK(cudaMalloc((void **)&d_ShiftData_cufft_batch, sizeof(cufftComplex) * batch_size_pixels * totalElements_per_pixel_sdelay));
    CUDA_CHECK(cudaMalloc((void **)&d_ShiftData_time_cufft_batch, sizeof(cufftComplex) * batch_size_pixels * totalElements_per_pixel_sdelay));
    CUDA_CHECK(cudaMalloc((void **)&d_realShiftDataTime_sdelay_batch, sizeof(float) * batch_size_pixels * totalElements_per_pixel_sdelay));
    CUDA_CHECK(cudaMalloc((void **)&d_delay_batch, sizeof(float) * batch_size_pixels * Nchannels));

    // IFFT plan for batched, shifted data
    // We have `batch_size_pixels * Nchannels` individual 1D IFFTs of length `Nsamples`.
    cufftHandle plan_ifft_batch;
    int n_ifft[] = {Nsamples}; // Length of each 1D IFFT
    // Data layout for d_ShiftData_cufft_batch: P1(C1(S...S)C2(S...S)...)P2(...)...
    // Each 1D IFFT (length Nsamples) is contiguous.
    // Distance between start of one IFFT and the next (e.g. P1C1 to P1C2, or P1CN to P2C1) is Nsamples.
    CUFFT_CHECK(cufftPlanMany(&plan_ifft_batch, 1, n_ifft,
                              nullptr, 1, Nsamples, // input embed, istride, idist
                              nullptr, 1, Nsamples, // output embed, ostride, odist
                              CUFFT_C2C, batch_size_pixels * Nchannels)); // type, total # of FFTs in one call

    // Device buffers for covariance matrices and RCB outputs for the batch
    float *d_R_syrk_batch_buffer; // Stores all real R_syrk matrices (lower tri) for the batch
    CUDA_CHECK(cudaMalloc((void **)&d_R_syrk_batch_buffer, sizeof(float) * batch_size_pixels * Nchannels * Nchannels));

    thrust::complex<float> *d_Rs_rcb_batch_buffer; // Stores all complex Hermitian Rs_rcb matrices
    CUDA_CHECK(cudaMalloc((void **)&d_Rs_rcb_batch_buffer, sizeof(thrust::complex<float>) * batch_size_pixels * Nchannels * Nchannels));

    float *d_psi_out_batch; // RCB output (power) for each pixel in the batch
    CUDA_CHECK(cudaMalloc((void **)&d_psi_out_batch, sizeof(float) * batch_size_pixels));
    std::vector<float> h_psi_out_batch(batch_size_pixels); // Host buffer for batch results

    // --- Main Processing Loop (Batch over Pixels) ---
    for (int batch_idx = 0; batch_idx < total_batches; ++batch_idx)
    {
        int pixel_start_offset_global = batch_idx * batch_size_pixels; // Global flat index of the first pixel in this batch
        int current_batch_numpixels = std::min(batch_size_pixels, Npixels - pixel_start_offset_global); // Actual num pixels in this batch

        if (current_batch_numpixels <= 0) continue;

        // 1. Compute delays for the current batch of pixels
        dim3 block_delay(256); // Threads per block for delay calculation
        dim3 grid_delay((current_batch_numpixels + block_delay.x - 1) / block_delay.x);
        compute_delay_batch_kernel<<<grid_delay, block_delay>>>(
            d_delay_batch, d_image_Range_X_um_gpu, d_image_Range_Z_um_gpu, d_element_Pos_Array_um_X_flat_gpu,
            num_x_points, Nchannels, speed_Of_Sound_umps, sampling_Freq,
            pixel_start_offset_global, current_batch_numpixels);
        CUDA_CHECK(cudaGetLastError());

        // 2. Apply phase shifts in frequency domain (batched over pixels)
        // Kernel launch configuration:
        // Grid Dim Z: current_batch_numpixels (each block processes one pixel's data)
        // Grid Dim X: for elements within a pixel (Nsamples*Nchannels)
        // Block Dim X: threads for elements within a pixel
        dim3 threads_per_block_shift(256); 
        dim3 num_blocks_shift( (totalElements_per_pixel_sdelay + threads_per_block_shift.x -1) / threads_per_block_shift.x, // X-dim for data within a pixel
                                1,                                                                                            // Y-dim (unused)
                                current_batch_numpixels);                                                                     // Z-dim for pixels in batch
        
        // Correction to RFFT indexing in shiftDataKernelBatch was made.
        // d_RFFT_cufft is Nsamples x Nchannels, column major.
        // Each thread in kernel: idx_in_pixel_allchannels_samples from 0 to Ns*Nc-1
        // r_sample = idx % Ns; c_channel = idx / Ns.
        // RFFT access: RFFT[r_sample + c_channel * Ns]
        shiftDataKernelBatch<<<num_blocks_shift, threads_per_block_shift>>>(
            d_RFFT_cufft, d_fftBin, d_delay_batch,
            Nsamples, Nchannels, current_batch_numpixels, 
            d_ShiftData_cufft_batch);
        CUDA_CHECK(cudaGetLastError());
        
        // 3. IFFT the phase-shifted data
        // If current_batch_numpixels < batch_size_pixels, cufftExecC2C will process only the relevant part
        // if the plan was made for batch_size_pixels * Nchannels.
        // Or, more robustly, make plan for max, and if current_batch_numpixels is smaller,
        // one might need to call cufftExec with adjusted pointers or make a specific plan.
        // For now, assuming cufftExecC2C with a larger plan on smaller data subset is okay.
        // The plan is for batch_size_pixels * Nchannels.
        // If current_batch_numpixels is less, we only care about the first current_batch_numpixels * Nchannels transforms.
        // This requires that plan_ifft_batch is for batch_size_pixels*Nchannels and not a smaller number.
        CUFFT_CHECK(cufftExecC2C(plan_ifft_batch, 
                                 d_ShiftData_cufft_batch, // Input: up to batch_size_pixels * totalElements_per_pixel_sdelay
                                 d_ShiftData_time_cufft_batch, // Output
                                 CUFFT_INVERSE));

        // 4. Extract real part and scale by 1/Nsamples
        float inv_Nsamples_scale = 1.0f / static_cast<float>(Nsamples);
        // Use same kernel launch config as shiftDataKernelBatch (num_blocks_shift, threads_per_block_shift)
        extractRealPartAndScaleKernelBatch<<<num_blocks_shift, threads_per_block_shift>>>(
            d_ShiftData_time_cufft_batch, // Unscaled IFFT output
            d_realShiftDataTime_sdelay_batch, // Output: scaled real part
            totalElements_per_pixel_sdelay, 
            current_batch_numpixels,
            inv_Nsamples_scale);
        CUDA_CHECK(cudaGetLastError());

        // --- Per-pixel operations within the batch (covariance, RCB) ---
        for (int p_batch_idx = 0; p_batch_idx < current_batch_numpixels; ++p_batch_idx)
        {
            // 5. Compute spatial covariance matrix R_syrk = (1/Nsamples) * Y^T * Y
            // Y is d_realShiftDataTime_sdelay_batch for this pixel (Nsamples x Nchannels)
            const float alpha_syrk = 1.0f / static_cast<float>(Nsamples); // Scaling for sample covariance
            const float beta_syrk = 0.0f;
            
            // Pointer to current pixel's data in the batch buffer for Y
            float *current_d_realShiftDataTime_pixel = d_realShiftDataTime_sdelay_batch + p_batch_idx * totalElements_per_pixel_sdelay;
            // Pointer to current pixel's output covariance matrix R_syrk
            float *current_d_R_syrk_pixel = d_R_syrk_batch_buffer + p_batch_idx * Nchannels * Nchannels;

            // A is (Nsamples x Nchannels), LDA=Nsamples. Output C is (Nchannels x Nchannels), LDC=Nchannels.
            // op(A) is A^T (Nchannels x Nsamples). Result is A^T * A.
            CUBLAS_CHECK(cublasSsyrk(cublasH_sdelay, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                                     Nchannels, Nsamples, &alpha_syrk,
                                     current_d_realShiftDataTime_pixel, Nsamples, 
                                     &beta_syrk, current_d_R_syrk_pixel, Nchannels));

            // 6. Convert real lower-triangular R_syrk to full complex Hermitian Rs_rcb
            thrust::complex<float> *current_d_Rs_rcb_pixel = d_Rs_rcb_batch_buffer + p_batch_idx * Nchannels * Nchannels;
            dim3 block_conv(16, 16);
            dim3 grid_conv((Nchannels + block_conv.x - 1) / block_conv.x, (Nchannels + block_conv.y - 1) / block_conv.y);
            convert_real_lower_to_complex_hermitian_kernel<<<grid_conv, block_conv>>>(
                current_d_R_syrk_pixel, current_d_Rs_rcb_pixel, Nchannels);
            CUDA_CHECK(cudaGetLastError());

            // 7. Perform Robust Capon Beamforming for the current pixel
            robustCaponBeamformingCUDA_onDevice_f(
                current_d_Rs_rcb_pixel, p_epsilon, n_elements, // n_elements is Nchannels
                d_psi_out_batch + p_batch_idx, // Output for this pixel
                cublasH_rcb, cusolverH_rcb, rcb_workspace);
            // No cudaDeviceSynchronize needed here typically unless RCB has internal async issues or for timing the loop.
        } // End loop over pixels in batch

        // Copy results for the completed batch from device to host
        CUDA_CHECK(cudaMemcpy(h_psi_out_batch.data(), d_psi_out_batch, current_batch_numpixels * sizeof(float), cudaMemcpyDeviceToHost));

        // Store batch results into the final MATLAB output array (column-major)
        for (int p_batch_idx = 0; p_batch_idx < current_batch_numpixels; ++p_batch_idx) {
            int pixel_flat_global = pixel_start_offset_global + p_batch_idx;
            // Image is num_z_points rows, num_x_points columns.
            // Pixel (zi, xi) where 0 <= zi < num_z_points, 0 <= xi < num_x_points
            // If pixel_flat_global scans x-first (raster scan):
            int zi_idx = pixel_flat_global / num_x_points;
            int xi_idx = pixel_flat_global % num_x_points;
            // MATLAB column-major index: zi_idx + xi_idx * num_z_points
            beamformed_Image_out[zi_idx + xi_idx * num_z_points] = h_psi_out_batch[p_batch_idx];
        }
    } // End loop over batches

    // --- Cleanup ---
    CUDA_CHECK(cudaFree(d_fftBin));
    CUDA_CHECK(cudaFree(d_RData_cufft));
    CUDA_CHECK(cudaFree(d_RFFT_cufft));
    CUDA_CHECK(cudaFree(d_ShiftData_cufft_batch));
    CUDA_CHECK(cudaFree(d_ShiftData_time_cufft_batch));
    CUDA_CHECK(cudaFree(d_realShiftDataTime_sdelay_batch));
    CUDA_CHECK(cudaFree(d_delay_batch));
    CUDA_CHECK(cudaFree(d_element_Pos_Array_um_X_flat_gpu));
    CUDA_CHECK(cudaFree(d_image_Range_X_um_gpu));
    CUDA_CHECK(cudaFree(d_image_Range_Z_um_gpu));

    CUDA_CHECK(cudaFree(d_R_syrk_batch_buffer));
    CUDA_CHECK(cudaFree(d_Rs_rcb_batch_buffer));
    CUDA_CHECK(cudaFree(d_psi_out_batch));

    CUBLAS_CHECK(cublasDestroy(cublasH_sdelay));
    CUBLAS_CHECK(cublasDestroy(cublasH_rcb));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH_rcb));
    CUFFT_CHECK(cufftDestroy(plan_ifft_batch));
}