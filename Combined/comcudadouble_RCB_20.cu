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
#include <thrust/sort.h> // For thrust::reverse

#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

// --- CUDA Kernels (Double Precision) ---
__global__ void computeFftBinKernel_d(double *fftBin, int nfft, int binStart)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nfft)
    {
        int shift = (nfft % 2 == 0) ? nfft / 2 : (nfft - 1) / 2;
        int srcIndex = (idx + shift) % nfft;
        double value = static_cast<double>(srcIndex) - static_cast<double>(binStart);
        fftBin[idx] = (2.0 * M_PI * value) / static_cast<double>(nfft);
    }
}

__global__ void shiftDataKernelBatch_d(
    const cufftDoubleComplex *RFFT, 
    const double *fftBin,           
    const double *delay_batch,      
    int Nsamples, int Nchannels, int Npixels_in_batch, 
    cufftDoubleComplex *ShiftData 
)
{
    int pixel_idx = blockIdx.z; 
    int idx_in_pixel_allchannels_samples = blockIdx.x * blockDim.x + threadIdx.x; 
    int total_samples_channels_per_pixel = Nsamples * Nchannels;

    if (idx_in_pixel_allchannels_samples < total_samples_channels_per_pixel && pixel_idx < Npixels_in_batch)
    {
        int r_sample = idx_in_pixel_allchannels_samples % Nsamples; 
        int c_channel = idx_in_pixel_allchannels_samples / Nsamples; 

        double angle = -delay_batch[pixel_idx * Nchannels + c_channel] * fftBin[r_sample];
        double cosval = cos(angle);
        double sinval = sin(angle);
        
        cufftDoubleComplex factor;
        factor.x = cosval;
        factor.y = sinval;
        
        cufftDoubleComplex val = RFFT[c_channel * Nsamples + r_sample]; // Assumes RFFT is Chan0_data, Chan1_data ...
                                                                      // which means access is RFFT[channel_offset + sample_in_channel]
                                                                      // If MATLAB style col-major (Nsamples x Nchannels): RFFT[r_sample + c_channel * Nsamples]
                                                                      // Based on cufftPlanMany setup, it should be Chan0_data, Chan1_data
                                                                      // So access `RFFT[c_channel * Nsamples + r_sample]` is for this layout.
        
        cufftDoubleComplex res;
        res.x = val.x * factor.x - val.y * factor.y;
        res.y = val.x * factor.y + val.y * factor.x;
        
        ShiftData[pixel_idx * total_samples_channels_per_pixel + idx_in_pixel_allchannels_samples] = res;
    }
}

__global__ void extractRealPartAndScaleKernelBatch_d(
    const cufftDoubleComplex *ShiftData_time_complex_unscaled,
    double *realOutput,
    int totalSize_per_pixel, 
    int Npixels_in_batch,
    double scale_factor)     
{
    int pixel_idx = blockIdx.z; 
    int idx_in_pixel_data = blockIdx.x * blockDim.x + threadIdx.x; 

    if (idx_in_pixel_data < totalSize_per_pixel && pixel_idx < Npixels_in_batch)
    {
        int global_idx = pixel_idx * totalSize_per_pixel + idx_in_pixel_data;
        realOutput[global_idx] = ShiftData_time_complex_unscaled[global_idx].x * scale_factor;
    }
}

__global__ void compute_delay_batch_kernel_d(
    double *d_delay_batch, 
    const double *d_image_Range_X_um_global,
    const double *d_image_Range_Z_um_global,
    const double *d_element_Pos_Array_um_X_flat, 
    int num_x_points_img,                        
    int Nchannels_dev,
    double speed_Of_Sound_umps_dev,
    double sampling_Freq_dev,
    int pixel_start_offset_batch, 
    int current_batch_size)
{
    int p_in_batch = blockIdx.x * blockDim.x + threadIdx.x; 

    if (p_in_batch < current_batch_size)
    {
        int pixel_flat_global = pixel_start_offset_batch + p_in_batch;
        int zi_idx = pixel_flat_global / num_x_points_img;
        int xi_idx = pixel_flat_global % num_x_points_img;

        double current_image_Z_um = d_image_Range_Z_um_global[zi_idx];
        double current_image_X_um = d_image_Range_X_um_global[xi_idx];

        for (int ch = 0; ch < Nchannels_dev; ++ch)
        {
            double dx = current_image_X_um - d_element_Pos_Array_um_X_flat[ch * 2 + 0]; 
            double dz = current_image_Z_um - d_element_Pos_Array_um_X_flat[ch * 2 + 1]; 
            double distance = sqrt(dx * dx + dz * dz);
            double time_Pt_Along_RF = distance / speed_Of_Sound_umps_dev;
            d_delay_batch[p_in_batch * Nchannels_dev + ch] = -(time_Pt_Along_RF * sampling_Freq_dev);
        }
    }
}

__global__ void convert_real_lower_to_complex_hermitian_kernel_d(
    const double *R_syrk_lower_col_major,      
    thrust::complex<double> *Rs_rcb_col_major, 
    int N_dim)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x; 
    int r = blockIdx.y * blockDim.y + threadIdx.y; 

    if (r < N_dim && c < N_dim)
    {
        int flat_idx_cm = r + c * N_dim;
        if (r >= c)
        { 
            Rs_rcb_col_major[flat_idx_cm] = thrust::complex<double>(R_syrk_lower_col_major[flat_idx_cm], 0.0);
        }
        else
        { 
            Rs_rcb_col_major[flat_idx_cm] = thrust::complex<double>(R_syrk_lower_col_major[c + r * N_dim], 0.0);
        }
    }
}

// --- Robust Capon Functors (Double Precision) ---
struct fill_ones_functor_d
{
    __host__ __device__
        thrust::complex<double>
        operator()(const int &) const
    {
        return thrust::complex<double>(1.0, 0.0);
    }
};

struct g_transform_op_d
{
    const double x0_lambda; 
    g_transform_op_d(double _x0) : x0_lambda(_x0) {}
    __host__ __device__ double operator()(const thrust::tuple<double, thrust::complex<double>> &t) const
    {
        double gamma_val = thrust::get<0>(t); 
        thrust::complex<double> z2_val = thrust::get<1>(t); 
        double abs_z2_sq = thrust::norm(z2_val); 
        double den_base = 1.0 + gamma_val * x0_lambda;
        if (fabs(den_base) < 1e-12) // Adjusted tolerance
        {
            if (abs_z2_sq < 1e-20) return 0.0; // Adjusted tolerance
            return HUGE_VAL; 
        }
        return abs_z2_sq / (den_base * den_base);
    }
};

struct gprime_transform_op_d
{
    const double current_lambda; 
    gprime_transform_op_d(double _current_lambda) : current_lambda(_current_lambda) {}
    __host__ __device__ double operator()(const thrust::tuple<double, thrust::complex<double>> &t) const
    {
        double gamma_k_val = thrust::get<0>(t); 
        thrust::complex<double> z2_k_val = thrust::get<1>(t); 
        double abs_z2_k_sq = thrust::norm(z2_k_val);
        double den_base = 1.0 + gamma_k_val * current_lambda;
        if (fabs(den_base) < 1e-12) // Adjusted tolerance
             return 0.0; 
        return (-2.0 * gamma_k_val * abs_z2_k_sq) / (den_base * den_base * den_base);
    }
};

struct b_sum_transform_op_d
{
    __host__ __device__ double operator()(const thrust::tuple<double, thrust::complex<double>> &t) const
    {
        double b_diag_val = thrust::get<0>(t);
        thrust::complex<double> temp_vec3_val = thrust::get<1>(t); 
        return b_diag_val * thrust::norm(temp_vec3_val); 
    }
};

struct inv_term_times_z2_lambda_op_d
{
    const double lambda_val_member; 
    inv_term_times_z2_lambda_op_d(double _lambda_val) : lambda_val_member(_lambda_val) {}
    __host__ __device__
        thrust::complex<double>
        operator()(const thrust::tuple<double, thrust::complex<double>> &t) const
    {
        double gamma_val = thrust::get<0>(t);
        thrust::complex<double> z2_val = thrust::get<1>(t);
        double den = 1.0 + lambda_val_member * gamma_val;
        if (fabs(den) < 1e-12) { // Adjusted tolerance
            if (thrust::norm(z2_val) < 1e-20 * thrust::norm(z2_val)) return thrust::complex<double>(0.0,0.0); 
            return thrust::complex<double>(0.0, 0.0); 
        }
        return (1.0 / den) * z2_val;
    }
};

struct abar_minus_temp_vec_op_d
{
    __host__ __device__
        thrust::complex<double>
        operator()(const thrust::complex<double> &abar_val, const thrust::complex<double> &temp_val) const
    {
        return abar_val - temp_val;
    }
};

struct b_diag_op_d
{
    const double lambda_val; 
    b_diag_op_d(double _lambda) : lambda_val(_lambda) {}
    __host__ __device__ double operator()(const double &gamma_val) const
    {
        if (fabs(lambda_val) < 1e-14) // Adjusted tolerance
        {
            return 0.0;
        }
        double lambda_inv = 1.0 / lambda_val;
        double den_term = lambda_inv + gamma_val;
        double den_sq = den_term * den_term;
        if (fabs(den_sq) < 1e-14) // Adjusted tolerance
        {
            if (fabs(gamma_val) < 1e-18) return 0.0; // Adjusted tolerance
            return HUGE_VAL; 
        }
        return gamma_val / den_sq;
    }
};

__global__ void flip_columns_kernel_d(const cufftDoubleComplex *A_in, cufftDoubleComplex *A_out, int N_rcb)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N_rcb && col < N_rcb)
    {
        A_out[row + col * N_rcb] = A_in[row + (N_rcb - 1 - col) * N_rcb];
    }
}

// --- RCB Workspace Structure (Double Precision) ---
struct RCB_Workspace_d
{
    thrust::device_vector<thrust::complex<double>> V_d;
    thrust::device_vector<double> gamma_d;
    thrust::device_vector<thrust::complex<double>> abar_d;
    thrust::device_vector<thrust::complex<double>> z2_d;
    thrust::device_vector<thrust::complex<double>> ahat_d;
    thrust::device_vector<thrust::complex<double>> Rs_copy_d;
    thrust::device_vector<thrust::complex<double>> work_d_eig; 
    thrust::device_vector<int> devInfo_d;
    thrust::device_vector<double> temp_terms_d_newton;
    thrust::device_vector<thrust::complex<double>> V_flipped_d;
    thrust::device_vector<thrust::complex<double>> temp_vec1_d;
    thrust::device_vector<thrust::complex<double>> temp_vec2_d;
    thrust::device_vector<double> b_diag_vals_d;
    thrust::device_vector<thrust::complex<double>> temp_vec3_d;
    int lwork_eig_val;

    RCB_Workspace_d(int n_capon, cusolverDnHandle_t cusolverH_rcb_init) : lwork_eig_val(0)
    {
        if (n_capon <= 0) {
             devInfo_d.resize(1); 
             work_d_eig.resize(1); 
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
        
        CUSOLVER_CHECK(cusolverDnZheevd_bufferSize(cusolverH_rcb_init, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n_capon,
                                                   (cuDoubleComplex *)thrust::raw_pointer_cast(Rs_copy_d.data()), n_capon,
                                                   thrust::raw_pointer_cast(gamma_d.data()), 
                                                   &lwork_eig_val));
        work_d_eig.resize(lwork_eig_val > 0 ? lwork_eig_val : 1); 
    }
};

// --- Modified Robust Capon Beamforming Function (Double Precision, operates on device data) ---
void robustCaponBeamformingCUDA_onDevice_d(
    const thrust::complex<double> *d_Rs_in, 
    double epsilon, int n_capon,
    double *d_psi_out, 
    cublasHandle_t cublasH, cusolverDnHandle_t cusolverH,
    RCB_Workspace_d &ws) 
{
    if (n_capon <= 0)
    {
        double nan_val = std::numeric_limits<double>::quiet_NaN();
        CUDA_CHECK(cudaMemcpy(d_psi_out, &nan_val, sizeof(double), cudaMemcpyHostToDevice));
        return;
    }

    CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(ws.Rs_copy_d.data()), d_Rs_in,
                          static_cast<size_t>(n_capon) * n_capon * sizeof(thrust::complex<double>),
                          cudaMemcpyDeviceToDevice));

    thrust::counting_iterator<int> first(0);
    thrust::counting_iterator<int> last = first + n_capon;
    thrust::transform(thrust::cuda::par, first, last, ws.abar_d.begin(), fill_ones_functor_d());

    double anorm = sqrt(static_cast<double>(n_capon));
    if (epsilon < 0.0) 
        mexErrMsgIdAndTxt("RobustCaponInternal:EpsilonNegative", "Epsilon cannot be negative.");
    double sqrt_epsilon = sqrt(epsilon); 
    double bound_val;
    if (fabs(sqrt_epsilon) < 1e-18) { // Adjusted tolerance
         bound_val = (anorm > 1e-12) ? HUGE_VAL : 0.0; // Adjusted tolerance
    } else {
        bound_val = (anorm - sqrt_epsilon) / sqrt_epsilon;
    }

    CUSOLVER_CHECK(cusolverDnZheevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n_capon,
                                    (cuDoubleComplex *)thrust::raw_pointer_cast(ws.Rs_copy_d.data()), n_capon,
                                    thrust::raw_pointer_cast(ws.gamma_d.data()), 
                                    (cuDoubleComplex *)thrust::raw_pointer_cast(ws.work_d_eig.data()), ws.lwork_eig_val,
                                    thrust::raw_pointer_cast(ws.devInfo_d.data())));
    
    CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(ws.V_d.data()), thrust::raw_pointer_cast(ws.Rs_copy_d.data()),
                          static_cast<size_t>(n_capon) * n_capon * sizeof(thrust::complex<double>),
                          cudaMemcpyDeviceToDevice));

    int devInfo_h;
    CUDA_CHECK(cudaMemcpy(&devInfo_h, thrust::raw_pointer_cast(ws.devInfo_d.data()), sizeof(int), cudaMemcpyDeviceToHost));
    if (devInfo_h != 0)
    {
        // mexPrintf("RCB cuSOLVER zheevd failed for a pixel: info = %d\n", devInfo_h);
        double nan_val = std::numeric_limits<double>::quiet_NaN();
        CUDA_CHECK(cudaMemcpy(d_psi_out, &nan_val, sizeof(double), cudaMemcpyHostToDevice));
        return;
    }

    thrust::reverse(thrust::cuda::par, ws.gamma_d.begin(), ws.gamma_d.end()); 

    if (n_capon > 0) {
        dim3 threadsPerBlock_fc(16, 16); 
        dim3 numBlocks_fc((n_capon + threadsPerBlock_fc.x - 1) / threadsPerBlock_fc.x,
                          (n_capon + threadsPerBlock_fc.y - 1) / threadsPerBlock_fc.y);
        flip_columns_kernel_d<<<numBlocks_fc, threadsPerBlock_fc>>>(
            (const cufftDoubleComplex *)thrust::raw_pointer_cast(ws.V_d.data()), 
            (cufftDoubleComplex *)thrust::raw_pointer_cast(ws.V_flipped_d.data()), 
            n_capon);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize()); 
        
        CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(ws.V_d.data()), 
                              thrust::raw_pointer_cast(ws.V_flipped_d.data()),
                              static_cast<size_t>(n_capon) * n_capon * sizeof(thrust::complex<double>),
                              cudaMemcpyDeviceToDevice));
    }

    cuDoubleComplex cu_alpha_blas = {1.0, 0.0};
    cuDoubleComplex cu_beta_blas = {0.0, 0.0};

    if (n_capon > 0) {
        CUBLAS_CHECK(cublasZgemv(cublasH, CUBLAS_OP_C, n_capon, n_capon, 
                                 &cu_alpha_blas,
                                 (const cuDoubleComplex *)thrust::raw_pointer_cast(ws.V_d.data()), n_capon,
                                 (const cuDoubleComplex *)thrust::raw_pointer_cast(ws.abar_d.data()), 1,
                                 &cu_beta_blas,
                                 (cuDoubleComplex *)thrust::raw_pointer_cast(ws.z2_d.data()), 1));
    }

    double min_gamma_val_h = 0.0, max_gamma_val_h = 0.0; 
    if (n_capon > 0) {
        CUDA_CHECK(cudaMemcpy(&max_gamma_val_h, thrust::raw_pointer_cast(ws.gamma_d.data()), sizeof(double), cudaMemcpyDeviceToHost)); 
        CUDA_CHECK(cudaMemcpy(&min_gamma_val_h, thrust::raw_pointer_cast(ws.gamma_d.data() + (n_capon - 1)), sizeof(double), cudaMemcpyDeviceToHost));
    }

    double lower_bound, upper_bound;
    if (fabs(epsilon) < 1e-14) // Adjusted tolerance
    {
        lower_bound = (fabs(max_gamma_val_h) < 1e-14) ? 0.0 : (-1.0 / max_gamma_val_h); // Adjusted tolerance
        upper_bound = 0.0;
    }
    else
    {
        lower_bound = (fabs(max_gamma_val_h) < 1e-14) ? ((bound_val > 0) ? HUGE_VAL : ((bound_val < 0) ? -HUGE_VAL : 0.0)) : (bound_val / max_gamma_val_h); // Adjusted tolerance
        upper_bound = (fabs(min_gamma_val_h) < 1e-14) ? ((bound_val > 0) ? HUGE_VAL : ((bound_val < 0) ? -HUGE_VAL : 0.0)) : (bound_val / min_gamma_val_h); // Adjusted tolerance
    }

    if (lower_bound > upper_bound) 
        std::swap(lower_bound, upper_bound);
    
    if (max_gamma_val_h > 1e-14) // Adjusted tolerance
    {
        double min_lambda_from_constraint = (-1.0 / max_gamma_val_h);
        lower_bound = std::max(lower_bound, min_lambda_from_constraint + fabs(min_lambda_from_constraint)*1e-10 + 1e-14 ); // Adjusted offsets/tolerances
    }

    double x0 = lower_bound; 
    if (!std::isfinite(x0)) x0 = 0.0;

    if (n_capon > 0 && max_gamma_val_h > 1e-14 && (1.0 + max_gamma_val_h * x0 <= 1e-10) ) // Adjusted tolerances
    {
        double min_lambda_from_constraint = (-1.0 / max_gamma_val_h);
        x0 = min_lambda_from_constraint + fabs(min_lambda_from_constraint)*1e-8 + 1e-12; // Adjusted offsets/tolerances
    }
    double x1 = x0; 

    int iter = 0;
    const int MAX_ITER_NEWTON = 50; 
    const double TOL_NEWTON_D = 1e-8; // Adjusted tolerance for double

    if (n_capon > 0 && fabs(epsilon) > 1e-14) // Adjusted tolerance
    {
        for (iter = 0; iter < MAX_ITER_NEWTON; ++iter)
        {
            thrust::transform(thrust::cuda::par,
                              thrust::make_zip_iterator(thrust::make_tuple(ws.gamma_d.begin(), ws.z2_d.begin())),
                              thrust::make_zip_iterator(thrust::make_tuple(ws.gamma_d.end(), ws.z2_d.end())),
                              ws.temp_terms_d_newton.begin(), g_transform_op_d(x0));
            double g_sum = thrust::reduce(thrust::cuda::par, ws.temp_terms_d_newton.begin(), ws.temp_terms_d_newton.end(), 0.0, thrust::plus<double>());
            double g_val = g_sum - epsilon; 

            if (fabs(g_val) < TOL_NEWTON_D * epsilon + 1e-14) // Adjusted tolerance
                break;

            thrust::transform(thrust::cuda::par,
                              thrust::make_zip_iterator(thrust::make_tuple(ws.gamma_d.begin(), ws.z2_d.begin())),
                              thrust::make_zip_iterator(thrust::make_tuple(ws.gamma_d.end(), ws.z2_d.end())),
                              ws.temp_terms_d_newton.begin(), gprime_transform_op_d(x0));
            double gprime_sum = thrust::reduce(thrust::cuda::par, ws.temp_terms_d_newton.begin(), ws.temp_terms_d_newton.end(), 0.0, thrust::plus<double>());

            if (fabs(gprime_sum) < 1e-14) { // Adjusted tolerance
                // mexPrintf("Warning: RCB Newton gprime_sum too small (%g) at iter %d, lambda_try %g. Stopping.\n", gprime_sum, iter, x0);
                break; 
            }
            x1 = x0 - (g_val / gprime_sum);

            if (max_gamma_val_h > 1e-14 && (1.0 + max_gamma_val_h * x1 <= 1e-10)) // Adjusted tolerances
            {
                double min_lambda_from_constraint = (-1.0 / max_gamma_val_h);
                x1 = x0 * 0.5 + (min_lambda_from_constraint + fabs(min_lambda_from_constraint)*1e-8 + 1e-12) * 0.5; // Adjusted offsets/tolerances
            }
            
            if (fabs(x1 - x0) < TOL_NEWTON_D * (1e-10 + fabs(x1))) // Adjusted tolerance
                break;
            x0 = x1;
        }
        // if (iter == MAX_ITER_NEWTON)
            // mexPrintf("Warning: RCB Newton did not converge for a pixel. Lambda = %g.\n", x1);
    }
    else if (n_capon > 0) 
    {
        thrust::transform(thrust::cuda::par,
                          thrust::make_zip_iterator(thrust::make_tuple(ws.gamma_d.begin(), ws.z2_d.begin())),
                          thrust::make_zip_iterator(thrust::make_tuple(ws.gamma_d.end(), ws.z2_d.end())),
                          ws.temp_terms_d_newton.begin(), g_transform_op_d(0.0));
        double g_at_zero = thrust::reduce(thrust::cuda::par, ws.temp_terms_d_newton.begin(), ws.temp_terms_d_newton.end(), 0.0, thrust::plus<double>());
        
        if (g_at_zero <= epsilon + 1e-14)  // Adjusted tolerance
            x1 = 0.0;
        else 
        {
            if (max_gamma_val_h > 1e-14) { // Adjusted tolerance
                 double min_lambda_from_constraint = (-1.0 / max_gamma_val_h);
                 x1 = min_lambda_from_constraint + fabs(min_lambda_from_constraint)*1e-8 + 1e-12; // Adjusted offsets/tolerances
            } else { 
                 x1 = 0.0; 
            }
        }
    } else { 
        x1 = 0.0;
    }

    double final_lambda_val = x1;
    if (!std::isfinite(final_lambda_val))
    {
        // mexPrintf("Warning: RCB final lambda is not finite (%g). Defaulting to 0.\n", final_lambda_val);
        final_lambda_val = 0.0; 
    }
    if (n_capon > 0 && max_gamma_val_h > 1e-14 && (1.0 + max_gamma_val_h * final_lambda_val <= 1e-10)) // Adjusted tolerances
    {
        // mexPrintf("Warning: RCB final lambda %g is invalid. Adjusting.\n", final_lambda_val);
        double min_lambda_from_constraint = (-1.0 / max_gamma_val_h);
        final_lambda_val = min_lambda_from_constraint + fabs(min_lambda_from_constraint)*1e-8 + 1e-12; // Adjusted offsets/tolerances
    }

    double host_psi_val_calc;
    if (n_capon > 0) {
        thrust::transform(thrust::cuda::par,
                          thrust::make_zip_iterator(thrust::make_tuple(ws.gamma_d.begin(), ws.z2_d.begin())),
                          thrust::make_zip_iterator(thrust::make_tuple(ws.gamma_d.end(), ws.z2_d.end())),
                          ws.temp_vec1_d.begin(), inv_term_times_z2_lambda_op_d(final_lambda_val));

        CUBLAS_CHECK(cublasZgemv(cublasH, CUBLAS_OP_N, n_capon, n_capon,
                                 &cu_alpha_blas, (const cuDoubleComplex *)thrust::raw_pointer_cast(ws.V_d.data()), n_capon,
                                 (const cuDoubleComplex *)thrust::raw_pointer_cast(ws.temp_vec1_d.data()), 1,
                                 &cu_beta_blas, (cuDoubleComplex *)thrust::raw_pointer_cast(ws.temp_vec2_d.data()), 1));
        
        thrust::transform(thrust::cuda::par, ws.abar_d.begin(), ws.abar_d.end(), ws.temp_vec2_d.begin(), ws.ahat_d.begin(), abar_minus_temp_vec_op_d());
        
        thrust::transform(thrust::cuda::par, ws.gamma_d.begin(), ws.gamma_d.end(), ws.b_diag_vals_d.begin(), b_diag_op_d(final_lambda_val));

        CUBLAS_CHECK(cublasZgemv(cublasH, CUBLAS_OP_C, n_capon, n_capon,
                                 &cu_alpha_blas, (const cuDoubleComplex *)thrust::raw_pointer_cast(ws.V_d.data()), n_capon,
                                 (const cuDoubleComplex *)thrust::raw_pointer_cast(ws.ahat_d.data()), 1,
                                 &cu_beta_blas, (cuDoubleComplex *)thrust::raw_pointer_cast(ws.temp_vec3_d.data()), 1));
        
        thrust::transform(thrust::cuda::par,
                          thrust::make_zip_iterator(thrust::make_tuple(ws.b_diag_vals_d.begin(), ws.temp_vec3_d.begin())),
                          thrust::make_zip_iterator(thrust::make_tuple(ws.b_diag_vals_d.end(), ws.temp_vec3_d.end())),
                          ws.temp_terms_d_newton.begin(), b_sum_transform_op_d()); 
        double b_scalar = thrust::reduce(thrust::cuda::par, ws.temp_terms_d_newton.begin(), ws.temp_terms_d_newton.end(), 0.0, thrust::plus<double>());

        if (fabs(b_scalar) < 1e-18) // Adjusted tolerance
        {
            host_psi_val_calc = (fabs(b_scalar) < 1e-50) ? std::numeric_limits<double>::infinity() : (static_cast<double>(n_capon) * static_cast<double>(n_capon)) / b_scalar; // Adjusted tolerance
        }
        else
        {
            host_psi_val_calc = (static_cast<double>(n_capon) * static_cast<double>(n_capon)) / b_scalar;
        }

        if (!std::isfinite(host_psi_val_calc)) { 
            // mexPrintf("Warning: RCB psi is Inf/NaN. b_scalar=%g, lambda=%g\n", b_scalar, final_lambda_val);
            if (std::isinf(host_psi_val_calc) && host_psi_val_calc < 0) { 
                 host_psi_val_calc = 0.0; 
            }
        } else if (host_psi_val_calc < 0) {
            // mexPrintf("Warning: RCB psi is negative (%g). Clamping to 0. b_scalar=%g, lambda=%g\n", host_psi_val_calc, b_scalar, final_lambda_val);
            host_psi_val_calc = 0.0; 
        }
    } else { 
        host_psi_val_calc = std::numeric_limits<double>::quiet_NaN();
    }
    
    CUDA_CHECK(cudaMemcpy(d_psi_out, &host_psi_val_calc, sizeof(double), cudaMemcpyHostToDevice));
}


// --- Main CUDA MEX Entry Point: Batched PCI Imaging (Double Precision) ---
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 9)
    {
        mexErrMsgIdAndTxt("Combcuda:nrhs", "Nine inputs required: RF_Arr, element_Pos_Array_um_X, speed_Of_Sound_umps, RF_Start_Time, sampling_Freq, image_Range_X_um, image_Range_Z_um, p_epsilon, n_elements.");
    }
    if (nlhs > 1)
    {
        mexErrMsgIdAndTxt("Combcuda:nlhs", "One output argument (beamformed_Image_out) allowed.");
    }

    // --- Parse Inputs (Double Precision) ---
    if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]))
        mexErrMsgIdAndTxt("Combcuda:RF_ArrNotRealDouble", "Input RF_Arr must be a real double matrix.");
    mwSize Nsamples_mw = mxGetM(prhs[0]);
    mwSize Nchannels_mw = mxGetN(prhs[0]);
    int Nsamples = static_cast<int>(Nsamples_mw);
    int Nchannels = static_cast<int>(Nchannels_mw);
    const double *hostRData_ptr = (const double*)mxGetData(prhs[0]);

    if (!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) || mxGetM(prhs[1]) != 2 || mxGetN(prhs[1]) != Nchannels)
        mexErrMsgIdAndTxt("Combcuda:element_Pos_Array_um_X", "element_Pos_Array_um_X must be 2 x Nchannels real double matrix.");
    const double *element_Pos_Array_um_X = (const double*)mxGetData(prhs[1]);

    double speed_Of_Sound_umps = static_cast<double>(mxGetScalar(prhs[2]));
    if (speed_Of_Sound_umps <= 0.0)
        mexErrMsgIdAndTxt("Combcuda:SoSInvalid", "speed_Of_Sound_umps must be positive.");

    double sampling_Freq = static_cast<double>(mxGetScalar(prhs[4]));
    if (sampling_Freq <= 0.0)
        mexErrMsgIdAndTxt("Combcuda:fsInvalid", "sampling_Freq must be positive.");

    if (!mxIsDouble(prhs[5]) || mxIsComplex(prhs[5]) || mxGetM(prhs[5]) * mxGetN(prhs[5]) < 1)
        mexErrMsgIdAndTxt("Combcuda:image_Range_X_um", "image_Range_X_um must be a non-empty real double vector.");
    int num_x_points = static_cast<int>(mxGetNumberOfElements(prhs[5]));
    const double *image_Range_X_um = (const double*)mxGetData(prhs[5]);

    if (!mxIsDouble(prhs[6]) || mxIsComplex(prhs[6]) || mxGetM(prhs[6]) * mxGetN(prhs[6]) < 1)
        mexErrMsgIdAndTxt("Combcuda:image_Range_Z_um", "image_Range_Z_um must be a non-empty real double vector.");
    int num_z_points = static_cast<int>(mxGetNumberOfElements(prhs[6]));
    const double *image_Range_Z_um = (const double*)mxGetData(prhs[6]);

    double p_epsilon = static_cast<double>(mxGetScalar(prhs[7]));
    
    int n_elements = static_cast<int>(mxGetScalar(prhs[8])); 
    if (n_elements != Nchannels)
        mexErrMsgIdAndTxt("Combcuda:n_elements", "n_elements must match number of RF_Arr columns (Nchannels).");
     if (n_elements <= 0) 
        mexErrMsgIdAndTxt("Combcuda:n_elements_positive", "n_elements (and Nchannels) must be positive.");

    // --- Output Allocation (Double Precision) ---
    mwSize dims[2] = {(mwSize)num_z_points, (mwSize)num_x_points}; 
    plhs[0] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
    double *beamformed_Image_out = (double*)mxGetData(plhs[0]);
    int Npixels = num_x_points * num_z_points;

    if (Nsamples == 0 || Npixels == 0 || Nchannels == 0) { 
        size_t totalOutputElements = static_cast<size_t>(num_z_points) * num_x_points;
        for (size_t i = 0; i < totalOutputElements; ++i) beamformed_Image_out[i] = std::numeric_limits<double>::quiet_NaN();
        return;
    }

    // --- Precompute FFT bin (Double Precision) ---
    std::vector<double> fftBin_h(Nsamples);
    { 
        int binStart = Nsamples / 2; 
        for (int idx = 0; idx < Nsamples; ++idx) {
            int shift = (Nsamples % 2 == 0) ? Nsamples / 2 : (Nsamples - 1) / 2;
            int srcIndex = (idx + shift) % Nsamples; 
            double value = static_cast<double>(srcIndex) - static_cast<double>(binStart);
            fftBin_h[idx] = (2.0 * M_PI * value) / static_cast<double>(Nsamples);
        }
    }
    double *d_fftBin;
    CUDA_CHECK(cudaMalloc((void **)&d_fftBin, Nsamples * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_fftBin, fftBin_h.data(), Nsamples * sizeof(double), cudaMemcpyHostToDevice));

    std::vector<cufftDoubleComplex> RData_complex_h(static_cast<size_t>(Nsamples) * Nchannels);
    for (size_t r = 0; r < Nsamples; ++r) {
        for (size_t c = 0; c < Nchannels; ++c) {
            RData_complex_h[c * Nsamples + r].x = hostRData_ptr[r + c * Nsamples_mw]; 
            RData_complex_h[c * Nsamples + r].y = 0.0;
        }
    }
    
    cufftDoubleComplex *d_RData_cufft; 
    CUDA_CHECK(cudaMalloc((void **)&d_RData_cufft, sizeof(cufftDoubleComplex) * Nsamples * Nchannels));
    CUDA_CHECK(cudaMemcpy(d_RData_cufft, RData_complex_h.data(), sizeof(cufftDoubleComplex) * Nsamples * Nchannels, cudaMemcpyHostToDevice));

    cufftHandle plan_fft_rf;
    int Nsamples_int = Nsamples; // cufftPlanMany wants int* for n
    CUFFT_CHECK(cufftPlanMany(&plan_fft_rf, 1, &Nsamples_int, 
                              nullptr, 1, Nsamples, 
                              nullptr, 1, Nsamples, 
                              CUFFT_Z2Z, Nchannels)); // Changed to Z2Z
    
    cufftDoubleComplex *d_RFFT_cufft; 
    CUDA_CHECK(cudaMalloc((void **)&d_RFFT_cufft, sizeof(cufftDoubleComplex) * Nsamples * Nchannels));
    CUFFT_CHECK(cufftExecZ2Z(plan_fft_rf, d_RData_cufft, d_RFFT_cufft, CUFFT_FORWARD)); // Changed to Z2Z
    CUFFT_CHECK(cufftDestroy(plan_fft_rf));

    cublasHandle_t cublasH_sdelay, cublasH_rcb; 
    cusolverDnHandle_t cusolverH_rcb;
    CUBLAS_CHECK(cublasCreate(&cublasH_sdelay));
    CUBLAS_CHECK(cublasCreate(&cublasH_rcb));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH_rcb));

    RCB_Workspace_d rcb_workspace(n_elements, cusolverH_rcb); 

    double *d_element_Pos_Array_um_X_flat_gpu, *d_image_Range_X_um_gpu, *d_image_Range_Z_um_gpu;
    CUDA_CHECK(cudaMalloc((void **)&d_element_Pos_Array_um_X_flat_gpu, 2 * Nchannels * sizeof(double))); 
    CUDA_CHECK(cudaMemcpy(d_element_Pos_Array_um_X_flat_gpu, element_Pos_Array_um_X, 2 * Nchannels * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void **)&d_image_Range_X_um_gpu, num_x_points * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_image_Range_X_um_gpu, image_Range_X_um, num_x_points * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void **)&d_image_Range_Z_um_gpu, num_z_points * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_image_Range_Z_um_gpu, image_Range_Z_um, num_z_points * sizeof(double), cudaMemcpyHostToDevice));

    int batch_size_pixels = 128; 
    if (Npixels < batch_size_pixels) batch_size_pixels = Npixels; 
    int total_batches = (Npixels + batch_size_pixels - 1) / batch_size_pixels;

    int totalElements_per_pixel_sdelay = Nsamples * Nchannels; 
    
    cufftDoubleComplex *d_ShiftData_cufft_batch; 
    cufftDoubleComplex *d_ShiftData_time_cufft_batch; 
    double *d_realShiftDataTime_sdelay_batch; 
    double *d_delay_batch; 

    CUDA_CHECK(cudaMalloc((void **)&d_ShiftData_cufft_batch, sizeof(cufftDoubleComplex) * batch_size_pixels * totalElements_per_pixel_sdelay));
    CUDA_CHECK(cudaMalloc((void **)&d_ShiftData_time_cufft_batch, sizeof(cufftDoubleComplex) * batch_size_pixels * totalElements_per_pixel_sdelay));
    CUDA_CHECK(cudaMalloc((void **)&d_realShiftDataTime_sdelay_batch, sizeof(double) * batch_size_pixels * totalElements_per_pixel_sdelay));
    CUDA_CHECK(cudaMalloc((void **)&d_delay_batch, sizeof(double) * batch_size_pixels * Nchannels));

    cufftHandle plan_ifft_batch;
    int n_ifft[] = {Nsamples}; 
    CUFFT_CHECK(cufftPlanMany(&plan_ifft_batch, 1, n_ifft,
                              nullptr, 1, Nsamples, 
                              nullptr, 1, Nsamples, 
                              CUFFT_Z2Z, batch_size_pixels * Nchannels)); // Changed to Z2Z

    double *d_R_syrk_batch_buffer; 
    CUDA_CHECK(cudaMalloc((void **)&d_R_syrk_batch_buffer, sizeof(double) * batch_size_pixels * Nchannels * Nchannels));

    thrust::complex<double> *d_Rs_rcb_batch_buffer; 
    CUDA_CHECK(cudaMalloc((void **)&d_Rs_rcb_batch_buffer, sizeof(thrust::complex<double>) * batch_size_pixels * Nchannels * Nchannels));

    double *d_psi_out_batch; 
    CUDA_CHECK(cudaMalloc((void **)&d_psi_out_batch, sizeof(double) * batch_size_pixels));
    std::vector<double> h_psi_out_batch(batch_size_pixels); 

    for (int batch_idx = 0; batch_idx < total_batches; ++batch_idx)
    {
        int pixel_start_offset_global = batch_idx * batch_size_pixels; 
        int current_batch_numpixels = std::min(batch_size_pixels, Npixels - pixel_start_offset_global); 

        if (current_batch_numpixels <= 0) continue;

        dim3 block_delay(256); 
        dim3 grid_delay((current_batch_numpixels + block_delay.x - 1) / block_delay.x);
        compute_delay_batch_kernel_d<<<grid_delay, block_delay>>>(
            d_delay_batch, d_image_Range_X_um_gpu, d_image_Range_Z_um_gpu, d_element_Pos_Array_um_X_flat_gpu,
            num_x_points, Nchannels, speed_Of_Sound_umps, sampling_Freq,
            pixel_start_offset_global, current_batch_numpixels);
        CUDA_CHECK(cudaGetLastError());

        dim3 threads_per_block_shift(256); 
        dim3 num_blocks_shift( (totalElements_per_pixel_sdelay + threads_per_block_shift.x -1) / threads_per_block_shift.x, 
                                1,                                                                                            
                                current_batch_numpixels);                                                                     
        
        shiftDataKernelBatch_d<<<num_blocks_shift, threads_per_block_shift>>>(
            d_RFFT_cufft, d_fftBin, d_delay_batch,
            Nsamples, Nchannels, current_batch_numpixels, 
            d_ShiftData_cufft_batch);
        CUDA_CHECK(cudaGetLastError());
        
        CUFFT_CHECK(cufftExecZ2Z(plan_ifft_batch, // Changed to Z2Z
                                 d_ShiftData_cufft_batch, 
                                 d_ShiftData_time_cufft_batch, 
                                 CUFFT_INVERSE));

        double inv_Nsamples_scale = 1.0 / static_cast<double>(Nsamples);
        extractRealPartAndScaleKernelBatch_d<<<num_blocks_shift, threads_per_block_shift>>>(
            d_ShiftData_time_cufft_batch, 
            d_realShiftDataTime_sdelay_batch, 
            totalElements_per_pixel_sdelay, 
            current_batch_numpixels,
            inv_Nsamples_scale);
        CUDA_CHECK(cudaGetLastError());

        for (int p_batch_idx = 0; p_batch_idx < current_batch_numpixels; ++p_batch_idx)
        {
            const double alpha_syrk = 1.0 / static_cast<double>(Nsamples); 
            const double beta_syrk = 0.0;
            
            double *current_d_realShiftDataTime_pixel = d_realShiftDataTime_sdelay_batch + p_batch_idx * totalElements_per_pixel_sdelay;
            double *current_d_R_syrk_pixel = d_R_syrk_batch_buffer + p_batch_idx * Nchannels * Nchannels;

            CUBLAS_CHECK(cublasDsyrk(cublasH_sdelay, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, // Changed to Dsyrk
                                     Nchannels, Nsamples, &alpha_syrk,
                                     current_d_realShiftDataTime_pixel, Nsamples, 
                                     &beta_syrk, current_d_R_syrk_pixel, Nchannels));

            thrust::complex<double> *current_d_Rs_rcb_pixel = d_Rs_rcb_batch_buffer + p_batch_idx * Nchannels * Nchannels;
            dim3 block_conv(16, 16);
            dim3 grid_conv((Nchannels + block_conv.x - 1) / block_conv.x, (Nchannels + block_conv.y - 1) / block_conv.y);
            convert_real_lower_to_complex_hermitian_kernel_d<<<grid_conv, block_conv>>>(
                current_d_R_syrk_pixel, current_d_Rs_rcb_pixel, Nchannels);
            CUDA_CHECK(cudaGetLastError());

            robustCaponBeamformingCUDA_onDevice_d( // Changed to _d version
                current_d_Rs_rcb_pixel, p_epsilon, n_elements, 
                d_psi_out_batch + p_batch_idx, 
                cublasH_rcb, cusolverH_rcb, rcb_workspace);
        } 

        CUDA_CHECK(cudaMemcpy(h_psi_out_batch.data(), d_psi_out_batch, current_batch_numpixels * sizeof(double), cudaMemcpyDeviceToHost));

        for (int p_batch_idx = 0; p_batch_idx < current_batch_numpixels; ++p_batch_idx) {
            int pixel_flat_global = pixel_start_offset_global + p_batch_idx;
            int zi_idx = pixel_flat_global / num_x_points;
            int xi_idx = pixel_flat_global % num_x_points;
            beamformed_Image_out[zi_idx + xi_idx * num_z_points] = h_psi_out_batch[p_batch_idx];
        }
    } 

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