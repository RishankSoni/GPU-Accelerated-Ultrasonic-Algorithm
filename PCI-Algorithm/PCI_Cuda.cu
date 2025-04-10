#include "mex.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>
#include <cstdio>
#include <algorithm>
#ifndef M_PI
#define M_PI 3.141592653589
#endif
#define CUDA_CHECK(err)                                                                                    \
    {                                                                                                      \
        cudaError_t err_ = (err);                                                                          \
        if (err_ != cudaSuccess)                                                                           \
        {                                                                                                  \
            mexPrintf("CUDA error %d at %s:%d: %s\n", err_, __FILE__, __LINE__, cudaGetErrorString(err_)); \
            mexErrMsgIdAndTxt("PCIimagingSparseCUDA:cudaError", cudaGetErrorString(err_));                 \
        }                                                                                                  \
    }
#define KERNEL_CHECK()                  \
    {                                   \
        CUDA_CHECK(cudaGetLastError()); \
    }
#define CUFFT_CHECK(err)                                                                             \
    {                                                                                                \
        cufftResult err_ = (err);                                                                    \
        if (err_ != CUFFT_SUCCESS)                                                                   \
        {                                                                                            \
            const char *cufftErrorString;                                                            \
            switch (err_)                                                                            \
            {                                                                                        \
            case CUFFT_SETUP_FAILED:                                                                 \
                cufftErrorString = "CUFFT_SETUP_FAILED";                                             \
                break;                                                                               \
            case CUFFT_INVALID_PLAN:                                                                 \
                cufftErrorString = "CUFFT_INVALID_PLAN";                                             \
                break;                                                                               \
            case CUFFT_INVALID_VALUE:                                                                \
                cufftErrorString = "CUFFT_INVALID_VALUE";                                            \
                break;                                                                               \
            case CUFFT_INTERNAL_ERROR:                                                               \
                cufftErrorString = "CUFFT_INTERNAL_ERROR";                                           \
                break;                                                                               \
            case CUFFT_EXEC_FAILED:                                                                  \
                cufftErrorString = "CUFFT_EXEC_FAILED";                                              \
                break;                                                                               \
            case CUFFT_ALLOC_FAILED:                                                                 \
                cufftErrorString = "CUFFT_ALLOC_FAILED";                                             \
                break;                                                                               \
            case CUFFT_INVALID_TYPE:                                                                 \
                cufftErrorString = "CUFFT_INVALID_TYPE";                                             \
                break;                                                                               \
            case CUFFT_INVALID_SIZE:                                                                 \
                cufftErrorString = "CUFFT_INVALID_SIZE";                                             \
                break;                                                                               \
            case CUFFT_INCOMPLETE_PARAMETER_LIST:                                                    \
                cufftErrorString = "CUFFT_INCOMPLETE_PARAMETER_LIST";                                \
                break;                                                                               \
            case CUFFT_INVALID_DEVICE:                                                               \
                cufftErrorString = "CUFFT_INVALID_DEVICE";                                           \
                break;                                                                               \
            case CUFFT_PARSE_ERROR:                                                                  \
                cufftErrorString = "CUFFT_PARSE_ERROR";                                              \
                break;                                                                               \
            case CUFFT_NO_WORKSPACE:                                                                 \
                cufftErrorString = "CUFFT_NO_WORKSPACE";                                             \
                break;                                                                               \
            case CUFFT_NOT_IMPLEMENTED:                                                              \
                cufftErrorString = "CUFFT_NOT_IMPLEMENTED";                                          \
                break;                                                                               \
            case CUFFT_LICENSE_ERROR:                                                                \
                cufftErrorString = "CUFFT_LICENSE_ERROR";                                            \
                break;                                                                               \
            case CUFFT_NOT_SUPPORTED:                                                                \
                cufftErrorString = "CUFFT_NOT_SUPPORTED";                                            \
                break;                                                                               \
            default:                                                                                 \
                cufftErrorString = "UNKNOWN_CUFFT_ERROR";                                            \
                break;                                                                               \
            }                                                                                        \
            mexPrintf("CUFFT error %d (%s) at %s:%d\n", err_, cufftErrorString, __FILE__, __LINE__); \
            mexErrMsgIdAndTxt("PCIimagingSparseCUDA:cufftError", cufftErrorString);                  \
        }                                                                                            \
    }
__global__ void computeFftBinKernel(double *fftBin, int nfft, int binStart)
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

__global__ void realToComplexKernel(const double *realInput, cufftDoubleComplex *complexOutput, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        complexOutput[idx].x = realInput[idx];
        complexOutput[idx].y = 0.0;
    }
}
__global__ void shiftDataBatchKernel(const cufftDoubleComplex *RFFT, const double *fftBin, const double *batch_delays, int nfft, int ncols, int current_batch_size, cufftDoubleComplex *ShiftDataBatch)
{
    int idx_out = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = (size_t)nfft * ncols * current_batch_size;
    if (idx_out < total_elements)
    {
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

__global__ void scaleIFFTBatchKernel(cufftDoubleComplex *ShiftData_time_Batch, double scale, size_t total_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements)
    {
        ShiftData_time_Batch[idx].x *= scale;
        ShiftData_time_Batch[idx].y *= scale;
    }
}

__global__ void extractRealPartBatchKernel(const cufftDoubleComplex *Scaled_ShiftData_time_Batch, size_t total_elements, double *tempBatch)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements)
    {
        tempBatch[idx] = Scaled_ShiftData_time_Batch[idx].x;
    }
}

__global__ void pthrootElementwiseBatchKernel(const double *tempBatch, double pp, int nfft, int ncols, int current_batch_size, double *QpDASBatch)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = (size_t)nfft * ncols * current_batch_size;
    if (idx < total_elements)
    {
        double val = tempBatch[idx];
        double s = (val >= 0.0) ? 1.0 : -1.0;
        double abs_val = fabs(val);
        QpDASBatch[idx] = s * ((abs_val == 0.0) ? 0.0 : pow(abs_val, pp));
    }
}

__global__ void pthrootRowSumBatchKernel(const double *QpDASBatch, int nfft, int ncols, int current_batch_size, double *d_batch_rowSum)
{
    extern __shared__ double sdata[];
    int r = blockIdx.x;
    int p = blockIdx.y;
    int tid = threadIdx.x;
    int threads_per_block = blockDim.x;
    if (r >= nfft || p >= current_batch_size)
        return;
    double sum = 0.0;
    for (int c = tid; c < ncols; c += threads_per_block)
    {
        size_t idx_in = r + c * nfft + p * nfft * ncols;
        sum += QpDASBatch[idx_in];
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = threads_per_block / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        size_t idx_out = r + p * nfft;
        d_batch_rowSum[idx_out] = sdata[0];
    }
}

__global__ void pthrootFinalBatchKernel(const double *d_batch_rowSum, double inv_pp, int nfft, int current_batch_size, double *d_batch_pDAS)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = (size_t)nfft * current_batch_size;
    if (idx < total_elements)
    {
        double val = d_batch_rowSum[idx];
        double s = (val >= 0.0) ? 1.0 : -1.0;
        double abs_val = fabs(val);
        d_batch_pDAS[idx] = s * ((abs_val == 0.0) ? 0.0 : pow(abs_val, inv_pp));
    }
}

__global__ void calculateDCoffsetBatchKernel(const double *tempBatch, int nfft, int ncols, int current_batch_size, double *d_batch_DCoffset)
{
    extern __shared__ double sdata[];
    int r = blockIdx.x;
    int p = blockIdx.y;
    int tid = threadIdx.x;
    int threads_per_block = blockDim.x;
    if (r >= nfft || p >= current_batch_size)
        return;
    double sum_sq = 0.0;
    for (int c = tid; c < ncols; c += threads_per_block)
    {
        size_t idx_in = r + c * nfft + p * nfft * ncols;
        double val = tempBatch[idx_in];
        sum_sq += val * val;
    }
    sdata[tid] = sum_sq;
    __syncthreads();
    for (int s = threads_per_block / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        size_t idx_out = r + p * nfft;
        d_batch_DCoffset[idx_out] = sdata[0];
    }
}

__global__ void finalPixelValueBatchKernel(const double *d_batch_pDAS, const double *d_batch_DCoffset, int nfft, int current_batch_size, int batch_start_idx, int numX, double *d_beamformed_Image)
{
    extern __shared__ double sdata[];
    int p = blockIdx.x;
    int tid = threadIdx.x;
    int threads_per_block = blockDim.x;
    if (p >= current_batch_size)
        return;
    double pixel_sum = 0.0;
    for (int r = tid; r < nfft; r += threads_per_block)
    {
        size_t data_idx = r + p * nfft;
        double pDAS_val = d_batch_pDAS[data_idx];
        double dc_offset_val = d_batch_DCoffset[data_idx];
        pixel_sum += (pDAS_val * pDAS_val) - dc_offset_val;
    }
    sdata[tid] = pixel_sum;
    __syncthreads();
    for (int s = threads_per_block / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        int global_pixel_idx = batch_start_idx + p;
        d_beamformed_Image[global_pixel_idx] = sdata[0];
    }
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    if (nrhs != 9)
        mexErrMsgIdAndTxt("PCIimagingSparseCUDA:nrhs", "Nine inputs required: RF_Arr, element_Pos_Array_um, speed_Of_Sound_umps, RF_Start_Time, sampling_Freq, image_Range_X_um, image_Range_Z_um, p, all_delays_matrix");
    if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]))
        mexErrMsgIdAndTxt("PCIimagingSparseCUDA:input", "RF_Arr must be a real double matrix.");
    if (!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]))
        mexErrMsgIdAndTxt("PCIimagingSparseCUDA:input", "element_Pos_Array_um must be a real double matrix.");
    if (!mxIsScalar(prhs[2]) || !mxIsDouble(prhs[2]))
        mexErrMsgIdAndTxt("PCIimagingSparseCUDA:input", "speed_Of_Sound_umps must be a real double scalar.");
    if (!mxIsScalar(prhs[4]) || !mxIsDouble(prhs[4]))
        mexErrMsgIdAndTxt("PCIimagingSparseCUDA:input", "sampling_Freq must be a real double scalar.");
    if (!mxIsDouble(prhs[5]) || mxIsComplex(prhs[5]))
        mexErrMsgIdAndTxt("PCIimagingSparseCUDA:input", "image_Range_X_um must be a real double vector.");
    if (!mxIsDouble(prhs[6]) || mxIsComplex(prhs[6]))
        mexErrMsgIdAndTxt("PCIimagingSparseCUDA:input", "image_Range_Z_um must be a real double vector.");
    if (!mxIsScalar(prhs[7]) || !mxIsDouble(prhs[7]))
        mexErrMsgIdAndTxt("PCIimagingSparseCUDA:input", "p must be a real double scalar.");
    if (!mxIsDouble(prhs[8]) || mxIsComplex(prhs[8]))
        mexErrMsgIdAndTxt("PCIimagingSparseCUDA:input", "all_delays_matrix must be a real double matrix.");

    const double *h_RF_Arr = mxGetPr(prhs[0]);
    size_t nfft = mxGetM(prhs[0]);
    size_t ncols = mxGetN(prhs[0]);

    const double *h_image_Range_X = mxGetPr(prhs[5]);
    size_t numX = mxGetNumberOfElements(prhs[5]);
    const double *h_image_Range_Z = mxGetPr(prhs[6]);
    size_t numZ = mxGetNumberOfElements(prhs[6]);
    size_t total_pixels = numX * numZ;

    const mxArray *mxAllDelays = prhs[8];
    const double *h_all_delays = mxGetPr(mxAllDelays);
    size_t delay_rows = mxGetM(mxAllDelays);
    size_t delay_cols = mxGetN(mxAllDelays);

    if (delay_rows != ncols || delay_cols != total_pixels)
    {
        mexPrintf("Expected delay matrixsize: %zu x %zu, Got: %zu x %zu\n", ncols, total_pixels, delay_rows, delay_cols);
        mexErrMsgIdAndTxt("PCIimagingSparseCUDA:input", "all_delays_matrix has incorrect dimensions. Should be ncols x total_pixels.");
    }

    double p_param = mxGetScalar(prhs[7]);
    double pp = 1.0 / p_param;
    double inv_pp = p_param;

    mexPrintf("Starting CUDA PCI Imaging (v6 - Host Delays): Image=%zu(X) x %zu(Z), RF=%zu(nfft) x %zu(ncols), p=%.2f\n", numX, numZ, nfft, ncols, p_param);

    plhs[0] = mxCreateDoubleMatrix(numX, numZ, mxREAL);
    double *h_beamformed_Image = mxGetPr(plhs[0]);

    cufftDoubleComplex *d_RFFT;
    double *d_fftBin;
    double *d_beamformed_Image;

    CUDA_CHECK(cudaMalloc(&d_RFFT, nfft * ncols * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_fftBin, nfft * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_beamformed_Image, total_pixels * sizeof(double)));

    CUDA_CHECK(cudaMemset(d_beamformed_Image, 0, total_pixels * sizeof(double)));

    int blockSize = 256;

    int binStart = nfft / 2;
    int gridSize_nfft = (nfft + blockSize - 1) / blockSize;
    computeFftBinKernel<<<gridSize_nfft, blockSize>>>(d_fftBin, nfft, binStart);
    KERNEL_CHECK();

    cufftDoubleComplex *d_RData_complex_temp;
    double *d_RF_Arr_temp;
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

    int batch_size = 256;
    mexPrintf("Using pixel batch size: %d\n", batch_size);

    double *d_batch_delays;
    cufftDoubleComplex *d_batch_ShiftData, *d_batch_ShiftData_time;
    double *d_batch_temp, *d_batch_QpDAS, *d_batch_rowSum, *d_batch_pDAS, *d_batch_DCoffset;
    size_t batch_delay_size = (size_t)ncols * batch_size * sizeof(double);
    size_t batch_complex_size = (size_t)nfft * ncols * batch_size * sizeof(cufftDoubleComplex);
    size_t batch_real_size = (size_t)nfft * ncols * batch_size * sizeof(double);
    size_t batch_nfft_size = (size_t)nfft * batch_size * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_batch_delays, batch_delay_size));
    CUDA_CHECK(cudaMalloc(&d_batch_ShiftData, batch_complex_size));
    CUDA_CHECK(cudaMalloc(&d_batch_ShiftData_time, batch_complex_size));
    CUDA_CHECK(cudaMalloc(&d_batch_temp, batch_real_size));
    CUDA_CHECK(cudaMalloc(&d_batch_QpDAS, batch_real_size));
    CUDA_CHECK(cudaMalloc(&d_batch_rowSum, batch_nfft_size));
    CUDA_CHECK(cudaMalloc(&d_batch_pDAS, batch_nfft_size));
    CUDA_CHECK(cudaMalloc(&d_batch_DCoffset, batch_nfft_size));

    int rank = 1;
    int n[] = {(int)nfft};
    int istride = 1, ostride = 1;
    int idist = nfft, odist = nfft;
    int inembed[] = {(int)nfft};
    int onembed[] = {(int)nfft};
    cufftHandle plan_ifft_batch_full;
    int batch_count_full = ncols * batch_size;
    CUFFT_CHECK(cufftPlanMany(&plan_ifft_batch_full, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2Z, batch_count_full));
    cufftHandle plan_ifft_batch_last = 0;
    int last_batch_size_actual = total_pixels % batch_size;
    if (last_batch_size_actual == 0 && total_pixels > 0)
        last_batch_size_actual = batch_size;
    if (last_batch_size_actual != batch_size && last_batch_size_actual > 0)
    {
        int batch_count_last = ncols * last_batch_size_actual;
        CUFFT_CHECK(cufftPlanMany(&plan_ifft_batch_last, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2Z, batch_count_last));
    }
    else
    {
        plan_ifft_batch_last = plan_ifft_batch_full;
    }

    for (size_t batch_start_idx = 0; batch_start_idx < total_pixels; batch_start_idx += batch_size)
    {
        int current_batch_size = std::min((int)batch_size, (int)(total_pixels - batch_start_idx));
        if (current_batch_size <= 0)
            continue;

        size_t current_total_elements_ncols = (size_t)nfft * ncols * current_batch_size;
        size_t current_total_elements_nfft = (size_t)nfft * current_batch_size;

        cufftHandle current_plan = plan_ifft_batch_full;
        if ((batch_start_idx + current_batch_size >= total_pixels) && plan_ifft_batch_last != plan_ifft_batch_full)
        {
            current_plan = plan_ifft_batch_last;
        }

        const double *h_delay_source_ptr = h_all_delays + batch_start_idx * ncols;
        size_t copy_size_bytes = (size_t)ncols * current_batch_size * sizeof(double);
        CUDA_CHECK(cudaMemcpy(d_batch_delays, h_delay_source_ptr, copy_size_bytes, cudaMemcpyHostToDevice));

        int gridSize_shift = (current_total_elements_ncols + blockSize - 1) / blockSize;
        shiftDataBatchKernel<<<gridSize_shift, blockSize>>>(d_RFFT, d_fftBin, d_batch_delays, nfft, ncols, current_batch_size, d_batch_ShiftData);
        KERNEL_CHECK();

        CUFFT_CHECK(cufftExecZ2Z(current_plan, d_batch_ShiftData, d_batch_ShiftData_time, CUFFT_INVERSE));

        double scale = 1.0 / (double)nfft;
        int gridSize_scale_extract = (current_total_elements_ncols + blockSize - 1) / blockSize;
        scaleIFFTBatchKernel<<<gridSize_scale_extract, blockSize>>>(d_batch_ShiftData_time, scale, current_total_elements_ncols);
        KERNEL_CHECK();

        extractRealPartBatchKernel<<<gridSize_scale_extract, blockSize>>>(d_batch_ShiftData_time, current_total_elements_ncols, d_batch_temp);
        KERNEL_CHECK();

        pthrootElementwiseBatchKernel<<<gridSize_scale_extract, blockSize>>>(d_batch_temp, pp, nfft, ncols, current_batch_size, d_batch_QpDAS);
        KERNEL_CHECK();

        int threads_for_reduce_ncols = 256;
        dim3 gridDimReduceCol(nfft, current_batch_size);
        dim3 blockDimReduceCol(threads_for_reduce_ncols, 1);
        size_t sharedMemReduceCol = threads_for_reduce_ncols * sizeof(double);
        if (gridDimReduceCol.x > 0 && gridDimReduceCol.y > 0 && blockDimReduceCol.x > 0)
        {
            pthrootRowSumBatchKernel<<<gridDimReduceCol, blockDimReduceCol, sharedMemReduceCol>>>(d_batch_QpDAS, nfft, ncols, current_batch_size, d_batch_rowSum);
            KERNEL_CHECK();
        }

        int gridSize_finalP = (current_total_elements_nfft + blockSize - 1) / blockSize;
        if (gridSize_finalP > 0 && blockSize > 0 && current_total_elements_nfft > 0)
        {
            pthrootFinalBatchKernel<<<gridSize_finalP, blockSize>>>(d_batch_rowSum, inv_pp, nfft, current_batch_size, d_batch_pDAS);
            KERNEL_CHECK();
        }

        if (gridDimReduceCol.x > 0 && gridDimReduceCol.y > 0 && blockDimReduceCol.x > 0)
        {
            calculateDCoffsetBatchKernel<<<gridDimReduceCol, blockDimReduceCol, sharedMemReduceCol>>>(d_batch_temp, nfft, ncols, current_batch_size, d_batch_DCoffset);
            KERNEL_CHECK();
        }

        int threads_for_reduce_nfft = 256;
        if (nfft <= 32)
            threads_for_reduce_nfft = 32;
        else if (nfft <= 64)
            threads_for_reduce_nfft = 64;
        else if (nfft <= 128)
            threads_for_reduce_nfft = 128;
        else if (nfft <= 256)
            threads_for_reduce_nfft = 256;
        else if (nfft <= 512)
            threads_for_reduce_nfft = 512;
        else
            threads_for_reduce_nfft = 1024;
        dim3 gridDimFinalSum(current_batch_size);
        dim3 blockDimFinalSum(threads_for_reduce_nfft);
        size_t sharedMemFinalSum = threads_for_reduce_nfft * sizeof(double);
        if (gridDimFinalSum.x > 0 && blockDimFinalSum.x > 0)
        {
            finalPixelValueBatchKernel<<<gridDimFinalSum, blockDimFinalSum, sharedMemFinalSum>>>(d_batch_pDAS, d_batch_DCoffset, nfft, current_batch_size, batch_start_idx, numX, d_beamformed_Image);
            KERNEL_CHECK();
        }
    }
    mexPrintf("Finished processing all batches.\n");

    CUDA_CHECK(cudaMemcpy(h_beamformed_Image, d_beamformed_Image, total_pixels * sizeof(double), cudaMemcpyDeviceToHost));

    CUFFT_CHECK(cufftDestroy(plan_ifft_batch_full));
    if (plan_ifft_batch_last != plan_ifft_batch_full && plan_ifft_batch_last != 0)
    {
        CUFFT_CHECK(cufftDestroy(plan_ifft_batch_last));
    }
    CUDA_CHECK(cudaFree(d_RFFT));
    CUDA_CHECK(cudaFree(d_fftBin));
    CUDA_CHECK(cudaFree(d_beamformed_Image));
    CUDA_CHECK(cudaFree(d_batch_delays));
    CUDA_CHECK(cudaFree(d_batch_ShiftData));
    CUDA_CHECK(cudaFree(d_batch_ShiftData_time));
    CUDA_CHECK(cudaFree(d_batch_temp));
    CUDA_CHECK(cudaFree(d_batch_QpDAS));
    CUDA_CHECK(cudaFree(d_batch_rowSum));
    CUDA_CHECK(cudaFree(d_batch_pDAS));
    CUDA_CHECK(cudaFree(d_batch_DCoffset));

    mexPrintf("CUDA PCI Imaging complete.\n");
}