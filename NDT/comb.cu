#include "mex.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define MAX_ELEMENTS 64
#define MAX_SUBAPERTURES 128

// CUDA kernel for normalization
__global__ void norm_kernel(float* data, float norm, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) data[i] *= norm;
}

// CUDA kernel for Hilbert transform frequency domain manipulation
__global__ void hilbert_kernel(cufftComplex* data, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N/2 + 1) return;
    
    if (i > 0 && i < N/2) { 
        data[i].x *= 2.0f;
        data[i].y *= 2.0f;
    }
}

// Hilbert transform using cuFFT
void hilbertTransform(float* d_signal, float* d_output, int width, int height) {
    int N = width;
    cufftHandle planR2C, planC2R;
    cufftComplex* d_spectrum;
    
    cudaMalloc((void**)&d_spectrum, sizeof(cufftComplex) * (N/2 + 1) * height);
    
    cufftPlan1d(&planR2C, N, CUFFT_R2C, height);
    cufftPlan1d(&planC2R, N, CUFFT_C2R, height);
    
    // Forward FFT
    cufftExecR2C(planR2C, d_signal, d_spectrum);
    
    // Modify spectrum to create Hilbert transform
    int blocksPerGrid = (N/2 + 1 + 255) / 256;
    hilbert_kernel<<<blocksPerGrid, 256>>>(d_spectrum, N);
    
    // Inverse FFT
    cufftExecC2R(planC2R, d_spectrum, d_output);
    
    // Normalize
    float normFactor = 1.0f / N;
    norm_kernel<<<(N * height + 255) / 256, 256>>>(d_output, normFactor, N * height);
    
    // Clean up
    cufftDestroy(planR2C);
    cufftDestroy(planC2R);
    cudaFree(d_spectrum);
}

// Device function for p-th order coherence factor
__device__ float pthcoherencefactor(const float* temp, int len, float p) {
    float sum1 = 0.0f, sum2 = 0.0f;
    for (int i = 0; i < len; ++i) {
        float val = temp[i];
        float signval = (val > 0.0f) - (val < 0.0f);
        float absval = fabsf(val);
        sum1 += signval * powf(absval, 1.0f/p);
        sum2 += absval * absval;
    }
    float nR = (sum1 > 0.0f ? 1.0f : (sum1 < 0.0f ? -1.0f : 0.0f)) * powf(fabsf(sum1), p);
    float Nr = nR * nR;
    float Dr = sum2;
    if (Dr == 0.0f) Dr = 1.1920929e-07f; // epsilon for float
    float CF = (1.0f/len) * (Nr / Dr);
    return CF;
}

// CUDA kernel for beamforming (matches the pthcoherenceNDT function)
__global__ void beamform_kernel(
    const float* __restrict__ RData, int RData_rows, int RData_cols,
    const float* __restrict__ element_Pos, int nElements,
    float speed_Of_Sound_umps, float RF_Start_Time, float fs,
    const float* __restrict__ BeamformX, int nBeamformX,
    const float* __restrict__ BeamformZ, int nBeamformZ,
    const float* __restrict__ element_loc,
    float p,
    float* __restrict__ BeamformData)
{
    int Xi = blockIdx.x * blockDim.x + threadIdx.x;
    int Zi = blockIdx.y * blockDim.y + threadIdx.y;
    if (Xi >= nBeamformX || Zi >= nBeamformZ) return;

    float sum = 0.0f;
    float temp[MAX_ELEMENTS];

    for (int ex = 0; ex < RData_cols; ++ex) {  // Changed from nElements to RData_cols
        float dx1 = BeamformX[Xi] - element_loc[0];
        float dz1 = BeamformZ[Zi];
        float dx2 = BeamformX[Xi] - element_Pos[ex];
        float dz2 = BeamformZ[Zi];
        float distance_Along_RF = sqrtf(dx1*dx1 + dz1*dz1) + sqrtf(dx2*dx2 + dz2*dz2);
        float time_Pt_Along_RF = distance_Along_RF / speed_Of_Sound_umps;
        int samples = __float2int_rn((time_Pt_Along_RF - RF_Start_Time) * fs) + 1;
        if (samples > RData_rows || samples < 1) {
            temp[ex] = 0.0f;
        } else {
            float val = RData[IDX2C(samples-1, ex, RData_rows)];
            float signval = (val > 0.0f) - (val < 0.0f);
            float absval = fabsf(val);
            temp[ex] = signval * absval;
            sum += temp[ex];
        }
    }

    float pCF = pthcoherencefactor(temp, RData_cols, p);  // Changed from nElements to RData_cols
    BeamformData[IDX2C(Xi, Zi, nBeamformX)] = sum * pCF;
}

// CUDA error check
void checkCudaError(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        mexErrMsgIdAndTxt("comb_mex:CUDAError", "%s: %s", msg, cudaGetErrorString(err));
    }
}

// MEX gateway
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 10)
        mexErrMsgIdAndTxt("comb_mex:nrhs", "Ten inputs required.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("comb_mex:nlhs", "One output allowed.");

    for (int i = 0; i < 10; ++i) {
        if (i == 2 || i == 3) continue; // l is int32, Co is int32
        if (mxGetClassID(prhs[i]) != mxSINGLE_CLASS && i != 2 && i != 3)
            mexErrMsgIdAndTxt("comb_mex:type", "All inputs except l, Co must be single.");
    }

    const float* RF_Data = (const float*)mxGetData(prhs[0]);
    int RF_rows = (int)mxGetM(prhs[0]);
    int RF_cols = (int)mxGetN(prhs[0]);

    const float* element_Pos_Array_um_X = (const float*)mxGetData(prhs[1]);
    int nElements = (int)mxGetNumberOfElements(prhs[1]);
    
    const int* l = (const int*)mxGetData(prhs[2]);
    int nSubApertures = (int)mxGetNumberOfElements(prhs[2]);
    int Co = *((const int*)mxGetData(prhs[3]));

    float speed_Of_Sound_umps = *((const float*)mxGetData(prhs[4]));
    float RF_Start_Time = *((const float*)mxGetData(prhs[5]));
    float fs = *((const float*)mxGetData(prhs[6]));

    const float* BeamformX = (const float*)mxGetData(prhs[7]);
    int nBeamformX = (int)mxGetNumberOfElements(prhs[7]);

    const float* BeamformZ = (const float*)mxGetData(prhs[8]);
    int nBeamformZ = (int)mxGetNumberOfElements(prhs[8]);

    float p = *((const float*)mxGetData(prhs[9]));

    // Validate array sizes and memory requirement
    if (nElements > MAX_ELEMENTS) {
        mexErrMsgIdAndTxt("comb_mex:maxElements", "Number of elements (%d) exceeds MAX_ELEMENTS (%d)", 
                         nElements, MAX_ELEMENTS);
    }
    
    int max_subrf_cols = 64/Co;
    if (max_subrf_cols > MAX_ELEMENTS) {
        mexErrMsgIdAndTxt("comb_mex:maxElements", "Number of sub-aperture columns (%d) exceeds MAX_ELEMENTS (%d)", 
                         max_subrf_cols, MAX_ELEMENTS);
    }

    plhs[0] = mxCreateNumericMatrix(nBeamformX, nBeamformZ, mxSINGLE_CLASS, mxREAL);
    float* tem_pcf = (float*)mxGetData(plhs[0]);
    
    // Allocate host memory
    float* h_sub_RF_Data = (float*)malloc(RF_rows * max_subrf_cols * sizeof(float));
    if (!h_sub_RF_Data) {
        mexErrMsgIdAndTxt("comb_mex:malloc", "Failed to allocate memory for h_sub_RF_Data");
    }
    
    float* h_BeamformData = (float*)malloc(nBeamformX * nBeamformZ * sizeof(float));
    if (!h_BeamformData) {
        free(h_sub_RF_Data);
        mexErrMsgIdAndTxt("comb_mex:malloc", "Failed to allocate memory for h_BeamformData");
    }
    
    float* h_BeamformData_hilbert = (float*)malloc(nBeamformX * nBeamformZ * sizeof(float));
    if (!h_BeamformData_hilbert) {
        free(h_sub_RF_Data);
        free(h_BeamformData);
        mexErrMsgIdAndTxt("comb_mex:malloc", "Failed to allocate memory for h_BeamformData_hilbert");
    }
    
    float* h_tem_pcf = (float*)calloc(nBeamformX * nBeamformZ, sizeof(float));
    if (!h_tem_pcf) {
        free(h_sub_RF_Data);
        free(h_BeamformData);
        free(h_BeamformData_hilbert);
        mexErrMsgIdAndTxt("comb_mex:malloc", "Failed to allocate memory for h_tem_pcf");
    }
    
    float h_element_loc[2] = {0.0f, 0.0f};
    
    // Allocate device memory
    float *d_RF_Data = NULL;
    if (cudaMalloc(&d_RF_Data, RF_rows * RF_cols * sizeof(float)) != cudaSuccess) {
        free(h_sub_RF_Data); free(h_BeamformData);
        free(h_BeamformData_hilbert); free(h_tem_pcf);
        mexErrMsgIdAndTxt("comb_mex:cudaMalloc", "Failed to allocate device memory for d_RF_Data");
    }
    
    float *d_element_Pos_Array_um_X = NULL;
    if (cudaMalloc(&d_element_Pos_Array_um_X, nElements * sizeof(float)) != cudaSuccess) {
        cudaFree(d_RF_Data);
        free(h_sub_RF_Data); free(h_BeamformData);
        free(h_BeamformData_hilbert); free(h_tem_pcf);
        mexErrMsgIdAndTxt("comb_mex:cudaMalloc", "Failed to allocate device memory for d_element_Pos_Array_um_X");
    }
    
    float *d_BeamformX = NULL;
    if (cudaMalloc(&d_BeamformX, nBeamformX * sizeof(float)) != cudaSuccess) {
        cudaFree(d_RF_Data); cudaFree(d_element_Pos_Array_um_X);
        free(h_sub_RF_Data); free(h_BeamformData);
        free(h_BeamformData_hilbert); free(h_tem_pcf);
        mexErrMsgIdAndTxt("comb_mex:cudaMalloc", "Failed to allocate device memory for d_BeamformX");
    }
    
    float *d_BeamformZ = NULL;
    if (cudaMalloc(&d_BeamformZ, nBeamformZ * sizeof(float)) != cudaSuccess) {
        cudaFree(d_RF_Data); cudaFree(d_element_Pos_Array_um_X); cudaFree(d_BeamformX);
        free(h_sub_RF_Data); free(h_BeamformData);
        free(h_BeamformData_hilbert); free(h_tem_pcf);
        mexErrMsgIdAndTxt("comb_mex:cudaMalloc", "Failed to allocate device memory for d_BeamformZ");
    }
    
    float *d_sub_RF_Data = NULL;
    if (cudaMalloc(&d_sub_RF_Data, RF_rows * max_subrf_cols * sizeof(float)) != cudaSuccess) {
        cudaFree(d_RF_Data); cudaFree(d_element_Pos_Array_um_X);
        cudaFree(d_BeamformX); cudaFree(d_BeamformZ);
        free(h_sub_RF_Data); free(h_BeamformData);
        free(h_BeamformData_hilbert); free(h_tem_pcf);
        mexErrMsgIdAndTxt("comb_mex:cudaMalloc", "Failed to allocate device memory for d_sub_RF_Data");
    }
    
    float *d_element_loc = NULL;
    if (cudaMalloc(&d_element_loc, 2 * sizeof(float)) != cudaSuccess) {
        cudaFree(d_RF_Data); cudaFree(d_element_Pos_Array_um_X);
        cudaFree(d_BeamformX); cudaFree(d_BeamformZ); cudaFree(d_sub_RF_Data);
        free(h_sub_RF_Data); free(h_BeamformData);
        free(h_BeamformData_hilbert); free(h_tem_pcf);
        mexErrMsgIdAndTxt("comb_mex:cudaMalloc", "Failed to allocate device memory for d_element_loc");
    }
    
    float *d_BeamformData = NULL;
    if (cudaMalloc(&d_BeamformData, nBeamformX * nBeamformZ * sizeof(float)) != cudaSuccess) {
        cudaFree(d_RF_Data); cudaFree(d_element_Pos_Array_um_X);
        cudaFree(d_BeamformX); cudaFree(d_BeamformZ);
        cudaFree(d_sub_RF_Data); cudaFree(d_element_loc);
        free(h_sub_RF_Data); free(h_BeamformData);
        free(h_BeamformData_hilbert); free(h_tem_pcf);
        mexErrMsgIdAndTxt("comb_mex:cudaMalloc", "Failed to allocate device memory for d_BeamformData");
    }
    
    float *d_BeamformData_hilbert = NULL;
    if (cudaMalloc(&d_BeamformData_hilbert, nBeamformX * nBeamformZ * sizeof(float)) != cudaSuccess) {
        cudaFree(d_RF_Data); cudaFree(d_element_Pos_Array_um_X);
        cudaFree(d_BeamformX); cudaFree(d_BeamformZ);
        cudaFree(d_sub_RF_Data); cudaFree(d_element_loc); cudaFree(d_BeamformData);
        free(h_sub_RF_Data); free(h_BeamformData);
        free(h_BeamformData_hilbert); free(h_tem_pcf);
        mexErrMsgIdAndTxt("comb_mex:cudaMalloc", "Failed to allocate device memory for d_BeamformData_hilbert");
    }

    // Copy constant data to device
    if (cudaMemcpy(d_element_Pos_Array_um_X, element_Pos_Array_um_X, nElements * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        mexErrMsgIdAndTxt("comb_mex:cudaMemcpy", "Failed to copy element_Pos_Array_um_X to device");
    }
    
    if (cudaMemcpy(d_BeamformX, BeamformX, nBeamformX * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        mexErrMsgIdAndTxt("comb_mex:cudaMemcpy", "Failed to copy BeamformX to device");
    }
    
    if (cudaMemcpy(d_BeamformZ, BeamformZ, nBeamformZ * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        mexErrMsgIdAndTxt("comb_mex:cudaMemcpy", "Failed to copy BeamformZ to device");
    }

    dim3 block(16, 16);
    dim3 grid((nBeamformX + block.x - 1) / block.x, (nBeamformZ + block.y - 1) / block.y);

    // Loop over sub-apertures (just like in MATLAB)
    for (int j = 0; j < nSubApertures; ++j) {
        // Extract RF data for this sub-aperture
        int baseCol = 64 * (l[j] - 1);
        int subRFCols = 0;
        
        // Extract sub-RF data (equivalent to rf_Data = RF_Data(:, 64 * l(j) - 63 : 64 * l(j)); rf_Data = rf_Data(:, 1:Co:64);)
        for (int k = 0; k < 64; k += Co) {
            int srcCol = baseCol + k;
            if (srcCol >= 0 && srcCol < RF_cols) {
                for (int r = 0; r < RF_rows; ++r) {
                    h_sub_RF_Data[IDX2C(r, subRFCols, RF_rows)] = RF_Data[IDX2C(r, srcCol, RF_rows)];
                }
                subRFCols++;
            }
        }
        
        // Set Single_element_loc (equivalent to Single_element_loc = [element_Pos_Array_um_X(j), 0];)
        h_element_loc[0] = element_Pos_Array_um_X[j];
        h_element_loc[1] = 0.0f;
        
        // Copy sub-RF data and Single_element_loc to device
        if (cudaMemcpy(d_sub_RF_Data, h_sub_RF_Data, RF_rows * subRFCols * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            mexErrMsgIdAndTxt("comb_mex:cudaMemcpy", "Failed to copy sub_RF_Data to device");
        }
        
        if (cudaMemcpy(d_element_loc, h_element_loc, 2 * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            mexErrMsgIdAndTxt("comb_mex:cudaMemcpy", "Failed to copy element_loc to device");
        }
        
        // Call beamform_kernel (equivalent to [I] = pthcoherenceNDT(...))
        beamform_kernel<<<grid, block>>>(
            d_sub_RF_Data, RF_rows, subRFCols,
            d_element_Pos_Array_um_X, nElements,
            speed_Of_Sound_umps, RF_Start_Time, fs,
            d_BeamformX, nBeamformX,
            d_BeamformZ, nBeamformZ,
            d_element_loc,
            p,
            d_BeamformData);
        checkCudaError("beamform_kernel");
        
        // Take absolute value of Hilbert transform (equivalent to temp = abs(hilbert(I));)
        hilbertTransform(d_BeamformData, d_BeamformData_hilbert, nBeamformX, nBeamformZ);
        
        // Copy result back to host
        if (cudaMemcpy(h_BeamformData_hilbert, d_BeamformData_hilbert, nBeamformX * nBeamformZ * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            mexErrMsgIdAndTxt("comb_mex:cudaMemcpy", "Failed to copy BeamformData_hilbert from device");
        }
        
        // Accumulate result (equivalent to tem_pcf = tem_pcf + temp;)
        for (int i = 0; i < nBeamformX * nBeamformZ; ++i) {
            h_tem_pcf[i] += fabsf(h_BeamformData_hilbert[i]); // Take abs
        }
    }
    
    // Copy accumulated result back to output
    memcpy(tem_pcf, h_tem_pcf, nBeamformX * nBeamformZ * sizeof(float));
    
    // Free memory
    cudaFree(d_RF_Data);
    cudaFree(d_element_Pos_Array_um_X);
    cudaFree(d_BeamformX);
    cudaFree(d_BeamformZ);
    cudaFree(d_sub_RF_Data);
    cudaFree(d_element_loc);
    cudaFree(d_BeamformData);
    cudaFree(d_BeamformData_hilbert);
    
    free(h_sub_RF_Data);
    free(h_BeamformData);
    free(h_BeamformData_hilbert);
    free(h_tem_pcf);
}