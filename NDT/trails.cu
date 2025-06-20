#include "mex.h"
#include <cuda_runtime.h>
#include <math.h>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define MAX_ELEMENTS 128

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
    if (Dr == 0.0f) Dr = 1.1920929e-07f; // eps('single')
    float CF = (1.0f/len) * (Nr / Dr);
    return CF;
}

__global__ void pthcoherenceNDT_kernel(
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

    for (int ex = 0; ex < nElements; ++ex) {
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

    float pCF = pthcoherencefactor(temp, nElements, p);
    BeamformData[IDX2C(Xi, Zi, nBeamformX)] = sum * pCF;
}

void checkCudaError(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        mexErrMsgIdAndTxt("pthcoherenceNDT_mex:CUDAError", "%s: %s", msg, cudaGetErrorString(err));
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 9)
        mexErrMsgIdAndTxt("pthcoherenceNDT_mex:nrhs", "Nine inputs required.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("pthcoherenceNDT_mex:nlhs", "One output allowed.");

    for (int i = 0; i < 9; ++i) {
        if (mxGetClassID(prhs[i]) != mxSINGLE_CLASS)
            mexErrMsgIdAndTxt("pthcoherenceNDT_mex:type", "All inputs must be single.");
    }

    const float* RData = (const float*)mxGetData(prhs[0]);
    int RData_rows = (int)mxGetM(prhs[0]);
    int RData_cols = (int)mxGetN(prhs[0]);

    const float* element_Pos = (const float*)mxGetData(prhs[1]);
    int nElements = (int)mxGetNumberOfElements(prhs[1]);
    if (nElements > MAX_ELEMENTS)
        mexErrMsgIdAndTxt("pthcoherenceNDT_mex:maxElements", "Increase MAX_ELEMENTS in kernel.");

    float speed_Of_Sound_umps = *((const float*)mxGetData(prhs[2]));
    float RF_Start_Time = *((const float*)mxGetData(prhs[3]));
    float fs = *((const float*)mxGetData(prhs[4]));

    const float* BeamformX = (const float*)mxGetData(prhs[5]);
    int nBeamformX = (int)mxGetNumberOfElements(prhs[5]);

    const float* BeamformZ = (const float*)mxGetData(prhs[6]);
    int nBeamformZ = (int)mxGetNumberOfElements(prhs[6]);

    const float* element_loc = (const float*)mxGetData(prhs[7]); // length 2

    float p = *((const float*)mxGetData(prhs[8]));

    plhs[0] = mxCreateNumericMatrix(nBeamformX, nBeamformZ, mxSINGLE_CLASS, mxREAL);
    float* BeamformData = (float*)mxGetData(plhs[0]);

    float *d_RData, *d_element_Pos, *d_BeamformX, *d_BeamformZ, *d_element_loc, *d_BeamformData;
    cudaMalloc(&d_RData, RData_rows*RData_cols*sizeof(float));
    cudaMalloc(&d_element_Pos, nElements*sizeof(float));
    cudaMalloc(&d_BeamformX, nBeamformX*sizeof(float));
    cudaMalloc(&d_BeamformZ, nBeamformZ*sizeof(float));
    cudaMalloc(&d_element_loc, 2*sizeof(float));
    cudaMalloc(&d_BeamformData, nBeamformX*nBeamformZ*sizeof(float));
    checkCudaError("cudaMalloc");

    cudaMemcpy(d_RData, RData, RData_rows*RData_cols*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_element_Pos, element_Pos, nElements*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_BeamformX, BeamformX, nBeamformX*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_BeamformZ, BeamformZ, nBeamformZ*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_element_loc, element_loc, 2*sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError("cudaMemcpy H2D");

    dim3 block(32,32);
    dim3 grid((nBeamformX+block.x-1)/block.x, (nBeamformZ+block.y-1)/block.y);
    pthcoherenceNDT_kernel<<<grid, block>>>(
        d_RData, RData_rows, RData_cols,
        d_element_Pos, nElements,
        speed_Of_Sound_umps, RF_Start_Time, fs,
        d_BeamformX, nBeamformX,
        d_BeamformZ, nBeamformZ,
        d_element_loc,
        p,
        d_BeamformData);
    checkCudaError("Kernel launch");

    cudaMemcpy(BeamformData, d_BeamformData, nBeamformX*nBeamformZ*sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaError("cudaMemcpy D2H");

    cudaFree(d_RData);
    cudaFree(d_element_Pos);
    cudaFree(d_BeamformX);
    cudaFree(d_BeamformZ);
    cudaFree(d_element_loc);
    cudaFree(d_BeamformData);
}