#include "mex.h"
#include <cuda_runtime.h>
#include <math.h>

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))
#define MAX_ELEMENTS 128
#define MAX_SUBAPERTURES 64

__device__ float pthcoherencefactor(const float *temp, int len, float p)
{
    float sum1 = 0.0f, sum2 = 0.0f;
    for (int i = 0; i < len; ++i)
    {
        float val = temp[i];
        float signval = (val > 0.0f) - (val < 0.0f);
        float absval = fabsf(val);
        sum1 += signval * powf(absval, 1.0f / p);
        sum2 += absval * absval;
    }
    float nR = (sum1 > 0.0f ? 1.0f : (sum1 < 0.0f ? -1.0f : 0.0f)) * powf(fabsf(sum1), p);
    float Nr = nR * nR;
    float Dr = sum2;
    if (Dr == 0.0f)
        Dr = 1.1920929e-07f; // eps('single')
    float CF = (1.0f / len) * (Nr / Dr);
    return CF;
}

__global__ void tem_pcf_kernel(
    const float *__restrict__ RF_Data, int RData_rows, int RData_cols,
    const float *__restrict__ element_Pos_Array_um_X, int nElements,
    const float *__restrict__ l, int nSubApertures,
    int Co,
    float speed_Of_Sound_umps, float RF_Start_Time, float fs,
    const float *__restrict__ BeamformX, int nBeamformX,
    const float *__restrict__ BeamformZ, int nBeamformZ,
    float p,
    float *__restrict__ tem_pcf)
{
    int Xi = blockIdx.x * blockDim.x + threadIdx.x;
    int Zi = blockIdx.y * blockDim.y + threadIdx.y;
    if (Xi >= nBeamformX || Zi >= nBeamformZ)
        return;

    float acc = 0.0f;

    for (int j = 0; j < nSubApertures; ++j)
    {
        int l_j = (int)l[j];               // l is 1-based in MATLAB
        int start_col = 64 * l_j - 63 - 1; // -1 for 0-based indexing
        int nCols = 64 / Co;
        if (nCols > MAX_ELEMENTS)
            nCols = MAX_ELEMENTS;

        float temp[MAX_ELEMENTS];
        float sum = 0.0f;

        // Subsampled columns
        for (int ex = 0; ex < nCols; ++ex)
        {
            int col_idx = start_col + ex * Co;
            if (col_idx >= RData_cols)
            {
                temp[ex] = 0.0f;
                continue;
            }
            float dx1 = BeamformX[Xi] - element_Pos_Array_um_X[j];
            float dz1 = BeamformZ[Zi];
            float dx2 = BeamformX[Xi] - element_Pos_Array_um_X[col_idx];
            float dz2 = BeamformZ[Zi];
            float distance_Along_RF = sqrtf(dx1 * dx1 + dz1 * dz1) + sqrtf(dx2 * dx2 + dz2 * dz2);
            float time_Pt_Along_RF = distance_Along_RF / speed_Of_Sound_umps;
            int samples = __float2int_rn((time_Pt_Along_RF - RF_Start_Time) * fs) + 1;
            if (samples > RData_rows || samples < 1)
            {
                temp[ex] = 0.0f;
            }
            else
            {
                float val = RF_Data[IDX2C(samples - 1, col_idx, RData_rows)];
                float signval = (val > 0.0f) - (val < 0.0f);
                float absval = fabsf(val);
                temp[ex] = signval * absval;
                sum += temp[ex];
            }
        }

        float pCF = pthcoherencefactor(temp, nCols, p);
        float I = sum * pCF;
        acc += fabsf(I); // Envelope (abs) -- for true Hilbert, post-process in MATLAB
    }

    tem_pcf[IDX2C(Xi, Zi, nBeamformX)] = acc;
}

void checkCudaError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        mexErrMsgIdAndTxt("tem_pcf_cuda:CUDAError", "%s: %s", msg, cudaGetErrorString(err));
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Inputs: RF_Data, element_Pos_Array_um_X, l, Co, speed_Of_Sound_umps, RF_Start_Time, fs, BeamformX, BeamformZ, p
    if (nrhs != 10)
        mexErrMsgIdAndTxt("tem_pcf_cuda:nrhs", "Ten inputs required.");
    if (nlhs > 1)
        mexErrMsgIdAndTxt("tem_pcf_cuda:nlhs", "One output allowed.");

    for (int i = 0; i < 10; ++i)
    {
        if (mxGetClassID(prhs[i]) != mxSINGLE_CLASS)
            mexErrMsgIdAndTxt("tem_pcf_cuda:type", "All inputs must be single.");
    }

    const float *RF_Data = (const float *)mxGetData(prhs[0]);
    int RData_rows = (int)mxGetM(prhs[0]);
    int RData_cols = (int)mxGetN(prhs[0]);

    const float *element_Pos_Array_um_X = (const float *)mxGetData(prhs[1]);
    int nElements = (int)mxGetNumberOfElements(prhs[1]);

    const float *l = (const float *)mxGetData(prhs[2]);
    int nSubApertures = (int)mxGetNumberOfElements(prhs[2]);

    int Co = (int)(*((const float *)mxGetData(prhs[3])));

    float speed_Of_Sound_umps = *((const float *)mxGetData(prhs[4]));
    float RF_Start_Time = *((const float *)mxGetData(prhs[5]));
    float fs = *((const float *)mxGetData(prhs[6]));

    const float *BeamformX = (const float *)mxGetData(prhs[7]);
    int nBeamformX = (int)mxGetNumberOfElements(prhs[7]);

    const float *BeamformZ = (const float *)mxGetData(prhs[8]);
    int nBeamformZ = (int)mxGetNumberOfElements(prhs[8]);

    float p = *((const float *)mxGetData(prhs[9]));

    plhs[0] = mxCreateNumericMatrix(nBeamformX, nBeamformZ, mxSINGLE_CLASS, mxREAL);
    float *tem_pcf = (float *)mxGetData(plhs[0]);

    float *d_RF_Data, *d_element_Pos_Array_um_X, *d_l, *d_BeamformX, *d_BeamformZ, *d_tem_pcf;
    cudaMalloc(&d_RF_Data, RData_rows * RData_cols * sizeof(float));
    cudaMalloc(&d_element_Pos_Array_um_X, nElements * sizeof(float));
    cudaMalloc(&d_l, nSubApertures * sizeof(float));
    cudaMalloc(&d_BeamformX, nBeamformX * sizeof(float));
    cudaMalloc(&d_BeamformZ, nBeamformZ * sizeof(float));
    cudaMalloc(&d_tem_pcf, nBeamformX * nBeamformZ * sizeof(float));
    checkCudaError("cudaMalloc");

    cudaMemcpy(d_RF_Data, RF_Data, RData_rows * RData_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_element_Pos_Array_um_X, element_Pos_Array_um_X, nElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, l, nSubApertures * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_BeamformX, BeamformX, nBeamformX * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_BeamformZ, BeamformZ, nBeamformZ * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError("cudaMemcpy H2D");

    dim3 block(16, 16);
    dim3 grid((nBeamformX + block.x - 1) / block.x, (nBeamformZ + block.y - 1) / block.y);
    tem_pcf_kernel<<<grid, block>>>(
        d_RF_Data, RData_rows, RData_cols,
        d_element_Pos_Array_um_X, nElements,
        d_l, nSubApertures,
        Co,
        speed_Of_Sound_umps, RF_Start_Time, fs,
        d_BeamformX, nBeamformX,
        d_BeamformZ, nBeamformZ,
        p,
        d_tem_pcf);
    checkCudaError("Kernel launch");

    cudaMemcpy(tem_pcf, d_tem_pcf, nBeamformX * nBeamformZ * sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaError("cudaMemcpy D2H");

    cudaFree(d_RF_Data);
    cudaFree(d_element_Pos_Array_um_X);
    cudaFree(d_l);
    cudaFree(d_BeamformX);
    cudaFree(d_BeamformZ);
    cudaFree(d_tem_pcf);
}