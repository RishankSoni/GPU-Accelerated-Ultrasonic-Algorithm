#include "mex.h"
#include <cuda_runtime.h>
#include <math.h>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

__device__ double pthcoherencefactor(const double* temp, int len, double p) {
    double sum1 = 0.0, sum2 = 0.0;
    for (int i = 0; i < len; ++i) {
        double val = temp[i];
        double signval = (val > 0) - (val < 0);
        double absval = fabs(val);
        sum1 += signval * pow(absval, 1.0/p);
        sum2 += absval * absval;
    }
    double nR = (sum1 > 0 ? 1 : (sum1 < 0 ? -1 : 0)) * pow(fabs(sum1), p);
    double Nr = nR * nR;
    double Dr = sum2;
    if (Dr == 0) Dr = 2.220446049250313e-16; // eps
    double CF = (1.0/len) * (Nr / Dr);
    return CF;
}

__global__ void pthcoherenceNDT_kernel(
    const double* RData, mwSize RData_rows, mwSize RData_cols,
    const double* element_Pos, mwSize nElements,
    double speed_Of_Sound_umps, double RF_Start_Time, double fs,
    const double* BeamformX, mwSize nBeamformX,
    const double* BeamformZ, mwSize nBeamformZ,
    const double* element_loc,
    double p,
    double* BeamformData)
{
    int Xi = blockIdx.x * blockDim.x + threadIdx.x;
    int Zi = blockIdx.y * blockDim.y + threadIdx.y;
    if (Xi >= nBeamformX || Zi >= nBeamformZ) return;

    double sum = 0.0;
    double temp[128]; // max elements supported
    int valid_ex = 0;

    for (int ex = 0; ex < nElements; ++ex) {
        double dx1 = BeamformX[Xi] - element_loc[0];
        double dz1 = BeamformZ[Zi];
        double dx2 = BeamformX[Xi] - element_Pos[ex];
        double dz2 = BeamformZ[Zi];
        double distance_Along_RF = sqrt(dx1*dx1 + dz1*dz1) + sqrt(dx2*dx2 + dz2*dz2);
        double time_Pt_Along_RF = distance_Along_RF / speed_Of_Sound_umps;
        int samples = round((time_Pt_Along_RF - RF_Start_Time) * fs) + 1;
        if (samples > RData_rows || samples < 1) {
            temp[ex] = 0.0;
            continue;
        } else {
            double val = RData[IDX2C(samples-1, ex, RData_rows)];
            double signval = (val > 0) - (val < 0);
            double absval = fabs(val);
            temp[ex] = signval * absval;
            sum += temp[ex];
            valid_ex++;
        }
    }
    double pCF = pthcoherencefactor(temp, nElements, p);
    BeamformData[IDX2C(Xi, Zi, nBeamformX)] = sum * pCF;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Input checks omitted for brevity
    // Inputs: RData, element_Pos, speed_Of_Sound_umps, RF_Start_Time, fs, BeamformX, BeamformZ, element_loc, p

    const double* RData = mxGetPr(prhs[0]);
    mwSize RData_rows = mxGetM(prhs[0]);
    mwSize RData_cols = mxGetN(prhs[0]);

    const double* element_Pos = mxGetPr(prhs[1]);
    mwSize nElements = mxGetNumberOfElements(prhs[1]);

    double speed_Of_Sound_umps = mxGetScalar(prhs[2]);
    double RF_Start_Time = mxGetScalar(prhs[3]);
    double fs = mxGetScalar(prhs[4]);

    const double* BeamformX = mxGetPr(prhs[5]);
    mwSize nBeamformX = mxGetNumberOfElements(prhs[5]);

    const double* BeamformZ = mxGetPr(prhs[6]);
    mwSize nBeamformZ = mxGetNumberOfElements(prhs[6]);

    const double* element_loc = mxGetPr(prhs[7]); // length 2

    double p = mxGetScalar(prhs[8]);

    // Output
    plhs[0] = mxCreateDoubleMatrix(nBeamformX, nBeamformZ, mxREAL);
    double* BeamformData = mxGetPr(plhs[0]);

    // Copy data to device
    double *d_RData, *d_element_Pos, *d_BeamformX, *d_BeamformZ, *d_element_loc, *d_BeamformData;
    cudaMalloc(&d_RData, RData_rows*RData_cols*sizeof(double));
    cudaMalloc(&d_element_Pos, nElements*sizeof(double));
    cudaMalloc(&d_BeamformX, nBeamformX*sizeof(double));
    cudaMalloc(&d_BeamformZ, nBeamformZ*sizeof(double));
    cudaMalloc(&d_element_loc, 2*sizeof(double));
    cudaMalloc(&d_BeamformData, nBeamformX*nBeamformZ*sizeof(double));

    cudaMemcpy(d_RData, RData, RData_rows*RData_cols*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_element_Pos, element_Pos, nElements*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_BeamformX, BeamformX, nBeamformX*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_BeamformZ, BeamformZ, nBeamformZ*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_element_loc, element_loc, 2*sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 block(8,8);
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

    cudaMemcpy(BeamformData, d_BeamformData, nBeamformX*nBeamformZ*sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_RData);
    cudaFree(d_element_Pos);
    cudaFree(d_BeamformX);
    cudaFree(d_BeamformZ);
    cudaFree(d_element_loc);
    cudaFree(d_BeamformData);
}