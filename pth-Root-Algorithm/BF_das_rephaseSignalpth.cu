/*
 * BF_das_rephaseSignalpth.cu
 *
 * This mex file implements a simplified version of the beamforming 
 * rephasing algorithm from BF_das_rephaseSignalpth.m using CUDA.
 *
 * Assumptions / Limitations:
 *   - The RF signal SIG is assumed to be a double‐precision, real array 
 *     with dimensions [nSamples x Nelements x nAngles] stored in MATLAB’s 
 *     column–major order.
 *   - The “IQ rephasing” (complex exponential multiplication) is omitted 
 *     here. (If needed, see the comments below.)
 *   - Only a subset of the PARAM structure fields is used.
 *
 * Compile with: mexcuda BF_das_rephaseSignalpth.cu
 *
 * Inputs:
 *   0: SIG      – RF data [nSamples x Nelements x nAngles]
 *   1: PARAM    – structure with fields:
 *                   - f0      : central frequency (fc)
 *                   - fs      : sampling frequency
 *                   - t0      : start time of receiving signal
 *                   - c       : speed of sound in the medium
 *                   - Nelements: number of elements
 *                   - pitch   : distance between element centers
 *                   - fnumber : f-number for reception apodization
 *                   - theta   : emission angle (in radians)
 *                   - xe      : vector of element positions (length = Nelements)
 *   2: X        – grid (flattened) for lateral positions (same size as Z)
 *   3: Z        – grid (flattened) for axial positions (same size as X)
 *
 * Output:
 *   0: migSIG1 – Beamformed image, reshaped to the same size as X and Z.
 */

#include "mex.h"
#include "matrix.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// CUDA kernel: each thread computes one output pixel for one angle.
__global__ void beamformKernel(const double *sig, int nSamples, int nElements, int nAngles,
                               const double *xe, const double *X, const double *Z, int nPixels,
                               double theta, double c, double fs, double t0, double fc, double fnumber,
                               double add_term, double *out, double p)
{

    // Determine pixel and angle indices from 2D thread block.
    int pixel = blockIdx.x * blockDim.x + threadIdx.x;  // index in X,Z arrays
    int angle = blockIdx.y * blockDim.y + threadIdx.y;    // angle index (third dim of SIG)
    if (pixel < nPixels && angle < nAngles)
    {
        // Get spatial coordinates for this pixel.
        double x = X[pixel];
        double z = Z[pixel];

        // Compute transmit delay term.
        double dtx = sin(theta)*x + cos(theta)*z + add_term;
        double sum = 0.0;
        double epsilon_0 = 1e-30;
        // Loop over each receive element.
        for (int k = 0; k < nElements; k++) {
            // Compute receive delay (distance) from element k to pixel.
            double dx = x - xe[k];
            double drx = sqrtf(dx*dx + z*z);
            double tau = (dtx + drx) / c;
            
            // Convert delay tau into a sample index.
            double idx = (tau - t0) * fs + 1.0;
            if (idx < 1.0 || idx > (double)(nSamples - 1)) {
                continue;  // out-of-bound indices contribute zero
            }
            int iidx = (int)floor(idx);
            double frac = idx - iidx;
            
            // Compute the linear indices for the two samples to interpolate.
            // MATLAB stores arrays in column-major order.
            // SIG has dimensions [nSamples x nElements x nAngles].
            int offset1 = iidx + k * nSamples + angle * (nSamples * nElements);
            int offset2 = (iidx + 1) + k * nSamples + angle * (nSamples * nElements);
            
            double sample1 = sig[offset1];
            double sample2 = sig[offset2];
            double sample = sample1 * (1.0 - frac) + sample2 * frac;
            double weights = powf(fabs(sample + epsilon_0), (1-p)/p);
            sample = sample * weights;
            
          
            // Apply the F-number mask.
            // (Here we assume that the mask is 1 when abs(x - xe[k]) < z/(2*fnumber))
            double mask = (fabs(x - xe[k]) < (z / (2.0 * fnumber))) ? 1.0 : 0.0;
            
            sum += sample * mask;
        } // end for each element

        // Write the accumulated result to the output.
        // The output is stored as a [nPixels x nAngles] array.
        out[pixel + angle * nPixels] = sum;
    }
}

// Gateway routine.
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    // Check for proper number of inputs.
    if (nrhs != 4) {
        mexErrMsgIdAndTxt("pth:InvalidInput",
                          "Four inputs required: SIG, PARAM, X, Z.");
    }

    // Input 0: SIG
    const mxArray *mxSIG = prhs[0];
    if (!mxIsDouble(mxSIG) || mxIsComplex(mxSIG)) {
        mexErrMsgIdAndTxt("BF_das_rephaseSignal:InputNotRealDouble",
                          "SIG must be a real double array.");
    }
    double *h_sig = mxGetPr(mxSIG);
    const mwSize *sigDims = mxGetDimensions(mxSIG);
    int nDimsSIG = mxGetNumberOfDimensions(mxSIG);
    if (nDimsSIG < 2) {
        mexErrMsgIdAndTxt("BF_das_rephaseSignal:InvalidSIG",
                          "SIG must have at least 2 dimensions.");
    }
    int nSamples = (int)sigDims[0];
    int nElements = (int)sigDims[1];
    int nAngles = (nDimsSIG >= 3) ? (int)sigDims[2] : 1;

    // Input 1: PARAM (a MATLAB struct)
    const mxArray *mxPARAM = prhs[1];
    if (!mxIsStruct(mxPARAM)) {
        mexErrMsgIdAndTxt("BF_das_rephaseSignal:InvalidPARAM",
                          "PARAM must be a structure.");
    }
    // Extract required fields.
    // Field names: 'f0', 'fs', 't0', 'c', 'fnumber', 'theta', 'xe'
    mxArray *field;
    double fc, fs, t0, c, fnumber, theta,p;
    double *xe = nullptr;
    
    field = mxGetField(mxPARAM, 0, "f0");
    if (field == NULL) mexErrMsgIdAndTxt("BF_das_rephaseSignal:MissingField", "PARAM.f0 missing.");
    fc = mxGetScalar(field);
    
    field = mxGetField(mxPARAM, 0, "fs");
    if (field == NULL) mexErrMsgIdAndTxt("BF_das_rephaseSignal:MissingField", "PARAM.fs missing.");
    fs = mxGetScalar(field);
    
    field = mxGetField(mxPARAM, 0, "t0");
    if (field == NULL) mexErrMsgIdAndTxt("BF_das_rephaseSignal:MissingField", "PARAM.t0 missing.");
    t0 = mxGetScalar(field);
    
    field = mxGetField(mxPARAM, 0, "c");
    if (field == NULL) mexErrMsgIdAndTxt("BF_das_rephaseSignal:MissingField", "PARAM.c missing.");
    c = mxGetScalar(field);
    field = mxGetField(mxPARAM, 0, "p_pDAS");
    if (field == NULL) mexErrMsgIdAndTxt("BF_das_rephaseSignal:MissingField", "PARAM.p missing.");
    p = mxGetScalar(field);
    
    field = mxGetField(mxPARAM, 0, "fnumber");
    if (field == NULL) mexErrMsgIdAndTxt("BF_das_rephaseSignal:MissingField", "PARAM.fnumber missing.");
    fnumber = mxGetScalar(field);
    
    field = mxGetField(mxPARAM, 0, "theta");
    if (field == NULL) mexErrMsgIdAndTxt("BF_das_rephaseSignal:MissingField", "PARAM.theta yo missing.");
    theta = mxGetScalar(field);
    
    field = mxGetField(mxPARAM, 0, "xe");
    if (field == NULL) mexErrMsgIdAndTxt("BF_das_rephaseSignal:MissingField", "PARAM.xe missing.");
    xe = mxGetPr(field);
    mwSize nxe = mxGetNumberOfElements(field);
    if ((int)nxe != nElements) {
        mexErrMsgIdAndTxt("BF_das_rephaseSignal:xeSizeMismatch", "Length of xe must equal Nelements.");
    }
    
    // For the transmit delay, compute TXdelay = (1/c)*tan(theta)*abs(xe - xe[0])
    // and then add_term = mean(TXdelay - min(TXdelay)) * c.
    double txdelay_min = fabs(xe[0] - xe[0]); // will be zero.
    double txdelay_sum = 0.0;
    double txdelay;
    double txdelay_min_found = 1e12;
    for (int k = 0; k < nElements; k++) {
        txdelay = (1.0 / c) * tan(theta) * fabs(xe[k] - xe[0]);
        if (txdelay < txdelay_min_found) txdelay_min_found = txdelay;
        txdelay_sum += txdelay;
    }
    double txdelay_mean = txdelay_sum / nElements;
    double add_term = (txdelay_mean - txdelay_min_found) * c;
    
    // Input 2: X grid.
    const mxArray *mxX = prhs[2];
    if (!mxIsDouble(mxX)) {
        mexErrMsgIdAndTxt("BF_das_rephaseSignal:InvalidX",
                          "X must be a double array.");
    }
    double *h_X = mxGetPr(mxX);
    mwSize nPixels_X = mxGetNumberOfElements(mxX);
    
    // Input 3: Z grid.
    const mxArray *mxZ = prhs[3];
    if (!mxIsDouble(mxZ)) {
        mexErrMsgIdAndTxt("BF_das_rephaseSignal:InvalidZ",
                          "Z must be a double array.");
    }
    double *h_Z = mxGetPr(mxZ);
    mwSize nPixels_Z = mxGetNumberOfElements(mxZ);
    
    // We assume that X and Z have the same number of elements.
    int nPixels = (int)nPixels_X;
    if (nPixels != (int)nPixels_Z) {
        mexErrMsgIdAndTxt("BF_das_rephaseSignal:GridSizeMismatch",
                          "X and Z must have the same number of elements.");
    }
    
    // Determine the size of the output.
    // In the MATLAB code, migSIG is accumulated over elements and then reshaped to the size of X.
    // Here, we produce an output array with dimensions [nPixels x nAngles].
    mwSize outDims[2] = { (mwSize)nPixels, (mwSize)nAngles };
    plhs[0] = mxCreateNumericArray(2, outDims, mxDOUBLE_CLASS, mxREAL);
    double *h_out = mxGetPr(plhs[0]);
    
    // Allocate GPU memory and copy input arrays.
    double *d_sig = nullptr, *d_xe = nullptr, *d_X = nullptr, *d_Z = nullptr, *d_out = nullptr;
    size_t sigSize = nSamples * nElements * nAngles * sizeof(double);
    size_t xeSize = nElements * sizeof(double);
    size_t gridSize = nPixels * sizeof(double);
    size_t outSize = nPixels * nAngles * sizeof(double);
    
    cudaMalloc((void**)&d_sig, sigSize);
    cudaMalloc((void**)&d_xe, xeSize);
    cudaMalloc((void**)&d_X, gridSize);
    cudaMalloc((void**)&d_Z, gridSize);
    cudaMalloc((void**)&d_out, outSize);
    
    cudaMemcpy(d_sig, h_sig, sigSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_xe, xe, xeSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, h_X, gridSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Z, h_Z, gridSize, cudaMemcpyHostToDevice);
    
    // Set up the CUDA grid.
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((nPixels + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (nAngles + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    // Launch the kernel.
    beamformKernel<<<numBlocks, threadsPerBlock>>>(d_sig, nSamples, nElements, nAngles,
                                                     d_xe, d_X, d_Z, nPixels,
                                                     theta, c, fs, t0, fc, fnumber,
                                                     add_term, d_out,p);

 
    
    // Wait for the kernel to complete.
    cudaDeviceSynchronize();
    
    // Copy the result back to host.
    cudaMemcpy(h_out, d_out, outSize, cudaMemcpyDeviceToHost);
    
    // (Optional) If you want to reshape the output to the same size as X and Z (i.e. a 2D image)
    // and if nAngles==1, then h_out is already of size [nPixels x 1]. 
    // If nAngles > 1, you might need additional compounding.
    
    // Free GPU memory.
    cudaFree(d_sig);
    cudaFree(d_xe);
    cudaFree(d_X);
    cudaFree(d_Z);
    cudaFree(d_out);
}
