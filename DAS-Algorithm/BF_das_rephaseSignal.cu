// BF_das_rephaseSignal.cu
//
// This CUDA mex file performs delay‐and‐sum beamforming.
// It assumes the input SIG is a double array of size [nSamples x nElements],
// the imaging grids X and Z are double arrays of size [M x N] (from meshgrid),
// and that PARAM is a MATLAB structure with fields: f0, fs, t0, c, fnumber, theta, and xe.
//
// Compile with: mexcuda BF_das_rephaseSignal.cu

#include "mex.h"
#include "matrix.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

//----------------------------------------------------------------------
// CUDA Kernel: beamformKernel
//
// Each thread computes one pixel of the output beamformed image.
// The grid (X and Z) is assumed to have dimensions M (rows) x N (columns),
// stored in MATLAB’s column‐major order. (The linear index is computed as:
//    index = row + col*M )
//
// For each pixel (x,z), the kernel loops over all receive elements.
// It computes the total delay, converts it to a fractional sample index,
// does linear interpolation of the RF data, applies a simple F-number mask,
// and accumulates the result.
//----------------------------------------------------------------------
__global__ void beamformKernel(const double *sig, int nSamples, int nElements,
                                 const double *xe,
                                 const double *X, const double *Z,
                                 int M, int N,
                                 double theta, double c, double fs, double t0, double fnumber,
                                 double add_term,
                                 double *out)
{
    // Compute 2D indices (row, col) from block and thread indices.
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(row < M && col < N) {
        // Compute the linear index (MATLAB uses column-major order).
        int idx = row + col * M;
        double x = X[idx];
        double z = Z[idx];
        
        // Compute the transmit delay term:
        double dtx = sin(theta) * x + cos(theta) * z + add_term;
        double sum = 0.0;
        
        // Loop over each receive element.
        for (int k = 0; k < nElements; k++){
            double dx = x - xe[k];
            double drx = sqrt(dx*dx + z*z);
            double tau = (dtx + drx) / c;
            
            // Convert delay (tau) into a sample index.
            // (tau-t0)*fs gives an offset in samples. Adding 1 converts from C (0-index)
            // to MATLAB indexing. Later we subtract 1 for C indexing.
            double sample_index = (tau - t0) * fs + 1.0;
            // For interpolation, we require sample_index in [1, nSamples-1]
            if (sample_index < 1.0 || sample_index > (double)(nSamples - 1))
                continue;
            
            int iidx = (int)floor(sample_index);  // iidx in [1, nSamples-1]
            double frac = sample_index - (double)iidx;
            // Convert MATLAB 1-index to C 0-index:
            int offset1 = (iidx - 1) + k * nSamples;
            // Safety check (should not be needed if sample_index is in bounds).
            if (offset1 < 0 || offset1 + 1 >= nSamples * nElements)
                continue;
            
            double sample1 = sig[offset1];
            double sample2 = sig[offset1 + 1];
            double sample = sample1 * (1.0 - frac) + sample2 * frac;
            
            // Apply an F-number mask:
            double mask = (fabs(x - xe[k]) < (z / (2.0 * fnumber))) ? 1.0 : 0.0;
            
            sum += sample * mask;
        }
        // Write the accumulated value to the output image.
        out[idx] = sum;
    }
}

//----------------------------------------------------------------------
// mexFunction: Gateway for MATLAB
//----------------------------------------------------------------------
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    // Check number of inputs.
    if(nrhs != 4) {
        mexErrMsgIdAndTxt("BF_das_rephaseSignal:InvalidInput", 
                          "Four inputs required: SIG, PARAM, X, Z.");
    }
    
    //--------------------------------------------------------------------------
    // Input 0: SIG (RF data)
    //--------------------------------------------------------------------------
    const mxArray *mxSIG = prhs[0];
    if (!mxIsDouble(mxSIG) || mxIsComplex(mxSIG))
        mexErrMsgIdAndTxt("BF_das_rephaseSignal:InputNotRealDouble",
                          "SIG must be a real double array.");
    double *h_sig = mxGetPr(mxSIG);
    const mwSize *sigDims = mxGetDimensions(mxSIG);
    int nDims = mxGetNumberOfDimensions(mxSIG);
    if(nDims < 2)
        mexErrMsgIdAndTxt("BF_das_rephaseSignal:InvalidSIG",
                          "SIG must have at least 2 dimensions.");
    int nSamples = (int)sigDims[0];
    int nElements = (int)sigDims[1];
    
    //--------------------------------------------------------------------------
    // Input 1: PARAM (structure)
    //--------------------------------------------------------------------------
    const mxArray *mxPARAM = prhs[1];
    if (!mxIsStruct(mxPARAM))
        mexErrMsgIdAndTxt("BF_das_rephaseSignal:InvalidPARAM", "PARAM must be a structure.");
    
    // Extract required fields.
    double fc = mxGetScalar(mxGetField(mxPARAM, 0, "f0"));       // central frequency [Hz]
    double fs = mxGetScalar(mxGetField(mxPARAM, 0, "fs"));       // sampling frequency [Hz]
    double t0 = mxGetScalar(mxGetField(mxPARAM, 0, "t0"));       // start time [s]
    double c = mxGetScalar(mxGetField(mxPARAM, 0, "c"));         // speed of sound [m/s]
    double fnumber = mxGetScalar(mxGetField(mxPARAM, 0, "fnumber")); // f-number
    double theta = mxGetScalar(mxGetField(mxPARAM, 0, "theta"));   // transmit angle [rad]
    
    mxArray *mxxe = mxGetField(mxPARAM, 0, "xe");
    if(mxxe == NULL)
        mexErrMsgIdAndTxt("BF_das_rephaseSignal:MissingField", "PARAM.xe missing.");
    double *h_xe = mxGetPr(mxxe);
    mwSize num_xe = mxGetNumberOfElements(mxxe);
    if ((int)num_xe != nElements)
        mexErrMsgIdAndTxt("BF_das_rephaseSignal:xeSizeMismatch",
                          "Length of xe must equal the number of elements.");
    
    // Compute the transmit delay term.
    double txdelay_sum = 0.0;
    double txdelay_min = 1e12;
    for (int k = 0; k < nElements; k++){
        double txdelay = (1.0/c)*tan(theta)*fabs(h_xe[k] - h_xe[0]);
        if(txdelay < txdelay_min) txdelay_min = txdelay;
        txdelay_sum += txdelay;
    }
    double txdelay_mean = txdelay_sum / nElements;
    double add_term = (txdelay_mean - txdelay_min) * c;
    
    //--------------------------------------------------------------------------
    // Input 2: X (grid)
    //--------------------------------------------------------------------------
    const mxArray *mxX = prhs[2];
    if(!mxIsDouble(mxX))
        mexErrMsgIdAndTxt("BF_das_rephaseSignal:InvalidX", "X must be a double array.");
    double *h_X = mxGetPr(mxX);
    mwSize M = mxGetM(mxX); // number of rows
    mwSize N = mxGetN(mxX); // number of columns
    int nPixels = (int)(M * N);
    
    //--------------------------------------------------------------------------
    // Input 3: Z (grid)
    //--------------------------------------------------------------------------
    const mxArray *mxZ = prhs[3];
    if(!mxIsDouble(mxZ))
        mexErrMsgIdAndTxt("BF_das_rephaseSignal:InvalidZ", "Z must be a double array.");
    double *h_Z = mxGetPr(mxZ);
    if (mxGetM(mxZ) != M || mxGetN(mxZ) != N)
        mexErrMsgIdAndTxt("BF_das_rephaseSignal:GridSizeMismatch", "X and Z must have the same dimensions.");
    
    //--------------------------------------------------------------------------
    // Allocate device memory for inputs and output.
    //--------------------------------------------------------------------------
    double *d_sig, *d_xe, *d_X, *d_Z, *d_out;
    size_t sigSize = nSamples * nElements * sizeof(double);
    size_t xeSize  = nElements * sizeof(double);
    size_t gridSize = nPixels * sizeof(double);
    
    cudaMalloc((void**)&d_sig, sigSize);
    cudaMalloc((void**)&d_xe, xeSize);
    cudaMalloc((void**)&d_X, gridSize);
    cudaMalloc((void**)&d_Z, gridSize);
    cudaMalloc((void**)&d_out, gridSize);
    
    cudaMemcpy(d_sig, h_sig, sigSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_xe, h_xe, xeSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, h_X, gridSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Z, h_Z, gridSize, cudaMemcpyHostToDevice);
    
    //--------------------------------------------------------------------------
    // Launch the kernel.
    //--------------------------------------------------------------------------
    // Use a 2D block/grid so that each thread computes one pixel.
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    beamformKernel<<<numBlocks, threadsPerBlock>>>(d_sig, nSamples, nElements, d_xe,
                                                     d_X, d_Z,
                                                     (int)M, (int)N,
                                                     theta, c, fs, t0, fnumber,
                                                     add_term,
                                                     d_out);
    cudaDeviceSynchronize();
    
    //--------------------------------------------------------------------------
    // Copy the output from device to host.
    //--------------------------------------------------------------------------
    plhs[0] = mxCreateDoubleMatrix(M, N, mxREAL);
    double *h_out = mxGetPr(plhs[0]);
    cudaMemcpy(h_out, d_out, gridSize, cudaMemcpyDeviceToHost);
    
    //--------------------------------------------------------------------------
    // Free device memory.
    //--------------------------------------------------------------------------
    cudaFree(d_sig);
    cudaFree(d_xe);
    cudaFree(d_X);
    cudaFree(d_Z);
    cudaFree(d_out);
}
