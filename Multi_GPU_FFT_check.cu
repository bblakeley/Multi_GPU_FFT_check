// Multiple GPU version of cuFFT_check that uses multiple GPU's
// This program creates a real-valued 3D function sin(x)*cos(y)*cos(z) and then 
// takes the forward and inverse Fourier Transform, with the necessary scaling included. 
// The output of this process should match the input function

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>

// includes, project
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <cuComplex.h>
//CUFFT Header file
#include <cufftXt.h>

#define NX 512
#define NY 512
#define NZ 512
#define NZ2 (NZ/2+1)
#define NN (NX*NY*NZ)
#define L (2*M_PI)
#define TX 8
#define TY 8
#define TZ 8

int divUp(int a, int b) { return (a + b - 1) / b; }

__device__
int idxClip(int idx, int idxMax){
    return idx > (idxMax - 1) ? (idxMax - 1) : (idx < 0 ? 0 : idx);
}

__device__
int flatten(int col, int row, int stack, int width, int height, int depth){
    return idxClip(stack, depth) + idxClip(row, height)*depth + idxClip(col, width)*depth*height;
    // Note: using column-major indexing format
}

__global__ 
void initialize(int NX_per_GPU, int gpuNum, cufftDoubleComplex *f1, cufftDoubleComplex *f2)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if ((i >= NX_per_GPU) || (j >= NY) || (k >= NZ)) return;
    const int idx = flatten(i, j, k, NX, NY, NZ);

    // Create physical vectors in temporary memory
    double x = i * (double)L / NX + (double)gpuNum*NX_per_GPU*L / NX;
    double y = j * (double)L / NY;
    double z = k * (double)L / NZ;

    // Initialize starting array
    f1[idx].x = sin(x)*cos(y)*cos(z);
    f1[idx].y = 0.0;

    f2[idx].x = 0.0;
    f2[idx].y = 0.0;

    return;
}

__global__
void scaleResult(int NX_per_GPU, cufftDoubleComplex *f)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if ((i >= NX_per_GPU) || (j >= NY) || (k >= NZ)) return;
    const int idx = flatten(i, j, k, NX, NY, NZ);

    f[idx].x = f[idx].x / ( (double)NN );
    f[idx].y = f[idx].y / ( (double)NN );

    return;
}

int main (void)
{
    int i, j, k, idx, NX_per_GPU;
    // double complex test;

    // Set GPU's to use and list device properties
    int nGPUs = 2, deviceNum[nGPUs];
    for(i = 0; i<nGPUs; ++i)
    {
        deviceNum[i] = i;

        cudaSetDevice(deviceNum[i]);

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, deviceNum[i]);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
            2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }

    printf("Running Multi_GPU_FFT_check using %d GPUs on a %dx%dx%d grid.\n",nGPUs,NX,NY,NZ);

    // Initialize input data
    // Split data according to number of GPUs
    NX_per_GPU = NX/nGPUs;              // This is not a good solution long-term; needs more work for arbitrary grid sizes/nGPUs

    // Declare variables
    cufftDoubleComplex *u;
    cufftDoubleComplex *u_fft;

    // Allocate memory for arrays
    cudaMallocManaged(&u, sizeof(cufftDoubleComplex)*NN );
    cudaMallocManaged(&u_fft, sizeof(cufftDoubleComplex)*NN );
    // Launch CUDA kernel to initialize velocity field
    const dim3 blockSize(TX, TY, TZ);
    const dim3 gridSize(divUp(NX_per_GPU, TX), divUp(NY, TY), divUp(NZ, TZ));
    for (i = 0; i<nGPUs; ++i){
        cudaSetDevice(deviceNum[i]);
        int idx = i*NX_per_GPU*NY*NZ;                // sets the index value of the data to send to each gpu
        initialize<<<gridSize, blockSize>>>(NX_per_GPU, deviceNum[i], &u[idx], &u_fft[idx]);
    }

    // Synchronize both GPUs before moving forward
    for (i = 0; i<nGPUs; ++i){
        cudaSetDevice(deviceNum[i]);
        cudaDeviceSynchronize();
    }

    // Initialize CUFFT for multiple GPUs //
    // Initialize result variable used for error checking
    cufftResult result;

    // Create empty plan that will be used for the FFT
    cufftHandle plan;
    result = cufftCreate(&plan);
    if (result != CUFFT_SUCCESS) { printf ("*Create failed\n"); return 1; }

    // Tell cuFFT which GPUs to use
    result = cufftXtSetGPUs (plan, nGPUs, deviceNum);
    if (result != CUFFT_SUCCESS) { printf ("*XtSetGPUs failed: code %i\n", result); return 1; }

    // Create the plan for the FFT
    size_t *worksize;                                   // Initializes the worksize variable
    worksize =(size_t*)malloc(sizeof(size_t) * nGPUs);  // Allocates memory for the worksize variable, which tells cufft how many GPUs it has to work with
    
    // Create the plan for cufft
    result = cufftMakePlan3d(plan, NX, NY, NZ, CUFFT_Z2Z, worksize);
    if (result != CUFFT_SUCCESS) { printf ("*MakePlan* failed: code %d \n",(int)result); exit (EXIT_FAILURE) ; }

    printf("The size of the worksize is %lu\n", worksize[0]);

    // Initialize transform array - to be split among GPU's and transformed in place using cufftX
    cudaLibXtDesc *u_prime;
    // Allocate data on multiple gpus using the cufft routines
    result = cufftXtMalloc(plan, (cudaLibXtDesc **)&u_prime, CUFFT_XT_FORMAT_INPLACE);
    if (result != CUFFT_SUCCESS) { printf ("*XtMalloc failed\n"); exit (EXIT_FAILURE) ; }

    // Copy the data from 'host' to device using cufftXt formatting
    result = cufftXtMemcpy(plan, u_prime, u, CUFFT_COPY_HOST_TO_DEVICE);
    if (result != CUFFT_SUCCESS) { printf ("*XtMemcpy failed, code: %d\n",result); exit (EXIT_FAILURE); }

    // Perform FFT on multiple GPUs
    printf("Forward 3d FFT on multiple GPUs\n");
    result = cufftXtExecDescriptorZ2Z(plan, u_prime, u_prime, CUFFT_FORWARD);
    if (result != CUFFT_SUCCESS) { printf ("*XtExecZ2Z  failed\n"); exit (EXIT_FAILURE); }

////////// Apparently re-ordering the data prior to the IFFT is not necessary (gives incorrect results)////////////////////
    // cudaLibXtDesc *u_reorder;
    // result = cufftXtMalloc(plan, (cudaLibXtDesc **)&u_reorder, CUFFT_XT_FORMAT_INPLACE);
    // if (result != CUFFT_SUCCESS) { printf ("*XtMalloc failed\n"); exit (EXIT_FAILURE) ; }
    // // Re-order data on multiple GPUs to natural order
    // printf("Reordering the data on the GPUs\n");
    // result = cufftXtMemcpy (plan, u_reorder, u_prime, CUFFT_COPY_DEVICE_TO_DEVICE);
    // if (result != CUFFT_SUCCESS) { printf ("*XtMemcpy failed\n"); exit (EXIT_FAILURE); }
/////////////////////////////////////////////////////////////////////////////////////////////

    // Perform inverse FFT on multiple GPUs
    printf("Inverse 3d FFT on multiple GPUs\n");
    result = cufftXtExecDescriptorZ2Z(plan, u_prime, u_prime, CUFFT_INVERSE);
    if (result != CUFFT_SUCCESS) { printf ("*XtExecZ2Z  failed\n"); exit (EXIT_FAILURE); }

    // Copy the output data from multiple gpus to the 'host' result variable (automatically reorders the data from output to natural order)
    result = cufftXtMemcpy (plan, u_fft, u_prime, CUFFT_COPY_DEVICE_TO_HOST);
    if (result != CUFFT_SUCCESS) { printf ("*XtMemcpy failed\n"); exit (EXIT_FAILURE); }

    // Scale output to match input (cuFFT does not automatically scale FFT output by 1/N)
    for (i = 0; i<nGPUs; ++i){
        cudaSetDevice(deviceNum[i]);
        idx = i*NX_per_GPU*NY*NZ;                // sets the index value of the data to send to each gpu
        scaleResult<<<gridSize, blockSize>>>(NX_per_GPU, &u_fft[idx]);
    }

    // Synchronize GPUs
    for (i = 0; i<nGPUs; ++i){
        cudaSetDevice(deviceNum[i]);
        cudaDeviceSynchronize();
    }

    // Test results to make sure that u = u_fft
    double error = 0.0;
    for (i = 0; i<NX; ++i){
        for (j = 0; j<NY; ++j){
            for (k = 0; k<NZ; ++k){
                idx = k + j*NZ + NZ*NY*i;
                // error += (double)u[idx].x - sin(x)*cos(y)*cos(z);
                error += (double)u[idx].x - (double)u_fft[idx].x;
                // printf("At idx = %d, the value of the error is %f\n",idx,(double)u[idx].x - (double)u_fft[idx].x);
                // printf("At idx = %d, the value of the error is %f\n",idx,error);

            }
        }
    }
    printf("The sum of the error is %4.4g\n",error);

    // Deallocate variables

    // Free malloc'ed variables
    free(worksize);
    // Free cuda malloc'ed variables
    cudaFree(u);
    cudaFree(u_fft);
    // Free cufftX malloc'ed variables
    result = cufftXtFree(u_prime);
    if (result != CUFFT_SUCCESS) { printf ("*XtFree failed\n"); exit (EXIT_FAILURE); }
    // result = cufftXtFree(u_reorder);
    // if (result != CUFFT_SUCCESS) { printf ("*XtFree failed\n"); exit (EXIT_FAILURE); }
    // Destroy FFT plan
    result = cufftDestroy(plan);
    if (result != CUFFT_SUCCESS) { printf ("cufftDestroy failed: code %d\n",(int)result); exit (EXIT_FAILURE); }

    return 0;

}