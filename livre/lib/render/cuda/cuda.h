#ifndef _Cuda_h_
#define _Cuda_h_

#ifdef __CUDACC__
    #define CUDA_CALL __host__ __device__
    #define CUDA_CONSTANT __constant__
#else
    #define CUDA_CALL
    #define CUDA_CONSTANT
#endif

#endif // _Cuda_h_

