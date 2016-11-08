# Copyright (c) 2011-2016, ahmetbilgili@gmail.com
#
# This file is part of Livre <https://github.com/bilgili/Libre>
#

if(NOT CUDA_FOUND)
  return()
endif()

foreach(DEPEND Livre Lunchbox Servus vmmlib Collage Pression Lexis ZeroBuf)
  cuda_include_directories(${${DEPEND}_DIR}/include
                           ${${DEPEND}_SOURCE_DIR}
                           ${${DEPEND}_BINARY_DIR}/include)
endforeach()
set(CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52
                    --std c++11
                    --use_fast_math
                    --default-stream per-thread
                    -Xcompiler "-fPIC")

if(NOT CMAKE_BUILD_TYPE MATCHES Debug)
    list(APPEND CUDA_NVCC_FLAGS -O2)
endif()

set(CUDA_SOURCES cuda/ClipPlanes.cu
                 cuda/ColorMap.cu
                 cuda/PixelBufferObject.cu
                 cuda/Renderer.cu
                 cuda/TexturePool.cu )

cuda_compile(CUDA_FILES ${CUDA_SOURCES})
set(LIVRECUDARAYCAST_HEADERS CudaRaycastRenderer.h
                             CudaRaycastPipeline.h
                             CudaRenderUploadFilter.h
                             CudaTextureObject.h
                             CudaTexturePool.h
                             CudaTextureUploadFilter.h)
set(LIVRECUDARAYCAST_SOURCES CudaRaycastRenderer.cpp
                             CudaRaycastPipeline.cpp
                             CudaRenderUploadFilter.cpp
                             CudaTextureObject.cpp
                             CudaTexturePool.cpp
                             CudaTextureUploadFilter.cpp
                             ${CUDA_FILES})
set(LIVRECUDARAYCAST_LINK_LIBRARIES LivreLib ${CUDA_CUDART_LIBRARY})
set(LIVRECUDARAYCAST_INCLUDE_NAME livre/renderer/cuda)
common_library(LivreCUDARaycast)
