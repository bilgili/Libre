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

set(CUDA_VERBOSE_BUILD ON)
# set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_NVCC_FLAGS -gencode=arch=compute_61,code=sm_61
                    --std=c++11
                    -lineinfo
                    --use_fast_math
                    -Xcompiler='-fPIC'
                    #-Xcompiler='-fno-strict-aliasing'
                    #-Xcompiler='-std=gnu++11'
                    #-G
                    )
#-fPIC   -Wnon-virtual-dtor -Wsign-promo -Wvla -fno-strict-aliasing -Wall -Wextra -Winvalid-pch -Winit-self -Wno-unknown-pragmas -Wshadow -Werror -fmax-errors=5 -fPIC -std=gnu++11 -MMD -MT
if(CMAKE_BUILD_TYPE MATCHES Debug)
    # list(APPEND CUDA_NVCC_FLAGS "-G")
else()
    list(APPEND CUDA_NVCC_FLAGS -O2)
endif()

set(CUDA_SOURCES cuda/ClipPlanes.cu
                 cuda/ColorMap.cu
                 cuda/IrradianceCompute.cu
                 cuda/IrradianceTexture.cu
                 cuda/PixelBufferObject.cu
                 cuda/Renderer.cu
                 cuda/TexturePool.cu )


cuda_compile(CUDA_FILES ${CUDA_SOURCES})
# cuda_compute_separable_compilation_object_file_name(LINK_OBJS LivreCUDARaycast ${CUDA_FILES})
# cuda_link_separable_compilation_objects(${LINK_OBJS} LivreCUDARaycast
#                                        ${GENCODE} ${CUDA_FILES})
# message(FATAL_ERROR ".....${CUDA_FILES}.....${LINK_OBJS}")
set(LIVRECUDARAYCAST_HEADERS CudaIrradianceCompute.h
                             CudaRaycastRenderer.h
                             CudaRaycastPipeline.h
                             CudaRenderUploadFilter.h
                             CudaTextureObject.h
                             CudaTexturePool.h
                             CudaTextureUploadFilter.h)
set(LIVRECUDARAYCAST_SOURCES CudaIrradianceCompute.cpp
                             CudaRaycastRenderer.cpp
                             CudaRaycastPipeline.cpp
                             CudaRenderUploadFilter.cpp
                             CudaTextureObject.cpp
                             CudaTexturePool.cpp
                             CudaTextureUploadFilter.cpp
                             ${CUDA_FILES}
                             ${LINK_OBJS})
set(LIVRECUDARAYCAST_LINK_LIBRARIES LivreLib ${CUDA_CUDART_LIBRARY})
set(LIVRECUDARAYCAST_INCLUDE_NAME livre/renderer/cuda)
common_library(LivreCUDARaycast)
