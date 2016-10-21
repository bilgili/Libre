/* Copyright (c) 2011-2016  Ahmet Bilgili <ahmetbilgili@gmail.com>
 *
 * This file is part of Livre <https://github.com/bilgili/Livre>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include "cuda.h"
#include "TexturePool.cuh"

#include <cuda_runtime.h>
#include <cuda_texture_types.h>

#include <boost/thread/mutex.hpp>

#include <livre/core/data/DataSource.h>
#include <vector>
#include <algorithm>

namespace livre
{
namespace cuda
{
namespace
{
cudaChannelFormatDesc getCudaChannelFormatDesc( const size_t dataTypeSize,
                                                const bool isSigned,
                                                const bool isFloat,
                                                const size_t nComponents )
{

    cudaChannelFormatKind format = cudaChannelFormatKindUnsigned;

    if( isFloat )
        format = cudaChannelFormatKindFloat;
    else if( isSigned )
        format = cudaChannelFormatKindSigned;

    int x = 0, y = 0, z = 0, w = 0;
    int bitSize = dataTypeSize * 8;
    switch( nComponents )
    {
    case 1:
        x = bitSize;
        break;
    case 2:
        x = y = bitSize;
        break;
    case 3:
        x = y = z = bitSize;
        break;
    case 4:
        x = y = z = w = bitSize;
        break;
    default:
        throw std::runtime_error( "Channel number cannot be 0 or larger than 4" );
    }


    size_t availableMemory, totalMemory;
    cudaMemGetInfo( &availableMemory, &totalMemory) ;
    return { x, y, z, w, format };
}

size_t findSingleBlockCudaMemorySize( const size_t dataTypeSize,
                                      const bool isSigned,
                                      const bool isFloat,
                                      const size_t nComponents,
                                      const Vector3ui& maxBlockSize )
{

    cudaChannelFormatDesc channelDesc = getCudaChannelFormatDesc( dataTypeSize,
                                                                  isSigned,
                                                                  isFloat,
                                                                  nComponents );
    cudaArray_t array ;
    const cudaExtent blockSize = { maxBlockSize[ 0 ],
                                   maxBlockSize[ 1 ],
                                   maxBlockSize[ 2 ]};
    size_t preAvailableMemory, totalMemory;
    checkCudaErrors( cudaMemGetInfo( &preAvailableMemory, &totalMemory));
    checkCudaErrors( cudaMalloc3DArray( &array, &channelDesc, blockSize ));
    size_t postAvailableMemory;
    checkCudaErrors( cudaMemGetInfo( &postAvailableMemory, &totalMemory ));
    checkCudaErrors( cudaFreeArray( array ));
    return preAvailableMemory - postAvailableMemory;

}
const Vector3f INVALID_SLOT_POSITION( -1.0f );
}

struct TexturePool::Impl
{
    Impl( const size_t dataTypeSize,
          const bool isSigned,
          const bool isFloat,
          const size_t nComponents,
          const Vector3ui& maxBlockSize,
          const size_t maxGpuMemory )
        : _cacheSlotsSize( Vector3ui( 1u ))
        , _dataTypeSize( dataTypeSize )
        , _isSigned( isSigned )
        , _isFloat( isFloat )
        , _nComponents( nComponents )
        , _maxBlockSize( maxBlockSize )
        , _cudaBlockSize( findSingleBlockCudaMemorySize( dataTypeSize,
                                                         isSigned,
                                                         isFloat,
                                                         nComponents,
                                                         maxBlockSize ))
    {
        size_t availableMemory, totalMemory;
        checkCudaErrors( cudaMemGetInfo( &availableMemory, &totalMemory ));

        const uint32_t maxMemory = std::min( availableMemory, maxGpuMemory );
        const uint32_t maxBlocks = maxMemory / _cudaBlockSize;

        cudaDeviceProp deviceProp;
        checkCudaErrors( cudaGetDeviceProperties( &deviceProp, 0 ));

        _cacheSlotsSize[ 0 ] = std::min( deviceProp.maxTexture3D[ 0 ] / maxBlockSize[ 0 ],
                                         std::max( maxBlocks, 1u ));

        _cacheSlotsSize[ 1 ] = std::min( deviceProp.maxTexture3D[ 1 ] / maxBlockSize[ 1 ],
                                         std::max( maxBlocks / _cacheSlotsSize[ 0 ], 1u ));

        _cacheSlotsSize[ 2 ] = std::min( deviceProp.maxTexture3D[ 2 ] / maxBlockSize[ 2 ],
                    std::max( maxBlocks / ( _cacheSlotsSize[ 0 ] * _cacheSlotsSize[ 1 ] ), 1u ));

        for( int i = _cacheSlotsSize[ 0 ] - 1; i >= 0; --i )
            for( int j = _cacheSlotsSize[ 1 ] - 1; j >= 0; --j )
                for( int k = _cacheSlotsSize[ 2 ] - 1; k >= 0; --k )
                {
                    _emptySlotsList.emplace_back( (float)i / _cacheSlotsSize[ 0 ],
                                                  (float)j / _cacheSlotsSize[ 1 ],
                                                  (float)k / _cacheSlotsSize[ 2 ] );
                }

        const cudaChannelFormatDesc& desc = getCudaChannelFormatDesc( dataTypeSize,
                                                                      isSigned,
                                                                      isFloat,
                                                                      nComponents );
        const Vector3ui volumeSize = _cacheSlotsSize * maxBlockSize;
        checkCudaErrors( cudaMalloc3DArray( &_cudaArray, &desc,
                                          { volumeSize[ 0 ],
                                            volumeSize[ 1 ],
                                            volumeSize[ 2 ] }));


        // create texture object
        cudaResourceDesc resDesc;
        ::memset( &resDesc, 0, sizeof( cudaResourceDesc ));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = _cudaArray;

        cudaTextureDesc texDesc;
        ::memset( &texDesc, 0, sizeof( cudaTextureDesc ));
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 1;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.addressMode[ 0 ] = cudaAddressModeClamp;
        texDesc.addressMode[ 1 ] = cudaAddressModeClamp;
        texDesc.addressMode[ 2 ] = cudaAddressModeClamp;

        checkCudaErrors( cudaCreateTextureObject( &_texture, &resDesc, &texDesc, NULL ));
    }

    Vector3f copyToSlot(  const unsigned char* ptr, const Vector3ui& size )
    {
        Vector3f slot( INVALID_SLOT_POSITION );
        {
            boost::unique_lock< boost::mutex > lock( _mutex );
            if( _emptySlotsList.empty( ))
                return slot;

            slot = _emptySlotsList.back();
            _emptySlotsList.pop_back();

        }

        cudaMemcpy3DParms params = { 0 };
        params.srcPtr = make_cudaPitchedPtr( const_cast< void* >( ( const void *)ptr ),
                                             size[ 0 ] * _dataTypeSize * _nComponents,
                                             size[ 0 ],
                                             size[ 1 ] );

        const Vector3f volumeSize = _cacheSlotsSize * _maxBlockSize;
        params.dstPos = { std::lround( slot[ 0 ] *  volumeSize[ 0 ] ) * _dataTypeSize * _nComponents,
                          std::lround( slot[ 1 ] *  volumeSize[ 1 ] ),
                          std::lround( slot[ 2 ] *  volumeSize[ 2 ] )};
        params.dstArray = _cudaArray;
        params.kind = cudaMemcpyHostToDevice;
        params.extent = { size[ 0 ], size[ 1 ], size[ 2 ] };
        checkCudaErrors( cudaMemcpy3DAsync( &params ));
        return slot;
    }

    void releaseSlot( const Vector3f& pos )
    {
        boost::unique_lock< boost::mutex > lock( _mutex );
        _emptySlotsList.push_back( pos );
    }

    Vector3ui _cacheSlotsSize;
    std::vector< Vector3f > _emptySlotsList;
    const size_t _dataTypeSize;
    const bool _isSigned;
    const bool _isFloat;
    const size_t _nComponents;
    const Vector3ui _maxBlockSize;
    boost::mutex _mutex;
    const size_t _cudaBlockSize;
    cudaArray_t _cudaArray;
    cudaTextureObject_t _texture;
};

TexturePool::TexturePool( const size_t dataTypeSize,
                          const bool isSigned,
                          const bool isFloat,
                          const size_t nComponents,
                          const Vector3ui& maxBlockSize,
                          const size_t maxGpuMemory )
    : _impl( new TexturePool::Impl( dataTypeSize,
                                    isSigned,
                                    isFloat,
                                    nComponents,
                                    maxBlockSize,
                                    maxGpuMemory ))
{}

Vector3f TexturePool::copyToSlot(  const unsigned char* ptr, const Vector3ui& size )
{
    return _impl->copyToSlot( ptr, size );
}

void TexturePool::releaseSlot( const Vector3f& pos )
{
    _impl->releaseSlot( pos );
}

cudaTextureObject_t TexturePool::getTexture() const
{
    return _impl->_texture;
}

TexturePool::~TexturePool()
{}

size_t TexturePool::getSlotMemSize() const
{
    return _impl->_cudaBlockSize;
}

Vector3ui TexturePool::getTextureSize() const
{
    return _impl->_cacheSlotsSize * _impl->_maxBlockSize;
}

size_t TexturePool::getTextureMem() const
{
    return ( _impl->_cacheSlotsSize ).product() * _impl->_cudaBlockSize;
}
}
}


