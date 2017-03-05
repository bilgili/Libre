/* Copyright (c) 2011-2016  Ahmet Bilgili <ahmetbilgili@gmail.com>
 *
 * This file is part of Livre <https://github.com/bilgili/Libre>
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

#include "IrradianceTexture.cuh"

#include "debug.cuh"

namespace livre
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

    return { x, y, z, w, format };
}
}
namespace cuda
{
IrradianceTexture::IrradianceTexture(size_t dataTypeSize,
                                      bool isSigned,
                                      bool isFloat,
                                      size_t nComponents,
                                      const Vector3ui& overlap,
                                      const Vector3ui& coarseVolumeSize )
    : _dataTypeSize( dataTypeSize )
    , _isSigned( isSigned )
    , _isFloat( isFloat )
    , _nComponents( nComponents )
    , _overlap( overlap )
    , _volumeSize( coarseVolumeSize )
    , _irradianceVolumeSize( coarseVolumeSize )
{
    const cudaChannelFormatDesc& volDesc = getCudaChannelFormatDesc( dataTypeSize,
                                                                     isSigned,
                                                                     isFloat,
                                                                     nComponents );
    checkCudaErrors( cudaMalloc3DArray( &_volumeArray, &volDesc,
                                      { coarseVolumeSize[ 0 ],
                                        coarseVolumeSize[ 1 ],
                                        coarseVolumeSize[ 2 ] }));

    const cudaChannelFormatDesc irrDesc = { 8, 0, 0, 0, cudaChannelFormatKindUnsigned };
    cudaResourceDesc surfRes;
    memset( &surfRes, 0, sizeof( cudaResourceDesc ));
    surfRes.resType = cudaResourceTypeArray;

    cudaTextureDesc texDesc;
    ::memset( &texDesc, 0, sizeof( cudaTextureDesc ));
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.addressMode[ 0 ] = cudaAddressModeClamp;
    texDesc.addressMode[ 1 ] = cudaAddressModeClamp;
    texDesc.addressMode[ 2 ] = cudaAddressModeClamp;

    for( unsigned int i = 0; i < 3; ++i )
    {
        checkCudaErrors( cudaMalloc3DArray( &_irradianceArray[ i ], &irrDesc,
                                           { _irradianceVolumeSize[ 0 ],
                                             _irradianceVolumeSize[ 1 ],
                                             _irradianceVolumeSize[ 2 ] }));

        surfRes.res.array.array = _irradianceArray[ i ];
        checkCudaErrors( cudaCreateSurfaceObject( &_irradianceSurf[ i ], &surfRes ));
        checkCudaErrors( cudaCreateTextureObject( &_irradianceTexture[ i ], &surfRes, &texDesc, NULL ));
    }

    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    surfRes.res.array.array = _volumeArray;
    checkCudaErrors( cudaCreateTextureObject( &_volumeTexture, &surfRes, &texDesc, NULL ));
}

IrradianceTexture::~IrradianceTexture()
{}

void IrradianceTexture::clear()
{
    for( unsigned int i = 0; i < 3; ++i )
        cudaFreeArray( _irradianceArray[ i ] );
    cudaFreeArray( _volumeArray );
}

void IrradianceTexture::upload( const unsigned char* ptr,
                                const Vector3ui& pos,
                                const Vector3ui& size )
{
    cudaMemcpy3DParms params = { 0 };
    params.srcPtr = make_cudaPitchedPtr( const_cast< void* >( ( const void *)ptr ),
                                         size[ 0 ] * _dataTypeSize * _nComponents,
                                         size[ 0 ],
                                         size[ 1 ] );
    params.srcPos = { _overlap[ 0 ] * _dataTypeSize * _nComponents, _overlap[ 1 ], _overlap[ 2 ] };
    params.dstPos = { pos[ 0 ] * _dataTypeSize * _nComponents, pos[ 1 ], pos[ 2 ] };
    params.dstArray = _volumeArray;
    params.kind = cudaMemcpyHostToDevice;
    params.extent = { size[ 0 ] - 2 * _overlap[ 0 ],
                      size[ 1 ] - 2 * _overlap[ 1 ],
                      size[ 2 ] - 2 * _overlap[ 2 ]};
    checkCudaErrors( cudaMemcpy3D( &params ));
}

}
}
