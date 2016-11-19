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

#include "IrradianceCompute.cuh"
#include "LightSource.cuh"
#include "Renderer.cuh"
#include "commonMath.cuh"
#include "Allocate.cuh"
#include <livre/core/render/LightSource.h>

namespace livre
{
namespace
{
int iDivUp( const int a, const int b )
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}
}
namespace cuda
{

inline __device__ float3 getWorldPosition( const float3& pos,
                                           const float3& volSize,
                                           const float3& globalBoxMin,
                                           const float3& globalBoxSize )
{
    return ( pos / volSize ) * globalBoxSize + globalBoxMin;
}

inline __device__ float4 integrate( const float3& pos,
                                    const float3& dir,
                                    const float& maxDistance,
                                    const cudaTextureObject_t& volumeTex,
                                    const cudaTextureObject_t& tfTex,
                                    const cudaTextureObject_t* irrTextures,
                                    const float3& globalBoxMin,
                                    const float3& globalBoxMax,
                                    const float3& globalBoxSize,
                                    const float mult,
                                    const float add,
                                    const float stepSize,
                                    const float alphaCorrection )
{

    float4 colorPrev = { 0.0f, 0.0f, 0.0f, 0.0f };
    float tNear = 0;
    float tFar = 0;
    float3 rayStart;
    float3 rayStop;
    if( !intersectBox( pos, dir, globalBoxMin, globalBoxMax, tNear, tFar ))
        return colorPrev;

    // Starting position is always inside
    rayStart = pos;
    rayStop = pos + ( dir * tNear );

    if( dir.x == 0.0 )
        rayStop.x = pos.x;

    if( dir.y == 0.0 )
        rayStop.y = pos.y;

    if( dir.z == 0.0 )
        rayStop.z = pos.z;

    const float3& step = normalize( rayStop - rayStart ) * stepSize;
    const float dist = fminf( distance( rayStop, rayStart ), maxDistance );

    // Front-to-back absorption-emission integrator
    bool isEarlyExit = false;
    for( float travel = dist; travel > 0.0; rayStart += step, travel -= stepSize )
    {
        const float3& texPos = ( rayStart -  globalBoxMin ) / globalBoxSize;
        const float density = tex3D< unsigned char >( volumeTex, texPos.x, texPos.y, texPos.z );
        float4 color = tex1D< float4 >( tfTex, density * mult + add );
        if( irrTextures )
        {
            color.x += tex3D< float >( irrTextures[ 0 ], texPos.x, texPos.y, texPos.z );
            color.y += tex3D< float >( irrTextures[ 1 ], texPos.x, texPos.y, texPos.z );
            color.z += tex3D< float >( irrTextures[ 2 ], texPos.x, texPos.y, texPos.z );
        }

        colorPrev = composite( color, colorPrev, alphaCorrection );
        isEarlyExit = colorPrev.w > EARLY_EXIT;
        if( isEarlyExit )
            break;

    }
    return colorPrev;
}

__global__ void computeIrradiance( IrradianceTexture texture,
                                   const ColorMap colorMap,
                                   const LightSource** lightSources,
                                   const unsigned int lightCount,
                                   const ::livre::cuda::ViewData viewData,
                                   const ::livre::cuda::RenderData renderData )

{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    const uint3& volSize = make_uint3FromArray( texture.getIrradianceVolumeSize().array );

    if( i >= volSize.x || j >= volSize.y || k >= volSize.z )
        return;

    __shared__ float3 directions[ 128 ];
    __shared__ float dirWeights[ 128 ];
    __shared__ unsigned int directionCount;

    const float3& globalBoxMin = make_float3FromArray( viewData.aabbMin.array );
    const float3& globalBoxMax = make_float3FromArray( viewData.aabbMax.array );
    const float3& globalBoxSize = globalBoxMax - globalBoxMin;
    const float3 fPos = { (float)i + 0.5f, (float)j + 0.5f, (float)k + 0.5f };
    const float3 fVolSize = { (float)volSize.x, (float)volSize.y, (float)volSize.z };
    const float3 worldPos = getWorldPosition( fPos, fVolSize, globalBoxMin, globalBoxSize );

    const float2& dataSourceRange = make_float2FromArray( renderData.dataSourceRange.array );
    const float multiplyer = 1.0f / ( dataSourceRange.y - dataSourceRange.x );
    const float addedValue = -dataSourceRange.x / ( dataSourceRange.y - dataSourceRange.x );

    const float alphaCorrection = float( renderData.maxSamplesPerRay ) /
                                  float( renderData.samplesPerRay );

    const float stepSize = 1.0 / float( renderData.samplesPerRay );
    const cudaTextureObject_t& volumeTex = texture.getVolumeTexture();
    cudaSurfaceObject_t* irradianceBuffer = texture.getIrradianceBuffer();

    LightData lighDatas[ 16 ];

    // Collect light data
    unsigned int lightDataCount = 0;
    for( unsigned int lsInd = 0; lsInd < lightCount; ++lsInd )
    {
        const LightSource* ls = lightSources[ lsInd ];
        const unsigned int samples = ls->getSampleCount();
        ls->getSamples( lighDatas + lightDataCount, worldPos );
        lightDataCount += samples;
    }
    const float pi = 3.141592654f;
    if( threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 )
    {
        const float thetaSamples = 4.0f;
        const float phiSamples = 4.0f;

        const float deltaTheta = pi / thetaSamples;
        const float deltaPhi = 2.0 * pi / phiSamples;
        directionCount = 0;
        for( float theta = deltaTheta / 2.0; theta < ( pi + deltaTheta / 2.0 ); theta += deltaTheta )
        {
            for( float phi = deltaPhi / 2.0; phi < ( 2.0 * pi + deltaPhi / 2.0 ); phi += deltaPhi )
            {
                const float sinTheta = sin( theta );
                directions[ directionCount ].x = cos( phi ) * sinTheta;
                directions[ directionCount ].y = sin( phi ) * sinTheta;
                directions[ directionCount ].z = cos( theta );
                dirWeights[ directionCount ] = ( deltaPhi * deltaTheta * fabs( sinTheta )) / ( 4 * pi );
                directionCount++;
            }
        }
    }
    __syncthreads();

    // printf("lightDataCount %d\n", lightDataCount );
    // Light contribution
    float4 lightRadiance = { 0.0f, 0.0f, 0.0f, 1.0f };
    const float scale = 1000.0f;
    for( unsigned int ldInd = 0; ldInd < lightDataCount; ++ldInd )
    {
        const LightData& lightData = lighDatas[ ldInd ];
        float4 color = integrate( worldPos,
                                         lightData.dir * -1.0f,
                                         lightData.distance,
                                         volumeTex,
                                         colorMap.getTexture(),
                                         0,
                                         globalBoxMin,
                                         globalBoxMax,
                                         globalBoxSize,
                                         multiplyer,
                                         addedValue,
                                         stepSize,
                                         alphaCorrection );

        if( color.w <= EARLY_EXIT )
        {
            const float4 lightColor = { lightData.color.x * scale,
                                        lightData.color.y * scale,
                                        lightData.color.z * scale,
                                        1.0f };
            color.x = 0; color.y = 0; color.z = 0;
            lightRadiance = lightRadiance + composite( lightColor, color, alphaCorrection );
        }

    }
    lightRadiance.w = 0.0f;

    float4 radiance = lightRadiance / ( 4.0 * pi );
    uchar3 radiance8bit = { fminf( radiance.x * 255.0, 255.0 ),
                            fminf( radiance.y * 255.0, 255.0 ),
                            fminf( radiance.z * 255.0, 255.0 ) };
    surf3Dwrite( radiance8bit.x, irradianceBuffer[ 0 ], i, j, k );
    surf3Dwrite( radiance8bit.y, irradianceBuffer[ 1 ], i, j, k );
    surf3Dwrite( radiance8bit.z, irradianceBuffer[ 2 ], i, j, k );

     __syncthreads();
    // Volume contribution
    float4 emissionRadiance = { 0.0f, 0.0f, 0.0f, 0.0f };
    for( unsigned int volInd = 0; volInd < directionCount; ++volInd )
    {
        const float4 color = integrate( worldPos,
                                        directions[ volInd ],
                                        10000000.0f,
                                        volumeTex,
                                        colorMap.getTexture(),
                                        texture.getIrradianceTexture(),
                                        globalBoxMin,
                                        globalBoxMax,
                                        globalBoxSize,
                                        multiplyer,
                                        addedValue,
                                        stepSize,
                                        alphaCorrection );
        emissionRadiance = emissionRadiance + color * dirWeights[ volInd ];
    }

    radiance = ( lightRadiance + emissionRadiance ) / ( 4.0 * pi );
    radiance8bit = { fminf( radiance.x * 255.0, 255.0 ),
                     fminf( radiance.y * 255.0, 255.0 ),
                     fminf( radiance.z * 255.0, 255.0 ) };
    surf3Dwrite( radiance8bit.x, irradianceBuffer[ 0 ], i, j, k );
    surf3Dwrite( radiance8bit.y, irradianceBuffer[ 1 ], i, j, k );
    surf3Dwrite( radiance8bit.z, irradianceBuffer[ 2 ], i, j, k );

}

IrradianceCompute::IrradianceCompute( size_t dataTypeSize,
                                      bool isSigned,
                                      bool isFloat,
                                      size_t nComponents,
                                      const Vector3ui& overlap,
                                      const Vector3ui& coarseVolumeSize )
    : _texture( dataTypeSize, isSigned, isFloat, nComponents, overlap, coarseVolumeSize )
{
     checkCudaErrors( cudaMalloc( &_cudaLightSources, sizeof( LightSource* ) * 32 ));
}

IrradianceCompute::~IrradianceCompute()
{
    _texture.clear();
}

LightSource* getPointCudaLight( const ::livre::LightSource& ls )
{
    switch( ls.getType( ))
    {
        case POINT:
        {
            const ::livre::PointLightSource& pls =
                    static_cast< const ::livre::PointLightSource& >( ls );
            const float3 pos = { pls.position[ 0 ], pls.position[ 1 ], pls.position[ 2 ] };
            const float3 color = { pls.color[ 0 ], pls.color[ 1 ], pls.color[ 2 ] };
            return allocateCudaObjectInDevice< PointLightSource >( pos, color );
        }
        case DIRECT:
        {
            const ::livre::DirectLightSource& dls =
                    static_cast< const ::livre::DirectLightSource& >( ls );
            const float3 dir = { dls.direction[ 0 ], dls.direction[ 1 ], dls.direction[ 2 ] };
            const float3 color = { dls.color[ 0 ], dls.color[ 1 ], dls.color[ 2 ] };
            return allocateCudaObjectInDevice< DirectLightSource >( normalize( dir ), color );
        }
    default:
        std::cout << "Unsupported light source type " << std::endl;
    }

    return 0;
}

void IrradianceCompute::update( const RenderData& renderData,
                                const ViewData& viewData,
                                const lexis::render::ColorMap& colorMap,
                                const ::livre::ConstLightSources& lightSources )
{
    _renderData = renderData;
    _viewData = viewData;
    _colorMap.upload( colorMap );
    _lightSources.clear();
    for( const auto& ls: lightSources )
    {
        LightSource* cls = getPointCudaLight( *ls );
        if( cls )
            _lightSources.push_back( cls );
    }
}

IrradianceTexture IrradianceCompute::compute()
{
    checkCudaErrors( cudaMemcpy( _cudaLightSources,
                                 _lightSources.data(),
                                 sizeof( LightSource *) * _lightSources.size( ),
                                 cudaMemcpyHostToDevice ));

    const Vector3ui& volumeSize = _texture.getIrradianceVolumeSize();
    const dim3 blockDim( 4, 4, 4 );
    const dim3 gridDim( iDivUp( volumeSize[ 0 ], blockDim.x ),
                        iDivUp( volumeSize[ 1 ], blockDim.y ),
                        iDivUp( volumeSize[ 2 ], blockDim.z ));
    computeIrradiance<<< gridDim, blockDim >>>( _texture,
                                                _colorMap,
                                                _cudaLightSources,
                                                 _lightSources.size(),
                                                _viewData,
                                                _renderData );
    checkCudaErrors( cudaThreadSynchronize( ));
    for( auto& ls: _lightSources )
        deallocateCudaObjectInDevice( ls );

    return _texture;
}

void IrradianceCompute::upload( const unsigned char* ptr,
                                const Vector3ui& pos,
                                const Vector3ui& size)
{
    _texture.upload( ptr, pos, size );
}
}
}
