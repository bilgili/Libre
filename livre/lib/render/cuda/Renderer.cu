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

#include "Renderer.h"

#include <cuda_runtime.h>
#include <device_functions.h>
#include "math.cuh"

#include "PixelBufferObject.cuh"
#include "ColorMap.cuh"
#include "ClipPlanes.cuh"

namespace livre
{
namespace
{
#define EARLY_EXIT 0.999f
#define EPSILON 0.0000000001f
#define SH_UINT 0u
#define SH_INT 1u
#define SH_FLOAT 2u

inline __device__ float4 calcPositionInEyeSpaceFromWindowSpace( const float2 windowSpace,
                                                                const uint4 viewport,
                                                                const float* invProjMatrix )
{
    const float2 viewportzw = make_float2( viewport.z, viewport.w );
    const float2 ndcPosxy = 2.0 * ( windowSpace - make_float2( viewport.x, viewport.y ) -
                            ( viewportzw / 2.0f )) / viewportzw;

    const float4 ndcPos = make_float4( ndcPosxy.x, ndcPosxy.y, 1.0f, 1.0f );
    const float4 eyeSpacePos = invProjMatrix * ndcPos;
    return eyeSpacePos / eyeSpacePos.w;
}


inline __device__ bool intersectBox( const float3& origin,
                                     float3 dir,
                                     const float3& aabbMin,
                                     const float3& aabbMax,
                                     float& t0,
                                     float& t1 )
{
    //We need to avoid division by zero in "vec3 invR = 1.0 / r.Dir;"
    if( dir.x == 0 )
        dir.x = EPSILON;

    if( dir.y == 0 )
        dir.y = EPSILON;

    if( dir.z == 0 )
        dir.z = EPSILON;

    float3 invR = recp( dir );
    float3 tbot = invR * ( aabbMin - origin );
    float3 ttop = invR * ( aabbMax - origin );
    float3 tmin = min( ttop, tbot );
    float3 tmax = max( ttop, tbot );
    float2 t = max( make_float2( tmin.x, tmin.x ), make_float2( tmin.y, tmin.z ));
    t0 = max( t.x, t.y );
    t = min( make_float2( tmax.x, tmax.x ), make_float2( tmax.y, tmax.z ));
    t1 = min( t.x, t.y );
    return t0 <= t1;
}

inline __device__ float4 composite( const float4& src,
                                    const float4& dst,
                                    const float alphaCorrection )
{
    // The alpha correction function behaves badly around maximum alpha
    float alpha = 1.0 - pow( 1.0 - min( src.w, 1.0 - 1.0 / 256.0), alphaCorrection );
    float3 xyz = make_float3( dst.x, dst.y, dst.z );
    xyz += make_float3( src.x, src.y, src.z ) * alpha * ( 1.0 - dst.w );
    float dstw = alpha * ( 1.0 - dst.w );
    return make_float4( xyz.x, xyz.y, xyz.z, dstw );
}

// Compute texture position.
inline __device__ unsigned int calcTextureIndexFromAABBPos( const float3& pos,
                                                            const float3& aabbMin,
                                                            const float3& aabbMax,
                                                            const uint3& blockSize,
                                                            const uint3& overlap )
{
    const float3 dataSize = make_float3( blockSize + 2 * overlap );
    const float3 overlapTexture = make_float3( overlap ) / dataSize;
    const float3 normalizedPos = ( pos - aabbMin ) / ( aabbMax - aabbMin );
    const float3 normalizedIndexPos = normalizedPos + overlapTexture;
    const uint3 indexPos = make_uint3( normalizedIndexPos * dataSize );
    return indexPos.x * dataSize.y * dataSize.z
         + indexPos.y * dataSize.z
         + indexPos.z;
}

__global__ void rayCast( const cuda::PixelBufferObject pbo,
                         const cuda::ClipPlanes clipPlanes,
                         const cuda::ColorMap colorMap,
                         const cuda::ViewData viewData,
                         const unsigned int nodeCount,
                         const cuda::NodeData* nodeDatas,
                         const cuda::RenderData renderData )
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    const float4 pixelEyeSpacePos =
            calcPositionInEyeSpaceFromWindowSpace( make_float2( x, y ),
                                                   make_uint4FromArray( viewData.glViewport.array ),
                                                   viewData.invProjMatrix.array );
    const float4 pixelWorldSpacePos = viewData.invViewMatrix.array * pixelEyeSpacePos;
    const float3 dir = normalize( make_float3( pixelWorldSpacePos ) -
                                  make_float3FromArray( viewData.eyePosition.array ));

    float* pixelBuffer = pbo.getBuffer();

    const unsigned int pixelPos = ( y * pbo.getWidth() + x ) * 4;
    const float alpha = pixelBuffer[ pixelPos + 3 ];

    if( alpha > EARLY_EXIT )
        return;

    float4 color = make_float4( pixelBuffer[ pixelPos ],
                                pixelBuffer[ pixelPos + 1 ],
                                pixelBuffer[ pixelPos + 2 ],
                                alpha );
    float tNearGlobal, tFarGlobal;

    const float3 boxMin = make_float3FromArray( viewData.aabbMin.array );
    const float3 boxMax = make_float3FromArray( viewData.aabbMax.array );
    const float3 origin = make_float3FromArray( viewData.eyePosition.array );
    if( !intersectBox( origin, dir, boxMin, boxMax, tNearGlobal, tFarGlobal ))
        return;

    const float3 nearPlaneNormal = { 0.0f, 0.0f, 1.0f };
    float tNearPlane = dot( nearPlaneNormal, make_float3( 0.0, 0.0, -viewData.nearPlane ))
                       / dot( nearPlaneNormal, make_float3( pixelEyeSpacePos ));

    for( unsigned int i = 0; i < nodeCount; ++i  )
    {
        const ::livre::cuda::NodeData& nodeData = nodeDatas[ i ];
        const float3 boxMin = make_float3FromArray( nodeData.aabbMin.array );
        const float3 boxMax = make_float3FromArray( nodeData.aabbMax.array );

        float tNear = 0.0f; float tFar = 0.0f;
        if( !intersectBox( origin, dir, boxMin, boxMax, tNear, tFar ))
            continue;

        if( tNear < tNearPlane )
            tNear = tNearPlane;

        const float stepSize = 1.0 / float( renderData.samplesPerRay );
        const float residu = mod( tNear - tNearGlobal, stepSize );

        if( residu > 0.0f )
            tNear += stepSize - residu;

        if( tNear > tFar )
            continue;

        const float4* clipPlanesArray = clipPlanes.getClipPlanes();
        for( int j = 0; j < clipPlanes.getNPlanes(); j++ )
        {
            const float4 clipPlane = clipPlanesArray[ j ];
            const float3 planeNormal = make_float3( clipPlane );
            const float rn = max( dot( dir, planeNormal ), EPSILON );
            const float d = clipPlane.w;
            float t = -( dot( planeNormal,
                              make_float3FromArray( viewData.eyePosition.array )) + d ) / rn;
            if( rn > 0.0 ) // opposite direction plane
                tNear = max( tNear, t );
            else
                tFar = min( tFar, t );
        }

        if( tNear > tFar )
            continue;

        const float3 rayStart = origin + dir * tNear;
        const float3 rayStop = origin + dir * tFar;

        // http://stackoverflow.com/questions/12494439/opacity-correction-in-raycasting-volume-rendering
        float alphaCorrection = float( renderData.maxSamplesPerRay ) /
                                float( renderData.samplesPerRay );

        float3 pos = rayStart;
        const float3 step = normalize( rayStop - rayStart ) * stepSize;

        //Used later for MAD optimization in the raymarching loop
        const float2 dataSourceRange = make_float2FromArray( renderData.dataSourceRange.array );
        const float multiplyer = 1.0f / ( dataSourceRange.y - dataSourceRange.x );
        const float addedValue = -dataSourceRange.x / ( dataSourceRange.y - dataSourceRange.x );

        // Front-to-back absorption-emission integrator
        for( float travel = distance( rayStop, rayStart );
             travel > 0.0; pos += step, travel -= stepSize )
        {
            const unsigned int texPos =
                    calcTextureIndexFromAABBPos( pos,
                                                 boxMin,
                                                 boxMax,
                                                 make_uint3FromArray( nodeData.blockSize.array ),
                                                 make_uint3FromArray( renderData.overlap.array ));

            float density = 0;
            if( renderData.datatype == SH_UINT )
                density = ((unsigned char*)nodeData.data)[ texPos ] * multiplyer + addedValue;
            else if( renderData.datatype == SH_INT )
                density = ((char *)nodeData.data)[ texPos ] * multiplyer + addedValue;
            else if( renderData.datatype == SH_FLOAT )
                density = ((float *)nodeData.data)[ texPos ] * multiplyer + addedValue;

            float4 transferFn  = tex1D( colorMap.getTexture(), density );
            color = composite( transferFn, color, alphaCorrection );

            if( color.w > EARLY_EXIT )
                break;
        }
     }

    pixelBuffer[ pixelPos ] = color.x;
    pixelBuffer[ pixelPos + 1 ] = color.y;
    pixelBuffer[ pixelPos + 2 ] = color.z;
    pixelBuffer[ pixelPos + 3 ] = color.w;
}
}

namespace  cuda
{
struct Renderer::Impl
{
    Impl()
    {
         cudaGLSetGLDevice(0);
    }

    ~Impl()
    {}

    void update( const lexis::render::ColorMap& colorMap )
    {
        _cudaColorMap.upload( colorMap );
    }

    void preRender( const ViewData& viewData )
    {
        _cudaRenderPBO.resize( viewData.glViewport[ 2 ] - viewData.glViewport[ 0 ],
                               viewData.glViewport[ 3 ] - viewData.glViewport[ 1 ]);
        _cudaRenderPBO.mapBuffer();
    }

    void copyNodeDatasToCudaMemory( const NodeDatas& nodeDatas )
    {
        _nodeDatas = nodeDatas;

        for( NodeData& nodeData: _nodeDatas )
        {
            unsigned char* data = nodeData.data;
            cudaMalloc( &nodeData.data, nodeData.size );
            cudaMemcpy( nodeData.data, data, nodeData.size, cudaMemcpyHostToDevice );
        }

        cudaMalloc( &_cudaNodeDatas, sizeof(NodeData) * _nodeDatas.size( ));
        cudaMemcpy( _cudaNodeDatas, _nodeDatas.data(),
                    sizeof(NodeData) * _nodeDatas.size(), cudaMemcpyHostToDevice);

        _nodeCount = nodeDatas.size();
    }

    void render( const lexis::render::ClipPlanes& clipPlanes,
                 const ViewData& viewData,
                 const NodeDatas& nodeDatas,
                 const RenderData& renderData )
    {

        const unsigned int width = _cudaRenderPBO.getWidth();
        const unsigned int height = _cudaRenderPBO.getHeight();

        copyNodeDatasToCudaMemory( nodeDatas );

        dim3 gridDim((width % 16 != 0) ? (width / 16 + 1) : (width / 16),
                     (height % 16 != 0) ? (height / 16 + 1) : (height / 16));
        dim3 blockDim(16, 16);
        _cudaClipPlanes.upload( clipPlanes );
        rayCast<<< gridDim, blockDim >>>( _cudaRenderPBO,
                                          _cudaClipPlanes,
                                          _cudaColorMap,
                                          viewData,
                                          _nodeCount,
                                          _cudaNodeDatas,
                                          renderData );
    }

    void postRender()
    {
        _cudaRenderPBO.unmapBuffer();

        for( const NodeData& nodeData: _nodeDatas )
            cudaFree( nodeData.data );
        cudaFree( _cudaNodeDatas );
    }

    ::livre::cuda::ClipPlanes _cudaClipPlanes;
    ::livre::cuda::ColorMap _cudaColorMap;
    ::livre::cuda::PixelBufferObject _cudaRenderPBO;
    NodeDatas _nodeDatas;
    NodeData* _cudaNodeDatas;
    unsigned int _nodeCount;
};

Renderer::Renderer()
    : _impl( new Renderer::Impl( ))
{}

Renderer::~Renderer()
{}

void Renderer::preRender( const ViewData& viewData )
{
    _impl->preRender( viewData );
}

void Renderer::render( const lexis::render::ClipPlanes& clipPlanes,
                       const ViewData& viewData,
                       const NodeDatas& nodeData,
                       const RenderData& renderData )
{
    _impl->render( clipPlanes, viewData, nodeData, renderData );
}

void Renderer::postRender()
{
    _impl->postRender();
}


void Renderer::update( const lexis::render::ColorMap& colorMap )
{
    _impl->update( colorMap );
}
}
}





