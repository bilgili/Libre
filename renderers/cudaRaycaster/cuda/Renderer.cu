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

#include "Renderer.cuh"

#include <cuda_runtime.h>
#include <device_functions.h>
#include "math.cuh"

#include "PixelBufferObject.cuh"
#include "ColorMap.cuh"
#include "ClipPlanes.cuh"
#include "TexturePool.cuh"

namespace livre
{
namespace cuda
{
#define EARLY_EXIT 0.999f
#define EPSILON 0.0000000001f
#define SH_UINT 0u
#define SH_INT 1u
#define SH_FLOAT 2u

inline __device__ float4 calcPositionInEyeSpaceFromWindowSpace( const float2& windowSpace,
                                                                const uint4& viewport,
                                                                const float* invProjMatrix )
{
    const float2& viewportzw = make_float2( viewport.z, viewport.w );
    const float2& ndcPosxy = 2.0 * ( windowSpace - make_float2( viewport.x, viewport.y ) -
                            ( viewportzw / 2.0f )) / viewportzw;

    const float4& ndcPos = make_float4( ndcPosxy.x, ndcPosxy.y, 1.0f, 1.0f );
    const float4& eyeSpacePos = invProjMatrix * ndcPos;
    return eyeSpacePos / eyeSpacePos.w;
}

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
// implementation from nvidia sample
__device__ bool intersectBox( const float3& origin,
                              const float3& dir,
                              const float3& boxMin,
                              const float3& boxMax,
                              float *tnear,
                              float *tfar )
{
    // compute intersection of ray with all six bbox planes
    const float3& invR = recp( dir );
    const float3& tbot = invR * ( boxMin - origin );
    const float3& ttop = invR * ( boxMax - origin );

    // re-order intersections to find smallest and largest on each axis
    const float3& tmin = fminf( ttop, tbot );
    const float3& tmax = fmaxf( ttop, tbot );

    // find the largest tmin and the smallest tmax
    const float largestTmin = fmaxf( fmaxf( tmin.x, tmin.y ), fmaxf( tmin.x, tmin.z ));
    const float smallestTmax = fminf( fminf( tmax.x, tmax.y ), fminf( tmax.x, tmax.z ));

    *tnear = largestTmin;
    *tfar = smallestTmax;

    return smallestTmax > largestTmin;
}


inline __device__ float4 composite( const float4& src,
                                    const float4& dst,
                                    const float alphaCorrection )
{
    // The alpha correction function behaves badly around maximum alpha
    const float alpha = 1.0f - pow( 1.0f - min( src.w, 1.0f - 1.0f / 256.0 ), alphaCorrection );
    const float3& xyz = make_float3( dst ) + make_float3( src ) * alpha * ( 1.0 - dst.w );
    const float dstw = dst.w + alpha * ( 1.0f - dst.w );
    return {  xyz.x, xyz.y, xyz.z, dstw };
}

__global__ void rayCast( const cudaTextureObject_t dataTexture,
                         float4* pixelBuffer,
                         const unsigned int pixelBufferWidth,
                         const unsigned int pixelBufferHeight,
                         const ::livre::cuda::ClipPlanes clipPlanes,
                         const cudaTextureObject_t tfTexture,
                         const ::livre::cuda::ViewData viewData,
                         const unsigned int nodeCount,
                         const ::livre::cuda::NodeData* nodeDatas,
                         const ::livre::cuda::RenderData renderData )
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (( x >= pixelBufferWidth ) || ( y >= pixelBufferHeight ))
        return;

    const float4& pixelEyeSpacePos =
            calcPositionInEyeSpaceFromWindowSpace( make_float2( x, y ),
                                                   make_uint4FromArray( viewData.glViewport.array ),
                                                   viewData.invProjMatrix.array );

    const float4& pixelWorldSpacePos = viewData.invViewMatrix.array * pixelEyeSpacePos;
    const float3& eyePos = make_float3FromArray( viewData.eyePosition.array );
    float3 dir = normalize( make_float3( pixelWorldSpacePos ) -  eyePos );

    if( dir.x == 0.0f ) dir.x = EPSILON;
    if( dir.y == 0.0f ) dir.y = EPSILON;
    if( dir.z == 0.0f ) dir.z = EPSILON;

    const unsigned int pixelPos = y * pixelBufferWidth + x;
    const float alpha = pixelBuffer[ pixelPos ].w;

    if( alpha > EARLY_EXIT )
        return;

    float4 color = pixelBuffer[ pixelPos ];
    float tNearGlobal, tFarGlobal;

    const float3& globalBoxMin = make_float3FromArray( viewData.aabbMin.array );
    const float3& globalBoxMax = make_float3FromArray( viewData.aabbMax.array );
    const float3& origin = eyePos;
    if( !intersectBox( origin, dir, globalBoxMin, globalBoxMax, &tNearGlobal, &tFarGlobal ))
        return;

    const float4* clipPlanesArray = clipPlanes.getClipPlanes();
    for( unsigned int i = 0; i < clipPlanes.getNPlanes(); i++ )
    {
        const float4& clipPlane = clipPlanesArray[ i ];
        const float3& planeNormal = make_float3( clipPlane );
        float rn = dot( dir, planeNormal );
        if( rn == 0.0f )
            rn = EPSILON;
        const float d = clipPlane.w;
        float t = -( dot( planeNormal, eyePos) + d ) / rn;
        if( rn > 0.0 )
            tNearGlobal = fmaxf( tNearGlobal, t );
        else
            tFarGlobal = fminf( tFarGlobal, t );
    }

    if( tNearGlobal > tFarGlobal )
        return;

    const float3 nPixelEyeSpacePos = normalize( make_float3( pixelEyeSpacePos ));
    const float tNearPlane = -viewData.nearPlane / nPixelEyeSpacePos.z;

    const float2& dataSourceRange = make_float2FromArray( renderData.dataSourceRange.array );
    const float multiplyer = 1.0f / ( dataSourceRange.y - dataSourceRange.x );
    const float addedValue = -dataSourceRange.x / ( dataSourceRange.y - dataSourceRange.x );

    // http://stackoverflow.com/questions/12494439/opacity-correction-in-raycasting-volume-rendering
    const float alphaCorrection = float( renderData.maxSamplesPerRay ) /
                                  float( renderData.samplesPerRay );

    const float stepSize = 1.0 / float( renderData.samplesPerRay );

    for( unsigned int i = 0; i < nodeCount; ++i  )
    {
        const ::livre::cuda::NodeData nodeData = nodeDatas[ i ];
        const float3& boxMin = make_float3FromArray( nodeData.aabbMin.array );
        const float3& boxSize = make_float3FromArray( nodeData.aabbSize.array );
        const float3& boxMax = boxMin + boxSize;

        float tNear = 0.0f; float tFar = 0.0f;
        if( !intersectBox( origin, dir, boxMin, boxMax, &tNear, &tFar ))
            continue;

        if( tNear > tFarGlobal )
            break;

        tNear = fmaxf( fmaxf( tNearPlane, tNear ), tNearGlobal );
        tFar = fminf( tFar, tFarGlobal );

        if( tNear > tFar )
            continue;

        const float3& rayStart = origin + dir * tNear;
        const float3& rayStop = origin + dir * tFar;

        float3 pos = rayStart;
        const float3& step = normalize( rayStop - rayStart ) * stepSize;

        const float dist = distance( rayStop, rayStart );

        const float3& texMin = make_float3FromArray( nodeData.textureMin.array );
        const float3& texSize = make_float3FromArray( nodeData.textureSize.array );

        bool isEarlyExit = false;
        // Front-to-back absorption-emission integrator
        for( float travel = dist; travel > 0.0; pos += step, travel -= stepSize )
        {
            const float3& texPos = ((( pos - boxMin ) / boxSize) * texSize ) + texMin;
            const float density = tex3D< unsigned char >( dataTexture,
                                                          texPos.x,
                                                          texPos.y,
                                                          texPos.z );
            const float4& transferFn = tex1D< float4 >( tfTexture,
                                                        density * multiplyer + addedValue );

            color = composite( transferFn, color, alphaCorrection );
            isEarlyExit = color.w > EARLY_EXIT;

            if( isEarlyExit )
                break;
        }

        if( isEarlyExit )
            break;

     }
     pixelBuffer[ pixelPos ] = color;
}

struct Renderer::Impl
{
    Impl()
    {
       cudaGLSetGLDevice( 0 );
       cudaMalloc( &_cudaNodeDatas, sizeof( ::livre::cuda::NodeData ) * 16384 );
    }

    ~Impl()
    {
        cudaFree( _cudaNodeDatas );
    }

    void update( const lexis::render::ClipPlanes& clipPlanes,
                 const lexis::render::ColorMap& colorMap )
    {
        _cudaClipPlanes.upload( clipPlanes );
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
         checkCudaErrors( cudaMemcpy( _cudaNodeDatas,
                                      nodeDatas.data(),
                                      sizeof( NodeData ) * nodeDatas.size(),
                                      cudaMemcpyHostToDevice ));

        _nodeCount = nodeDatas.size();
    }

    int iDivUp( const int a, const int b ) const
    {
        return (a % b != 0) ? (a / b + 1) : (a / b);
    }

    void render( const cuda::ViewData& viewData,
                 const cuda::NodeDatas& nodeDatas,
                 const cuda::RenderData& renderData,
                 const cuda::TexturePool& texturePool )
    {

        const unsigned int width = _cudaRenderPBO.getWidth();
        const unsigned int height = _cudaRenderPBO.getHeight();

        copyNodeDatasToCudaMemory( nodeDatas );

        const dim3 blockDim( 16, 16 );
        const dim3 gridDim( iDivUp( width, blockDim.x ), iDivUp( height, blockDim.y ));
        rayCast<<< gridDim, blockDim >>>( texturePool.getTexture(),
                                          _cudaRenderPBO.getBuffer(),
                                          _cudaRenderPBO.getWidth(),
                                          _cudaRenderPBO.getHeight(),
                                          _cudaClipPlanes,
                                          _cudaColorMap.getTexture(),
                                          viewData,
                                          _nodeCount,
                                          _cudaNodeDatas,
                                          renderData );
    }

    void postRender()
    {
        _cudaRenderPBO.unmapBuffer();

        GLfloat modelview[16];
        GLfloat projection[16];
        glGetFloatv(GL_MODELVIEW_MATRIX, modelview);
        glGetFloatv(GL_PROJECTION_MATRIX, projection);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

        glDisable(GL_DEPTH_TEST);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glRasterPos2i(0, 0);
        glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, _cudaRenderPBO.getId( ));
        glDrawPixels( _cudaRenderPBO.getWidth(),
                      _cudaRenderPBO.getHeight(),
                      GL_RGBA, GL_FLOAT, 0);
        glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );

        glMatrixMode(GL_PROJECTION);
        glLoadMatrixf( projection );
        glMatrixMode(GL_MODELVIEW);
        glLoadMatrixf( modelview );
    }

    ::livre::cuda::ClipPlanes _cudaClipPlanes;
    ::livre::cuda::ColorMap _cudaColorMap;
    ::livre::cuda::PixelBufferObject _cudaRenderPBO;
    unsigned int _nodeCount;
    ::livre::cuda::NodeData* _cudaNodeDatas;
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

void Renderer::render( const ViewData& viewData,
                       const NodeDatas& nodeData,
                       const RenderData& renderData,
                       const cuda::TexturePool& texturePool )
{
    _impl->render( viewData, nodeData, renderData, texturePool );
}

void Renderer::postRender()
{
    _impl->postRender();
}

void Renderer::update( const lexis::render::ClipPlanes& clipPlanes,
                       const lexis::render::ColorMap& colorMap )
{
    _impl->update( clipPlanes, colorMap );
}
}
}





