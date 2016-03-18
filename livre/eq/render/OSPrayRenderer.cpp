/* Copyright (c) 2011-2016, EPFL/Blue Brain Project
 *                          Ahmet Bilgili <ahmet.bilgili@epfl.ch>
 *
 * This file is part of Livre <https://github.com/BlueBrain/Livre>
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

#include <livre/core/data/DataSource.h>
#include <livre/core/data/VolumeInformation.h>
#include <livre/core/render/GLSLShaders.h>
#include <livre/core/render/Frustum.h>
#include <livre/core/render/TransferFunction1D.h>
#include <livre/core/data/LODNode.h>
#include <livre/core/maths/maths.h>
#include <livre/core/render/GLContext.h>

#include <livre/lib/configuration/VolumeRendererParameters.h>
#include <livre/lib/cache/TextureDataCache.h>

#include <livre/eq/FrameData.h>
#include <livre/eq/render/OSPrayRenderer.h>
#include <livre/eq/settings/RenderSettings.h>

#include <livre/eq/render/ospray/OSPrayVolume.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcomment"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <ospray/render/Renderer.h>
#include <ospray/fb/FrameBuffer.h>
#include <ospray/transferFunction/TransferFunction.h>
#pragma GCC diagnostic pop

#include <eq/eq.h>
#include <eq/gl.h>

namespace livre
{

// Sort helper function for sorting the textures with their distances to viewpoint
struct DistanceOperator
{
    explicit DistanceOperator( const DataSource& dataSource, const Frustum& frustum )
        : _frustum( frustum )
        , _dataSource( dataSource )
    { }

    bool operator()( const NodeId& rb1,
                     const NodeId& rb2 )
    {
        const LODNode& lodNode1 = _dataSource.getNode( rb1 );
        const LODNode& lodNode2 = _dataSource.getNode( rb2 );

        const float distance1 = ( _frustum.getMVMatrix() *
                                  lodNode1.getWorldBox().getCenter() ).length();
        const float distance2 = ( _frustum.getMVMatrix() *
                                  lodNode2.getWorldBox().getCenter() ).length();
        return  distance1 < distance2;
    }
    const Frustum& _frustum;
    const DataSource& _dataSource;
};

#define glewGetContext() GLContext::getCurrent()->glewGetContext()

namespace
{
std::string where( const char* file, const int line )
{
    return std::string( " in " ) + std::string( file ) + ":" +
           boost::lexical_cast< std::string >( line );
}

const uint32_t maxSamplesPerRay = 32;
const uint32_t minSamplesPerRay = 512;

const GLfloat fullScreenQuad[] = { -1.0f, -1.0f, 0.0f,
                                    1.0f, -1.0f, 0.0f,
                                   -1.0f,  1.0f, 0.0f,
                                   -1.0f,  1.0f, 0.0f,
                                    1.0f, -1.0f, 0.0f,
                                    1.0f,  1.0f, 0.0f };

}

struct OSPrayRenderer::Impl
{
    Impl( const TextureDataCache& dataCache,
          uint32_t samplesPerRay,
          uint32_t samplesPerPixel )
        : _nSamplesPerRay( samplesPerRay )
        , _nSamplesPerPixel( samplesPerPixel )
        , _renderTexture( GL_TEXTURE_RECTANGLE_ARB, glewGetContext( ))
        , _dataCache( dataCache )
        , _dataSource( dataCache.getDataSource( ))
        , _volInfo( _dataSource.getVolumeInfo( ))
        , _ospVolume( ospNewVolume( "livre_ospray_volume" ))
        , _ospRenderer( ospNewRenderer( "dvr" ))
        , _ospTransferFunction( ospNewTransferFunction( "piecewise_linear" ))
        , _ospAmbientLight( ospNewLight( _ospRenderer.get(), "AmbientLight" ))
        , _ospDirectionalLight( ospNewLight( _ospRenderer.get(), "DirectionalLight" ))
        , _ospCamera( ospNewCamera( "perspective" ))
        , _ospModel( ospNewModel( ))

    {
        setDefaultOsprayParameters();

        glGenBuffers( 1, &_quadVBO );
        glBindBuffer( GL_ARRAY_BUFFER, _quadVBO );
        glBufferData( GL_ARRAY_BUFFER, sizeof( fullScreenQuad ), fullScreenQuad, GL_STATIC_DRAW );
    }

    ~Impl()
    {
        _renderTexture.flush();
        glDeleteBuffers( 1, &_quadVBO );
    }


    NodeIds order( const NodeIds& bricks,
                   const Frustum& frustum ) const
    {
        NodeIds rbs = bricks;
        DistanceOperator distanceOp( _dataSource, frustum );
        std::sort( rbs.begin(), rbs.end(), distanceOp );
        return rbs;
    }

    void createFBIfViewportChanges( const PixelViewport& viewport )
    {
        const int32_t w = viewport[ 2 ] - viewport[ 0 ];
        const int32_t h = viewport[ 3 ] - viewport[ 1 ];

        if( _renderTexture.getWidth() != w || _renderTexture.getHeight() != h )
        {
            _renderTexture.flush();
            _renderTexture.init( GL_RGBA32F, w, h );
            _ospFrameBuffer.reset( ospNewFrameBuffer( { w, h },
                                                      OSP_FB_RGBA8,
                                                      OSP_FB_COLOR | OSP_FB_ACCUM ));

            ospSet1f( _ospFrameBuffer.get(), "gamma", 2.2f );
            ospCommit( _ospFrameBuffer.get( ));
            ospFrameBufferClear( _ospFrameBuffer.get(), OSP_FB_ACCUM );
        }

    }

    void updateOSPTf( const TransferFunction1D& tf )
    {
        Floats colors;
        Floats opacities;

        const size_t lutSize = tf.getLutSize() / 4;
        colors.resize( 3 * lutSize ); // RGB * 256
        opacities.resize( lutSize );
        const uint8_t* lut = tf.getLut();
        for( size_t i = 0; i < lutSize; ++i )
        {
            colors[ i ] = float( lut[ i * 4 + 3 ] / 256.0f );
            for( size_t j = 0; j < 3; ++j )
                colors[ i * 3 + j ] = float( lut[ i * 4 + j ] ) / 256.0f;
        }

        const OSPData& ospColors = ospNewData( colors.size(), OSP_FLOAT3, colors.data( ));
        const OSPData& ospOpacities = ospNewData( opacities.size(),
                                                  OSP_FLOAT,
                                                  opacities.data( ));
        ospSetData( _ospTransferFunction.get(), "colors", ospColors );
        ospSetData( _ospTransferFunction.get(), "opacities", ospOpacities );
        ospCommit( _ospTransferFunction.get( ));
        ospSetObject( _ospVolume.get(), "transferFunction", _ospTransferFunction.get( ));
        ospCommit( _ospVolume.get( ));
    }

    void setDefaultOsprayParameters()
    {
        if( !_ospRenderer )
            LBTHROW( std::runtime_error( "OSP renderer cannot be created" ));

        ospCommit( _ospAmbientLight.get( ));
        ospCommit( _ospDirectionalLight.get( ));


        ospSetObject( _ospRenderer.get(), "camera", _ospCamera.get( ));
        std::array< osp::Light*, 2 > lights = { _ospAmbientLight.get(),
                                                _ospDirectionalLight.get( )};
        ospSetData( _ospRenderer.get(), "lights", ospNewData( lights.size(),
                                                              OSP_OBJECT,
                                                              lights.data( )));

        const Boxf& boundingBox = _dataSource.getVolumeInfo().boundingBox;
        const Vector3f& bbMin = boundingBox.getMin();
        const Vector3f& bbMax = boundingBox.getMax();
        const osp::box3f ospBBox = {{ bbMin.x(), bbMin.y(), bbMin.z( )},
                                    { bbMax.x(), bbMax.y(), bbMax.z( )}};

        ospSetVec3f( _ospVolume.get(), "boundingBoxMin", ospBBox.lower );
        ospSetVec3f( _ospVolume.get(), "boundingBoxMax", ospBBox.upper );
        ospCommit( _ospVolume.get( ));

        ospAddVolume( _ospModel.get(), _ospVolume.get( ));
        ospSetObject( _ospRenderer.get(), "model", _ospModel.get( ));
        ospCommit( _ospRenderer.get( ));
        ospSetData( _ospVolume.get(), "dataCache", ospNewData( 1,
                                                               OSP_OBJECT,
                                                               &_dataCache ));

        updateOSPTf( TransferFunction1D( ));
    }

    void update( const FrameData& frameData )
    {
        updateOSPTf( frameData.getRenderSettings().getTransferFunction( ));
        _nSamplesPerRay = frameData.getVRParameters().getSamplesPerRay();
        ospSet1f( _ospVolume.get(), "samplingRate", _nSamplesPerRay );
        ospCommit( _ospVolume.get( ));
    }

    void onFrameStart( const Frustum& frustum,
                       const PixelViewport& view,
                       const NodeIds& orderedBricks )
    {
        ospSetData(  _ospVolume.get(), "nodeIds", ospNewData( orderedBricks.size(),
                                                              OSP_ULONG,
                                                              orderedBricks.data( )));

        const Vector3f& eyePos = frustum.getEyePos();
        const osp::vec3f eye = { eyePos.x(), eyePos.y(), eyePos.z() };

        const Vector3f& direction = frustum.getViewDir();
        const osp::vec3f dir = { direction.x(), direction.y(), direction.z() };

        const osp::vec3f up = { 0.0f, 1.0f, 0.0f };

        ospSetVec3f( _ospCamera.get(), "pos", eye );
        ospSetVec3f( _ospCamera.get(), "dir", dir );
        ospSetVec3f( _ospCamera.get(), "up", up );

        const float w = view[ 2 ] - view[ 0 ];
        const float h = view[ 3 ] - view[ 1 ];
        ospSetf( _ospCamera.get(), "aspect", w / h );

        const float near = frustum.nearPlane();
        const float fovy = 2.0f * std::atan2( h / 2.0, near );

        ospSetf( _ospCamera.get(), "fovy", fovy );
        ospCommit( _ospCamera.get( ));

        createFBIfViewportChanges( view );
    }

    void onFrameRender( const Frustum&,
                        const PixelViewport&,
                        const NodeIds& )
    {
        ospRenderFrame( _ospFrameBuffer.get(),
                        _ospRenderer.get(),
                        OSP_FB_COLOR | OSP_FB_ACCUM );
    }

    void copyTexToFrameBufAndClear()
    {
        GLSLShaders::Handle program = _texCopyShaders.getProgram( );
        LBASSERT( program );

        glUseProgram( program );
        glBindImageTexture( 0, _renderTexture.getName(),
                            0, GL_FALSE, 0, GL_READ_WRITE,
                            _renderTexture.getInternalFormat( ));
        GLint tParamNameGL = glGetUniformLocation( program, "renderTexture" );
        glUniform1i( tParamNameGL, 0 );

        glBindBuffer( GL_ARRAY_BUFFER, _quadVBO );
        glEnableVertexAttribArray( 0 );
        glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, 0, NULL );

        glDisable( GL_CULL_FACE );
        glDrawArrays( GL_TRIANGLES, 0, 6 );

        glUseProgram( 0 );
    }

    void onFrameEnd( const Frustum& ,
                     const PixelViewport&,
                     const NodeIds& )
    {
        const void* fb = (void *)ospMapFrameBuffer( _ospFrameBuffer.get(), OSP_FB_COLOR);
        _renderTexture.upload( _renderTexture.getWidth(),
                               _renderTexture.getHeight(),
                               fb );

        ospUnmapFrameBuffer( fb, _ospFrameBuffer.get( ));
        copyTexToFrameBufAndClear();

    }


    uint32_t _nSamplesPerRay;
    uint32_t _nSamplesPerPixel;
    uint32_t _computedSamplesPerRay;

    GLSLShaders _texCopyShaders;
    GLuint _quadVBO;
    eq::util::Texture _renderTexture;

    const TextureDataCache& _dataCache;
    const DataSource& _dataSource;
    const VolumeInformation& _volInfo;

    std::unique_ptr< osp::Volume > _ospVolume;
    std::unique_ptr< osp::Renderer > _ospRenderer;
    std::unique_ptr< osp::TransferFunction > _ospTransferFunction;
    std::unique_ptr< osp::Light > _ospAmbientLight;
    std::unique_ptr< osp::Light > _ospDirectionalLight;
    std::unique_ptr< osp::FrameBuffer > _ospFrameBuffer;
    std::unique_ptr< osp::Camera > _ospCamera;
    std::unique_ptr< osp::Texture2D > _ospMaxDepthTexture;
    std::unique_ptr< osp::Model > _ospModel;
};

OSPrayRenderer::OSPrayRenderer( const TextureDataCache& dataCache,
                                uint32_t samplesPerRay,
                                uint32_t samplesPerPixel  )
    : _impl( new OSPrayRenderer::Impl( dataCache,
                                       samplesPerRay,
                                       samplesPerPixel ))
{}

OSPrayRenderer::~OSPrayRenderer()
{}

void OSPrayRenderer::update( const FrameData& frameData )
{
    _impl->update( frameData );
}

NodeIds OSPrayRenderer::_order( const NodeIds& bricks,
                                const Frustum& frustum ) const
{
    return _impl->order( bricks, frustum );

}

void OSPrayRenderer::_onFrameStart( const Frustum& frustum,
                                    const PixelViewport& view,
                                    const NodeIds& orderedBricks )
{
    _impl->onFrameStart( frustum, view, orderedBricks );
}

void OSPrayRenderer::_onFrameRender( const Frustum& frustum,
                                     const PixelViewport& view,
                                     const NodeIds& orderedBricks )
{
     _impl->onFrameRender( frustum, view, orderedBricks );
}

void OSPrayRenderer::_onFrameEnd( const Frustum& frustum,
                                  const PixelViewport& view,
                                  const NodeIds& orderedBricks )
{
    _impl->onFrameEnd( frustum, view, orderedBricks );
}


}
