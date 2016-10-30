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

#include <livre/lib/cache/DataObject.h>
#include <livre/lib/cache/TextureObject.h>

#include <livre/core/cache/Cache.h>
#include <livre/core/data/LODNode.h>
#include <livre/core/data/DataSource.h>
#include <livre/core/render/TexturePool.h>

#include <GL/glew.h>

namespace livre
{
namespace
{
size_t getTextureSize( const DataSource& dataSource )
{
    const Vector3ui& textureSize = dataSource.getVolumeInfo().maximumBlockSize;
    return textureSize.product() * dataSource.getVolumeInfo().getBytesPerVoxel();
}
}

/**
 * The TextureObject class holds the informarmation for the data which is on the GPU.
  */
struct TextureObject::Impl
{

    Impl( const CacheId& cacheId,
          const Cache& dataCache,
          const DataSource& dataSource,
          TexturePool& texturePool )
       : _texturePool( texturePool )
       , _texture( texturePool.generate( ))
       , _size( getTextureSize( dataSource ))
   {
        if( !load( cacheId, dataCache, dataSource, texturePool ))
            LBTHROW( CacheLoadException( cacheId, "Unable to construct texture cache object" ));
    }

    ~Impl()
    {
        _texturePool.release( _texture );
    }

    bool load( const CacheId& cacheId,
               const Cache& dataCache,
               const DataSource& dataSource,
               const TexturePool& texturePool )
    {
        ConstDataObjectPtr data = dataCache.get< DataObject >( cacheId );
        if( !data )
            return false;

        initialize( cacheId, dataSource, texturePool, data );
        return true;
    }

    void initialize( const CacheId& cacheId,
                     const DataSource& dataSource,
                     const TexturePool& texturePool,
                     const ConstDataObjectPtr& data )
    {
        // TODO: The internal format size should be calculated correctly
        const Vector3f& overlap = dataSource.getVolumeInfo().overlap;
        const LODNode& lodNode = dataSource.getNode( NodeId( cacheId ));
        const Vector3f& size = lodNode.getVoxelBox().getSize();
        const Vector3f& maxSize = dataSource.getVolumeInfo().maximumBlockSize;
        const Vector3f& overlapf = overlap / maxSize;
        _texturePos = overlapf;
        _textureSize = ( overlapf + size / maxSize ) - _texturePos;

        loadTextureToGPU( lodNode, dataSource, texturePool, data );
    }

    void bind() const
    {
        glBindTexture( GL_TEXTURE_3D, _texture );
    }

    bool loadTextureToGPU( const LODNode& lodNode,
                           const DataSource& dataSource,
                           const TexturePool& texturePool,
                           const ConstDataObjectPtr& data ) const
    {
    #ifdef LIVRE_DEBUG_RENDERING
        std::cout << "Upload "  << lodNode.getNodeId().getLevel() << ' '
                  << lodNode.getRelativePosition() << " to "
                  << _texture << std::endl;
    #endif
        const Vector3ui& overlap = dataSource.getVolumeInfo().overlap;
        const Vector3ui& voxSizeVec = lodNode.getBlockSize() + overlap * 2;
        bind();
        glTexSubImage3D( GL_TEXTURE_3D, 0, 0, 0, 0,
                         voxSizeVec[0], voxSizeVec[1], voxSizeVec[2],
                         texturePool.getFormat() ,
                         texturePool.getTextureType(),
                         data->getDataPtr( ));

        const GLenum glErr = glGetError();
        if ( glErr != GL_NO_ERROR )
        {
            LBERROR << "Error loading the texture into GPU, error number : "  << glErr << std::endl;
            return false;
        }

        return true;
    }


    TexturePool& _texturePool;
    Vector3f _texturePos; //!< Minimum texture coordinates in the maximum texture block.
    Vector3f _textureSize; //!< The texture size.
    GLuint _texture; //!< The OpenGL texture id.
    size_t _size;
};

TextureObject::TextureObject( const CacheId& cacheId,
                              const Cache& dataCache,
                              const DataSource& dataSource,
                              TexturePool& texturePool )
   : CacheObject( cacheId )
   , _impl( new Impl( cacheId, dataCache, dataSource, texturePool ))
{}

TextureObject::~TextureObject()
{}

size_t TextureObject::getSize() const
{
    return _impl->_size;
}


/** @return The texture position in normalized space.*/
Vector3f TextureObject::getTexPosition() const
{
    return _impl->_texturePos;
}

/** @return The texture size in normalized space.*/
Vector3f TextureObject::getTexSize() const
{
    return _impl->_textureSize;
}

/** OpenGL bind() the texture. */
void TextureObject::bind() const
{
    _impl->bind();
}

}
