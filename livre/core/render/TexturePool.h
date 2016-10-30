/* Copyright (c) 2011-2014, EPFL/Blue Brain Project
 *                     Ahmet Bilgili <ahmet.bilgili@epfl.ch>
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

#ifndef _TexturePool_h_
#define _TexturePool_h_

#include <livre/core/api.h>
#include <livre/core/types.h>
#include <livre/core/mathTypes.h>

#include <GL/glew.h>

namespace livre
{

/**
 * The TexturePool class is responsible for allocating texture slots and copying
 * textures into the texture slots. The methods are not thread safe.
 */
class TexturePool
{
public:

    /**
     * Constructor. Preallocates texture blocks
     * @param dataSource the data source
     * @throws std::runtime_error if data has multiple channels
     */
    LIVRECORE_API TexturePool( const DataSource& dataSource );
    LIVRECORE_API ~TexturePool();

    /** @return The OpenGL GPU internal format of the texture data. */
    LIVRECORE_API int32_t getInternalFormat() const { return _internalTextureFormat; }

    /** @return The OpenGL data type of the texture data. */
    LIVRECORE_API uint32_t getFormat() const { return _format; }

    /** @return The OpenGL data type of the texture data. */
    LIVRECORE_API uint32_t getTextureType() const { return _textureType; };

    /**
     * Generates / uses a preallocated a 3D OpenGL texture based on OpenGL parameters.
     * @return open gl texture id
     */
    LIVRECORE_API GLuint generate();

    /**
     * Releases a texture slot.
     * @param texture the given open gl texture id.
     */
    LIVRECORE_API void release( GLuint texture );

private:

    UInt32s _textureStack;

    const Vector3ui _maxBlockSize;
    int32_t _internalTextureFormat;
    uint32_t _format;
    uint32_t _textureType;

    boost::mutex _mutex;
};


}

#endif // _TexturePool_h_
