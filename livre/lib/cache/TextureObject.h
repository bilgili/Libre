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

#ifndef _TextureObject_h_
#define _TextureObject_h_

#include <livre/lib/api.h>
#include <livre/lib/types.h>
#include <livre/core/mathTypes.h>
#include <livre/core/cache/CacheObject.h>

namespace livre
{

/**
 * The TextureObject class holds the informarmation for the data which is on the GPU.
  */
class TextureObject : public CacheObject
{
    friend class TextureCache;

public:

    LIVRE_API virtual ~TextureObject();

    /**
     * @return The texture state.
     */
    LIVRE_API TextureStatePtr getTextureState();

    /**
     * @return The texture state ( const ).
     */
    LIVRE_API ConstTextureStatePtr getTextureState() const;

private:

    /**
     * Constructor
     * @param cacheId is the unique identifier
     * @param textureCache the cache the texture object is created
     * @throws CacheLoadException when the data cache does not have the data for cache id
     */
    TextureObject( const CacheId& cacheId,
                   TextureCache& textureCache );

    size_t _getSize() const final;

    struct Impl;
    std::unique_ptr<Impl> _impl;
};

}

#endif // _TextureObject_h_
