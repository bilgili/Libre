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

#ifndef _CudaTextureObject_h_
#define _CudaTextureObject_h_

#include "CudaTexturePool.h"

#include <livre/lib/api.h>
#include <livre/lib/types.h>
#include <livre/core/mathTypes.h>
#include <livre/core/cache/CacheObject.h>

namespace livre
{

/**
 * The CudaObject class holds the informarmation for the data which is on the GPU.
  */
class CudaTextureObject : public CacheObject
{
public:

    /**
     * Constructor
     * @param cacheId is the unique identifier
     * @param dataCache source for the raw data
     * @param dataSource provides information about spatial structure of texture
     * @throws CacheLoadException when the data cache does not have the data for cache id
     */
    CudaTextureObject( const CacheId& cacheId,
                       const Cache& dataCache,
                       const DataSource& dataSource,
                       CudaTexturePool& pool );

    virtual ~CudaTextureObject();

    /** @copydoc livre::CacheObject::getSize */
    size_t getSize() const final;

    /** @return The texture position in normalized space.*/
    Vector3f getTexPosition() const;

    /** @return The texture size in normalized space.*/
    Vector3f getTexSize() const;

    /** @return texture pool */
    CudaTexturePool& getTexturePool() const;

private:

    struct Impl;
    std::unique_ptr<Impl> _impl;
};

}

#endif // _CudaTextureObject_h_
