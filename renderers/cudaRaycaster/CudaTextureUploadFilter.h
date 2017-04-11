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

#ifndef _CudaTextureUploadFilter_h_
#define _CudaTextureUploadFilter_h_

#include "types.h"

#include <livre/lib/types.h>
#include <tuyau/filter.h>
#include <livre/core/render/RenderInputs.h>

namespace livre
{

/** CudaTextureUploadFilter class implements the CudaTextureObject uploading */
class CudaTextureUploadFilter : public tuyau::Filter
{
public:

    /**
     * Constructor
     * @param data cache data cache
     * @param textureCache texture cache
     * @param texturePool the pool for 3D textures
     */
    CudaTextureUploadFilter( const DataCache& dataCache,
                             CudaTextureCache& textureCache,
                             CudaTexturePool& texturePool );
    ~CudaTextureUploadFilter();

    /** @copydoc Filter::execute */
    void execute( const tuyau::FutureMap& input, tuyau::PromiseMap& output ) const final;

    /** @copydoc Filter::getInputDataInfos */
    tuyau::DataInfos getInputDataInfos() const final
    {
        return
        {
            { "RenderInputs", tuyau::getType< RenderInputs >() },
            { "DataCacheObjects", tuyau::getType< ConstCacheObjects >() },
        };
    }

    /** @copydoc Filter::getOutputDataInfos */
    tuyau::DataInfos getOutputDataInfos() const final
    {
        return
        {
            { "CudaTextureCacheObjects", tuyau::getType< ConstCacheObjects >() },
        };
    }

private:

    struct Impl;
    std::unique_ptr<Impl> _impl;
};
}

#endif //_CudaTextureUploadFilter_h_
