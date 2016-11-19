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

#ifndef _CudaTextureUploadFilter_h_
#define _CudaTextureUploadFilter_h_

#include "types.h"

#include <livre/lib/types.h>
#include <livre/core/pipeline/Filter.h>
#include <livre/core/render/RenderInputs.h>

namespace livre
{

/** CudaTextureUploadFilter class implements the CudaTextureObject uploading */
class CudaTextureUploadFilter : public Filter
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
    void execute( const FutureMap& input, PromiseMap& output ) const final;

    /** @copydoc Filter::getInputDataInfos */
    DataInfos getInputDataInfos() const final
    {
        return
        {
            { "RenderInputs", getType< RenderInputs >() },
            { "DataCacheObjects", getType< ConstCacheObjects >() },
        };
    }

    /** @copydoc Filter::getOutputDataInfos */
    DataInfos getOutputDataInfos() const final
    {
        return
        {
            { "CudaTextureCacheObjects", getType< ConstCacheObjects >() },
        };
    }

private:

    struct Impl;
    std::unique_ptr<Impl> _impl;
};
}

#endif //_CudaTextureUploadFilter_h_
