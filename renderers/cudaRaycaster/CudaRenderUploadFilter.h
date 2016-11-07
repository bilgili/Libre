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

#ifndef _CudaRenderUploadFilter_h_
#define _CudaRenderUploadFilter_h_

#include "types.h"

#include <livre/lib/types.h>
#include <livre/core/render/RenderInputs.h>
#include <livre/core/pipeline/Filter.h>

namespace livre
{


/**
 * CudaRenderUploadFilter class implements the parallel data loading for raw volume data and
 * textures. A group of uploaders is executed in rendering pipeline and each uploader
 * has an id in the group.
 */
class CudaRenderUploadFilter : public Filter
{
public:

    /**
     * Constructor
     * @param textureCache texture cache
     * @param texturePool pool for textures
     * @param nUploadThreads mumber of data upload thread
     * @param executor that runs the upload operations
     */
    CudaRenderUploadFilter( DataCache& dataCache,
                            CudaTextureCache& cudaCache,
                            CudaTexturePool& texturePool,
                            size_t nUploadThreads,
                            Executor& executor );
    ~CudaRenderUploadFilter();

    /** @copydoc Filter::execute */
    void execute( const FutureMap& input, PromiseMap& output ) const final;

    /** @copydoc Filter::getInputDataInfos */
    DataInfos getInputDataInfos() const final
    {
        return
        {
            { "RenderInputs", getType< RenderInputs >() },
            { "NodeIds", getType< NodeIds >() },
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

#endif // _CudaRenderUploadFilter_h_
