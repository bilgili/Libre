/* Copyright (c) 2011-2015, EPFL/Blue Brain Project
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

#ifndef _RenderingSetGeneratorFilter_h_
#define _RenderingSetGeneratorFilter_h_

#include <livre/lib/types.h>

#include <livre/core/cache/Cache.h>
#include <livre/core/render/FrameInfo.h>
#include <tuyau/filter.h>
#include <livre/core/render/FrameInfo.h>

namespace livre
{

/**
 * RenderingSetGeneratorFilter class generates the rendering set given the visibles.
 */
template< class CacheObjectT >
class RenderingSetGeneratorFilter : public tuyau::Filter
{
public:

    /**
     * Constructor
     * @param cache the cache that holds the rendering objects
     */
    explicit RenderingSetGeneratorFilter( const Cache< CacheObjectT >& cache );
    ~RenderingSetGeneratorFilter();

    /** @copydoc Filter::execute */
    void execute( const tuyau::FutureMap& input, tuyau::PromiseMap& output ) const final;

    /** @copydoc Filter::getInputDataInfos */
    tuyau::DataInfos getInputDataInfos() const final
    {
        return
        {
            { "VisibleNodes", tuyau::getType< NodeIds >() }
        };
    }

    /** @copydoc Filter::getOutputDataInfos */
    tuyau::DataInfos getOutputDataInfos() const final
    {
        return
        {
            { "CacheObjects", tuyau::getType< ConstCacheObjects >() },
            { "NodeIds", tuyau::getType< NodeIds >() },
            { "RenderingDone", tuyau::getType< bool >() },
            { "RenderStatistics", tuyau::getType< RenderStatistics >() },
        };
    }

private:

    struct Impl;
    std::unique_ptr<Impl> _impl;
};
#include "RenderingSetGeneratorFilter.ipp"
}

#endif
