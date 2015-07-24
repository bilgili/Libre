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

#ifndef _HPXPipeline_h_
#define _HPXPipeline_h_

#include <livre/hpx/types.h>
#include <livre/core/Pipeline/Pipeline.h>


namespace livre
{

class HPXPipeline : public Pipeline
{

    /**
     * @param nDataUploaders Number of data uploader threads
     * @param nTextureUploaders Number of texture uploader threads
     */
    HPXPipeline( TextureDataCache& dataCache,
                 TextureCache& textureCache,
                 const size_t nDataUploaders,
                 const size_t nTextureUploaders,
                 GLContextPtr glContext );

    /**
     * Call to get the result of the request from Pipeline.
     * @param nodeIds The queried cache objects
     * @param cacheObjects Returned cache objects.
     * @param wait If wait is false, after submitting the work, the function
     * returns withe available objects, if false it waits all data to be loaded.
     */
    void getCacheObjects( const NodeIds& nodeIds,
                          ConstCacheObjects& cacheObjects,
                          bool wait = false ) final;

    /**
     * Clears the current work queue
     */
    virtual void clearWorkQueue() {}

private:

    class HPXPipelineImpl;
    HPXPipelineImpl *_impl;
};

}

#endif // _HPXPipeline_h_
