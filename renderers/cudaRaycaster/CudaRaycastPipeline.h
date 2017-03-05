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


#ifndef _CudaRaycasterPipeline_h_
#define _CudaRaycasterPipeline_h_

#include <livre/core/configuration/RendererParameters.h>
#include <livre/lib/types.h>

#include <livre/core/render/ClipPlanes.h>
#include <livre/core/render/FrameInfo.h>
#include <livre/core/render/RenderPipelinePlugin.h>
#include <livre/core/mathTypes.h>

namespace livre
{

/**
 * RenderPipeline generates and executes the rendering pipeline every frame
 */
class CudaRaycastPipeline : public RenderPipelinePlugin
{

public:
    /**
     * Constructor
     * @param name is the name of the pipeline
     */
    CudaRaycastPipeline( const std::string& name );
    ~CudaRaycastPipeline();

    /**
     * @param name is checked for the right plugin
     * @return true if renderer handles the name
     */
    static bool handles( const std::string& name ) { return name == "cuda"; }

    /** @return the data cache */
    DataCache& getDataCache();

private:

    RenderStatistics render( Renderer& renderer, const RenderInputs& renderInputs ) final;

    struct Impl;
    std::unique_ptr<Impl> _impl;
};

}

#endif // _CudaRaycasterPipeline_h_

