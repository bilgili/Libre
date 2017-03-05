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

#ifndef _Cuda_Renderer_h
#define _Cuda_Renderer_h

#include "common.cuh"
#include "types.cuh"

#include <livre/core/mathTypes.h>
#include <lexis/render/ColorMap.h>
#include <lexis/render/clipPlanes.h>

namespace livre
{
namespace cuda
{
class TexturePool;

/** Cuda representation of the renderer */
class Renderer
{
public:

    /** Constructor */
    Renderer();
    ~Renderer();

    /**
     * Updates the clip planes and the color map for rendering kernel
     * @param clipPlanes is the clip palnes
     * @param colorMap is the color map
     */
    void update( const lexis::render::ClipPlanes& clipPlanes,
                 const lexis::render::ColorMap& colorMap );

    /**
     * Updates the irradiance texture
     * @param irradianceCompute
     */
    void update( IrradianceCompute& irradianceCompute );

    /**
     * Called by the Libre renderer before rendering
     * @param viewData is the necessary view data settings
     */
    void preRender( const ViewData& viewData );

    /**
     * Called by the Libre renderer while rendering
     * @param viewData is the necessary view data settings
     * @param nodeDatas is the list of nodes to be rendered
     * @param renderData is the necessary render settings
     * @param texturePool is the cuda representation of the texture pool
     */
    void render( const ViewData& viewData,
                 const NodeDatas& nodeDatas,
                 const RenderData& renderData,
                 const cuda::TexturePool& texturePool );

    /**
     * Called by the libre renderer after rendering. The PBO is written
     * to draw buffer.
     */
    void postRender();

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
}
}
#endif // _Cuda_Renderer_h

