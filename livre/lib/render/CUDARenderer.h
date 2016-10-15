/* Copyright (c) 2011-2016  Ahmet Bilgili <ahmetbilgili@gmail.com>
 *
 * This file is part of Livre <https://github.com/bilgili/Livre>
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

#ifndef _CUDARenderer_h_
#define _CUDARenderer_h_

#include <livre/lib/types.h>
#include <livre/core/render/Renderer.h>

namespace livre
{

/** The CUDA class implements a single-pass ray caster in CUDA. */
class CUDARenderer : public Renderer
{
public:

    /**
     * Constructor
     * @param dataSource the data source
     * @param dataCache the source for cached data
     * @param samplesPerRay Number of samples per ray.
     * @param samplesPerPixel Number of samples per pixel.
     */
    CUDARenderer( const DataSource& dataSource,
                  const Cache& dataCache,
                  uint32_t samplesPerRay,
                  uint32_t samplesPerPixel );
    ~CUDARenderer();

    /**
     * Updates the renderer state with new values wrt samples per ray & pixel
     * and color map.
     * @param renderSettings the current render settings
     * @param renderParams the current render params
     */
    void update( const RenderSettings& renderSettings,
                 const VolumeRendererParameters& renderParams );

    /**
     * @copydoc Renderer::order
     */
    NodeIds order( const NodeIds& bricks, const Frustum& frustum ) const override;

protected:

    void _preRender( const Frustum& frustum,
                     const ClipPlanes& planes,
                     const PixelViewport& view,
                     const NodeIds& renderBricks ) override;

    void _render( const Frustum& frustum,
                  const ClipPlanes& planes,
                  const PixelViewport& view,
                  const NodeIds& orderedBricks ) final;

    void _postRender( const Frustum& frustum,
                      const ClipPlanes& planes,
                      const PixelViewport& view,
                      const NodeIds& renderBricks ) override;

    struct Impl;
    std::unique_ptr< Impl > _impl;
};

}

#endif // _CUDARenderer_h_
