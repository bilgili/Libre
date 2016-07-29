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

#ifndef _OSprayRenderer_h_
#define _OSprayRenderer_h_

#include <livre/eq/types.h>
#include <livre/core/render/Renderer.h>

namespace livre
{

/**
 * The RayCastRenderer class implements a single-pass ray caster.
 */
class OSPrayRenderer : public Renderer
{
public:

    /**
     * @param samplesPerRay Number of samples per ray.
     * @param samplesPerPixel Number of samples per pixel.
     * @param gpuDataType Data type of the texture data source.
     * @param internalFormat Internal format of the texture in GPU memory.
     * @param volInfo Volume information.
     */
    OSPrayRenderer( const DataCache& dataCache,
                    uint32_t samplesPerRay,
                    uint32_t samplesPerPixel );
    ~OSPrayRenderer();

    /**
     * Updates the renderer state with new values wrt samples per ray & pixel
     * and transfer function.
     * @param frameData the current frame data containing new values
     */
    void update( const FrameData& frameData );

protected:

    NodeIds _order( const NodeIds& bricks,
                    const Frustum& frustum ) const override;

    void _onFrameStart( const Frustum& frustum,
                        const ClipPlanes& planes,
                        const PixelViewport& view,
                        const NodeIds& orderedBricks ) override;

    void _onFrameRender( const Frustum& frustum,
                         const ClipPlanes& planes,
                         const PixelViewport& view,
                         const NodeIds& orderedBricks ) final;

    void _onFrameEnd( const Frustum& frustum,
                      const ClipPlanes& planes,
                      const PixelViewport& view,
                      const NodeIds& orderedBricks ) override;

    struct Impl;
    std::unique_ptr< Impl > _impl;
};

}

#endif // _OSprayRenderer_h_
