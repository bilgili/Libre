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

#ifndef _ComputeFullLODSet_h_
#define _ComputeFullLODSet_h_

#include <livre/core/Render/RenderingSetGenerator.h>

#include <livre/Lib/types.h>

namespace livre
{

/**
 * The FullSetGenerator class generates the complete rendering set for a given screen space error.
 */
class FullSetGenerator : public RenderingSetGenerator
{
public:

    /**
     * @param textureCache The \see TextureCache
     * @param lodTree LODTree.
     * @param screenSpaceError Screen space error.
     */
    FullSetGenerator( DashTreePtr tree,
                      const uint32_t windowHeight,
                      const float screenSpaceError );

    /**
     * Generates the rendering set according to the given frustum.
     * @param viewFrustum parameter is frustum to query HVD
     * @param renderNodeSet The list of nodes to be rendered.
     */
    void generateRenderingSet( const GLWidget& widget,
                               const Frustum& viewFrustum,
                               NodeIdSet& requestedNodeSet,
                               NodeIdSet& renderNodeSet ) override;

protected:

    void _generateRenderingSet( const GLWidget& widget,
                                const Frustum& viewFrustum,
                                NodeIdSet& requestedNodeSet );

    const TextureCachePtr _textureCachePtr;
    const float _screenSpaceError;
};


}
#endif // _ComputeFullLODSet_h_
