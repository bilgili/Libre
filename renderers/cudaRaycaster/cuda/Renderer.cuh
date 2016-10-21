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

#ifndef _Cuda_Renderer_h
#define _Cuda_Renderer_h

#include <livre/core/mathTypes.h>
#include <lexis/render/ColorMap.h>
#include <lexis/render/clipPlanes.h>

#include <vector>

namespace livre
{
namespace cuda
{
class TexturePool;

struct NodeData
{
    Vector3f textureMin;
    Vector3f textureSize;
    Vector3f aabbMin;
    Vector3f aabbSize;
};

typedef std::vector< NodeData > NodeDatas;

struct ViewData
{
    const Vector3f eyePosition;
    const Vector4ui glViewport;
    const Matrix4f invProjMatrix;
    const Matrix4f modelViewMatrix;
    const Matrix4f invViewMatrix;
    const Vector3f aabbMin;
    const Vector3f aabbMax;
    const float nearPlane;
};

struct RenderData
{
    const unsigned int samplesPerRay;
    const unsigned int samplesPerPixel;
    const unsigned int maxSamplesPerRay;
    const unsigned int datatype;
    const Vector2f dataSourceRange;
};

class Renderer
{
public:
    Renderer();
    ~Renderer();

    void update( const lexis::render::ClipPlanes& clipPlanes,
                 const lexis::render::ColorMap& colorMap );

    void preRender( const ViewData& viewData );
    void render( const ViewData& viewData,
                 const NodeDatas& nodeData,
                 const RenderData& renderData,
                 const cuda::TexturePool& texturePool );
    void postRender();

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
}
}
#endif // _Cuda_Renderer_h

