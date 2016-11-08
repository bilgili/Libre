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

#ifndef _Cuda_ClipPlanes_h_
#define _Cuda_ClipPlanes_h_

#include "cuda.h"

#include <lexis/render/clipPlanes.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <memory>

namespace livre
{
namespace cuda
{

/** CUDA representation of the clip planes */
class ClipPlanes
{
public:

    /** Constructor */
    ClipPlanes();
    ~ClipPlanes();

    /**
     * Uploads the clip planes to cuda structures
     * @param clipPlanes is the clip planes
     */
    void upload( const lexis::render::ClipPlanes& clipPlanes );

    /** @return the clip planes as cuda variables */
    CUDA_CALL const float4* getClipPlanes() const { return _clipPlanes; }

    /** @return the number of clip planes */
    CUDA_CALL unsigned int getNPlanes() const { return _nPlanes; };

private:

    float4 _clipPlanes[ 6 ];
    unsigned int _nPlanes;
};
}
}
#endif // _Cuda_ColorMap_h_
