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


#ifndef _CudaTypes_h_
#define _CudaTypes_h_

#include <livre/core/types.h>
#include <memory>

namespace livre
{
namespace cuda
{

class IrradianceCompute;
class IrradianceTexture;
class LightSource;
class PointLightSource;
class DirectLightSource;
class TexturePool;

typedef std::vector< LightSource* > LightSources;
}
}
#endif // _CudaRaycaster_Types_h_

