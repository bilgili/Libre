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

#ifndef _CudaIrradianceCompute_h_
#define _CudaIrradianceCompute_h_

#include "types.h"
#include <livre/lib/types.h>
#include <livre/core/types.h>
#include <livre/core/mathTypes.h>

#include <lexis/render/ColorMap.h>

namespace livre
{
namespace cuda
{
class IrradianceCompute;
}

/** Manages the texture pool allocation, data copies and deallocations */
class CudaIrradianceCompute
{

public:

     /**
     * Constructor
     * @param dataSource the data source
     * @param dataCache the data cache
     */
    CudaIrradianceCompute( DataSource& dataSource, DataCache& dataCache );
    ~CudaIrradianceCompute();

    /**
     * Updates the irradiance texture for a given frame id. If the texture is already computed
     * for the given timeStep id, it returns.
     * @param renderInputs inputs for the rendering
     */
    bool update( const RenderInputs& renderInputs );

    /** @return the cuda compute */
    ::livre::cuda::IrradianceCompute& getCudaCompute() const;

private:

    struct Impl;
    std::unique_ptr<Impl> _impl;
};
}

#endif // _CudaIrradianceCompute_h_

