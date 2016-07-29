/* Copyright (c) 2011-2016, EPFL/Blue Brain Project
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

#ifndef _OSPrayVolume_h_
#define _OSPrayVolume_h_

#include <livre/core/types.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcomment"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <ospray/volume/Volume.h>
#pragma GCC diagnostic pop

#include <livre/lib/types.h>

namespace livre
{

class OSPrayVolume : public ospray::Volume
{
public:

    //! Constructor.
    OSPrayVolume();

    //! Destructor.
    ~OSPrayVolume();

    //! A string description of this class.
    std::string toString() const final;

    //! Allocate storage and populate the volume, called through the OSPRay API.
    void commit() final;

    void setCurrentRenderNodes( const NodeIds& ordered );

    /**
     * @param renderBricks the list of render bricks
     * @param x the x world position
     * @param y the y world position
     * @param z the z world position
     * @return value at sample
     */
    float sample( const uint64_t* nodeIds,
                  const size_t count,
                  const float x,
                  const float y,
                  const float z  ) const;

private:

    void computeSamples( float** results,
                         const ospray::vec3f* worldCoordinates,
                         const size_t& count ) final;

    int setRegion( const void* source,
                   const ospray::vec3i& index,
                   const ospray::vec3i& count ) final;

    void createEquivalentISPC();

    const DataCache* _dataCache;
    const DataSource* _dataSource;
    const VolumeInformation* _volInfo;
};

}
namespace
{
extern "C" float sampleLivreData( void* osprayVolume,
                                  uint64_t* nodeIds,
                                  uint64_t count,
                                  float x,
                                  float y,
                                  float z );
}

#endif // _OSPrayVolume_h_
