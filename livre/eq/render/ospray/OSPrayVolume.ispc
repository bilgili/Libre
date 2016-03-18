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

// ======================================================================== //
// Copyright 2009-2016 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include <livre/eq/render/ospray/OSPrayVolume.ih>
#include <ospray/volume/Volume.ih>

extern "C" uniform float sampleLivreData( void* uniform cppEquivalent,
                                          uniform unsigned int64 *uniform nodeIds,
                                          uniform unsigned int64 count,
                                          uniform float x,
                                          uniform float y,
                                          uniform float z );

export void* uniform constructISPCVolume( void *uniform cppEquivalent )
{

    OSPrayISPCVolume *uniform volume = uniform new uniform OSPrayISPCVolume;
    volume->nodeIds = 0;
    volume->count = 0;
    Volume_Constructor( volume->super, cppEquivalent );
    return volume;
}

export void setListOfNodes( void *uniform self,
                            unsigned int64* uniform nodeIds,
                            unsigned int64 uniform count )
{
    uniform OSPrayISPCVolume* uniform volume = (uniform OSPrayISPCVolume* uniform)self;
    volume->nodeIds = nodeIds;
    volume->count = count;
}

export void computeSamples( void *uniform self,
                            uniform float* uniform results,
                            const uniform vec3f* uniform worldCoordinates,
                            const uniform size_t count)
{
    uniform OSPrayISPCVolume* uniform volume = (uniform OSPrayISPCVolume* uniform)self;
    foreach( i = 0 ... count )
    {
        float sample;
        float x; float y; float z;
        x =  worldCoordinates[ i ].x;
        y =  worldCoordinates[ i ].y;
        z =  worldCoordinates[ i ].z;
        foreach_active( lane )
        {
             insert( sample, lane, sampleLivreData( volume->cppEquivalent,
                                                    volume->nodeIds,
                                                    volume->count,
                                                    extract( x, lane ),
                                                    extract( y, lane ),
                                                    extract( z, lane )));
        }
        results[ i ] = sample;
    }
}
