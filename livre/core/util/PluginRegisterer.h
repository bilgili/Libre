
/* Copyright (c) 2013-2015, EPFL/Blue Brain Project
 *                          Raphael Dumusc <raphael.dumusc@epfl.ch>
 *                          Stefan.Eilemann@epfl.ch
 *                          ahmetbilgili@gmail.com
 *
 * This file is part of Lunchbox <https://github.com/Eyescale/Lunchbox>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 2.1 as published
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

#ifndef _PluginRegisterer_h_
#define _PluginRegisterer_h_

#include "Plugin.h" // used inline
#include "PluginFactory.h" // used inline

#include <boost/bind.hpp> // used inline
#include <boost/version.hpp>
#include <boost/functional/factory.hpp>

namespace livre
{

template< class Impl, class... Args >
class PluginRegisterer
{
public:

    /** Construct a registerer and register the Impl class. */
    PluginRegisterer()
    {
        Plugin< typename Impl::PluginT, Args... > plugin(
            []( Args&&... args )
            {
                return boost::factory< Impl* >()( std::forward< Args >( args )... );
            },
            []( Args&&... args )
            {
                return Impl::handles( std::forward< Args >( args )... );
            });
        PluginFactory< typename Impl::PluginT, Args... >::getInstance().
            register_( plugin );
    }
};
}

#endif
