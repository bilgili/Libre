
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

#ifndef _Plugin_h_
#define _Plugin_h_

#include <servus/uint128_t.h> // member
#include <boost/function.hpp> // Plugin functions
#include <boost/function_equal.hpp> // operator ==

namespace livre
{

template< class T, class... Args >
class PluginFactory;

/** Manages a class deriving from a PluginT interface. */
template< class PluginT, class... Args > class Plugin
{
public:
    /** The constructor method / concrete factory for Plugin objects */
    typedef boost::function< PluginT* ( Args... ) > Constructor;

    /** The method to check if the plugin can handle a given initData */
    typedef boost::function< bool ( Args... ) > HandlesFunc;

    /**
     * Construct a new Plugin.
     * @param constructor_ The constructor method for Plugin objects.
     * @param handles_ The method to check if the plugin can handle the
     * initData.
     */
    Plugin( const Constructor& constructor_, const HandlesFunc& handles_ )
        : constructor( constructor_ ), handles( handles_ )
        , tag( servus::make_UUID( )) {}

    /** @return true if the plugins wrap the same plugin. @version 1.11.0 */
    bool operator == ( const Plugin& rhs ) const
        { return tag == rhs.tag; }

    /** @return false if the plugins do wrap the same plugin. @version 1.11.0 */
    bool operator != ( const Plugin& rhs ) const { return !(*this == rhs); }

private:
    friend class PluginFactory< PluginT, Args... >;
    Constructor constructor;
    HandlesFunc handles;

    // Makes Plugin comparable. See http://stackoverflow.com/questions/18665515
    servus::uint128_t tag;
};

}

#endif //_Plugin_h_
