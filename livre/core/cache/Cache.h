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

#ifndef _Cache_h_
#define _Cache_h_

#include <livre/core/api.h>
#include <livre/core/types.h>
#include <livre/core/cache/CacheObject.h>
#include <livre/core/cache/CacheStatistics.h>
#include <deque>

#include <lunchbox/debug.h>

namespace livre
{

/**
 * The Cache class manages the \see CacheObjects according to LRU Policy, methods
 * are thread safe inserting/querying nodes. The type safety check is done in runtime.
 */
template< class CacheObjectT >
class Cache
{
public:

    /**
     * Constructor
     * @param name is the name of the cache.
     * @param maxMemBytes maximum memory.
     */
    LIVRECORE_API Cache( const std::string& name, size_t maxMemBytes );
    LIVRECORE_API ~Cache();

    /**
     * Gets the cached object from the cache with a given type and d
     * @param cacheId The object cache id to be queried.
     * @return The cache object from cache, if object is not in the list an empty cache
     * object is returned.
     */
    LIVRECORE_API std::shared_ptr< const CacheObjectT > get( const CacheId& cacheId ) const;

    /**
     * Unloads the object from the memory, if there are not any references. The
     * objects are removed from cache
     * @param cacheId The object cache id to be unloaded.
     * @return false if object is not unloaded or cacheId is invalid
     */
    LIVRECORE_API bool unload( const CacheId& cacheId );

    /** @return The number of cache objects managed. */
    LIVRECORE_API size_t getCount() const;

    /**
     * Loads the object to cache. If object is not in the cache it is created.
     * @param cacheId the id of the cache object to be loaded
     * @param args parameters of the cache object constructor. If there is already
     * a cache object with the same cache id, the args are not considered.
     * @return the loaded or previously loaded cache object. Return empty pointer
     * if cache id is invalid or object cannot be loaded.
     */
    template< class... Args >
    LIVRECORE_API std::shared_ptr< const CacheObjectT > load( const CacheId& cacheId,
                                                              Args&&... args );
    /** @return Statistics. */
    LIVRECORE_API const CacheStatistics& getStatistics() const;

    /**
     * Purges the cache by removing cached objects. The purged objects are not unloaded
     * and they will be in memory until no reference is left.
     */
    LIVRECORE_API void purge();

    /**
     * Purges a cached object from the cache. The purged object is not unloaded
     * and they will be in memory until no reference is left.
     * @param cacheId The object cache id to be purged.
     */
    LIVRECORE_API void purge( const CacheId& cacheId );

private:

    struct Impl;
    std::unique_ptr<Impl> _impl;
};

#include "Cache.ipp"

}

#endif // _Cache_h_
