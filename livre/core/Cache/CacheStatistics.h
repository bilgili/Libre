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

#ifndef _CacheStatistics_h_
#define _CacheStatistics_h_

#include <livre/core/Util/ThreadClock.h>
#include <livre/core/Cache/CacheObjectObserver.h>
#include <livre/core/Cache/CacheObject.h>

namespace livre
{
/**
 * The CacheStatistics struct keeps the statistics of the \see Cache.
 */
class CacheStatistics : public CacheObjectObserver
{
public:

    /**
     * @return Total number of objects in the corresponding \see Cache.
     */
    uint32_t getBlockCount( ) const { return totalBlockCount_; }

    /**
     * @return Total memory used by the \see Cache.
     */
    uint32_t getUsedMemory( ) const { return totalMemoryUsed_; }

    /**
     * @param statisticsName The name of the statistics.
     */
    void setStatisticsName( const std::string& statisticsName ) { statisticsName_ = statisticsName; }

    /**
     * @param stream Output stream.
     * @param cacheStatistics Input \see CacheStatistics
     * @return The output stream.
     */
    friend std::ostream& operator<<( std::ostream& stream, const CacheStatistics& cacheStatistics );

    virtual ~CacheStatistics();

private:

    friend class Cache;

    CacheStatistics( const std::string& statisticsName = "Cache Statistics",
                     const uint32_t queueSize = 1000000 );

    virtual void onLoaded_( const CacheObject& cacheObject );
    virtual void onPreUnload_( const CacheObject& cacheObject );

    virtual void onCacheMiss_( const CacheObject& cacheObject LB_UNUSED ) { ++cacheMiss_; }
    virtual void onCacheHit_( const CacheObject& cacheObject LB_UNUSED ) { ++cacheHit_; }

    lunchbox::Atomic< uint32_t > totalBlockCount_;
    lunchbox::Atomic< uint32_t > totalMemoryUsed_;

    std::string statisticsName_;

    lunchbox::Atomic< uint32_t > cacheHit_;
    lunchbox::Atomic< uint32_t > cacheMiss_;

    struct LoadInfo;
    typedef boost::shared_ptr< LoadInfo > LoadInfoPtr;
    typedef lunchbox::MTQueue< LoadInfoPtr > LoadInfoPtrQueue;

    LoadInfoPtrQueue ioQueue_;
    const uint32_t queueSize_;
};

}

#endif // _CacheStatistics_h_