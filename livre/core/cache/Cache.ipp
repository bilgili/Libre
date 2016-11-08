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



template< class CacheObjectT >
class Cache< CacheObjectT >::Impl final
{
    friend class Cache< CacheObjectT >;

    struct LRUCachePolicy
    {
        typedef std::deque< CacheId > LRUQueue;

        LRUCachePolicy( const size_t maxMemBytes )
            : _maxMemBytes( maxMemBytes )
            , _cleanUpRatio( 1.0f )
        {}

        bool isFull( const Cache< CacheObjectT >& cache ) const
        {
            const size_t usedMemBytes = cache.getStatistics().getUsedMemory();
            return usedMemBytes >= _maxMemBytes;
        }

        bool hasSpace( const Cache< CacheObjectT >& cache ) const
        {
            const size_t usedMemBytes = cache.getStatistics().getUsedMemory();
            return usedMemBytes < _cleanUpRatio * _maxMemBytes;
        }

        void insert( const CacheId& cacheId )
        {
            remove( cacheId );
            _lruQueue.push_back( cacheId );
        }

        void remove( const CacheId& cacheId )
        {
            typename LRUQueue::iterator it = _lruQueue.begin();
            while( it != _lruQueue.end( ))
            {
                if( *it == cacheId )
                {
                    _lruQueue.erase( it );
                    return;
                }
                else
                    ++it;
            }
        }

        CacheIds getObjects() const
        {
            CacheIds ids;
            ids.reserve( _lruQueue.size( ));
            ids.insert( ids.begin(), _lruQueue.begin(), _lruQueue.end());
            return ids;
        }

        void clear()
        {
            _lruQueue.clear();
        }

        const size_t _maxMemBytes;
        const float _cleanUpRatio;
        LRUQueue _lruQueue;
    };

    struct InternalCacheObject
    {
        InternalCacheObject()
            : _mutex( new ReadWriteMutex( ))
        {}

        InternalCacheObject( InternalCacheObject&& obj )
            : _mutex( std::move( obj._mutex ))
            , _obj( std::move( obj._obj ))
        {}

        template< class... Args >
        std::shared_ptr< const CacheObjectT > load( const CacheId& cacheId, Args&&... args )
        {
           WriteLock lock( *_mutex );
           if( _obj )
               return _obj;

           _obj.reset( new CacheObjectT( cacheId, args... ));
           return std::static_pointer_cast< const CacheObjectT >( _obj );
        }

        const std::shared_ptr< const CacheObjectT >& get() const
        {
           return _obj;
        }

        mutable std::shared_ptr< ReadWriteMutex > _mutex;
        std::shared_ptr< const CacheObjectT > _obj;
    };

    typedef std::unordered_map< CacheId, InternalCacheObject > CacheMap;

   Impl( Cache< CacheObjectT >& cache,
         const std::string& name,
         const size_t maxMemBytes )
    : _policy( maxMemBytes )
    , _cache( cache )
    , _statistics( name, maxMemBytes )
    , _cacheMap( 128 )
    {}

    void applyPolicy()
    {
        if( _cacheMap.empty() || !_policy.isFull( _cache ))
            return;

        // Objects are returned in delete order
        for( const CacheId& cacheId: _policy.getObjects( ))
        {
            unload( cacheId );
            if( _policy.hasSpace( _cache ))
                return;
        }
    }

    template< class... Args >
    std::shared_ptr< const CacheObjectT > load( const CacheId& cacheId, Args&&... args )
    {
        {   // If object is in cache, wait it is loading
            ReadLock readLock( _mutex );
            typename CacheMap::const_iterator it = _cacheMap.find( cacheId );
            if( it != _cacheMap.end() && it->second.get( ))
                return std::static_pointer_cast< const CacheObjectT >( it->second.get( ));
        }

        {
            WriteLock writeLock( _mutex );
            // If object is not in the cache, add do the cache and postpone loading to
            // get method
            typename CacheMap::const_iterator it = _cacheMap.find( cacheId );
            if( it == _cacheMap.end( ))
            {
                // If object is not cache, create the internal representation and
                // postpone the construction.
                // So the readers are not blocked with write lock ( i.e. concurrent
                // load, get method works )
                _cacheMap.emplace( std::piecewise_construct,
                                   std::forward_as_tuple( cacheId ),
                                   std::forward_as_tuple( InternalCacheObject()));
            }
            applyPolicy();
        }


            // If object is in cache, wait it is loading
        {
            ReadLock readLock( _mutex );
            typename CacheMap::iterator it = _cacheMap.find( cacheId );
            readLock.unlock();

            std::shared_ptr< const CacheObjectT > obj;
            try
            {
                // This can throw exception
                // Unlock read lock so nobody is blocked
                obj = it->second.load( cacheId, args... );
            }
            catch( const CacheLoadException& )
            {}

            WriteLock writeLock( _mutex );
            if( obj )
            {
                _statistics.notifyMiss();
                _statistics.notifyLoaded( *obj );
                _policy.insert( cacheId );
                applyPolicy();
            }
            else
                _cacheMap.erase( cacheId );
            return obj; // Blocks until loaded
        }
    }

    std::shared_ptr< const CacheObjectT > get( const CacheId& cacheId ) const
    {
        ReadLock readLock( _mutex );
        typename CacheMap::const_iterator it = _cacheMap.find( cacheId );
        if( it == _cacheMap.end( ))
           return std::shared_ptr< const CacheObjectT >();

        return it->second._obj; // Object may still be loaded
    }

    void unload( const CacheId& cacheId )
    {
        typename CacheMap::const_iterator it = _cacheMap.find( cacheId );
        if( it == _cacheMap.end( ))
            return;

        const ConstCacheObjectPtr& obj = it->second._obj; // +1 ref
        if( !obj || obj.use_count() > 2 ) // Object is still being loaded or referenced
            return;

        _policy.remove( cacheId );
        _statistics.notifyUnloaded( *obj );
        _cacheMap.erase( cacheId );
    }

    bool unloadSafe( const CacheId& cacheId )
    {
        ReadLock readLock( _mutex );
        typename CacheMap::iterator it = _cacheMap.find( cacheId );
        if( it == _cacheMap.end( ))
            return false;

        const ConstCacheObjectPtr& obj = it->second._obj; // +1 ref
        if( !obj || obj.use_count() > 2 ) // Object is still being loaded or referenced
            return false;
        readLock.unlock();

        WriteLock lock( _mutex );
        _policy.remove( cacheId );
        _statistics.notifyUnloaded( *obj );
        _cacheMap.erase( cacheId );
        return true;
    }

    size_t getCount() const
    {
        ReadLock lock( _mutex );
        return _cacheMap.size();
    }

    void purge()
    {
        WriteLock lock( _mutex );
        _statistics.clear();
        _policy.clear();
        _cacheMap.clear();
    }

    void purge( const CacheId& cacheId )
    {
        WriteLock lock( _mutex );
        _cacheMap.erase( cacheId );
    }

    mutable LRUCachePolicy _policy;
    Cache< CacheObjectT >& _cache;
    mutable CacheStatistics _statistics;
    CacheMap _cacheMap;
    mutable ReadWriteMutex _mutex;

public:
    ~Impl()
    {}
};

template< class CacheObjectT >
Cache< CacheObjectT >::Cache( const std::string& name, size_t maxMemBytes )
    : _impl( new Cache< CacheObjectT >::Impl( *this, name, maxMemBytes ))
{}

template< class CacheObjectT >
Cache< CacheObjectT >::~Cache()
{}

template< class CacheObjectT >
template< class... Args >
std::shared_ptr< const CacheObjectT >
Cache< CacheObjectT >::load( const CacheId& cacheId, Args&&... args )
{
    if( cacheId == INVALID_CACHE_ID )
        return std::shared_ptr< const CacheObjectT >();

    return _impl->load( cacheId, args... );
}

template< class CacheObjectT >
bool Cache< CacheObjectT >::unload( const CacheId& cacheId )
{
    if( cacheId == INVALID_CACHE_ID )
        return false;

    return _impl->unloadSafe( cacheId );
}

template< class CacheObjectT >
std::shared_ptr< const CacheObjectT > Cache< CacheObjectT >::get( const CacheId& cacheId ) const
{
    if( cacheId == INVALID_CACHE_ID )
        return false;

    std::shared_ptr< const CacheObjectT > obj = _impl->get( cacheId );
    if( !obj )
        return std::shared_ptr< const CacheObjectT >();

    return obj;
}

template< class CacheObjectT >
size_t Cache< CacheObjectT >::getCount() const
{
    return _impl->getCount();
}

template< class CacheObjectT >
const CacheStatistics& Cache< CacheObjectT >::getStatistics() const
{
    return _impl->_statistics;
}

template< class CacheObjectT >
void Cache< CacheObjectT >::purge()
{
    _impl->purge();
}

template< class CacheObjectT >
void Cache< CacheObjectT >::purge( const CacheId& cacheId )
{
    _impl->purge( cacheId );
}

