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

CacheLoadException::CacheLoadException( const Identifier& id,
                                        const std::string& message )
    : _id( id )
    , _message( message )
{}

const char* CacheLoadException::what() const throw()
{
    std::stringstream message;
    message << "Id: " << _id << " " << _message << std::endl;
    return message.str().c_str();
}


struct LRUCachePolicy
{
    typedef std::deque< CacheId > LRUQueue;

    LRUCachePolicy( const size_t maxMemBytes )
        : _maxMemBytes( maxMemBytes )
        , _cleanUpRatio( 1.0f )
    {}

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

template< class CacheObjectT, class Allocator >
class Cache< CacheObjectT, Allocator >::Impl final
{
    typedef std::shared_ptr< const CacheObjectT > ConstCacheObjectPtrT;

    struct InternalCacheObject
    {
        InternalCacheObject( Allocator& allocator )
            : _allocator( allocator )
        {}

        template< class... Args >
        ConstCacheObjectPtrT& load( const CacheId& id, Args&&... args )
        {
            WriteLock lock( _mutex );
            if( _obj )
                return _obj;
            _obj.reset( new( id, _allocator, std::forward( args )));
            return _obj;
        }

        bool unload()
        {
            WriteLock lock( _mutex );
            _obj.reset();
        }

        ConstCacheObjectPtrT tryGet() const
        {
            ReadLock lock( _mutex, boost::try_to_lock );
            if( !lock.owns_lock()) // Still loading the object
                return ConstCacheObjectPtrT();
            return _obj;
        }

        const ConstCacheObjectPtrT& get() const
        {
            ReadLock lock( _mutex );
            return _obj;
        }

        mutable ReadWriteMutex _mutex;
        ConstCacheObjectPtrT _obj;
        Allocator _allocator;
        typename Allocator::pointer _ptr;
    };

    typedef std::unordered_map< CacheId, InternalCacheObject > ConstCacheMap;

    Impl( Cache< CacheObjectT, Allocator >& cache,
          const std::string& name,
          const size_t maxMemBytes,
          const Allocator& allocator )
        : _policy( maxMemBytes )
        , _cache( cache )
        , _statistics( name, maxMemBytes )
        , _cacheMap( 128 )
        , _allocator( allocator )
    {}

    ~Impl()
    {}

    void applyPolicy()
    {
        if( _cacheMap.empty() || !_policy.isFull( _cache ))
            return;

        // Objects are returned in delete order
        for( const CacheId& cacheId: _policy.getObjects( ))
        {
            unloadFromCache( cacheId );
            if( _policy.hasSpace( _cache ))
                return;
        }
    }

    template< class... Args >
    ConstCacheObjectPtrT load( const CacheId& cacheId, Allocator& allocator, Args&& ... args )
    {

        {   // If object is in cache, wait it is loading
            ReadLock writeLock( _mutex );
            const CacheId& cacheId = obj->getId();
            ConstCacheMap::const_iterator it = _cacheMap.find( cacheId );
            if( it != _cacheMap.end( ))
                return it->second.get(); // Blocks until loaded
        }

        {
            WriteLock writeLock( _mutex );
            // If object is in the cache, do the loading
            const CacheId& cacheId = obj->getId();
            ConstCacheMap::const_iterator it = _cacheMap.find( cacheId );
            if( it != _cacheMap.end( ))
            {
                // If object is not cache, create the object postpone the construction
                // So the readers are not blocked with write lock ( i.e. concurrent
                // load, get method works )

                InternalCacheObject intCacheObject( allocator );
                _executableMap.emplace( std::piecewise_construct,
                                        std::forward_as_tuple( cacheId ),
                                        std::forward_as_tuple( std::move( intCacheObject )));
            }
        }

        {   // If object is in cache, wait it is loading
            ReadLock writeLock( _mutex );
            const CacheId& cacheId = obj->getId();
            ConstCacheMap::const_iterator it = _cacheMap.find( cacheId );
            it->second.load( cacheId, args ); // This can throw exception
            _statistics.notifyMiss();
            _policy.insert( cacheId );
            return it->second.get(); // Blocks until loaded
        }
    }

    bool unload( const CacheId& cacheId )
    {
        {
            ReadLock lock( _mutex );
            ConstCacheMap::iterator it = _cacheMap.find( cacheId );
            if( it == _cacheMap.end( ))
                return false;

            ConstCacheObjectPtrT& obj = it->second.tryGet();
            if( !obj || obj.use_count() > 1 ) // Object is still being loaded or referenced
                return false;

            obj->unload();
        }

        {
            WriteLock lock();
            _policy.remove( cacheId );
            _cacheMap.erase( cacheId );
        }
        return true;
    }

    ConstCacheObjectPtrT get( const CacheId& cacheId ) const
    {
        ReadLock readLock( _mutex );
        ConstCacheMap::const_iterator it = _cacheMap.find( cacheId );
        if( it == _cacheMap.end( ))
            return ConstCacheObjectPtrT();

        return it->second.tryGet(); // Object may still be loaded
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

    friend class Cache< Allocator >;

    mutable LRUCachePolicy _policy;
    Cache< CacheObjectT, Allocator >& _cache;
    mutable CacheStatistics _statistics;
    ConstCacheMap _cacheMap;
    mutable ReadWriteMutex _mutex;
    Allocator _allocator;
};

template< class CacheObjectT, class Allocator >
Cache< CacheObjectT, Allocator >::Cache( const std::string& name,
                                         size_t maxMemBytes,
                                         const Allocator& allocator )
    : _impl( new Cache< CacheObjectT, Allocator >::Impl( *this, name, maxMemBytes, allocator ))
{}

template< class CacheObjectT, class Allocator >
Cache< CacheObjectT, Allocator >::~Cache()
{}

template< class CacheObjectT, class Allocator >
std::shared_ptr< const Cache< Allocator >::CacheObjectT >
Cache< CacheObjectT, Allocator >::get( const CacheId& cacheId ) const
{
    if( cacheId == INVALID_CACHE_ID )
        return ConstCacheObjectPtrT();

    return _impl->get( cacheId );
}

template< class CacheObjectT, class Allocator >
template< class... Args >
std::shared_ptr< CacheObjectT >
Cache< CacheObjectT, Allocator >::load( const CacheId& cacheId, Args&&... args )
{
    try
    {
        const ConstCacheObjectPtrT& cacheObject
                = _impl->load( cacheId, args... );

        if( obj->getId() == INVALID_CACHE_ID )
            return ConstCacheObjectPtrT();

        return cacheObject;
    }
    catch( const CacheLoadException& )
    {}

    return obj;
}

template< class CacheObjectT, class Allocator >
size_t Cache< CacheObjectT, Allocator >::getCount() const
{
    return _impl->getCount();
}

template< class CacheObjectT, class Allocator >
const CacheStatistics& Cache< CacheObjectT, Allocator >::getStatistics() const
{
    return _impl->_statistics;
}

template< class CacheObjectT, class Allocator >
void Cache< CacheObjectT, Allocator >::purge()
{
    _impl->purge();
}

template< class CacheObjectT, class Allocator >
void Cache< CacheObjectT, Allocator >::purge( const CacheId& cacheId )
{
    _impl->purge( cacheId );
}

