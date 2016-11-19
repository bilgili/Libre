
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

template< class PluginT, class... Args >
PluginFactory< PluginT, Args... >&
PluginFactory< PluginT, Args... >::getInstance()
{
    static PluginFactory< PluginT, Args... > factory;
    return factory;
}

template< class PluginT, class... Args >
PluginFactory< PluginT, Args... >::~PluginFactory()
{
    // Do not do this: dtor is called in atexit(), at which point the other DSOs
    // might be unloaded already, causing dlclose to trip. It's pointless
    // anyways, we're in atexit, so the OS will dispose the DSOs for us anyways.
    // Let's call this a static deinitializer fiasco.
    //   deregisterAll(); // unload the DSO libraries
}

template< typename PluginT, class... Args >
PluginT* PluginFactory< PluginT, Args... >::create( Args&&... initData )
{
    for( PluginHolder& plugin: _plugins )
        if( plugin.handles( std::forward< Args >( initData )... ))
            return plugin.constructor( std::forward< Args >( initData )... );

    LBTHROW( std::runtime_error( "No plugin implementation available" ));
}

template< class PluginT, class... Args >
void PluginFactory< PluginT, Args... >::register_(
    const Plugin< PluginT, Args... >& plugin )
{
    _plugins.push_back( plugin );
}

template< class PluginT, class... Args >
bool PluginFactory< PluginT, Args... >::deregister(
    const Plugin< PluginT, Args... >& plugin )
{
    typename Plugins::iterator i =
        std::find( _plugins.begin(), _plugins.end(), plugin );
    if( i == _plugins.end( ))
        return false;

    _plugins.erase( i );
    return true;
}

template< class PluginT, class... Args >
void PluginFactory< PluginT, Args... >::deregisterAll()
{
    _plugins.clear();
    for( typename PluginMap::value_type& plugin: _libraries )
        delete plugin.first;
    _libraries.clear();
}

template< class PluginT, class... Args >
lunchbox::DSOs PluginFactory< PluginT, Args... >::load( const int version,
                                                        const Strings& paths,
                                                        const std::string& pattern )
{
    Strings unique = paths;
    lunchbox::usort( unique );

    lunchbox::DSOs result;
    for( const std::string& path: unique )
        _load( result, version, path, pattern );
    return result;
}

template< class PluginT, class... Args >
lunchbox::DSOs PluginFactory< PluginT, Args... >::load( const int version,
                                                        const std::string& path,
                                                        const std::string& pattern )
{
    lunchbox::DSOs loaded;
    _load( loaded, version, path, pattern );
    return loaded;
}

template< class PluginT, class... Args >
void PluginFactory< PluginT, Args... >::_load( lunchbox::DSOs& result,
                                               const int version,
                                               const std::string& path,
                                               const std::string& pattern )
{
#ifdef _MSC_VER
    const std::string regex( pattern + ".dll" );
#elif __APPLE__
    const std::string regex( "lib" + pattern + ".dylib" );
#else
    const std::string regex( "lib" + pattern + ".so" );
#endif
    const Strings& libs = lunchbox::searchDirectory( path, regex );

    for( const std::string& lib: libs )
    {
        lunchbox::DSO* dso = new lunchbox::DSO( path + "/" + lib );
        if( !dso->isOpen())
        {
            delete dso;
            continue;
        }

        typedef int( *GetVersion_t )();
        typedef bool( *Register_t )();

        GetVersion_t getVersion = dso->getFunctionPointer< GetVersion_t >(
            "LunchboxPluginGetVersion" );
        Register_t registerFunc = dso->getFunctionPointer< Register_t >(
            "LunchboxPluginRegister" );
        const bool matchesVersion = getVersion && (getVersion() == version);

        if( !getVersion || !registerFunc || !matchesVersion )
        {
            LBERROR << "Disable " << lib << ": "
                    << ( getVersion ? "" :
                        "Symbol for LunchboxPluginGetVersion missing " )
                    << ( registerFunc ? "" :
                        "Symbol for LunchboxPluginRegister missing " );
            if( getVersion && !matchesVersion )
                LBERROR << "Plugin version " << getVersion() << " does not"
                        << " match application version " << version;
            LBERROR << std::endl;

            delete dso;
            continue;
        }

        if( registerFunc( ))
        {
            _libraries.insert( std::make_pair( dso, _plugins.back( )));
            result.push_back( dso );
            LBINFO << "Enabled plugin " << lib << std::endl;
        }
        else
            delete dso;
    }
}

template< class PluginT, class... Args >
bool PluginFactory< PluginT, Args... >::unload( lunchbox::DSO* dso )
{
    typename PluginMap::iterator i = _libraries.find( dso );
    if( i == _libraries.end( ))
        return false;

    delete i->first;
    const bool ret = deregister( i->second );
    _libraries.erase( i );
    return ret;
}
