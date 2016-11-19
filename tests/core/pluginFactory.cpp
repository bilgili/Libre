
/* Copyright (c) 2013-2015, EPFL/Blue Brain Project
 *                          Raphael Dumusc <raphael.dumusc@epfl.ch>
 *                          Ahmet Bilgili <ahmetbilgili@gmail.com>
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

#define BOOST_TEST_MODULE PluginFactory

#include <livre/core/util/PluginFactory.h>
#include <livre/core/util/PluginRegisterer.h>
#include <servus/uri.h>

#include <boost/test/unit_test.hpp>
#include <boost/scoped_ptr.hpp>

#define VALID_VALUE    10
#define INVALID_VALUE   0

struct InitData
{
    servus::URI uri;
};

namespace boost
{
template<> inline std::string lexical_cast( const InitData& data )
{
    return lexical_cast< std::string >( data.uri );
}
}

class PluginInterface
{
public:
    typedef PluginInterface PluginT;
    virtual ~PluginInterface() {}
    virtual int getValue() = 0;
};

class DoubleParameterPluginInterface
{
public:
    typedef DoubleParameterPluginInterface PluginT;
    virtual ~DoubleParameterPluginInterface() {}
    virtual int getValue() = 0;
};

class MyPlugin : public PluginInterface
{
public:
    explicit MyPlugin( const servus::URI& ) {}
    static bool handles( const servus::URI& ) { return true; }
    int getValue() final { return VALID_VALUE; }
};

class MyDoubleParamPlugin : public DoubleParameterPluginInterface
{
public:
    explicit MyDoubleParamPlugin( const InitData&, const servus::URI& ) {}
    static bool handles( const InitData&, const servus::URI&  ) { return true; }
    int getValue() final { return VALID_VALUE; }
};

class MyDummyPlugin : public PluginInterface
{
public:
    explicit MyDummyPlugin( const servus::URI& ) {}
    static bool handles( const servus::URI& ) { return false; }
    int getValue() final { return INVALID_VALUE; }
};

class MyDoubleParamDummyPlugin : public DoubleParameterPluginInterface
{
public:
    explicit MyDoubleParamDummyPlugin( const InitData&, const servus::URI& ) {}
    static bool handles( const InitData&, const servus::URI& ) { return false; }
    int getValue() final { return INVALID_VALUE; }
};

typedef livre::PluginFactory< PluginInterface, const servus::URI& > MyPluginFactory;
typedef livre::PluginFactory< DoubleParameterPluginInterface,
                              const InitData&,
                              const servus::URI& > MyDoubleParamPluginFactory;

typedef std::unique_ptr< PluginInterface > PluginInterfacePtr;
typedef std::unique_ptr< DoubleParameterPluginInterface > DoubleParamPluginInterfacePtr;

void tryCreatePlugin( PluginInterfacePtr& plugin )
{
    MyPluginFactory& factory = MyPluginFactory::getInstance();
    plugin.reset( factory.create( servus::URI( "XYZ" )));
}

void tryCreateTypedPlugin( DoubleParamPluginInterfacePtr& plugin )
{
    MyDoubleParamPluginFactory& factory = MyDoubleParamPluginFactory::getInstance();
    plugin.reset( factory.create( InitData(), servus::URI( "XYZ" )));
}

BOOST_AUTO_TEST_CASE( testWhenNoPluginIsRegisteredCreateThrowsRuntimeError )
{
    MyPluginFactory::getInstance().deregisterAll();

    PluginInterfacePtr plugin;
    BOOST_CHECK_THROW( tryCreatePlugin( plugin ), std::runtime_error );
}

BOOST_AUTO_TEST_CASE( testWhenNoTypedPluginIsRegisteredCreateThrowsRuntimeErr )
{
    MyDoubleParamPluginFactory::getInstance().deregisterAll();

    DoubleParamPluginInterfacePtr plugin;
    BOOST_CHECK_THROW( tryCreateTypedPlugin( plugin ), std::runtime_error );
}


BOOST_AUTO_TEST_CASE( testWhenPluginRegistererIsInstantiatedPluginIsRegistered )
{
    MyPluginFactory::getInstance().deregisterAll();

    livre::PluginRegisterer< MyPlugin, const servus::URI& > registerer;

    PluginInterfacePtr plugin;
    BOOST_REQUIRE_NO_THROW( tryCreatePlugin( plugin ));
    BOOST_CHECK_EQUAL( plugin->getValue(), VALID_VALUE );
}

BOOST_AUTO_TEST_CASE(
                testWhenTypedPluginRegistererIsInstantiatedPluginIsRegistered )
{
    MyDoubleParamPluginFactory::getInstance().deregisterAll();

    livre::PluginRegisterer< MyDoubleParamPlugin, const InitData&, const servus::URI& > registerer;

    DoubleParamPluginInterfacePtr plugin;
    BOOST_REQUIRE_NO_THROW( tryCreateTypedPlugin( plugin ));
    BOOST_CHECK_EQUAL( plugin->getValue(), VALID_VALUE );
}

BOOST_AUTO_TEST_CASE( testWhenPluginsDontHandleURICreateThrowsRuntimeError )
{
    MyPluginFactory::getInstance().deregisterAll();

    livre::PluginRegisterer< MyDummyPlugin, const servus::URI& > registerer;

    PluginInterfacePtr plugin;
    BOOST_CHECK_THROW( tryCreatePlugin( plugin ), std::runtime_error );
}

BOOST_AUTO_TEST_CASE( testWhenTypedPlginsDontHandleURICreateThrowsRuntimeError )
{
    MyDoubleParamPluginFactory::getInstance().deregisterAll();

    livre::PluginRegisterer< MyDoubleParamDummyPlugin,
                             const InitData&,
                             const servus::URI& > registerer;

    DoubleParamPluginInterfacePtr plugin;
    BOOST_CHECK_THROW( tryCreateTypedPlugin( plugin ), std::runtime_error );
}

BOOST_AUTO_TEST_CASE( testWhenOnePluginHandlesURICreateInstanciesCorrectType )
{
    MyPluginFactory::getInstance().deregisterAll();

    livre::PluginRegisterer< MyDummyPlugin, const servus::URI& > registerer1;
    livre::PluginRegisterer< MyPlugin, const servus::URI& > registerer2;

    PluginInterfacePtr plugin;
    BOOST_REQUIRE_NO_THROW( tryCreatePlugin( plugin ));
    BOOST_CHECK_EQUAL( plugin->getValue(), VALID_VALUE );
}

BOOST_AUTO_TEST_CASE( testWhenOneTypedPluginHandlesURICreateInstCorrectType )
{
    MyDoubleParamPluginFactory::getInstance().deregisterAll();

    livre::PluginRegisterer< MyDoubleParamDummyPlugin,
                             const InitData&,
                             const servus::URI& > registerer1;
    livre::PluginRegisterer< MyDoubleParamPlugin,
                             const InitData&,
                             const servus::URI& > registerer2;

    DoubleParamPluginInterfacePtr plugin;
    BOOST_REQUIRE_NO_THROW( tryCreateTypedPlugin( plugin ));
    BOOST_CHECK_EQUAL( plugin->getValue(), VALID_VALUE );
}
