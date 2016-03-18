#
#
#       Ecole Polytechnique Federale de Lausanne
#       Brain Mind Institute,
#       Blue Brain Project
#       (c) 2005-2015. All rights reserved.
#
#       Main Authors: Ahmet Bilgili
#
#
<<<<<<< e0f41d0fe59e7cb190c9530a560345d42fffb837
=======

set(LIVREEQ_LINK_LIBRARIES PUBLIC Lexis LivreLib Equalizer)

if(OSPRAY_FOUND)
  # compiler selection for the non-MIC part
  if(APPLE)
    set(OSPRAY_COMPILER "CLANG" CACHE STRING "compiler to use (ICC,GCC, or CLANG)")
    set_property(CACHE OSPRAY_COMPILER PROPERTY STRINGS ICC GCC CLANG)
  else(APPLE)
    set(OSPRAY_COMPILER "GCC" CACHE STRING "compiler to use (ICC or GCC)")
    set_property(CACHE OSPRAY_COMPILER PROPERTY STRINGS ICC GCC)
  endif()
  # the arch we're targeting for the non-MIC/non-xeon phi part of ospray
  set(BRAYNS_BUILD_ISA "ALL" CACHE STRING "Target ISA (SSE,AVX,AVX2,or ALL)")
  set_property(CACHE BRAYNS_BUILD_ISA PROPERTY STRINGS ALL SSE AVX AVX2)
  list(APPEND LIVREEQ_LINK_LIBRARIES ${OSPRAY_LIBRARIES})
endif()

include(StringifyShaders)
include(Files.cmake)
>>>>>>> Ospray

set(LIVREEQ_HEADERS
      Channel.h
      Client.h
      Config.h
      ConnectionDefinitions.h
      Error.h
      events/EqEventHandlerFactory.h
      events/EqEventHandlerFactory.h
      events/EqEventInfo.h
      events/Events.h
      events/handlers/ChannelPointerHandler.h
      events/handlers/KeyboardHandler.h
      FrameData.h
      FrameGrabber.h
      Node.h
      Pipe.h
      render/EqContext.h
      render/RayCastRenderer.h
      settings/CameraSettings.h
      settings/FrameSettings.h
      settings/RenderSettings.h
      settings/VolumeSettings.h
      Window.h
      coTypes.h
      types.h)

set( LIVREEQ_SOURCES
      Channel.cpp
      Client.cpp
      Config.cpp
      Error.cpp
      events/EqEventHandler.cpp
      events/EqEventHandlerFactory.cpp
      events/handlers/ChannelPointerHandler.cpp
      events/handlers/KeyboardHandler.cpp
      FrameData.cpp
      FrameGrabber.cpp
      Node.cpp
      Pipe.cpp
      render/EqContext.cpp
      settings/CameraSettings.cpp
      settings/FrameSettings.cpp
      settings/RenderSettings.cpp
      settings/VolumeSettings.cpp
      Window.cpp
      )

include(StringifyShaders)
set(LIVREEQ_SHADERS render/shaders/vertRayCast.glsl
                    render/shaders/vertTexCopy.glsl
                    render/shaders/fragTexCopy.glsl )

if(APPLE)
    list(APPEND LIVREEQ_SHADERS render/shaders/fragRayCastGL2.glsl)
    list(APPEND LIVREEQ_SOURCES render/RayCastRendererGL2.cpp)
else()
    list(APPEND LIVREEQ_SHADERS render/shaders/fragRayCast.glsl)
    list(APPEND LIVREEQ_SOURCES render/RayCastRenderer.cpp)
endif()

stringify_shaders(${LIVREEQ_SHADERS})
list(APPEND LIVREEQ_SOURCES ${SHADER_SOURCES})
include_directories(${PROJECT_BINARY_DIR})

if(TARGET ZeroEQ)
  list(APPEND LIVREEQ_HEADERS zeroeq/communicator.h)
  list(APPEND LIVREEQ_SOURCES zeroeq/communicator.cpp)
  list(APPEND LIVREEQ_LINK_LIBRARIES PRIVATE ZeroEQ)
endif()

if(LibJpegTurbo_FOUND)
  list(APPEND LIVREEQ_LINK_LIBRARIES PRIVATE ${LibJpegTurbo_LIBRARIES})
endif()


set(LIVREEQ_INCLUDE_NAME livre/eq)
set(LIVREEQ_NAMESPACE livreeq)
common_library(LivreEq)
