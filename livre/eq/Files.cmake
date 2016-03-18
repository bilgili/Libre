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
  types.h
  )

set(LIVREEQ_SOURCES
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

if(APPLE)
    list(APPEND LIVREEQ_SOURCES render/RayCastRendererGL2.cpp)
else()
    list(APPEND LIVREEQ_SOURCES render/RayCastRenderer.cpp)
endif()

if(OSPRAY_FOUND)
    include(ispc)
    CONFIGURE_ISPC()
    INCLUDE_DIRECTORIES_ISPC(${OSPRAY_INCLUDE_DIRS})
    set(OSPRAY_EMBREE_SOURCE_DIR ${OSPRAY_ROOT}/include/ospray/embree)
    OSPRAY_ISPC_COMPILE(render/ospray/OSPrayVolume.ispc)
    list(APPEND LIVREEQ_HEADERS render/OSPrayRenderer.h render/ospray/OSPrayVolume.h)
    list(APPEND LIVREEQ_SOURCES render/OSPrayRenderer.cpp render/ospray/OSPrayVolume.cpp ${ISPC_OBJECTS})
endif()

