## ======================================================================== ##
## Copyright 2009-2016 Intel Corporation                                    ##
##                                                                          ##
## Licensed under the Apache License, Version 2.0 (the "License");          ##
## you may not use this file except in compliance with the License.         ##
## You may obtain a copy of the License at                                  ##
##                                                                          ##
##     http://www.apache.org/licenses/LICENSE-2.0                           ##
##                                                                          ##
## Unless required by applicable law or agreed to in writing, software      ##
## distributed under the License is distributed on an "AS IS" BASIS,        ##
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. ##
## See the License for the specific language governing permissions and      ##
## limitations under the License.                                           ##
## ======================================================================== ##

# ISPC versions to look for, in decending order (newest first)
IF(WIN32)
  SET(ISPC_VERSION_WORKING "1.9.0" "1.8.2")
ELSE()
  SET(ISPC_VERSION_WORKING "1.9.0" "1.8.2" "1.8.1")
ENDIF()
LIST(GET ISPC_VERSION_WORKING -1 ISPC_VERSION_REQUIRED)
SET(ISPC_VERSION_RECOMMENDED_KNC "1.8.1")

IF (NOT ISPC_EXECUTABLE)
  # try sibling folder as hint for path of ISPC
  IF (APPLE)
    SET(ISPC_DIR_SUFFIX "osx")
  ELSEIF(WIN32)
    SET(ISPC_DIR_SUFFIX "windows")
  ELSE()
    SET(ISPC_DIR_SUFFIX "linux")
  ENDIF()
  FOREACH(ver ${ISPC_VERSION_WORKING})
    LIST(APPEND ISPC_DIR_HINT ${PROJECT_SOURCE_DIR}/../ispc-v${ver}-${ISPC_DIR_SUFFIX})
  ENDFOREACH()

  FIND_PROGRAM(ISPC_EXECUTABLE ispc PATHS ${ISPC_DIR_HINT} DOC "Path to the ISPC executable.")
  IF (NOT ISPC_EXECUTABLE)
    MESSAGE("********************************************")
    MESSAGE("Could not find ISPC (looked in PATH and ${ISPC_DIR_HINT})")
    MESSAGE("")
    MESSAGE("This version of OSPRay expects you to have a binary install of ISPC minimum version ${ISPC_VERSION_REQUIRED}, and expects it to be found in 'PATH' or in the sibling directory to where the OSPRay source are located. Please go to https://ispc.github.io/downloads.html, select the binary release for your particular platform, and unpack it to ${PROJECT_SOURCE_DIR}/../")
    MESSAGE("")
    MESSAGE("If you insist on using your own custom install of ISPC, please make sure that the 'ISPC_EXECUTABLE' variable is properly set in CMake.")
    MESSAGE("********************************************")
    MESSAGE(FATAL_ERROR "Could not find ISPC. Exiting.")
  ELSE()
    MESSAGE(STATUS "Found Intel SPMD Compiler (ISPC): ${ISPC_EXECUTABLE}")
  ENDIF()
ENDIF()

IF(NOT ISPC_VERSION)
  EXECUTE_PROCESS(COMMAND ${ISPC_EXECUTABLE} --version OUTPUT_VARIABLE ISPC_OUTPUT)
  STRING(REGEX MATCH " ([0-9]+[.][0-9]+[.][0-9]+)(dev|knl)? " DUMMY "${ISPC_OUTPUT}")
  SET(ISPC_VERSION ${CMAKE_MATCH_1})

  IF (ISPC_VERSION VERSION_LESS ISPC_VERSION_REQUIRED)
    MESSAGE(FATAL_ERROR "Need at least version ${ISPC_VERSION_REQUIRED} of Intel SPMD Compiler (ISPC).")
  ENDIF()

  SET(ISPC_VERSION ${ISPC_VERSION} CACHE STRING "ISPC Version")
  MARK_AS_ADVANCED(ISPC_VERSION)
  MARK_AS_ADVANCED(ISPC_EXECUTABLE)
ENDIF()

# warn about recommended ISPC version on KNC
IF (OSPRAY_MIC AND NOT ISPC_VERSION VERSION_EQUAL ISPC_VERSION_RECOMMENDED_KNC
    AND NOT OSPRAY_WARNED_KNC_ISPC_VERSION)
  MESSAGE("Warning: use of ISPC v${ISPC_VERSION_RECOMMENDED_KNC} is recommended on KNC.")
  SET(OSPRAY_WARNED_KNC_ISPC_VERSION ON CACHE INTERNAL "Warned about recommended ISPC version with KNC.")
ENDIF()

GET_FILENAME_COMPONENT(ISPC_DIR ${ISPC_EXECUTABLE} PATH)



# ##################################################################
# add macro INCLUDE_DIRECTORIES_ISPC() that allows to specify search
# paths for ISPC sources
# ##################################################################
SET(ISPC_INCLUDE_DIR "")
MACRO (INCLUDE_DIRECTORIES_ISPC)
  SET(ISPC_INCLUDE_DIR ${ISPC_INCLUDE_DIR} ${ARGN})
ENDMACRO ()

MACRO (OSPRAY_ISPC_COMPILE)
  SET(ISPC_ADDITIONAL_ARGS "")
  SET(ISPC_TARGETS ${OSPRAY_ISPC_TARGET_LIST})

  IF (THIS_IS_MIC)
    SET(ISPC_TARGET_EXT .cpp)
    SET(ISPC_TARGET_ARGS generic-16)
    SET(ISPC_ADDITIONAL_ARGS ${ISPC_ADDITIONAL_ARGS} --opt=force-aligned-memory --emit-c++ --c++-include-file=${PROJECT_SOURCE_DIR}/ospray/common/ISPC_KNC_Backend.h )
  ELSE()
    SET(ISPC_TARGET_EXT ${CMAKE_CXX_OUTPUT_EXTENSION})
    STRING(REPLACE ";" "," ISPC_TARGET_ARGS "${ISPC_TARGETS}")
  ENDIF()

  IF (CMAKE_SIZEOF_VOID_P EQUAL 8)
    SET(ISPC_ARCHITECTURE "x86-64")
  ELSE()
    SET(ISPC_ARCHITECTURE "x86")
  ENDIF()

  SET(ISPC_TARGET_DIR ${CMAKE_CURRENT_BINARY_DIR})
  INCLUDE_DIRECTORIES(${ISPC_TARGET_DIR})

  IF(ISPC_INCLUDE_DIR)
    STRING(REPLACE ";" ";-I;" ISPC_INCLUDE_DIR_PARMS "${ISPC_INCLUDE_DIR}")
    SET(ISPC_INCLUDE_DIR_PARMS "-I" ${ISPC_INCLUDE_DIR_PARMS})
  ENDIF()

  IF (WIN32 OR "${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    SET(ISPC_OPT_FLAGS -O0 -g)
  ELSE()
    SET(ISPC_OPT_FLAGS -O0 -g)
  ENDIF()

  IF (NOT WIN32)
    SET(ISPC_ADDITIONAL_ARGS ${ISPC_ADDITIONAL_ARGS} --pic)
  ENDIF()

  SET(ISPC_OBJECTS "")

  FOREACH(src ${ARGN})
    GET_FILENAME_COMPONENT(fname ${src} NAME_WE)
    GET_FILENAME_COMPONENT(dir ${src} PATH)

    SET(input ${src})
    IF ("${dir}" MATCHES "^/") # absolute unix-style path to input
      SET(outdir "${ISPC_TARGET_DIR}/rebased${dir}")
    ELSEIF ("${dir}" MATCHES "^[A-Z]:") # absolute DOS-style path to input
      STRING(REGEX REPLACE "^[A-Z]:" "${ISPC_TARGET_DIR}/rebased/" outdir "${dir}")
    ELSE() # relative path to input
      SET(outdir "${ISPC_TARGET_DIR}/local_${dir}")
      SET(input ${CMAKE_CURRENT_SOURCE_DIR}/${src})
    ENDIF()

    SET(deps "")
    IF (EXISTS ${outdir}/${fname}.dev.idep)
      FILE(READ ${outdir}/${fname}.dev.idep contents)
      STRING(REPLACE " " ";"     contents "${contents}")
      STRING(REPLACE ";" "\\\\;" contents "${contents}")
      STRING(REPLACE "\n" ";"    contents "${contents}")
      FOREACH(dep ${contents})
        IF (EXISTS ${dep})
          SET(deps ${deps} ${dep})
        ENDIF (EXISTS ${dep})
      ENDFOREACH(dep ${contents})
    ENDIF ()

    SET(results "${outdir}/${fname}.dev${ISPC_TARGET_EXT}")

    # if we have multiple targets add additional object files
    IF (NOT THIS_IS_MIC)
      LIST(LENGTH ISPC_TARGETS NUM_TARGETS)
      IF (NUM_TARGETS EQUAL 1)
        # workaround link issues to Embree ISPC exports:
        # we add a 2nd target to force ISPC to add the ISA suffix during name
        # mangling
        SET(ISPC_TARGET_ARGS "${ISPC_TARGETS},sse2")
        LIST(APPEND ISPC_TARGETS sse2)
      ENDIF()
      FOREACH(target ${ISPC_TARGETS})
        # in v1.9.0 ISPC changed the ISA suffix of avx512knl-i32x16 to just 'avx512knl'
        IF (${target} STREQUAL "avx512knl-i32x16" AND NOT ISPC_VERSION VERSION_LESS "1.9.0")
          SET(target "avx512knl")
        ENDIF()
        SET(results ${results} "${outdir}/${fname}.dev_${target}${ISPC_TARGET_EXT}")
      ENDFOREACH()
    ENDIF()

    ADD_CUSTOM_COMMAND(
      OUTPUT ${results} ${ISPC_TARGET_DIR}/${fname}_ispc.h
      COMMAND ${CMAKE_COMMAND} -E make_directory ${outdir}
      COMMAND ${ISPC_EXECUTABLE}
      -I ${CMAKE_CURRENT_SOURCE_DIR}
      ${ISPC_INCLUDE_DIR_PARMS}
      --arch=${ISPC_ARCHITECTURE}
      --addressing=32
      ${ISPC_OPT_FLAGS}
      --target=${ISPC_TARGET_ARGS}
      --woff
      --opt=fast-math
      ${ISPC_ADDITIONAL_ARGS}
      -h ${ISPC_TARGET_DIR}/${fname}_ispc.h
      -MMM  ${outdir}/${fname}.dev.idep
      -o ${outdir}/${fname}.dev${ISPC_TARGET_EXT}
      ${input}
      DEPENDS ${input} ${deps}
      COMMENT "Building ISPC object ${outdir}/${fname}.dev${ISPC_TARGET_EXT}"
    )
    SET(ISPC_OBJECTS ${ISPC_OBJECTS} ${results})
  ENDFOREACH()
ENDMACRO()

MACRO(CONFIGURE_ISPC)
  # Embree common include directories; others may be added depending on build target.
  # this section could be sooo much cleaner if embree only used
  # fully-qualified include names...
  SET(EMBREE_INCLUDE_DIRECTORIES
    ${OSPRAY_EMBREE_SOURCE_DIR}/
    ${OSPRAY_EMBREE_SOURCE_DIR}/include
    ${OSPRAY_EMBREE_SOURCE_DIR}/common
    ${OSPRAY_EMBREE_SOURCE_DIR}/
    ${OSPRAY_EMBREE_SOURCE_DIR}/kernels
    )

  IF (OSPRAY_TARGET STREQUAL "mic")
    SET(OSPRAY_EXE_SUFFIX ".mic")
    SET(OSPRAY_LIB_SUFFIX "_mic")
    SET(OSPRAY_ISPC_SUFFIX ".cpp")
    SET(OSPRAY_ISPC_TARGET "mic")
    SET(THIS_IS_MIC ON)
    SET(__XEON__ OFF)
    INCLUDE(${PROJECT_SOURCE_DIR}/CMake/icc_xeonphi.cmake)

    # additional Embree include directory
    LIST(APPEND EMBREE_INCLUDE_DIRECTORIES ${OSPRAY_EMBREE_SOURCE_DIR}/kernels/xeonphi)

    SET(OSPRAY_TARGET_MIC ON PARENT_SCOPE)
  ELSE()
    SET(OSPRAY_EXE_SUFFIX "")
    SET(OSPRAY_LIB_SUFFIX "")
    SET(OSPRAY_ISPC_SUFFIX ".o")
    SET(THIS_IS_MIC OFF)
    SET(__XEON__ ON)
    IF (${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
      INCLUDE(${PROJECT_SOURCE_DIR}/CMake/icc.cmake)
    ELSEIF (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
      INCLUDE(${PROJECT_SOURCE_DIR}/CMake/gcc.cmake)
    ELSEIF (${CMAKE_CXX_COMPILER_ID} STREQUAL "AppleClang")
      INCLUDE(${PROJECT_SOURCE_DIR}/CMake/clang.cmake)
    ELSEIF (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
      INCLUDE(${PROJECT_SOURCE_DIR}/CMake/clang.cmake)
    ELSEIF (${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
      INCLUDE(${PROJECT_SOURCE_DIR}/CMake/msvc.cmake)
    ELSE()
      MESSAGE(FATAL_ERROR "Unsupported compiler specified: '${CMAKE_CXX_COMPILER_ID}'")
    ENDIF()

    # additional Embree include directory
    LIST(APPEND EMBREE_INCLUDE_DIRECTORIES ${OSPRAY_EMBREE_SOURCE_DIR}/kernels/xeon)

    SET(OSPRAY_EMBREE_ENABLE_AVX512 false)
    IF (BRAYNS_BUILD_ISA STREQUAL "ALL")
      # ------------------------------------------------------------------
      # in 'all' mode, we have a list of multiple targets for ispc,
      # and enable all targets for embree (we may still disable some
      # below if the compiler doesn't support them
      # ------------------------------------------------------------------
      SET(OSPRAY_ISPC_TARGET_LIST sse4 avx avx2)
      SET(OSPRAY_EMBREE_ENABLE_SSE  true)
      SET(OSPRAY_EMBREE_ENABLE_AVX  true)
      SET(OSPRAY_EMBREE_ENABLE_AVX2 true)
      IF (OSPRAY_BUILD_ENABLE_KNL)
        SET(OSPRAY_EMBREE_ENABLE_AVX512 true)
        SET(OSPRAY_ISPC_TARGET_LIST sse4 avx avx2 avx512knl-i32x16)
      ENDIF()

    ELSEIF (BRAYNS_BUILD_ISA STREQUAL "AVX512")
      # ------------------------------------------------------------------
      # in 'avx512' mode, we currently build only avx512, in generic
      # mode, but enable all embree targets to fall back to (currently
      # does not work since embree would require a 16-wide trace
      # function which it has in neither of the three targets)
      # ------------------------------------------------------------------
      SET(OSPRAY_ISPC_TARGET_LIST avx512knl-i32x16)
      SET(OSPRAY_EMBREE_ENABLE_SSE  true)
      SET(OSPRAY_EMBREE_ENABLE_AVX  true)
      SET(OSPRAY_EMBREE_ENABLE_AVX2 true)
      SET(OSPRAY_EMBREE_ENABLE_AVX512 true)

    ELSEIF (BRAYNS_BUILD_ISA STREQUAL "AVX2")
      # ------------------------------------------------------------------
      # in 'avx2' mode, we enable ONLY avx2 for ispc, and all targets
      # up to avx2 for embree. note that if the compiler doesn't
      # support AVX we will have a combination where embree uses AVX
      # (the most the compiler can do), while ispc still uses
      # avx. this works because both targets are 8 wide. it does
      # however require the compiler to understand AT LEAST AVX1.
      # ------------------------------------------------------------------
      SET(OSPRAY_ISPC_TARGET_LIST avx2)
      SET(OSPRAY_EMBREE_ENABLE_SSE  true)
      SET(OSPRAY_EMBREE_ENABLE_AVX  true)
      SET(OSPRAY_EMBREE_ENABLE_AVX2 true)

    ELSEIF (BRAYNS_BUILD_ISA STREQUAL "AVX")
      # ------------------------------------------------------------------
      # in 'avx' mode, we enable ONLY avx for ispc, and both sse and
      # avx for embree. note that this works ONLY works if the
      # compiler knows at least AVX
      # ------------------------------------------------------------------
      SET(OSPRAY_ISPC_TARGET_LIST avx)
      SET(OSPRAY_EMBREE_ENABLE_SSE  true)
      SET(OSPRAY_EMBREE_ENABLE_AVX  true)
      SET(OSPRAY_EMBREE_ENABLE_AVX2 false)

    ELSEIF (BRAYNS_BUILD_ISA STREQUAL "SSE")
      # ------------------------------------------------------------------
      # in 'sse' mode, we enable ONLY sse4 for ispc, and only sse for
      # embree
      # ------------------------------------------------------------------
      SET(OSPRAY_ISPC_TARGET_LIST sse4)
      SET(OSPRAY_EMBREE_ENABLE_SSE  true)
      SET(OSPRAY_EMBREE_ENABLE_AVX  false)
      SET(OSPRAY_EMBREE_ENABLE_AVX2 false)
    ELSE ()
      MESSAGE(ERROR "Invalid BRAYNS_BUILD_ISA value. Please select one of SSE, AVX, AVX2, or ALL.")
    ENDIF()

  ENDIF()

  IF (OSPRAY_EMBREE_ENABLE_AVX AND NOT OSPRAY_COMPILER_SUPPORTS_AVX)
    IF (NOT OSPRAY_WARNED_MISSING_AVX)
      MESSAGE("Warning: Need at least version ${GCC_VERSION_REQUIRED_AVX} of gcc for AVX. Disabling AVX.\nTo compile for AVX, please switch to either 'ICC'-compiler, or upgrade your gcc version.")
      SET(OSPRAY_WARNED_MISSING_AVX ON CACHE INTERNAL "Warned about missing AVX support.")
    ENDIF()
    SET(OSPRAY_EMBREE_ENABLE_AVX false)
  ENDIF()

  IF (OSPRAY_EMBREE_ENABLE_AVX2 AND NOT OSPRAY_COMPILER_SUPPORTS_AVX2)
    IF (NOT OSPRAY_WARNED_MISSING_AVX2)
      MESSAGE("Warning: Need at least version ${GCC_VERSION_REQUIRED_AVX2} of gcc for AVX2. Disabling AVX2.\nTo compile for AVX2, please switch to either 'ICC'-compiler, or upgrade your gcc version.")
      SET(OSPRAY_WARNED_MISSING_AVX2 ON CACHE INTERNAL "Warned about missing AVX2 support.")
    ENDIF()
    SET(OSPRAY_EMBREE_ENABLE_AVX2 false)
  ENDIF()

  IF (OSPRAY_EMBREE_ENABLE_AVX512 AND NOT OSPRAY_COMPILER_SUPPORTS_AVX512)
    IF (NOT OSPRAY_WARNED_MISSING_AVX2)
      MESSAGE("Warning: Need at least version ${GCC_VERSION_REQUIRED_AVX512} of gcc for AVX512. Disabling AVX512.\nTo compile for AVX512, please switch to either 'ICC'-compiler, or upgrade your gcc version.")
      SET(OSPRAY_WARNED_MISSING_AVX512 ON CACHE INTERNAL "Warned about missing AVX512 support.")
    ENDIF()
    SET(OSPRAY_EMBREE_ENABLE_AVX512 false)
  ENDIF()

  IF (THIS_IS_MIC)
    # whether to build in MIC/xeon phi support
    SET(OSPRAY_BUILD_COI_DEVICE OFF CACHE BOOL "Build COI Device for OSPRay's MIC support?")
  ENDIF()

  INCLUDE(${PROJECT_SOURCE_DIR}/CMake/ispc.cmake)

  INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR})
  INCLUDE_DIRECTORIES(${EMBREE_INCLUDE_DIRECTORIES})

  INCLUDE_DIRECTORIES_ISPC(${PROJECT_SOURCE_DIR})
  INCLUDE_DIRECTORIES_ISPC(${EMBREE_INCLUDE_DIRECTORIES})

  # for auto-generated cmakeconfig etc
  INCLUDE_DIRECTORIES(${PROJECT_BINARY_DIR})
  INCLUDE_DIRECTORIES_ISPC(${PROJECT_BINARY_DIR})

ENDMACRO()
