# CUDA 3.0 and newer nolonger supports emulation mode
#SET(CUDA_BUILD_TYPE "Emulation") #can be either: Emulation, Device
SET(CUDA_BUILD_TYPE "Device") #can be either: Emulation, Device

# INCLUDE(${OE_CURRENT_EXTENSION_DIR}/conf/cmake/cuda/FindCuda.cmake)

# INCLUDE(${OE_CURRENT_EXTENSION_DIR}/CMAKE/cuda/FindCuda.cmake)
# SET(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "/${OE_CURRENT_EXTENSION_DIR}/cmake/cuda")
# MESSAGE(${CMAKE_MODULE_PATH})
FIND_PACKAGE(CUDA REQUIRED)

#SET(CUDA_64_BIT_DEVICE_CODE OFF)

find_path(CUDA_CUDPP_INCLUDE_DIR
  cudpp/cudpp.h
  PATHS ${CUDA_SDK_SEARCH_PATH}
  ${PROJECT_SOURCE_DIR}/libraries/cudpp/include
  PATH_SUFFIXES "common/inc"
  DOC "Location of cudpp.h"
  NO_DEFAULT_PATH
  )

IF(CUDA_64_BIT_DEVICE_CODE)
  SET(CUDPP_NAME cudpp_x86_64)
  SET(CUT_NAME cutil_x86_64)
ELSE(CUDA_64_BIT_DEVICE_CODE)
  SET(CUDPP_NAME cudpp_i386)
  SET(CUT_NAME cutil_i386)
ENDIF(CUDA_64_BIT_DEVICE_CODE)

find_library(CUDA_CUDPP_LIBRARY
  NAMES cudpp ${CUDPP_NAME}
  PATHS ${CUDA_SDK_SEARCH_PATH}
  ${PROJECT_SOURCE_DIR}/libraries/cudpp/lib
  # The new version of the sdk shows up in common/lib, but the old one is in lib
  PATH_SUFFIXES "common/lib/darwin" "common/lib/linux" "common/lib" "lib" 
  DOC "Location of cutil library"
  NO_DEFAULT_PATH
  )


# Now search system paths
set(CUDA_CUT_LIBRARIES ${CUDA_CUT_LIBRARY})
 
find_path(CUDA_CUT_INCLUDE_DIR
  cutil.h
  PATHS ${CUDA_SDK_SEARCH_PATH}
  PATH_SUFFIXES "common/inc"
  DOC "Location of cutil.h"
  NO_DEFAULT_PATH
  )

find_library(CUDA_CUT_LIBRARY
  NAMES cutil ${CUT_NAME} ${cuda_cutil_name}
  PATHS ${CUDA_SDK_SEARCH_PATH}
  # The new version of the sdk shows up in common/lib, but the old one is in lib
  PATH_SUFFIXES "common/lib" "lib"
  DOC "Location of cutil library"
  NO_DEFAULT_PATH
  )
# Now search system paths
set(CUDA_CUT_LIBRARIES ${CUDA_CUT_LIBRARY})


IF (CMAKE_BUILD_TYPE MATCHES debug)
SET(CUDA_NVCC_FLAGS "-D_DEBUG;-DOE_SAFE=${OE_SAFE};-DOE_DEBUG_GL=${OE_DEBUG_GL};${CUDA_NVCC_FLAGS}")
MESSAGE("nvcc flags: ${CUDA_NVCC_FLAGS}")
MESSAGE("64-bit? ${CUDA_64_BIT_DEVICE_CODE}")
MESSAGE("cuda include dir: ${CUDA_INCLUDE_DIRS}")
#MESSAGE("cuda lib dir (cuda and cudart): ${CUDA_TARGET_LINK}")
MESSAGE("cuda sdk include dir: ${CUDA_CUT_INCLUDE_DIR}")
MESSAGE("CUDA_SDK_ROOT_DIR: ${CUDA_SDK_ROOT_DIR}")
MESSAGE("cuda sdk lib dir: ${CUDA_CUT_LIBRARIES} ${CUDA_CUT_TARGET_LINK}")
ENDIF (CMAKE_BUILD_TYPE MATCHES debug)

CUDA_INCLUDE_DIRECTORIES(
  "${OE_SOURCE_DIR}" # OE Base
  "${OE_CURRENT_EXTENSION_DIR}/../OpenGLRenderer/" # Meta/OpenGL.h
  ${OE_CURRENT_EXTENSION_DIR} # Meta/CUDA.h
  ${CUDA_INCLUDE_DIRS} # cuda
  ${CUDA_CUT_INCLUDE_DIR} # cutil
)

INCLUDE_DIRECTORIES(
  ${CUDA_INCLUDE_DIRS}
  ${CUDA_CUT_INCLUDE_DIR} #to include cutil.h
  ${CUDA_CUDPP_INCLUDE_DIR}
)
