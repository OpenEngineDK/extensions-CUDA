#SET(CUDA_BUILD_TYPE "Emulation") #can be either: Emulation, Device
SET(CUDA_BUILD_TYPE "Device") #can be either: Emulation, Device

INCLUDE(${OE_CURRENT_EXTENSION_DIR}/conf/cmake/cuda/FindCuda.cmake)

IF (CMAKE_BUILD_TYPE MATCHES debug)
SET(CUDA_NVCC_FLAGS "-D_DEBUG -DOE_SAFE=${OE_SAFE} -DOE_DEBUG_GL=${OE_DEBUG_GL} ${CUDA_NVCC_FLAGS}")
MESSAGE("cuda include dir: ${CUDA_INCLUDE}")
MESSAGE("cuda lib dir (cuda and cudart): ${CUDA_TARGET_LINK}")
MESSAGE("cuda sdk include dir: ${CUDA_CUT_INCLUDE}")
MESSAGE("cuda sdk lib dir: ${CUDA_CUT_TARGET_LINK}")
ENDIF (CMAKE_BUILD_TYPE MATCHES debug)

CUDA_INCLUDE_DIRECTORIES(
  "${OE_SOURCE_DIR}" # OE Base
  "${OE_CURRENT_EXTENSION_DIR}/../OpenGLRenderer/" # Meta/OpenGL.h
  ${OE_CURRENT_EXTENSION_DIR} # Meta/CUDA.h
  ${CUDA_INCLUDE} # cuda
  ${CUDA_CUT_INCLUDE} # cutil
)

INCLUDE_DIRECTORIES(
  ${CUDA_INCLUDE}
  ${CUDA_CUT_INCLUDE} #to include cutil.h
)
