SET ( EXTENSION_NAME "Extensions_CUDA")

ADD_LIBRARY( ${EXTENSION_NAME}
  Meta/CUDA.h
  Meta/CUDA.cpp
  Resources/CUDA/CUDADataBlock.h
)
