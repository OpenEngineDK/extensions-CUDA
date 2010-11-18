#include <Meta/CUDA.h>

cudaDeviceProp activeCudaDevice;

void INITIALIZE_CUDA() {
    //@todo: test that cuda is supported on the platform.
    
    CUdevice device = cutGetMaxGflopsDeviceId();
    cudaSetDevice(device);
    cudaGLSetGLDevice(device);
    CHECK_FOR_CUDA_ERROR();

    cuInit(0);
    CHECK_FOR_CUDA_ERROR();
 
    int version;
    cuDriverGetVersion(&version);
    cudaGetDeviceProperties(&activeCudaDevice, device);
    logger.info << "CUDA: version " << version/1000 << "." << version % 100 << ", using device " << std::string(activeCudaDevice.name) << logger.end;
    CHECK_FOR_CUDA_ERROR();
}
