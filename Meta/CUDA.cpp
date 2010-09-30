#include <Meta/CUDA.h>

cudaDeviceProp activeCudaDevice;

void INITIALIZE_CUDA() {
    //@todo: test that cuda is supported on the platform.
    
    cuInit(0);

    CUdevice device = cutGetMaxGflopsDeviceId();
    cudaSetDevice(device);
    cudaGLSetGLDevice(device);

    int version;
    cuDriverGetVersion(&version);
    cudaGetDeviceProperties(&activeCudaDevice, device);
    logger.info << "CUDA: version " << version/1000 << "." << version % 100 << ", using device " << std::string(activeCudaDevice.name) << logger.end;
}
