#include "input_filter.hh"

namespace holovibes
{
void InputFilter::normalize_filter(const cudaStream_t stream) {}

void InputFilter::interpolate_filter(size_t fd_width, size_t fd_height, const cudaStream_t stream) {}

InputFilter::InputFilter(std::string path) {
    gpu_filter = nullptr;
    width = 0;
    height = 0;
}

void InputFilter::apply_filter(cuComplex* gpu_input, size_t fd_width, size_t fd_height, const cudaStream_t stream) {}
}