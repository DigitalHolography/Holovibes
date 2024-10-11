#include "nppi_data.hh"

namespace holovibes
{
namespace cuda_tools
{
int NppiData::scratch_buffer_size_ = 0;
UniquePtr<Npp8u> NppiData::scratch_buffer_;

NppiData::NppiData(int width, int height)
{
    size_.width = width;
    size_.height = height;
}

const NppiSize& NppiData::get_size() const { return size_; }

void NppiData::set_size(int width, int height)
{
    size_.width = width;
    size_.height = height;
}

Npp8u* NppiData::get_scratch_buffer(std::function<NppStatus(NppiSize, int*)> size_function)
{
    int required_size = 0;
    if (size_function(size_, &required_size) != NPP_SUCCESS)
        return nullptr;
    return get_scratch_buffer(required_size);
}

Npp8u* NppiData::get_scratch_buffer() { return get_scratch_buffer(scratch_buffer_size_); }

Npp8u* NppiData::get_scratch_buffer(int size)
{
    if (scratch_buffer_.get() == nullptr || size > scratch_buffer_size_)
    {
        scratch_buffer_size_ = size;
        scratch_buffer_.resize(size);
    }
    return scratch_buffer_.get();
}

} // namespace cuda_tools
} // namespace holovibes