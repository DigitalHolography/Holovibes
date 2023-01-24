#include "compute_gsh_on_change.hh"
#include "common_gsh_on_change.hh"
#include "API.hh"

namespace holovibes
{

template <>
void ComputeGSHOnChange::operator()<BatchSize>(int& new_value)
{
    LOG_UPDATE_ON_CHANGE(BatchSize);

    if (new_value > api::get_input_buffer_size())
        new_value = api::get_input_buffer_size();

    auto time_stride = api::get_time_stride();
    if (time_stride < new_value)
        api::set_time_stride(new_value);
    else if (time_stride % new_value != 0)
        api::set_time_stride(time_stride - time_stride % new_value);
}

template <>
void ComputeGSHOnChange::operator()<TimeStride>(int& new_value)
{
    LOG_UPDATE_ON_CHANGE(TimeStride);

    auto batch_size = api::get_batch_size();
    if (batch_size > new_value)
        new_value = batch_size;
    else if (new_value % batch_size != 0)
        new_value = new_value - new_value % batch_size;
}

template <>
void ComputeGSHOnChange::operator()<TimeTransformationCutsEnable>(bool& new_value)
{
    LOG_UPDATE_ON_CHANGE(TimeTransformationCutsEnable);

    if (new_value == false)
        api::detail::set_value<CutsViewEnabled>(false);
}

template <>
void ComputeGSHOnChange::operator()<ComputeMode>(ComputeModeEnum& new_value)
{
    LOG_UPDATE_ON_CHANGE(ComputeMode);

    compute_output_fd(api::get_import_frame_descriptor(), new_value, api::get_image_type());
}

template <>
void ComputeGSHOnChange::operator()<ImageType>(ImageTypeEnum& new_value)
{
    LOG_UPDATE_ON_CHANGE(ImageType);

    compute_output_fd(api::get_import_frame_descriptor(), api::get_compute_mode(), new_value);
}

template <>
void ComputeGSHOnChange::operator()<Filter2D>(Filter2DStruct& new_value)
{
    LOG_UPDATE_ON_CHANGE(Filter2D);

    if (new_value.enabled == false)
        api::set_filter2d_view_enabled(false);
}

template <>
void ComputeGSHOnChange::operator()<SpaceTransformation>(SpaceTransformationEnum& new_value)
{
    if (new_value == SpaceTransformationEnum::FFT1)
        api::detail::set_value<FftShiftEnabled>(true);
    else
        api::detail::set_value<FftShiftEnabled>(false);
}
} // namespace holovibes