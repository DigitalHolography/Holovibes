#include "filter2d_api.hh"

#include "API.hh"
#include "input_filter.hh"
#include "tools.hh"

#define NOT_SAME_AND_NOT_RAW(old_val, new_val)                                                                         \
    if (old_val == new_val)                                                                                            \
        return ApiCode::NO_CHANGE;                                                                                     \
    if (api_->compute.get_compute_mode() == Computation::Raw)                                                          \
        return ApiCode::WRONG_COMP_MODE;

namespace holovibes::api
{

#pragma region Filter

ApiCode Filter2dApi::set_filter2d_enabled(bool checked) const
{
    NOT_SAME_AND_NOT_RAW(get_filter2d_enabled(), checked);

    UPDATE_SETTING(Filter2dEnabled, checked);

    return ApiCode::OK;
}

ApiCode Filter2dApi::set_filter2d_n1(int value) const
{
    NOT_SAME_AND_NOT_RAW(get_filter2d_n1(), value);

    if (value >= get_filter2d_n2() || value < 0)
    {
        LOG_WARN("Filter2dN1 must be in range [0, Filter2dN2[");
        return ApiCode::INVALID_VALUE;
    }

    UPDATE_SETTING(Filter2dN1, value);

    return ApiCode::OK;
}

ApiCode Filter2dApi::set_filter2d_smooth_high(int value) const
{
    NOT_SAME_AND_NOT_RAW(get_filter2d_smooth_high(), value);

    UPDATE_SETTING(Filter2dSmoothHigh, value);

    return ApiCode::OK;
}

ApiCode Filter2dApi::set_filter2d_n2(int value) const
{
    NOT_SAME_AND_NOT_RAW(get_filter2d_n2(), value);

    if (value <= get_filter2d_n1())
    {
        LOG_WARN("Filter2dN2 must be in range ]Filter2dN1, +inf[");
        return ApiCode::INVALID_VALUE;
    }

    UPDATE_SETTING(Filter2dN2, value);

    return ApiCode::OK;
}

ApiCode Filter2dApi::set_filter2d_smooth_low(int value) const
{
    NOT_SAME_AND_NOT_RAW(get_filter2d_smooth_low(), value);

    UPDATE_SETTING(Filter2dSmoothLow, value);

    return ApiCode::OK;
}

#pragma endregion

#pragma region Filter File

inline static const std::filesystem::path dir(GET_EXE_DIR);

std::vector<float> Filter2dApi::load_input_filter(const std::string& file) const
{
    if (file.empty())
        return {};

    auto path_file = dir / __INPUT_FILTER_FOLDER_PATH__ / file;

    if (!std::filesystem::exists(path_file))
    {
        LOG_WARN("Filter file not found : {}. Filter file deactivated", path_file.string());
        return {};
    }

    InputFilter input_filter(path_file.string(),
                             api_->compute.get_gpu_output_queue()->get_fd().width,
                             api_->compute.get_gpu_output_queue()->get_fd().height);

    return input_filter.get_input_filter();
}

ApiCode Filter2dApi::enable_filter(const std::string& filename) const
{
    if (api_->compute.get_is_computation_stopped())
        return ApiCode::NOT_STARTED;

    std::vector<float> input_filter = load_input_filter(filename);

    UPDATE_SETTING(InputFilter, input_filter);
    UPDATE_SETTING(FilterFileName, input_filter.empty() ? "" : filename);

    return ApiCode::OK;
}

#pragma endregion

} // namespace holovibes::api