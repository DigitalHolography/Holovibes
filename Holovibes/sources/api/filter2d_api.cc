#include "filter2d_api.hh"

#include "API.hh"
#include "input_filter.hh"
#include "tools.hh"

namespace holovibes::api
{

#pragma region Filter

void Filter2dApi::set_filter2d_enabled(bool checked) const
{
    if (api_->compute.get_compute_mode() == Computation::Raw)
        return;

    UPDATE_SETTING(Filter2dEnabled, checked);
    api_->compute.pipe_refresh();
}

void Filter2dApi::set_filter2d_n1(int value) const
{
    UPDATE_SETTING(Filter2dN1, value);
    api_->compute.pipe_refresh();
}

void Filter2dApi::set_filter2d_n2(int value) const
{
    UPDATE_SETTING(Filter2dN2, value);
    api_->compute.pipe_refresh();
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
    set_filter_file_name(input_filter.empty() ? "" : filename);

    api_->compute.pipe_refresh();

    return ApiCode::OK;
}

#pragma endregion

} // namespace holovibes::api