#include "filter2d_api.hh"

#include "input_filter.hh"
#include "tools.hh"

namespace holovibes::api
{

#pragma region Filter

void Filter2dApi::set_filter2d_enabled(bool checked)
{
    if (api_.compute.get_compute_mode() == Computation::Raw)
        return;

    UPDATE_SETTING(Filter2dEnabled, checked);
    api_.compute.pipe_refresh();
}

void Filter2dApi::set_filter2d_n1(int value)
{
    UPDATE_SETTING(Filter2dN1, value);
    api_.compute.pipe_refresh();
}

void Filter2dApi::set_filter2d_n2(int value)
{
    UPDATE_SETTING(Filter2dN2, value);
    api_.compute.pipe_refresh();
}

#pragma endregion

#pragma region Filter File

inline static const std::filesystem::path dir(GET_EXE_DIR);

void Filter2dApi::load_input_filter(const std::string& file)
{
    auto& holo = Holovibes::instance();
    try
    {
        auto path_file = dir / __INPUT_FILTER_FOLDER_PATH__ / file;
        InputFilter(get_input_filter(),
                    path_file.string(),
                    holo.get_gpu_output_queue()->get_fd().width,
                    holo.get_gpu_output_queue()->get_fd().height);
    }
    catch (std::exception& e)
    {
        LOG_ERROR("Couldn't load input filter : {}", e.what());
    }
}

void Filter2dApi::enable_filter(const std::string& filename)
{
    if (filename == get_filter_file_name())
        return;

    if (!api_.compute.get_compute_pipe_no_throw())
        return;

    set_filter_file_name(filename);
    UPDATE_SETTING(FilterEnabled, !filename.empty());

    // There is no file for filtering
    if (filename.empty())
        set_input_filter({});
    else
        load_input_filter(filename);

    api_.compute.pipe_refresh();
}

#pragma endregion

} // namespace holovibes::api