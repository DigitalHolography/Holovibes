#include "filter2d_api.hh"

#include "input_filter.hh"
#include "tools.hh"

namespace holovibes::api
{

#pragma region Filter

void set_filter2d_enabled(bool checked)
{
    if (api::get_compute_mode() == Computation::Raw)
        return;

    UPDATE_SETTING(Filter2dEnabled, checked);
    pipe_refresh();
}

void set_filter2d_n1(int value)
{
    UPDATE_SETTING(Filter2dN1, value);
    pipe_refresh();
}

void set_filter2d_n2(int value)
{
    UPDATE_SETTING(Filter2dN2, value);
    pipe_refresh();
}

#pragma endregion

#pragma region Filter File

inline static const std::filesystem::path dir(GET_EXE_DIR);

std::vector<float> get_input_filter() { return GET_SETTING(InputFilter); }

void set_input_filter(std::vector<float> value) { UPDATE_SETTING(InputFilter, value); }

void load_input_filter(const std::string& file)
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

void enable_filter(const std::string& filename)
{
    if (filename == api::get_filter_file_name())
        return;

    if (!get_compute_pipe_no_throw())
        return;

    api::set_filter_file_name(filename);
    UPDATE_SETTING(FilterEnabled, !filename.empty());

    // There is no file for filtering
    if (filename.empty())
        set_input_filter({});
    else
        load_input_filter(filename);

    pipe_refresh();
}

#pragma endregion

} // namespace holovibes::api