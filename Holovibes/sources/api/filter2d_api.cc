#include "filter2d_api.hh"

#include "input_filter.hh"
#include "tools.hh"

namespace holovibes::api
{

inline static const std::filesystem::path dir(GET_EXE_DIR);

void set_filter2d(bool checked)
{
    if (api::get_compute_mode() == Computation::Raw)
        return;

    set_filter2d_enabled(checked);
    pipe_refresh();
}

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

} // namespace holovibes::api