#include "compute_gsh_on_change.hh"
#include "common_on_change.hh"
#include "API.hh"
#include "import_gsh_on_change.hh"

namespace holovibes
{

template <>
void ImportGSHOnChange::operator()<ImportFrameDescriptor>(FrameDescriptor& new_value)
{
    LOG_UPDATE_ON_CHANGE(ImportFrameDescriptor);

    compute_output_fd(new_value, api::get_compute_mode(), api::get_image_type());
}

template <>
void ImportGSHOnChange::operator()<ImportType>(ImportTypeEnum& new_value)
{
    LOG_UPDATE_ON_CHANGE(ImportType);

    if (new_value != ImportTypeEnum::None)
    {
        Holovibes::instance().init_gpu_queues();
        Holovibes::instance().init_pipe();
        Holovibes::instance().start_compute();
        Holovibes::instance().start_information_display();
    }
    else
    {
        Holovibes::instance().stop_file_frame_read();
        Holovibes::instance().stop_camera_frame_read();
        Holovibes::instance().stop_information_display();
        Holovibes::instance().stop_compute();
        Holovibes::instance().destroy_pipe();
        Holovibes::instance().destroy_gpu_queues();
    }
}

template <>
void ImportGSHOnChange::operator()<ImportFilePath>(std::string& filename)
{
    LOG_UPDATE_ON_CHANGE(ImportFilePath);

    if (!filename.empty())
    {
        // Will throw if the file format (extension) cannot be handled
        io_files::InputFrameFile* input_file = io_files::InputFrameFileFactory::open(filename);

        input_file->import_compute_settings();
        input_file->import_info();

        api::detail::set_value<FileNumberOfFrame>(input_file->get_total_nb_frames());
        api::detail::set_value<ImportFrameDescriptor>(input_file->get_frame_descriptor());

        GSH::instance().notify();

        delete input_file;
    }
    else
        throw std::runtime_error("No filename");
}

template <>
bool ImportGSHOnChange::change_accepted<StartFrame>(uint new_value)
{
    return new_value <= api::get_end_frame();
}

template <>
bool ImportGSHOnChange::change_accepted<EndFrame>(uint new_value)
{
    return new_value >= api::get_start_frame();
}

} // namespace holovibes