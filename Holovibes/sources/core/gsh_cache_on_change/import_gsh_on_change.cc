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
        Holovibes::instance().start_compute();
        if (api::detail::get_value<IsGuiEnable>())
            Holovibes::instance().start_information_display();
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
void ImportGSHOnChange::operator()<StartFrame>(uint& new_value)
{
    if (api::get_end_frame() < new_value)
        api::set_end_frame(new_value);
}

template <>
void ImportGSHOnChange::operator()<EndFrame>(uint& new_value)
{
    if (api::get_start_frame() > new_value)
        api::set_start_frame(new_value);
}

template <>
void ImportGSHOnChange::operator()<FileNumberOfFrame>(uint& new_value)
{
    api::set_start_frame(1);
    api::set_end_frame(new_value);
}

} // namespace holovibes
