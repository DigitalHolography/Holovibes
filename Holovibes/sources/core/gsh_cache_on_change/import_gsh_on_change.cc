#include "compute_gsh_on_change.hh"
#include "common_gsh_on_change.hh"
#include "API.hh"
#include "import_gsh_on_change.hh"
#include "camera_dll.hh"

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
        Holovibes::instance().start_compute();
}

template <>
void ImportGSHOnChange::operator()<ImportFilePath>(std::string& filename)
{
    LOG_UPDATE_ON_CHANGE(ImportFilePath);

    if (filename.empty())
    {
        api::detail::set_value<ImportType>(ImportTypeEnum::None);
        return;
    }

    // Will throw if the file format (extension) cannot be handled
    io_files::InputFrameFile* input_file = io_files::InputFrameFileFactory::open(filename);

    input_file->import_compute_settings();
    input_file->import_info();

    api::detail::set_value<FileNumberOfFrame>(input_file->get_total_nb_frames());
    api::detail::set_value<ImportFrameDescriptor>(input_file->get_frame_descriptor());

    GSH::instance().notify();

    delete input_file;
}

template <>
void ImportGSHOnChange::operator()<CurrentCameraKind>(CameraKind& camera)
{
    if (camera == CameraKind::None)
    {
        api::detail::set_value<ImportType>(ImportTypeEnum::None);
        return;
    }

    try
    {
        const static std::map<CameraKind, LPCSTR> camera_dictionary = {
            {CameraKind::Adimec, "CameraAdimec.dll"},
            {CameraKind::BitflowCyton, "BitflowCyton.dll"},
            {CameraKind::IDS, "CameraIds.dll"},
            {CameraKind::Phantom, "CameraPhantom.dll"},
            {CameraKind::Hamamatsu, "CameraHamamatsu.dll"},
            {CameraKind::xiQ, "CameraXiq.dll"},
            {CameraKind::xiB, "CameraXib.dll"},
            {CameraKind::OpenCV, "CameraOpenCV.dll"},
        };
        Holovibes::instance().set_active_camera(camera::CameraDLL::load_camera(camera_dictionary.at(camera)));
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("Camera library cannot be loaded. (Exception: {})", e.what());
        throw;
    }

    api::detail::set_value<PixelSize>(Holovibes::instance().get_active_camera()->get_pixel_size());
    api::detail::set_value<ImportFrameDescriptor>(Holovibes::instance().get_active_camera()->get_fd());

    // FIXME : Camera should have the same workflow as File : a menu with selection and then the possibility to Start
    // and Stop. The Start and stop should handle only ImportType. This function should not do this line.
    api::detail::set_value<ImportType>(ImportTypeEnum::Camera);
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

template <>
bool ImportGSHOnChange::change_accepted<StartFrame>(uint new_value)
{
    return new_value > 0 && new_value <= api::detail::get_value<FileNumberOfFrame>();
}

template <>
bool ImportGSHOnChange::change_accepted<EndFrame>(uint new_value)
{
    return new_value > 0 && new_value <= api::detail::get_value<FileNumberOfFrame>();
}

template <>
bool ImportGSHOnChange::change_accepted<FileNumberOfFrame>(uint new_value)
{
    return new_value > 0;
}

} // namespace holovibes
