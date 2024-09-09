#include "camera_exception.hh"
#include <iostream>

#include "camera_alvium.hh"
#include "camera_logger.hh"

#include <chrono>
#include <VmbC/VmbCommonTypes.h>

namespace camera
{
CameraAlvium::CameraAlvium()
    : Camera("alvium.ini")
    , api_vmb_(VmbCPP::VmbSystem::GetInstance())
{
    name_ = "Alvium-1800-u/2050";
    Logger::camera()->info("Loading Alvium-1800-u/2050 ...");

    load_default_params();

    if (ini_file_is_open())
    {
        load_ini_params();
        ini_file_.close();
    }

    init_camera();
}

void CameraAlvium::init_camera()
{
    std::string name;
    VmbCPP::CameraPtrVector cameras;
    VmbCPP::FeaturePtr pFeature;       // Generic feature pointer
    VmbInt64_t nPLS;                   // Payload size value
    VmbCPP::FramePtrVector frames(15); // Frame array FIXME

    if (api_vmb_.Startup() == VmbErrorType::VmbErrorInternalFault ||
        api_vmb_.GetCameras(cameras) == VmbErrorType::VmbErrorInternalFault) // FIXME MAYBE add pathConfiguration
    {
        throw CameraException(CameraException::NOT_CONNECTED);
    }

    // Trying to connect the first available Camera
    for (VmbCPP::CameraPtrVector::iterator iter = cameras.begin(); cameras.end() != iter; ++iter)
    {
        if (VmbErrorType::VmbErrorSuccess != (*iter)->GetName(name))
        {
            continue;
        }

        Logger::camera()->info("Connected to {}", name_);

        // Open the camera
        camera_ptr_ = *iter;
        camera_ptr_->Open(VmbAccessModeFull);

        // Query the necessary buffer size
        camera_ptr_->GetFeatureByName("VmbPayloadsizeGet", pFeature);
        pFeature->GetValue(nPLS);
        for (VmbCPP::FramePtrVector::iterator iter = frames.begin(); frames.end() != iter; ++iter)
        {
            (*iter).reset(new VmbCPP::Frame(nPLS));
            /*(*iter)->RegisterObserver(VmbCPP::IFrameObserverPtr(new VmbCPP::FrameObserver(camera)));
            camera->AnnounceFrame(*iter);*/
        }

        return;
    }
    throw CameraException(CameraException::NOT_CONNECTED);
}

void CameraAlvium::start_acquisition()
{
    // FIXME
    return;
}

void CameraAlvium::stop_acquisition()
{
    // FIXME
    return;
}

struct camera::CapturedFramesDescriptor CameraAlvium::get_frames()
{
    // FIXME
    return {};
}

void CameraAlvium::load_default_params()
{
    // FIXME
    return;
}

void CameraAlvium::load_ini_params()
{
    // FIXME
    return;
};

void CameraAlvium::bind_params()
{
    // FIXME
    return;
};

void CameraAlvium::shutdown_camera() { api_vmb_.Shutdown(); }

} // namespace camera