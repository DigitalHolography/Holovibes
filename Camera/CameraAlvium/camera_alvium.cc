#include "camera_exception.hh"
#include <iostream>

#include "camera_alvium.hh"
#include "camera_logger.hh"

#include <chrono>
#include <span>
namespace camera
{
CameraAlvium::FrameObserver::FrameObserver(VmbCPP::CameraPtr camera_ptr,
                                           VmbInt64_t size_frame,
                                           CameraAlvium& camera_alvium)
    : VmbCPP::IFrameObserver(camera_ptr)
    , size_frame_(size_frame)
    , camera_alvium_(camera_alvium)
{
}

// Frame callback notifies about incoming frames
void CameraAlvium::FrameObserver::FrameReceived(const VmbCPP::FramePtr pFrame)
{
    // Send notification to working thread
    // Do not apply image processing within this callback (performance)

    unsigned char* buf;
    pFrame->GetImage(buf);
    camera_alvium_.waiting_queue_.push(buf);

    // When the frame has been processed, requeue it
    m_pCamera->QueueFrame(pFrame);
}

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
    VmbCPP::FeaturePtr pFeature; // Generic feature pointer
    VmbInt64_t nPLS;             // Payload size value (size of a frame)
    frames_ = VmbCPP::FramePtrVector(15);

    if (api_vmb_.Startup() != VmbErrorType::VmbErrorSuccess ||
        api_vmb_.GetCameras(cameras) != VmbErrorType::VmbErrorSuccess) // FIXME MAYBE add pathConfiguration
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

        // Open the camera, must be closed when finished
        camera_ptr_ = *iter;
        if (camera_ptr_->Open(VmbAccessModeFull) != VmbErrorType::VmbErrorSuccess)
        {
            throw CameraException(CameraException::NOT_CONNECTED);
        }

        // Query the necessary buffer size
        camera_ptr_->GetFeatureByName("VmbPayloadsizeGet", pFeature);
        pFeature->GetValue(nPLS);
        for (VmbCPP::FramePtrVector::iterator iter = frames_.begin(); frames_.end() != iter; ++iter)
        {
            (*iter).reset(new VmbCPP::Frame(nPLS));
            (*iter)->RegisterObserver(
                VmbCPP::IFrameObserverPtr(new CameraAlvium::FrameObserver(camera_ptr_, nPLS, *this)));
            camera_ptr_->AnnounceFrame(*iter);
        }

        // Start the capture engine (API)
        camera_ptr_->StartCapture();
        for (VmbCPP::FramePtrVector::iterator iter = frames_.begin(); frames_.end() != iter; ++iter)
        {
            // Put frame into the frame queue
            camera_ptr_->QueueFrame(*iter);
        }

        return;
    }
    throw CameraException(CameraException::NOT_CONNECTED);
}

void CameraAlvium::start_acquisition()
{
    // FIXME

    // Probably need to allocate a buffer like in Hamamatsu (host ?)
    VmbCPP::FeaturePtr pFeature; // Generic feature pointer
    camera_ptr_->GetFeatureByName("AcquisitionStart", pFeature);
    pFeature->RunCommand();
}

void CameraAlvium::stop_acquisition()
{
    // FIXME
    VmbCPP::FeaturePtr pFeature; // Generic feature pointer
    camera_ptr_->GetFeatureByName("AcquisitionStop", pFeature);
    pFeature->RunCommand();
}

struct camera::CapturedFramesDescriptor CameraAlvium::get_frames()
{
    // FIXME
    if (waiting_queue_.empty())
        return {};

    unsigned char* buf = waiting_queue_.back();
    waiting_queue_.pop();
    return camera::CapturedFramesDescriptor{buf};
}

void CameraAlvium::load_default_params()
{
    VmbCPP::FeaturePtr pFeature; // Generic feature pointer

    VmbInt64_t height;
    camera_ptr_->GetFeatureByName("Height", pFeature);
    pFeature->GetValue(height);
    fd_.height = height;

    VmbInt64_t width;
    camera_ptr_->GetFeatureByName("Width", pFeature);
    pFeature->GetValue(width);
    fd_.width = width;

    fd_.depth = 2; // FIXME 10-bit in theory, rounded to 2 bytes

    fd_.byteEndian = Endianness::LittleEndian; // FIXME Not sure, test this
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

void CameraAlvium::shutdown_camera()
{
    camera_ptr_->EndCapture();
    camera_ptr_->FlushQueue();
    camera_ptr_->RevokeAllFrames();
    for (VmbCPP::FramePtrVector::iterator iter = frames_.begin(); frames_.end() != iter; ++iter)
    {
        // Unregister the frame observer/callback
        (*iter)->UnregisterObserver();
    }
    if (camera_ptr_->Close() != VmbErrorType::VmbErrorSuccess)
    {
        throw CameraException(CameraException::CANT_SHUTDOWN);
    }

    api_vmb_.Shutdown();
}

ICamera* new_camera_device() { return new CameraAlvium(); }

} // namespace camera