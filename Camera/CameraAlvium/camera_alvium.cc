#include "camera_exception.hh"
#include <iostream>

#include "camera_alvium.hh"
#include "camera_logger.hh"

#include <chrono>
#include <span>
namespace camera
{

#define VMB_ERROR(cond) cond != VmbErrorType::VmbErrorSuccess

CameraAlvium::FrameObserver::FrameObserver(VmbCPP::CameraPtr camera_ptr, CameraAlvium& camera_alvium)
    : VmbCPP::IFrameObserver(camera_ptr)
    , camera_alvium_(camera_alvium)
{
}

// Frame callback notifies about incoming frames
void CameraAlvium::FrameObserver::FrameReceived(const VmbCPP::FramePtr pFrame)
{
    // Send notification to working thread
    unsigned char* buf;

    if (VMB_ERROR(pFrame->GetImage(buf)))
        throw CameraException(CameraException::CANT_GET_FRAME);

    camera_alvium_.waiting_queue_.push(buf);

    // When the frame has been processed, requeue it
    if (VMB_ERROR(m_pCamera->QueueFrame(pFrame)))
        throw CameraException(CameraException::CANT_GET_FRAME);
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

    if (VMB_ERROR(api_vmb_.Startup()) || VMB_ERROR(api_vmb_.GetCameras(cameras)))
        throw CameraException(CameraException::NOT_CONNECTED);

    // Trying to connect the first available Camera
    for (VmbCPP::CameraPtrVector::iterator iter = cameras.begin(); cameras.end() != iter; ++iter)
    {
        if (VMB_ERROR((*iter)->GetName(name)))
            continue;

        Logger::camera()->info("Connected to {}", name_);

        camera_ptr_ = *iter;
        if (VMB_ERROR(camera_ptr_->Open(VmbAccessModeFull)))
            throw CameraException(CameraException::NOT_CONNECTED);

        bind_params();

        return;
    }
    throw CameraException(CameraException::NOT_CONNECTED);
}

void CameraAlvium::start_acquisition()
{
    Logger::camera()->info("Start Acquisition");
    unsigned int n_buffers = 1; // Maybe need to be increase for processing

    auto frame_obs_ptr = VmbCPP::IFrameObserverPtr(new FrameObserver(camera_ptr_, *this));
    if (VMB_ERROR(camera_ptr_->StartContinuousImageAcquisition(n_buffers, frame_obs_ptr)))
        throw CameraException(CameraException::CANT_START_ACQUISITION);
}

void CameraAlvium::stop_acquisition()
{
    Logger::camera()->info("Stop Acquisition");
    if (VMB_ERROR(camera_ptr_->StopContinuousImageAcquisition()))
        throw CameraException(CameraException::CANT_STOP_ACQUISITION);
}

struct camera::CapturedFramesDescriptor CameraAlvium::get_frames()
{
    if (waiting_queue_.empty())
        return {};

    unsigned char* buf = waiting_queue_.front();
    waiting_queue_.pop();

    return camera::CapturedFramesDescriptor{buf};
}

void CameraAlvium::load_default_params()
{
    fd_.height = MAX_HEIGHT;
    fd_.width = MAX_WIDTH;
    height_ = MAX_HEIGHT;
    width_ = MAX_WIDTH;

    fd_.depth = 1;

    fd_.byteEndian = Endianness::LittleEndian;
}

void CameraAlvium::load_ini_params()
{
    // TODO : Add more params, check with mickael
    const boost::property_tree::ptree& pt = get_ini_pt();

    width_ = pt.get<unsigned short>("alvium.width", width_);
    height_ = pt.get<unsigned short>("alvium.height", height_);

    fd_.width = width_;
    fd_.height = height_;

    return;
};

void CameraAlvium::bind_params()
{
    VmbCPP::FeaturePtr fp; // Generic feature pointer

    /* TODO : Add more params, check with mickael
        \A -> \B => VMB_ERROR(camera_ptr_->GetFeatureByName(\A, fp)) || VMB_ERROR(fp->SetValue(\B))

        "DeviceLinkThroughputLimit" -> 450'000'000
        "AcquisitionFrameRateEnable" -> "true"
        "AcquisitionFrameRate" -> 10
        "ExposureAuto" -> "Off"
        "ExposureTime" -> EXP (I dont know what it is need to check API Manuel)
        "GainAuto" -> "Off"
        "SensorBitDepth" -> "Bpp12"
        "OffsetX" -> X0
        "OffsetY" -> Y0
    */

    if (VMB_ERROR(camera_ptr_->GetFeatureByName("PixelFormat", fp)) || VMB_ERROR(fp->SetValue("Mono8")) ||
        VMB_ERROR(camera_ptr_->GetFeatureByName("Width", fp)) || VMB_ERROR(fp->SetValue(width_)) ||
        VMB_ERROR(camera_ptr_->GetFeatureByName("Height", fp)) || VMB_ERROR(fp->SetValue(height_)))
        throw CameraException(CameraException::NOT_INITIALIZED);
};

void CameraAlvium::shutdown_camera()
{
    if (VMB_ERROR(camera_ptr_->FlushQueue()) || VMB_ERROR(camera_ptr_->RevokeAllFrames()) ||
        VMB_ERROR(camera_ptr_->Close()))
        throw CameraException(CameraException::CANT_SHUTDOWN);

    api_vmb_.Shutdown();
}

ICamera* new_camera_device() { return new CameraAlvium(); }

} // namespace camera