#include "camera_exception.hh"
#include <iostream>

#include "camera_alvium.hh"
#include "camera_logger.hh"

#include <chrono>
#include <span>
namespace camera
{
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
    VmbUint32_t nPLS; // Payload size value (size of a frame)

    if (api_vmb_.Startup() != VmbErrorType::VmbErrorSuccess ||
        api_vmb_.GetCameras(cameras) != VmbErrorType::VmbErrorSuccess)
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

        camera_ptr_ = *iter;
        if (camera_ptr_->Open(VmbAccessModeFull) != VmbErrorType::VmbErrorSuccess)
        {
            throw CameraException(CameraException::NOT_CONNECTED);
        }

        // Getting the necessary buffer size
        if (camera_ptr_->GetPayloadSize(nPLS) != VmbErrorType::VmbErrorSuccess)
        {
            throw CameraException(CameraException::NOT_CONNECTED);
        }

        bind_params();

        camera_ptr_->StartCapture();

        return;
    }
    throw CameraException(CameraException::NOT_CONNECTED);
}

void CameraAlvium::start_acquisition()
{
    Logger::camera()->info("Start Acquisition");
    unsigned int nBuffers = 1;

    camera_ptr_->StartContinuousImageAcquisition(nBuffers,
                                                 VmbCPP::IFrameObserverPtr(new FrameObserver(camera_ptr_, *this)));
}

void CameraAlvium::stop_acquisition()
{
    Logger::camera()->info("Stop Acquisition");
    camera_ptr_->StopContinuousImageAcquisition();
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
    if (
        // TODO : Add more params, check with mickael

        // camera_ptr_->GetFeatureByName("DeviceLinkThroughputLimit", fp) != OK || fp->SetValue(450'000'000) != OK ||
        // cam->GetFeatureByName("AcquisitionFrameRateEnable", fp) != OK ||  fp->SetValue("true") != OK  ||
        // cam->GetFeatureByName("AcquisitionFrameRate", fp) != OK       ||  fp->SetValue(10) != OK  ||
        // camera_ptr_->GetFeatureByName("ExposureAuto", fp) != OK || fp->SetValue("Off") != OK ||
        // camera_ptr_->GetFeatureByName("ExposureTime", fp) != OK || fp->SetValue(EXP) != OK ||
        // camera_ptr_->GetFeatureByName("GainAuto", fp) != OK || fp->SetValue("Off") != OK ||
        // camera_ptr_->GetFeatureByName("SensorBitDepth", fp) != VmbErrorType::VmbErrorSuccess ||
        // fp->SetValue("Bpp12") != VmbErrorType::VmbErrorSuccess ||
        camera_ptr_->GetFeatureByName("PixelFormat", fp) != VmbErrorType::VmbErrorSuccess ||
        fp->SetValue("Mono8") != VmbErrorType::VmbErrorSuccess ||
        camera_ptr_->GetFeatureByName("Width", fp) != VmbErrorType::VmbErrorSuccess ||
        fp->SetValue(width_) != VmbErrorType::VmbErrorSuccess ||
        camera_ptr_->GetFeatureByName("Height", fp) != VmbErrorType::VmbErrorSuccess ||
        fp->SetValue(height_) != VmbErrorType::VmbErrorSuccess
        // camera_ptr_->GetFeatureByName("OffsetX", fp) != OK || fp->SetValue(X0) != OK ||
        // camera_ptr_->GetFeatureByName("OffsetY", fp) != OK || fp->SetValue(Y0) != OK)
    )
    {
        throw CameraException(CameraException::NOT_INITIALIZED);
    }
};

void CameraAlvium::shutdown_camera()
{
    camera_ptr_->EndCapture();
    camera_ptr_->FlushQueue();
    camera_ptr_->RevokeAllFrames();

    if (camera_ptr_->Close() != VmbErrorType::VmbErrorSuccess)
    {
        throw CameraException(CameraException::CANT_SHUTDOWN);
    }

    api_vmb_.Shutdown();
}

ICamera* new_camera_device() { return new CameraAlvium(); }

} // namespace camera