#include "camera_exception.hh"
#include <iostream>

#include "camera_alvium.hh"
#include "camera_logger.hh"

#include <chrono>
#include <span>
namespace camera
{

#define VMB_ERROR(cond) (cond != VmbErrorType::VmbErrorSuccess)

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
    unsigned int n_buffers = 64; // Maybe need to be increase for processing

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
    while (waiting_queue_.empty())
        ;

    unsigned char* buf = waiting_queue_.front();
    waiting_queue_.pop();

    return camera::CapturedFramesDescriptor{buf};
}

void CameraAlvium::load_default_params()
{
    fd_.height = MAX_HEIGHT;
    fd_.width = MAX_WIDTH;
    fd_.depth = 1;
    fd_.byteEndian = Endianness::LittleEndian;

    height_ = MAX_HEIGHT;
    width_ = MAX_WIDTH;

    pixel_format_ = "Mono8";
    reverse_x_ = false;
    reverse_y_ = false;
    gamma_ = 1;
    gain_ = 0;
    lens_shading_value_ = 1;
    intensity_auto_precedence_ = "MinimizeNoise";
    exposure_active_mode_ = "FlashWindow";
    exposure_auto_ = "Off";
    exposure_time_ = 5007.18;
    correction_mode_ = "Off";
    correction_selector_ = "FixedPatternNoiseCorrection";
    contrast_bright_limit_ = 255;
    contrast_dark_limit_ = 0;
    contrast_enable_ = false;
    contrast_shape_ = 1;
    black_level = 0;
    binning_horizontal_ = 1;
    binning_horizontal_mode_ = "Sum";
    binning_vertical_ = 1;
    binning_vertical_mode_ = "Sum";
    adaptive_noise_suppression_factor_ = 1;
}

void CameraAlvium::load_ini_params()
{
    const boost::property_tree::ptree& pt = get_ini_pt();

    width_ = pt.get<unsigned short>("alvium.width", width_);
    height_ = pt.get<unsigned short>("alvium.height", height_);

    fd_.width = width_;
    fd_.height = height_;

    pixel_format_ = pt.get<std::string>("alvium.PixelFormat", pixel_format_);
    reverse_x_ = pt.get<bool>("alvium.ReverseX", reverse_x_);
    reverse_y_ = pt.get<bool>("alvium.ReverseY", reverse_y_);
    gamma_ = pt.get<double>("alvium.Gamma", gamma_);
    gain_ = pt.get<double>("alvium.Gain", gain_);
    lens_shading_value_ = pt.get<double>("alvium.LensShadingValue", lens_shading_value_);
    intensity_auto_precedence_ = pt.get<std::string>("alvium.IntensityAutoPrecedence", intensity_auto_precedence_);
    exposure_active_mode_ = pt.get<std::string>("alvium.ExposureActiveMode", exposure_active_mode_);
    exposure_auto_ = pt.get<std::string>("alvium.ExposureAuto", exposure_auto_);
    exposure_time_ = pt.get<double>("alvium.ExposureTime", exposure_time_);
    correction_mode_ = pt.get<std::string>("alvium.CorrectionMode", correction_mode_);
    correction_selector_ = pt.get<std::string>("alvium.CorrectionSelector", correction_selector_);
    contrast_bright_limit_ = pt.get<VmbInt64_t>("alvium.ContrastBrightLimit", contrast_bright_limit_);
    contrast_dark_limit_ = pt.get<VmbInt64_t>("alvium.ContrastDarkLimit", contrast_dark_limit_);
    contrast_enable_ = pt.get<bool>("alvium.ContrastEnable", contrast_enable_);
    contrast_shape_ = pt.get<VmbInt64_t>("alvium.ContrastShape", contrast_shape_);
    black_level = pt.get<double>("alvium.BlackLevel", black_level);
    binning_horizontal_ = pt.get<VmbInt64_t>("alvium.BinningHorizontal", binning_horizontal_);
    binning_horizontal_mode_ = pt.get<std::string>("alvium.BinningHorizontalMode", binning_horizontal_mode_);
    binning_vertical_ = pt.get<VmbInt64_t>("alvium.BinningVertical", binning_vertical_);
    binning_vertical_mode_ = pt.get<std::string>("alvium.BinningVerticalMode", binning_vertical_mode_);
    adaptive_noise_suppression_factor_ =
        pt.get<double>("alvium.AdaptiveNoiseSuppressionFactor", adaptive_noise_suppression_factor_);
};

#define VMB_IS_SET_OK(name, value) VMB_ERROR(camera_ptr_->GetFeatureByName(name, fp)) || VMB_ERROR(fp->SetValue(value))

void CameraAlvium::bind_params()
{

    VmbCPP::FeaturePtr fp; // Generic feature pointer use inside VMB_IS_SET_OK macro

    if (VMB_IS_SET_OK("Width", width_) || VMB_IS_SET_OK("Height", height_) || VMB_IS_SET_OK("PixelFormat", "Mono8") ||
        VMB_IS_SET_OK("ReverseX", reverse_x_) || VMB_IS_SET_OK("ReverseY", reverse_y_) ||
        VMB_IS_SET_OK("Gamma", gamma_) || VMB_IS_SET_OK("Gain", gain_) ||
        VMB_IS_SET_OK("LensShadingValue", lens_shading_value_) ||
        VMB_IS_SET_OK("IntensityAutoPrecedence", intensity_auto_precedence_.c_str()) ||
        VMB_IS_SET_OK("ExposureActiveMode", exposure_active_mode_.c_str()) ||
        VMB_IS_SET_OK("ExposureAuto", exposure_auto_.c_str()) || VMB_IS_SET_OK("ExposureTime", exposure_time_) ||
        VMB_IS_SET_OK("CorrectionMode", correction_mode_.c_str()) ||
        VMB_IS_SET_OK("CorrectionSelector", correction_selector_.c_str()) ||
        VMB_IS_SET_OK("ContrastBrightLimit", contrast_bright_limit_) ||
        VMB_IS_SET_OK("ContrastDarkLimit", contrast_dark_limit_) || VMB_IS_SET_OK("ContrastEnable", contrast_enable_) ||
        VMB_IS_SET_OK("ContrastShape", contrast_shape_) || VMB_IS_SET_OK("BlackLevel", black_level) ||
        VMB_IS_SET_OK("BinningHorizontal", binning_horizontal_) ||
        VMB_IS_SET_OK("BinningHorizontalMode", binning_horizontal_mode_.c_str()) ||
        VMB_IS_SET_OK("BinningVertical", binning_vertical_) ||
        VMB_IS_SET_OK("BinningVerticalMode", binning_vertical_mode_.c_str()) ||
        VMB_IS_SET_OK("AdaptiveNoiseSuppressionFactor", adaptive_noise_suppression_factor_))
    {
        Logger::camera()->info("Failed set some feature!");
        camera_ptr_->Close();
        api_vmb_.Shutdown();
        throw CameraException(CameraException::NOT_INITIALIZED);
    }
};

void CameraAlvium::shutdown_camera()
{
    bool queue_err = VMB_ERROR(camera_ptr_->FlushQueue());
    bool frames_err = VMB_ERROR(camera_ptr_->RevokeAllFrames());
    bool close_err = VMB_ERROR(camera_ptr_->Close());

    if (queue_err || frames_err || close_err)
        throw CameraException(CameraException::CANT_SHUTDOWN);

    api_vmb_.Shutdown();
}

ICamera* new_camera_device() { return new CameraAlvium(); }

} // namespace camera