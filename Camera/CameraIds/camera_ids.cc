#include "camera_exception.hh"

#include "camera_ids.hh"

namespace camera
{
CameraIds::CameraIds()
    : Camera("ids.ini")
{
    name_ = "IDS CMOSIS CMV 4000";

    load_default_params();

    if (ini_file_is_open())
    {
        load_ini_params();
        ini_file_.close();
    }

    init_camera();
}

void CameraIds::init_camera()
{
    int cameras_nb = 0;
    int result = is_GetNumberOfCameras(&cameras_nb);

    if (result == IS_SUCCESS && cameras_nb > 0)
    {
        // Assuming there's only one camera connected.
        cam_ = 0;
        result = is_InitCamera(&cam_, nullptr);

        if (result == IS_SUCCESS)
        {
            // Memory allocation
            is_AllocImageMem(cam_,
                             fd_.width,
                             fd_.height,
                             static_cast<int>(fd_.depth) << 3, // depth * 8
                             &frame_,
                             &frame_mem_pid_);
        }
        else
            throw CameraException(CameraException::NOT_INITIALIZED);
    }
    else
        throw CameraException(CameraException::NOT_CONNECTED);

    bind_params();
}

void CameraIds::start_acquisition()
{
    if (is_SetImageMem(cam_, frame_, frame_mem_pid_) != IS_SUCCESS)
        throw CameraException(CameraException::MEMORY_PROBLEM);
}

void CameraIds::stop_acquisition() {}

void CameraIds::shutdown_camera()
{
    if (is_FreeImageMem(cam_, frame_, frame_mem_pid_) != IS_SUCCESS)
        throw CameraException(CameraException::MEMORY_PROBLEM);

    if (is_ExitCamera(cam_) != IS_SUCCESS)
        throw CameraException(CameraException::CANT_SHUTDOWN);
}

CapturedFramesDescriptor CameraIds::get_frames()
{
    if (is_FreezeVideo(cam_, IS_WAIT) != IS_SUCCESS)
        throw CameraException(CameraException::CANT_GET_FRAME);

    return CapturedFramesDescriptor(frame_);
}

void CameraIds::load_default_params()
{
    fd_.width = 2048;
    fd_.height = 2048;
    fd_.depth = PixelDepth::Bits8;
    pixel_size_ = 5.5f;
    fd_.byteEndian = Endianness::LittleEndian;

    exposure_time_ = 49.91f;
    gain_ = 0;
    subsampling_ = -1;
    binning_ = -1;
    color_mode_ = IS_CM_SENSOR_RAW8;
    aoi_x_ = 0;
    aoi_y_ = 0;
    aoi_width_ = 2048;
    aoi_height_ = 2048;
    trigger_mode_ = IS_SET_TRIGGER_OFF;
}

void CameraIds::load_ini_params()
{
    const boost::property_tree::ptree& pt = get_ini_pt();

    fd_.width = pt.get<unsigned short>("ids.sensor_width", fd_.width);
    fd_.height = pt.get<unsigned short>("ids.sensor_height", fd_.height);

    exposure_time_ = pt.get<float>("ids.exposure_time", exposure_time_);
    gain_ = pt.get<int>("ids.gain", gain_);
    format_gain();
    subsampling_ = get_subsampling_mode(pt.get<std::string>("ids.subsampling", ""));
    binning_ = get_binning_mode(pt.get<std::string>("ids.binning", ""));
    color_mode_ = get_color_mode(pt.get<std::string>("ids.image_format", ""));
    aoi_x_ = pt.get<int>("ids.aoi_startx", aoi_x_);
    aoi_y_ = pt.get<int>("ids.aoi_starty", aoi_y_);
    aoi_width_ = pt.get<int>("ids.aoi_width", aoi_width_);
    aoi_height_ = pt.get<int>("ids.aoi_height", aoi_height_);
    trigger_mode_ = get_trigger_mode(pt.get<std::string>("ids.trigger", ""));
}

void CameraIds::bind_params()
{
    const boost::property_tree::ptree& pt = get_ini_pt();
    int status = IS_SUCCESS;

    // Exposure time
    // is_Exposure require a double as third argument.
    double exp = (double)exposure_time_;
    status |= is_Exposure(cam_, IS_EXPOSURE_CMD_SET_EXPOSURE, &exp, sizeof(&exp));

    // Gain
    if (is_SetGainBoost(cam_, IS_GET_SUPPORTED_GAINBOOST) == IS_SET_GAINBOOST_ON)
    {
        status |= is_SetGainBoost(cam_, IS_SET_GAINBOOST_ON);
        status |= is_SetHardwareGain(cam_, gain_, IS_IGNORE_PARAMETER, IS_IGNORE_PARAMETER, IS_IGNORE_PARAMETER);
    }

    // Subsampling
    if (subsampling_ != -1)
        status |= is_SetSubSampling(cam_, subsampling_);

    // Binning
    if (binning_ != -1)
    {
        status |= is_SetBinning(cam_, binning_);
        pixel_size_ *= std::atoi(pt.get<std::string>("ids.binning", "1x1").c_str());
    }

    // Image format/Color mode
    status |= is_SetColorMode(cam_, color_mode_);

    // Area Of Interest
    IS_RECT rectAOI;
    rectAOI.s32X = aoi_x_;
    rectAOI.s32Y = aoi_y_;
    rectAOI.s32Width = aoi_width_;
    rectAOI.s32Height = aoi_height_;

    status |= is_AOI(cam_, IS_AOI_IMAGE_SET_AOI, (void*)&rectAOI, sizeof(rectAOI));

    // Trigger
    status |= is_SetExternalTrigger(cam_, trigger_mode_);

    if (status != IS_SUCCESS)
        throw CameraException(CameraException::CANT_SET_CONFIG);
}

int CameraIds::format_gain() const
{
    if (gain_ < 0 || gain_ > 100)
        return 0;
    return gain_;
}

int CameraIds::get_subsampling_mode(const std::string ui) const
{
    if (ui == "2x2")
        return IS_SUBSAMPLING_2X_VERTICAL | IS_SUBSAMPLING_2X_HORIZONTAL;
    else if (ui == "3x3")
        return IS_SUBSAMPLING_3X_VERTICAL | IS_SUBSAMPLING_3X_HORIZONTAL;
    else if (ui == "4x4")
        return IS_SUBSAMPLING_4X_VERTICAL | IS_SUBSAMPLING_4X_HORIZONTAL;
    else if (ui == "5x5")
        return IS_SUBSAMPLING_5X_VERTICAL | IS_SUBSAMPLING_5X_HORIZONTAL;
    else if (ui == "6x6")
        return IS_SUBSAMPLING_6X_VERTICAL | IS_SUBSAMPLING_6X_HORIZONTAL;
    else if (ui == "8x8")
        return IS_SUBSAMPLING_8X_VERTICAL | IS_SUBSAMPLING_8X_HORIZONTAL;
    else if (ui == "16x16")
        return IS_SUBSAMPLING_16X_VERTICAL | IS_SUBSAMPLING_16X_HORIZONTAL;
    else
        return IS_SUBSAMPLING_DISABLE;
}

int CameraIds::get_binning_mode(const std::string ui)
{
    if (ui == "2x2")
        return IS_BINNING_2X_VERTICAL | IS_BINNING_2X_HORIZONTAL;
    else if (ui == "3x3")
        return IS_BINNING_3X_VERTICAL | IS_BINNING_3X_HORIZONTAL;
    else if (ui == "4x4")
        return IS_BINNING_4X_VERTICAL | IS_BINNING_4X_HORIZONTAL;
    else if (ui == "5x5")
        return IS_BINNING_5X_VERTICAL | IS_BINNING_5X_HORIZONTAL;
    else if (ui == "6x6")
        return IS_BINNING_6X_VERTICAL | IS_BINNING_6X_HORIZONTAL;
    else if (ui == "8x8")
        return IS_BINNING_8X_VERTICAL | IS_BINNING_8X_HORIZONTAL;
    else if (ui == "16x16")
        return IS_BINNING_16X_VERTICAL | IS_BINNING_16X_HORIZONTAL;
    else
        return IS_BINNING_DISABLE;
}

int CameraIds::get_color_mode(const std::string ui)
{
    fd_.depth = PixelDepth::Bits8;
    if (ui == "RAW8")
        return IS_CM_SENSOR_RAW8;
    if (ui == "MONO8")
        return IS_CM_MONO8;

    fd_.depth = PixelDepth::Bits16;
    if (ui == "RAW10")
        return IS_CM_SENSOR_RAW10;
    if (ui == "RAW12")
        return IS_CM_SENSOR_RAW12;
    if (ui == "RAW16")
        return IS_CM_SENSOR_RAW16;
    if (ui == "MONO10")
        return IS_CM_MONO10;
    if (ui == "MONO12")
        return IS_CM_MONO12;
    if (ui == "MONO16")
        return IS_CM_MONO16;

    fd_.depth = PixelDepth::Bits8;
    return IS_CM_SENSOR_RAW8;
}

int CameraIds::get_trigger_mode(const std::string ui) const
{
    if (ui == "Software")
        return IS_SET_TRIGGER_SOFTWARE;
    else if (ui == "HardwareHiLo")
        return IS_SET_TRIGGER_HI_LO;
    else if (ui == "HardwareLoHi")
        return IS_SET_TRIGGER_LO_HI;
    else
        return IS_SET_TRIGGER_OFF;
}

ICamera* new_camera_device() { return new CameraIds(); }
} // namespace camera
