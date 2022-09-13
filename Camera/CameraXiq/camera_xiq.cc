#include "camera_exception.hh"
#include <iostream>

#include "camera_xiq.hh"

#include <chrono>

namespace camera
{
CameraXiq::CameraXiq()
    : Camera("xiq.ini")
    , device_(nullptr)
{
    name_ = "XIQ MQ042MG-CM";

    load_default_params();
    if (ini_file_is_open())
        load_ini_params();

    if (ini_file_is_open())
        ini_file_.close();

    frame_.size = sizeof(XI_IMG);
    frame_.bp = nullptr;
    frame_.bp_size = 0;

    init_camera();
}

void CameraXiq::init_camera()
{
    if (xiOpenDevice(0, &device_) != XI_OK)
        throw CameraException(CameraException::NOT_INITIALIZED);

    /* Configure the camera API with given parameters. */
    bind_params();
}

void CameraXiq::start_acquisition()
{
    if (xiStartAcquisition(device_) != XI_OK)
        throw CameraException(CameraException::CANT_START_ACQUISITION);
}

void CameraXiq::stop_acquisition()
{
    if (xiStopAcquisition(device_) != XI_OK)
        throw CameraException(CameraException::CANT_STOP_ACQUISITION);
}

void CameraXiq::shutdown_camera()
{
    if (xiCloseDevice(device_) != XI_OK)
        throw CameraException(CameraException::CANT_SHUTDOWN);
}

CapturedFramesDescriptor CameraXiq::get_frames()
{
    xiGetImage(device_, FRAME_TIMEOUT, &frame_);

    return CapturedFramesDescriptor(frame_.bp);
}

void CameraXiq::load_default_params()
{
    /* Fill the frame descriptor. */
    fd_.width = 2048;
    fd_.height = 2048;
    pixel_size_ = 5.5f;
    fd_.depth = 1;
    fd_.byteEndian = Endianness::BigEndian;

    /* Custom parameters. */
    gain_ = 0.f;

    downsampling_rate_ = 1;
    downsampling_type_ = XI_BINNING;

    img_format_ = XI_RAW8;

    buffer_policy_ = XI_BP_UNSAFE;

    roi_x_ = 0;
    roi_y_ = 0;
    roi_width_ = 2048;
    roi_height_ = 2048;

    exposure_time_ = 0.0166666666666667f; // 1 / 60;
}

void CameraXiq::load_ini_params()
{
    const boost::property_tree::ptree& pt = get_ini_pt();

    gain_ = pt.get<float>("xiq.gain", gain_);

    downsampling_rate_ = pt.get<unsigned int>("xiq.downsampling_rate", downsampling_rate_);
    // Updating frame size, taking account downsampling.
    fd_.width = fd_.width / static_cast<unsigned short>(downsampling_rate_);
    fd_.height = fd_.height / static_cast<unsigned short>(downsampling_rate_);

    std::string str;
    str = pt.get<std::string>("xiq.downsampling_type", "");
    if (str == "BINNING")
        downsampling_type_ = XI_BINNING;
    else if (str == "SKIPPING")
        downsampling_type_ = XI_SKIPPING;

    str = pt.get<std::string>("xiq.format", "");
    if (str == "MONO8")
        img_format_ = XI_MONO8;
    else if (str == "MONO16")
        img_format_ = XI_MONO16;
    else if (str == "RAW8")
        img_format_ = XI_RAW8;
    else if (str == "RAW16")
        img_format_ = XI_RAW16;

    {
        const int tmp_roi_x = pt.get<int>("xiq.roi_x", roi_x_);
        const int tmp_roi_y = pt.get<int>("xiq.roi_y", roi_y_);
        const int tmp_roi_width = pt.get<int>("xiq.roi_width", roi_width_);
        const int tmp_roi_height = pt.get<int>("xiq.roi_height", roi_height_);

        /* Making sure ROI settings are valid.
         * Keep in mind that ROI area can't be larger than the
         * initial frame's area (after downsampling!). */
        if (tmp_roi_width > 0 && tmp_roi_height > 0 && tmp_roi_x < fd_.width && tmp_roi_y < fd_.height &&
            tmp_roi_width <= fd_.width && tmp_roi_height <= fd_.height)
        {
            roi_x_ = tmp_roi_x;
            roi_y_ = tmp_roi_y;
            roi_width_ = tmp_roi_width;
            roi_height_ = tmp_roi_height;

            // Don't forget to update the frame descriptor!
            fd_.width = static_cast<unsigned short>(roi_width_);
            fd_.height = static_cast<unsigned short>(roi_height_);
        }
        else
        {
            // std::cerr << "[CAMERA] Invalid ROI settings, ignoring ROI." << std::endl;
        }
    }

    trigger_src_ = (XI_TRG_SOURCE)pt.get<unsigned long>("xiq.trigger_src", XI_TRG_OFF);

    exposure_time_ = pt.get<float>("xiq.exposure_time", exposure_time_);
}

void CameraXiq::bind_params()
{
    XI_RETURN status = XI_OK;

    const unsigned int name_buffer_size = 32;
    char name[name_buffer_size];

    status |= xiGetParamString(device_, XI_PRM_DEVICE_NAME, &name, name_buffer_size);

    status |= xiSetParamInt(device_, XI_PRM_DOWNSAMPLING, downsampling_rate_);
    status |= xiSetParamInt(device_, XI_PRM_DOWNSAMPLING_TYPE, downsampling_type_);
    status |= xiSetParamInt(device_, XI_PRM_IMAGE_DATA_FORMAT, img_format_);
    status |= xiSetParamInt(device_, XI_PRM_OFFSET_X, roi_x_);
    status |= xiSetParamInt(device_, XI_PRM_OFFSET_Y, roi_y_);
    status |= xiSetParamInt(device_, XI_PRM_WIDTH, roi_width_);
    status |= xiSetParamInt(device_, XI_PRM_HEIGHT, roi_height_);

    status |= xiSetParamInt(device_, XI_PRM_BUFFER_POLICY, buffer_policy_);

    status |= xiSetParamFloat(device_, XI_PRM_EXPOSURE, 1.0e6f * exposure_time_);

    status |= xiSetParamFloat(device_, XI_PRM_GAIN, gain_);

    status |= xiSetParamInt(device_, XI_PRM_TRG_SOURCE, trigger_src_);

    if (status != XI_OK)
        throw CameraException(CameraException::CANT_SET_CONFIG);

    /* Update the frame descriptor. */
    if (img_format_ == XI_RAW16 || img_format_ == XI_MONO16)
        fd_.depth = 2;

    name_ = std::string(name);
}

ICamera* new_camera_device() { return new CameraXiq(); }
} // namespace camera
