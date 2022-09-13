#include "camera_xib.hh"

#include <chrono>
#include "camera_exception.hh"
#include <iostream>
#include <cstring>

namespace camera
{
CameraXib::CameraXib()
    : Camera("xib.ini")
    , device_(nullptr)
{
    name_ = "xiB-64";

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

void CameraXib::init_camera()
{

    auto status = xiOpenDevice(0, &device_);
    if (status != XI_OK)
        throw CameraException(CameraException::NOT_INITIALIZED);

    /* Configure the camera API with given parameters. */
    bind_params();
}

void CameraXib::start_acquisition()
{
    if (xiStartAcquisition(device_) != XI_OK)
        throw CameraException(CameraException::CANT_START_ACQUISITION);
}

void CameraXib::stop_acquisition()
{
    if (xiStopAcquisition(device_) != XI_OK)
        throw CameraException(CameraException::CANT_STOP_ACQUISITION);
}

void CameraXib::shutdown_camera()
{
    auto res = xiCloseDevice(device_);
    if (res != XI_OK)
        throw CameraException(CameraException::CANT_SHUTDOWN);
}

CapturedFramesDescriptor CameraXib::get_frames()
{
    xiGetImage(device_, FRAME_TIMEOUT, &frame_);

    return CapturedFramesDescriptor(frame_.bp);
}

void CameraXib::load_default_params()
{
    /* Fill the frame descriptor. */
    fd_.width = real_width_;
    fd_.height = real_height_;
    pixel_size_ = 5.5f;
    fd_.depth = 1;
    fd_.byteEndian = Endianness::BigEndian;

    /* Custom parameters. */
    gain_ = 0.f;

    downsampling_rate_ = 1;
    downsampling_type_ = XI_SKIPPING;

    img_format_ = XI_RAW8;

    buffer_policy_ = XI_BP_UNSAFE;

    roi_x_ = 0;
    roi_y_ = 0;
    roi_width_ = real_width_;
    roi_height_ = real_height_;

    fd_.width = static_cast<unsigned short>(roi_width_);
    fd_.height = static_cast<unsigned short>(roi_height_);

    exposure_time_ = 0; // free run
}

void CameraXib::load_ini_params()
{
    const boost::property_tree::ptree& pt = get_ini_pt();

    gain_ = pt.get<float>("xib.gain", gain_);

    downsampling_rate_ = pt.get<unsigned int>("xib.downsampling_rate", downsampling_rate_);
    // Updating frame size, taking account downsampling.
    fd_.width = fd_.width / static_cast<unsigned short>(downsampling_rate_);
    fd_.height = fd_.height / static_cast<unsigned short>(downsampling_rate_);

    std::string str;
    str = pt.get<std::string>("xib.downsampling_type", "");
    if (str == "BINNING")
        downsampling_type_ = XI_BINNING;
    else if (str == "SKIPPING")
        downsampling_type_ = XI_SKIPPING;

    str = pt.get<std::string>("xib.format", "");
    if (str == "MONO8")
        img_format_ = XI_MONO8;
    else if (str == "MONO16")
        img_format_ = XI_MONO16;
    else if (str == "RAW8")
        img_format_ = XI_RAW8;
    else if (str == "RAW16")
        img_format_ = XI_RAW16;

    {
        const int tmp_roi_x = pt.get<int>("xib.roi_shift_x", roi_x_);
        const int tmp_roi_y = pt.get<int>("xib.roi_shift_y", roi_y_);
        const int tmp_roi_width = pt.get<int>("xib.roi_width", roi_width_);
        const int tmp_roi_height = pt.get<int>("xib.roi_height", roi_height_);

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

    trigger_src_ = (XI_TRG_SOURCE)pt.get<unsigned long>("xib.trigger_src", XI_TRG_OFF);

    exposure_time_ = pt.get<float>("xib.exposure_time", exposure_time_);
}

void CameraXib::bind_params()
{
    XI_RETURN status = XI_OK;

    const unsigned int name_buffer_size = 32;
    char name[name_buffer_size];

    status = xiGetParamString(device_, XI_PRM_DEVICE_NAME, &name, name_buffer_size);

    // This camera does not support downsampling
    if (!strncmp(name, "CB160MG-LX-X8G3-R2", 18) || !strncmp(name, "CB013MG-LX-X8G3-R2", 18))
    {
        // std::cerr << "Detected camera is Ximea " << name << " which does not support downsampling options\n"
        // << "Skipping parameters setting of downsapling rate and "
        //    "downsampling type\n";
    }
    else
    {
        status = xiSetParamInt(device_, XI_PRM_DOWNSAMPLING, downsampling_rate_);

        if (status != XI_OK)
        {
            // std::cout << "Failed to set downsampling with err code " << status << std::endl;
        }
        status = xiSetParamInt(device_, XI_PRM_DOWNSAMPLING_TYPE, downsampling_type_);

        if (status != XI_OK)
        {
            // std::cout << "Failed to set downsampling type with err code " << status << std::endl;
        }
    }

    status = xiSetParamInt(device_, XI_PRM_IMAGE_DATA_FORMAT, img_format_);

    if (status != XI_OK)
    {
        // std::cout << "Failed to set image data format with err code " << status << std::endl;
    }
    status = xiSetParamInt(device_, XI_PRM_WIDTH, roi_width_);
    if (status != XI_OK)
    {
        // std::cout << "Failed to set roi width with err code " << status << std::endl;
    }
    status = xiSetParamInt(device_, XI_PRM_HEIGHT, roi_height_);
    if (status != XI_OK)
    {
        // std::cout << "Failed to set roi height with err code " << status << std::endl;
    }

    status = xiSetParamInt(device_, XI_PRM_OFFSET_X, roi_x_);
    if (status != XI_OK)
    {
        // std::cout << "Failed to set roi offset x with err code " << status << std::endl;
    }

    status = xiSetParamInt(device_, XI_PRM_OFFSET_Y, roi_y_);
    if (status != XI_OK)
    {
        // std::cout << "Failed to set roi offset y with err code " << status << std::endl;
    }

    status = xiSetParamInt(device_, XI_PRM_BUFFER_POLICY, buffer_policy_);
    if (status != XI_OK)
    {
        // std::cout << "Failed to set buffer policy with err code " << status << std::endl;
    }

    if (exposure_time_)
    {
        status = xiSetParamFloat(device_, XI_PRM_EXPOSURE, 1.0e6f * exposure_time_);
        if (status != XI_OK)
        {
            // std::cout << "Failed to set exposure with err code " << status << std::endl;
        }
    }
    else
    {
        status = xiSetParamFloat(device_, XI_PRM_ACQ_TIMING_MODE, XI_ACQ_TIMING_MODE_FREE_RUN);
        if (status != XI_OK)
        {
            // std::cout << "Failed to set timing mode with err code " << status << std::endl;
        }
    }

    status = xiSetParamFloat(device_, XI_PRM_GAIN, gain_);
    if (status != XI_OK)
    {
        // std::cout << "Failed to set gain with err code " << status << std::endl;
    }

    status = xiSetParamInt(device_, XI_PRM_TRG_SOURCE, trigger_src_);
    if (status != XI_OK)
    {
        // std::cout << "Failed to set trigger source with err code " << status << std::endl;
    }

    /* Update the frame descriptor. */
    if (img_format_ == XI_RAW16 || img_format_ == XI_MONO16)
        fd_.depth = 2;

    name_ = std::string(name);
}

ICamera* new_camera_device() { return new CameraXib(); }
} // namespace camera
