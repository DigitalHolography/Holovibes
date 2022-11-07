#include <iostream>
#include <chrono>
#include <cstring>
#include "spdlog/spdlog.h"

#include "camera_hamamatsu.hh"
#include "camera_logger.hh"

namespace camera
{
CameraHamamatsu::CameraHamamatsu()
    : Camera("hamamatsu.ini")
{
    name_ = "MISSING NAME";

    load_default_params();

    if (ini_file_is_open())
    {
        load_ini_params();
        ini_file_.close();
    }

    init_camera();
}

void CameraHamamatsu::init_camera()
{
    // initialize DCAM-API
    DCAMAPI_INIT param_init;
    std::memset(&param_init, 0, sizeof(DCAMAPI_INIT));
    param_init.size = sizeof(param_init); // This line is required by the API
    if (dcamapi_init(&param_init) != DCAMERR_SUCCESS || param_init.iDeviceCount <= 0)
    {
        throw CameraException(CameraException::NOT_CONNECTED);
    }

    // Trying to connect the first available Camera
    int32 nDevice = param_init.iDeviceCount;

    DCAMDEV_OPEN param_open;
    std::memset(&param_open, 0, sizeof(param_open));
    param_open.size = sizeof(param_open); // This line is required by the API
    for (int32 iDevice = 0; iDevice < nDevice; iDevice++)
    {
        param_open.index = iDevice;
        // Opening camera
        if (dcamdev_open(&param_open) != DCAMERR_SUCCESS)
        {
            continue;
        }

        hdcam_ = param_open.hdcam;

        // Gets and sets camera model name
        retrieve_camera_name();
        Logger::camera()->info("Connected to {}", name_);

        // Binding parameters
        try
        {
            bind_params();
        }
        catch (const CameraException& e)
        {
            // this closes Camera and releases DCAM driver resources
            shutdown_camera();
            throw e;
        }

        // Gets and sets camera pixel size in bytes
        retrieve_pixel_depth();

        // Allocates buffer for image reception
        allocate_host_frame_buffer();

        return; // SUCCESS
    }

    // Could not connect to any camera
    dcamapi_uninit();
    throw CameraException(CameraException::NOT_CONNECTED);
}

void CameraHamamatsu::retrieve_camera_name()
{
    constexpr int buf_size = 256;
    char buf[buf_size];

    DCAMDEV_STRING param_getstring;
    std::memset(&param_getstring, 0, sizeof(param_getstring));
    param_getstring.size = sizeof(param_getstring); // This line is required by the API
    param_getstring.iString = DCAM_IDSTR_MODEL;
    param_getstring.text = buf;
    param_getstring.textbytes = buf_size;
    if (dcamdev_getstring(hdcam_, &param_getstring) == DCAMERR_SUCCESS)
    {
        name_ = buf; // This calls the std::string constructor on buf, which is
                     // equal to param_getstring.text
    }
}

void CameraHamamatsu::retrieve_pixel_depth()
{
    double bits_per_channel;
    dcamprop_getvalue(hdcam_, DCAM_IDPROP_BITSPERCHANNEL, &bits_per_channel);
    fd_.depth = bits_per_channel / 8;
}

void CameraHamamatsu::allocate_host_frame_buffer()
{
    auto frame_size = fd_.get_frame_res();
    output_frame_ = std::make_unique<unsigned short[]>(frame_size);
}

// Should be called AFTER the camera has been setup ( init_camera() )
void CameraHamamatsu::set_frame_acq_info()
{
    std::memset(&dcam_frame_acq_info_, 0, sizeof(dcam_frame_acq_info_));

    // This separation is used to differentiate between API reserved fields and
    // Host writeable fields

    dcam_frame_acq_info_.size = sizeof(dcam_frame_acq_info_); // Line required by the API
    dcam_frame_acq_info_.iFrame = -1;                         // -1 to retrieve the latest captured image
    dcam_frame_acq_info_.buf = output_frame_.get();           // Pointer to host memory where the image will be copied
    dcam_frame_acq_info_.rowbytes = fd_.width * fd_.depth;    // Row size in bytes
    dcam_frame_acq_info_.width = fd_.width;
    dcam_frame_acq_info_.height = fd_.height;
    dcam_frame_acq_info_.left = 0;
    dcam_frame_acq_info_.top = 0;
}

void CameraHamamatsu::set_wait_info()
{
    std::memset(&dcam_wait_info_, 0, sizeof(dcam_wait_info_));

    dcam_wait_info_.size = sizeof(dcam_wait_info_);           // Line required by the API
    dcam_wait_info_.eventmask = DCAMWAIT_CAPEVENT_FRAMEREADY; // Waiting for event
    dcam_wait_info_.timeout = camera::FRAME_TIMEOUT;          // This field should be in milliseconds
}

void CameraHamamatsu::get_event_waiter_handle()
{
    DCAMWAIT_OPEN param_waitopen;
    std::memset(&param_waitopen, 0, sizeof(param_waitopen));
    param_waitopen.size = sizeof(param_waitopen);
    param_waitopen.hdcam = hdcam_;

    if (dcamwait_open(&param_waitopen) != DCAMERR_SUCCESS ||
        !(param_waitopen.supportevent & DCAMWAIT_CAPEVENT_FRAMEREADY))
    {
        // The Event Wait system is needed to capture frames
        // It would typically be used to wait for an FRAME_READY event
        // The supportevent field is set after the call to dcamwait_open and
        // contains the flags representing the supported events

        throw CameraException(CameraException::CANT_START_ACQUISITION);
    }
    else
    {
        hwait_ = param_waitopen.hwait;
    }
}

void CameraHamamatsu::start_acquisition()
{
    // allocate capturing buffer
    if (dcambuf_alloc(hdcam_, circ_buffer_frame_count_) != DCAMERR_SUCCESS)
    {
        throw CameraException(CameraException::CANT_START_ACQUISITION);
    }

    // Start acquisition in "SEQUENCE" mode (camera writes frames in its
    // internal circular buffer until dcamcap_stop() is called)
    if (dcamcap_start(hdcam_, DCAMCAP_START_SEQUENCE) != DCAMERR_SUCCESS)
    {
        throw CameraException(CameraException::CANT_START_ACQUISITION);
    }

    // Might throw a CameraException like the lines above
    get_event_waiter_handle();

    // Setting struct members values needed for acquisition
    set_frame_acq_info();
    set_wait_info();
}

void CameraHamamatsu::stop_acquisition()
{
    // stop capturing
    dcamcap_stop(hdcam_);

    // release capturing buffer
    dcambuf_release(hdcam_);

    // close event waiter
    dcamwait_close(hwait_);
}

void CameraHamamatsu::shutdown_camera()
{
    // close HDCAM handle
    dcamdev_close(hdcam_);

    // terminate DCAM-API
    dcamapi_uninit();
}

CapturedFramesDescriptor CameraHamamatsu::get_frames()
{
    // Waiting for frame to be ready
    auto err_code = dcamwait_start(hwait_, &dcam_wait_info_);
    if (err_code != DCAMERR_SUCCESS)
    {
        // Log err here
        if (err_code == DCAMERR_TIMEOUT)
        {
            Logger::camera()->error("Frame acquisition timed out");
        }
        throw CameraException(CameraException::CANT_GET_FRAME);
    }

    err_code = dcambuf_copyframe(hdcam_, &dcam_frame_acq_info_);
    if (err_code != DCAMERR_SUCCESS)
    {
        throw CameraException(CameraException::CANT_GET_FRAME);
    }

    return CapturedFramesDescriptor(output_frame_.get());
}

void CameraHamamatsu::load_default_params()
{
    fd_.width = MAX_WIDTH;
    fd_.height = MAX_WIDTH;
    fd_.depth = 2;
    fd_.byteEndian = Endianness::LittleEndian;

    pixel_size_ = 6.5f;

    exposure_time_ = 1000;

    srcox_ = 0;
    srcoy_ = 0;

    binning_ = 1;

    ext_trig_ = false;
    circ_buffer_frame_count_ = 64;
    trig_mode_ = DCAMPROP_TRIGGER_MODE__NORMAL;
    trig_connector_ = DCAMPROP_TRIGGER_CONNECTOR__BNC;
    trig_polarity_ = DCAMPROP_TRIGGERPOLARITY__NEGATIVE;
    trig_active_ = DCAMPROP_TRIGGERACTIVE__EDGE;

    readoutspeed_ = DCAMPROP_READOUTSPEED__FASTEST;
}

void CameraHamamatsu::load_ini_params()
{
    /* Use the default value in case of fail. */
    const boost::property_tree::ptree& pt = get_ini_pt();

    name_ = pt.get<std::string>("hamamatsu.name", name_);

    fd_.width = pt.get<unsigned short>("hamamatsu.roi_width", fd_.width);
    fd_.height = pt.get<unsigned short>("hamamatsu.roi_height", fd_.height);
    srcox_ = pt.get<long>("hamamatsu.roi_startx", srcox_);
    srcoy_ = pt.get<long>("hamamatsu.roi_starty", srcoy_);

    exposure_time_ = pt.get<float>("hamamatsu.exposure_time", exposure_time_);

    binning_ = pt.get<unsigned short>("hamamatsu.binning", binning_);

    ext_trig_ = pt.get<bool>("hamamatsu.ext_trig", ext_trig_);

    circ_buffer_frame_count_ = pt.get<int32>("hamamatsu.circ_buffer_frame_count", circ_buffer_frame_count_);

    std::string trig_mode = pt.get<std::string>("hamamatsu.trig_mode", "");
    if (trig_mode == "NORMAL")
        trig_mode_ = DCAMPROP_TRIGGER_MODE__NORMAL;
    else if (trig_mode == "START")
        trig_mode_ = DCAMPROP_TRIGGER_MODE__START;

    std::string trig_connector = pt.get<std::string>("hamamatsu.trig_connector", "");
    if (trig_connector == "INTERFACE")
        trig_connector_ = DCAMPROP_TRIGGER_CONNECTOR__INTERFACE;
    else if (trig_connector == "BNC")
        trig_connector_ = DCAMPROP_TRIGGER_CONNECTOR__BNC;

    std::string trig_polarity = pt.get<std::string>("hamamatsu.trig_polarity", "");
    if (trig_polarity == "POSITIVE")
        trig_polarity_ = DCAMPROP_TRIGGERPOLARITY__POSITIVE;
    else if (trig_polarity == "NEGATIVE")
        trig_polarity_ = DCAMPROP_TRIGGERPOLARITY__NEGATIVE;

    std::string readoutspeed = pt.get<std::string>("hamamatsu.readoutspeed", "");
    if (readoutspeed == "SLOWEST")
        readoutspeed_ = DCAMPROP_READOUTSPEED__SLOWEST;
    else if (readoutspeed == "FASTEST")
        readoutspeed_ = DCAMPROP_READOUTSPEED__FASTEST;

    std::string trig_active = pt.get<std::string>("hamamatsu.trig_active", "");
    if (trig_active == "EDGE")
        trig_active_ = DCAMPROP_TRIGGERACTIVE__EDGE;
    else if (trig_active == "LEVEL")
        trig_active_ = DCAMPROP_TRIGGERACTIVE__LEVEL;
    else if (trig_active == "SYNCREADOUT")
        trig_active_ = DCAMPROP_TRIGGERACTIVE__SYNCREADOUT;
}

void CameraHamamatsu::bind_params()
{
    // Hardcoded max width and height of the camera
    // Should change with the model
    if (fd_.width != MAX_WIDTH || fd_.height != MAX_HEIGHT) // SUBARRAY
    {
        dcamprop_setvalue(hdcam_, DCAM_IDPROP_SUBARRAYMODE, DCAMPROP_MODE__ON);
        dcamprop_setvalue(hdcam_, DCAM_IDPROP_SUBARRAYHSIZE, fd_.width);
        dcamprop_setvalue(hdcam_, DCAM_IDPROP_SUBARRAYVSIZE, fd_.height);
        dcamprop_setvalue(hdcam_, DCAM_IDPROP_SUBARRAYHPOS, srcox_);
        dcamprop_setvalue(hdcam_, DCAM_IDPROP_SUBARRAYVPOS, srcoy_);
    }

    if (dcamprop_setvalue(hdcam_, DCAM_IDPROP_EXPOSURETIME, exposure_time_ / 1E6) != DCAMERR_SUCCESS)
        throw CameraException(CameraException::CANT_SET_CONFIG);

    if (dcamprop_setvalue(hdcam_, DCAM_IDPROP_BINNING, binning_) != DCAMERR_SUCCESS)
        throw CameraException(CameraException::CANT_SET_CONFIG);
    fd_.width /= binning_;
    fd_.height /= binning_;

    if (dcamprop_setvalue(hdcam_,
                          DCAM_IDPROP_TRIGGERSOURCE,
                          ext_trig_ ? DCAMPROP_TRIGGERSOURCE__EXTERNAL : DCAMPROP_TRIGGERSOURCE__INTERNAL) !=
        DCAMERR_SUCCESS)
        throw CameraException(CameraException::CANT_SET_CONFIG);
    if (dcamprop_setvalue(hdcam_, DCAM_IDPROP_TRIGGER_CONNECTOR, trig_connector_) != DCAMERR_SUCCESS)
        throw CameraException(CameraException::CANT_SET_CONFIG);
    if (dcamprop_setvalue(hdcam_, DCAM_IDPROP_TRIGGERPOLARITY, trig_polarity_) != DCAMERR_SUCCESS)
        throw CameraException(CameraException::CANT_SET_CONFIG);
    if (dcamprop_setvalue(hdcam_, DCAM_IDPROP_READOUTSPEED, readoutspeed_) != DCAMERR_SUCCESS)
        throw CameraException(CameraException::CANT_SET_CONFIG);
    if (dcamprop_setvalue(hdcam_, DCAM_IDPROP_TRIGGERACTIVE, trig_active_) != DCAMERR_SUCCESS)
        throw CameraException(CameraException::CANT_SET_CONFIG);
}

ICamera* new_camera_device() { return new CameraHamamatsu(); }
} // namespace camera
