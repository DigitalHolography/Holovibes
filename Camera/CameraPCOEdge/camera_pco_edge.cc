#include "camera_pco_edge.hh"
#include <camera_exception.hh>

#include <PCO_err.h>
#include <sc2_defs.h>

namespace camera
{
  ICamera* new_camera_device()
  {
    return new CameraPCOEdge();
  }

  CameraPCOEdge::CameraPCOEdge()
    : CameraPCO("edge.ini", CAMERATYPE_PCO_EDGE_USB3)
  {
    name_ = "edge";
    load_default_params();
    if (ini_file_is_open())
      load_ini_params();
  }

  CameraPCOEdge::~CameraPCOEdge()
  {
  }

  static void initialize_pco_signal(PCO_Signal& sig, WORD index)
  {
    sig.wSize = sizeof(PCO_Signal);
    sig.wSignalNum = index; // { 0, 1, 2, 3}
    sig.wEnabled = 1; // on (activated)
    sig.wType = 1; // signal type : TTL
    sig.wPolarity = 1; // low level active
    sig.wFilterSetting = 1; // No signal filter
    sig.wSelected = 0; // standard signal

    /* TODO : I did NOT understand those! */
    sig.dwParameter[0] = 0;
    sig.dwParameter[1] = 0;
    sig.dwParameter[2] = 0;
    sig.dwParameter[3] = 0;

    sig.dwSignalFunctionality[0] = 0;
    sig.dwSignalFunctionality[1] = 0;
    sig.dwSignalFunctionality[2] = 0;
    sig.dwSignalFunctionality[3] = 0;
  }

  void CameraPCOEdge::load_default_params()
  {
    /* Various camera parameters. */
    exposure_time_ = 0.024f;

    triggermode_ = 0;

    framerate_ = 30 * 10e3;
    framerate_mode_ = 1;

    hz_binning_ = 1;
    vt_binning_ = 1;

    p0_x_ = 0;
    p0_y_ = 0;
    p1_x_ = CameraPCO::get_actual_res_x();
    p1_y_ = CameraPCO::get_actual_res_y();

    pixel_rate_ = 12 * 10e6;

    conversion_factor_ = 460;

    ad_converters_ = 1;

    timeouts_[0] = timeouts_[1] = timeouts_[2] =  50;

    initialize_pco_signal(io_1_conf, 0);
    initialize_pco_signal(io_2_conf, 1);
    initialize_pco_signal(io_3_conf, 2);
    initialize_pco_signal(io_4_conf, 3);

    /* Fill frame descriptor const values. */
    desc_.depth = 2;
    desc_.endianness = LITTLE_ENDIAN;
    desc_.pixel_size = 6.45f;
    desc_.width = 2048;
    desc_.height = 2048;
  }

  void CameraPCOEdge::load_ini_params()
  {
    const boost::property_tree::ptree& pt = get_ini_pt();

    exposure_time_ = pt.get<float>("pco-edge.exposure_time", exposure_time_);
    triggermode_ = pt.get<WORD>("pco-edge.trigger_mode", triggermode_);
    // TODO : Check if removing this solves the triggering problem!
    //if (triggermode_ < 3)
    //  triggermode_ = 0;
    framerate_ = pt.get<DWORD>("pco-edge.framerate", framerate_) * 10e3;
    framerate_mode_ = pt.get<WORD>("pco-edge.framerate_mode", framerate_mode_);
    p0_x_ = pt.get<WORD>("pco-edge.p0_x", p0_x_);
    p0_y_ = pt.get<WORD>("pco-edge.p0_y", p0_y_);
    p1_x_ = pt.get<WORD>("pco-edge.p1_x", p1_x_);
    p1_y_ = pt.get<WORD>("pco-edge.p1_y", p1_y_);
    hz_binning_ = pt.get<WORD>("pco-edge.binning_hz", hz_binning_);
    vt_binning_ = pt.get<WORD>("pco-edge.binning_vt", vt_binning_);
    pixel_rate_ = pt.get<DWORD>("pco-edge.pixel_rate", pixel_rate_) * 10e6;
    conversion_factor_ = static_cast<WORD>(
        pt.get<float>("pco-edge.conversion_factor", conversion_factor_) * 100);
    ad_converters_ = pt.get<WORD>("pco-edge.adc", ad_converters_);
    timeouts_[0] = pt.get<unsigned int>("pco-edge.timeout_command", timeouts_[0]);
    timeouts_[1] = pt.get<unsigned int>("pco-edge.timeout_img_acq", timeouts_[1]);
  }

  void CameraPCOEdge::bind_params()
  {
    int status = PCO_NOERROR;

    status |= PCO_ResetSettingsToDefault(device_);

    status |= PCO_SetTimeouts(device_, timeouts_, 2);
    status |= PCO_SetSensorFormat(device_, 0);
    status |= PCO_SetADCOperation(device_, ad_converters_);
    status |= PCO_SetConversionFactor(device_, ad_converters_);
    status |= PCO_SetBinning(device_, hz_binning_, vt_binning_);
    status |= PCO_SetROI(device_, p0_x_, p0_y_, p1_x_, p1_y_);
    status |= PCO_SetPixelRate(device_, pixel_rate_);
    status |= PCO_SetTriggerMode(device_, triggermode_);
    status |= PCO_SetNoiseFilterMode(device_, 0);

    {
      /* Convert exposure time in milliseconds. */
      exposure_time_ *= 1e3;

      /* base_time : 0x0002 = ms, 0x0001 = us, 0x0000 = ns */
      WORD base_time;

      for (base_time = 0x0002; base_time > 0 && exposure_time_ < 1.0f; --base_time)
        exposure_time_ *= 1e3;

      status |= PCO_SetDelayExposureTime(device_, 0, static_cast<DWORD>(exposure_time_), 0, base_time);
    }

    {
      WORD fps_change_status;
      DWORD exp_time = static_cast<DWORD>(exposure_time_ * 10e9);
      int res = PCO_SetFrameRate(device_, &fps_change_status, 0, &framerate_, &exp_time);
      // DEBUG : Not sure the camera actually implements this function.
      if (res != PCO_NOERROR)
        std::cerr << "PCO_SetFrameRate() call failed : error code " << res << std::endl;
      // ! DEBUG
      exposure_time_ = static_cast<float>(exp_time) * 10e-9; // Convert from nanoseconds
    }

    if (status != PCO_NOERROR)
      throw CameraException(CameraException::CANT_SET_CONFIG);
  }

}