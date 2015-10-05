#include "camera_pco_edge.hh"
#include <camera_exception.hh>
#include <boost/lexical_cast.hpp>

#include <PCO_err.h>
#include <sc2_defs.h>

namespace camera
{
  // DEBUG : remove these functions later

  static void soft_assert(const char* func, int& status)
  {
    if (status != PCO_NOERROR)
      std::cout << func << "() call failed : error " << status << std::endl;
    status = PCO_NOERROR;
  }

  // ! DEBUG

  static void initialize_pco_signal(PCO_Signal& sig, const WORD index)
  {
    sig.wSize = sizeof(PCO_Signal);
    sig.wSignalNum = index; // { 0, 1, 2, 3}
    sig.wEnabled = 1; // on (activated)
    sig.wType = 1; // signal type : TTL
    sig.wPolarity = 1; // low level active
    sig.wFilterSetting = 1; // no signal filter
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

  static void load_pco_signal_params(const boost::property_tree::ptree& pt,
      PCO_Signal& sig, const WORD sig_index)
  {
    // Field format for signal number X :   X_nameofthefield
    std::string prefix = "pco-edge." + boost::lexical_cast<std::string>(sig_index) + '_';

    sig.wEnabled = pt.get<WORD>(prefix + "state", sig.wEnabled);
    sig.wType = pt.get<WORD>(prefix + "type", sig.wType);
    sig.wPolarity = pt.get<WORD>(prefix + "polarity", sig.wPolarity);
    sig.wFilterSetting = pt.get<WORD>(prefix + "filter", sig.wFilterSetting);
    sig.wSelected = pt.get<WORD>(prefix + "subindex", sig.wSelected);

    /* TODO : Understand timing parameters (dwParameters) and
    ** signal functionalities (swSignalFunctionality) arrays.
    ** For now, they keep their initial values : zero-filled.
    */
  }

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

  void CameraPCOEdge::load_default_params()
  {
    /* Various camera parameters. */
    exposure_time_ = 0.024f;

    triggermode_ = 0;

    framerate_ = 30 * 1e3;
    framerate_mode_ = 1;

    hz_binning_ = 1;
    vt_binning_ = 1;

    p0_x_ = 1;
    p0_y_ = 1;
    p1_x_ = 2048;
    p1_y_ = 2048;

    pixel_rate_ = 110 * 1e6;

    conversion_factor_ = 46;

    timeouts_[0] = timeouts_[1] = timeouts_[2] = 50;

    initialize_pco_signal(io_0_conf, 0);
    initialize_pco_signal(io_1_conf, 1);
    initialize_pco_signal(io_2_conf, 2);
    initialize_pco_signal(io_3_conf, 3);

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

    framerate_ = pt.get<DWORD>("pco-edge.framerate", framerate_) * 1e3;
    framerate_mode_ = pt.get<WORD>("pco-edge.framerate_mode", framerate_mode_);

    hz_binning_ = pt.get<WORD>("pco-edge.binning_hz", hz_binning_);
    vt_binning_ = pt.get<WORD>("pco-edge.binning_vt", vt_binning_);
   
    p0_x_ = pt.get<WORD>("pco-edge.roi_x", p0_x_);
    p0_y_ = pt.get<WORD>("pco-edge.roi_y", p0_y_);
    p1_x_ = pt.get<WORD>("pco-edge.roi_width", p1_x_) + p0_x_;
    p1_y_ = pt.get<WORD>("pco-edge.roi_height", p1_y_) + p0_y_;

    pixel_rate_ = pt.get<DWORD>("pco-edge.pixel_rate", pixel_rate_) * 1e6;

    conversion_factor_ = static_cast<WORD>(
        pt.get<float>("pco-edge.conversion_factor", conversion_factor_) * 100);

    timeouts_[0] = pt.get<unsigned int>("pco-edge.timeout_command", timeouts_[0]) * 1e3;
    timeouts_[1] = pt.get<unsigned int>("pco-edge.timeout_img_acq", timeouts_[1]) * 1e3;

    load_pco_signal_params(pt, io_0_conf, 0);
    load_pco_signal_params(pt, io_1_conf, 1);
    load_pco_signal_params(pt, io_2_conf, 2);
    load_pco_signal_params(pt, io_3_conf, 3);
  }

  void CameraPCOEdge::bind_params()
  {
    int status = PCO_NOERROR;

    status |= PCO_ResetSettingsToDefault(device_);
    soft_assert("PCO_ResetSettingsToDefault", status);
    std::cout << "Resetted settings to default. Checking initial configuration...\n\n";

    {
      WORD h_binning, v_binning;

      status |= PCO_GetBinning(device_, &h_binning, &v_binning);
      soft_assert("(initial) PCO_GetBinning", status);
      std::cout << "binning (h x v)   = " << h_binning << "x" << v_binning << std::endl;
    }

    {
      WORD conv_f;

      status |= PCO_GetConversionFactor(device_, &conv_f);
      soft_assert("(initial) PCO_GetConversionFactor", status);
      std::cout << "conversion factor = " << conv_f << std::endl;
    }

    {
      DWORD delay, exposure;
      WORD timebase_del, timebase_exp;
      status |= PCO_GetDelayExposureTime(device_, &delay, &exposure, &timebase_del, &timebase_exp);
      soft_assert("(initial) PCO_GetDelayExposureTime", status);
      std::cout << "exposure time     = " << exposure << "[" << timebase_exp << "]\n" <<
        "delay time        = " << delay << "[" << timebase_del << "]" << std::endl;
    }

    {
      WORD tmp_status;
      DWORD tmp_fps;
      DWORD tmp_exp_time;

      status |= PCO_GetFrameRate(device_, &tmp_status, &tmp_fps, &tmp_exp_time);
      std::cout << "framerate         = " << tmp_fps << "\t{exposure time : " << tmp_exp_time <<
        "ns; status : " << tmp_status << "}" << std::endl;
    }

    for (unsigned i = 0; i < 4; ++i)
    {
      // Printing current signal configuration to try to understand...
      std::cout << "Pre-configuration of signal " << i << ":\n";
      PCO_Signal sig_data;
      sig_data.wSize = sizeof(PCO_Signal);
      status |= PCO_GetHWIOSignal(device_, i, &sig_data);
      std::string msg("(initial) PCO_getHWIOSignal_"); msg += boost::lexical_cast<std::string>(i);
      soft_assert(msg.c_str(), status);

      std::cout << "Port " << i << " : state[" << sig_data.wEnabled << "]\n" <<
        "         type[" << sig_data.wType << "]\n" <<
        "         polarity[" << sig_data.wPolarity << "]\n" <<
        "         filter[" << sig_data.wFilterSetting << "]\n" <<
        "         subindex[" << sig_data.wSelected << "]\n" <<
        "         dwParameters[" << sig_data.dwParameter[0] << "," << sig_data.dwParameter[1] << "," <<
        sig_data.dwParameter[2] << "," << sig_data.dwParameter[3] << "]\n" <<
        "         dwSignalFunctionality[" << sig_data.dwSignalFunctionality[0] << "," <<
        sig_data.dwSignalFunctionality[1] << "," << sig_data.dwSignalFunctionality[2] << "," <<
        sig_data.dwSignalFunctionality[3] << std::endl;
    }


    std::cout << "\n--------------\n\nSetting parameters manually now...\n";

    status |= PCO_SetTimeouts(device_, timeouts_, 2);
    soft_assert("PCO_SetTimeouts", status);

    status |= PCO_SetSensorFormat(device_, 0);
    soft_assert("PCO_SetSensorFormat", status);

    status |= PCO_SetConversionFactor(device_, conversion_factor_);
    soft_assert("PCO_SetConversionFactor", status);

    status |= PCO_SetBinning(device_, hz_binning_, vt_binning_);
    soft_assert("PCO_SetBinning", status);

    status |= PCO_SetROI(device_, p0_x_, p0_y_, p1_x_, p1_y_);
    soft_assert("PCO_SetROI", status);

    status |= PCO_SetPixelRate(device_, pixel_rate_);
    soft_assert("PCO_SetPixelRate", status);

    status |= PCO_SetTriggerMode(device_, triggermode_);
    soft_assert("PCO_SetTriggerMode", status);

    status |= PCO_SetNoiseFilterMode(device_, 0);
    soft_assert("PCO_SetNoiseFilterMode", status);

    {
      /* Convert exposure time in milliseconds. */
      float tmp_exp_time = exposure_time_;
      tmp_exp_time *= 1e3;

      /* base_time : 0x0002 = ms, 0x0001 = us, 0x0000 = ns */
      WORD base_time;

      // Why doing this?
      for (base_time = 0x0002; base_time > 0 && tmp_exp_time < 1.0f; --base_time)
        tmp_exp_time *= 1e3;

      std::cout << "Exposure time :" << tmp_exp_time << std::endl;
      status |= PCO_SetDelayExposureTime(device_, 0, static_cast<DWORD>(tmp_exp_time), 0, base_time);
      soft_assert("PCO_SetDelayExposureTime", status);
    }

    {
      WORD fps_change_status; // Knowing if some value was trimmed. Currently unused.
      DWORD tmp_exp_time = static_cast<DWORD>(exposure_time_ * 1e9); // SDK requires exp. time in ns

      status |= PCO_SetFrameRate(device_, &fps_change_status, framerate_mode_, &framerate_, &tmp_exp_time);
      soft_assert("PCO_SetFrameRate", status);

      std::cout << "FPS status after change : " << fps_change_status << std::endl;
      exposure_time_ = static_cast<float>(tmp_exp_time)* 1e-9; // Convert back exp. time to seconds
    }

    {
      status |= PCO_SetHWIOSignal(device_, 0, &io_0_conf);
      soft_assert("[Port 1] PCO_SetHWIOSignal", status);

      status |= PCO_SetHWIOSignal(device_, 1, &io_1_conf);
      soft_assert("[Port 2] PCO_SetHWIOSignal", status);

      status |= PCO_SetHWIOSignal(device_, 2, &io_2_conf);
      soft_assert("[Port 3] PCO_SetHWIOSignal", status);
       
      status |= PCO_SetHWIOSignal(device_, 3, &io_3_conf);
      soft_assert("[Port 4] PCO_SetHWIOSignal", status);
    }

    // Display final configuration
    std::cout << "\n\nFinal configuration : \n" <<
      "binning (h x v)   = " << hz_binning_ << "x" << vt_binning_ << "\n" <<
      "conversion factor = " << conversion_factor_ << "\n" <<
      "exposure time     = " << exposure_time_ << "\n" <<
      "trigger mode      = " << triggermode_ << "\n" <<
      "framerate         = " << framerate_ << "\n" <<
      "framerate mode    = " << framerate_mode_ << "\n" <<
      "Port 1 : state[" << io_0_conf.wEnabled << "]\n" <<
      "         type[" << io_0_conf.wType << "]\n" <<
      "         polarity[" << io_0_conf.wPolarity << "]\n" <<
      "         filter[" << io_0_conf.wFilterSetting << "]\n" <<
      "         subindex[" << io_0_conf.wSelected << "]\n" <<
      "Port 2 : state[" << io_1_conf.wEnabled << "]\n" <<
      "         type[" << io_1_conf.wType << "]\n" <<
      "         polarity[" << io_1_conf.wPolarity << "]\n" <<
      "         filter[" << io_1_conf.wFilterSetting << "]\n" <<
      "         subindex[" << io_1_conf.wSelected << "]\n" <<
      "Port 3 : state[" << io_2_conf.wEnabled << "]\n" <<
      "         type[" << io_2_conf.wType << "]\n" <<
      "         polarity[" << io_2_conf.wPolarity << "]\n" <<
      "         filter[" << io_2_conf.wFilterSetting << "]\n" <<
      "         subindex[" << io_2_conf.wSelected << "]\n" <<
      "Port 4 : state[" << io_3_conf.wEnabled << "]\n" <<
      "         type[" << io_3_conf.wType << "]\n" <<
      "         polarity[" << io_3_conf.wPolarity << "]\n" <<
      "         filter[" << io_3_conf.wFilterSetting << "]\n" <<
      "         subindex[" << io_3_conf.wSelected << "]\n" <<
      "ROI               = (" << p0_x_ << "," << p0_y_ << ") to (" << p1_x_ << "," << p1_y_ << "\n" <<
      "pixel rate        = " << pixel_rate_ << " Hz\n" <<
      "timeout           = " << timeouts_[0] << "(command)" << timeouts_[1] << "(img. request)" << std::endl;

    /* DEBUG : remove comments later
    if (status != PCO_NOERROR)
      throw CameraException(CameraException::CANT_SET_CONFIG);
    ** ! DEBUG */
  }

}