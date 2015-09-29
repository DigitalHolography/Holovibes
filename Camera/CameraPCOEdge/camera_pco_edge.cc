#include "camera_pco_edge.hh"
#include <camera_exception.hh>
#include <boost/lexical_cast.hpp>

// DEBUG : remove me later
#include <fstream>

#include <PCO_err.h>
#include <sc2_defs.h>

namespace camera
{
  static void soft_assert(char* func, int& status, std::ofstream& ostr)
  {
    if (status != PCO_NOERROR)
    {
      ostr << func << "() call failed : error " << status << std::endl;
      status = 0;
    }
  }

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
    // TODO : Check if removing this solves the triggering problem!
    //if (triggermode_ < 3)
    //  triggermode_ = 0;

    framerate_ = pt.get<DWORD>("pco-edge.framerate", framerate_) * 10e3;
    framerate_mode_ = pt.get<WORD>("pco-edge.framerate_mode", framerate_mode_);

    hz_binning_ = pt.get<WORD>("pco-edge.binning_hz", hz_binning_);
    vt_binning_ = pt.get<WORD>("pco-edge.binning_vt", vt_binning_);

    p0_x_ = pt.get<WORD>("pco-edge.p0_x", p0_x_);
    p0_y_ = pt.get<WORD>("pco-edge.p0_y", p0_y_);
    p1_x_ = pt.get<WORD>("pco-edge.p1_x", p1_x_);
    p1_y_ = pt.get<WORD>("pco-edge.p1_y", p1_y_);

    pixel_rate_ = pt.get<DWORD>("pco-edge.pixel_rate", pixel_rate_) * 10e6;

    conversion_factor_ = static_cast<WORD>(
        pt.get<float>("pco-edge.conversion_factor", conversion_factor_) * 100);

    ad_converters_ = pt.get<WORD>("pco-edge.adc", ad_converters_);

    timeouts_[0] = pt.get<unsigned int>("pco-edge.timeout_command", timeouts_[0]);
    timeouts_[1] = pt.get<unsigned int>("pco-edge.timeout_img_acq", timeouts_[1]);

    load_pco_signal_params(pt, io_0_conf, 0);
    load_pco_signal_params(pt, io_1_conf, 1);
    load_pco_signal_params(pt, io_2_conf, 2);
    load_pco_signal_params(pt, io_3_conf, 3);
  }

  void CameraPCOEdge::bind_params()
  {
    int status = PCO_NOERROR;
    // DEBUG : remove me later
    std::ofstream ostr("test_edge_4.2.log");
    assert(ostr.is_open());
    // ! DEBUG

    status |= PCO_ResetSettingsToDefault(device_);
    soft_assert("PCO_ResetSettingsToDefault", status, ostr);

    status |= PCO_SetTimeouts(device_, timeouts_, 2);
    soft_assert("PCO_SetTimeouts", status, ostr);

    status |= PCO_SetSensorFormat(device_, 0);
    soft_assert("PCO_SetSensorFormat", status, ostr);

    status |= PCO_SetADCOperation(device_, ad_converters_);
    soft_assert("PCO_SetADCOperation", status, ostr);

    status |= PCO_SetConversionFactor(device_, ad_converters_);
    soft_assert("PCO_SetConversionFactor", status, ostr);

    status |= PCO_SetBinning(device_, hz_binning_, vt_binning_);
    soft_assert("PCO_SetBinning", status, ostr);

    status |= PCO_SetROI(device_, p0_x_, p0_y_, p1_x_, p1_y_);
    soft_assert("PCO_SetROI", status, ostr);

    status |= PCO_SetPixelRate(device_, pixel_rate_);
    soft_assert("PCO_SetPixelRate", status, ostr);

    status |= PCO_SetTriggerMode(device_, triggermode_);
    soft_assert("PCO_SetTriggerMode", status, ostr);

    status |= PCO_SetNoiseFilterMode(device_, 0);
    soft_assert("PCO_SetNoiseFilterMode", status, ostr);

    {
      /* Convert exposure time in milliseconds. */
      exposure_time_ *= 1e3;

      /* base_time : 0x0002 = ms, 0x0001 = us, 0x0000 = ns */
      WORD base_time;

      for (base_time = 0x0002; base_time > 0 && exposure_time_ < 1.0f; --base_time)
        exposure_time_ *= 1e3;

      status |= PCO_SetDelayExposureTime(device_, 0, static_cast<DWORD>(exposure_time_), 0, base_time);
      soft_assert("PCO_SetDelayExposureTime", status, ostr);
    }

    {
      WORD fps_change_status;
      DWORD exp_time = static_cast<DWORD>(exposure_time_ * 10e9);
      status |= PCO_SetFrameRate(device_, &fps_change_status, 0, &framerate_, &exp_time);
      soft_assert("PCO_SetFrameRate", status, ostr);
      exposure_time_ = static_cast<float>(exp_time)* 10e-9; // Convert from nanoseconds
    }

    {
      status |= PCO_SetHWIOSignal(device_, 0, &io_0_conf);
      soft_assert("[Port 1] PCO_SetHWIOSignal", status, ostr);

      status |= PCO_SetHWIOSignal(device_, 1, &io_1_conf);
      soft_assert("[Port 2] PCO_SetHWIOSignal", status, ostr);

      status |= PCO_SetHWIOSignal(device_, 2, &io_2_conf);
      soft_assert("[Port 3] PCO_SetHWIOSignal", status, ostr);
       
      status |= PCO_SetHWIOSignal(device_, 3, &io_3_conf);
      soft_assert("[Port 4] PCO_SetHWIOSignal", status, ostr);
    }

    /* DEBUG : remove comments later
    if (status != PCO_NOERROR)
      throw CameraException(CameraException::CANT_SET_CONFIG);
    ** ! DEBUG */
  }

}