#include "camera_pco_edge.hh"
#include "camera_exception.hh"

#include <PCO_err.h>
#include <sc2_defs.h>

namespace camera
{
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
    exposure_time_ = 0.024f;
    triggermode_ = 0;

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
    if (triggermode_ < 3)
      triggermode_ = 0;
  }

  void CameraPCOEdge::bind_params()
  {
    int status = PCO_NOERROR;

    status |= PCO_ResetSettingsToDefault(device_);
    status |= PCO_SetSensorFormat(device_, 0);
    status |= PCO_SetTriggerMode(device_, triggermode_);

    {
      /* Convert exposure time in milliseconds. */
      exposure_time_ *= 1e3;

      /* base_time : 0x0002 = ms, 0x0001 = us, 0x0000 = ns */
      WORD base_time;

      for (base_time = 0x0002; base_time > 0 && exposure_time_ < 1.0f; --base_time)
        exposure_time_ *= 1e3;

      status |= PCO_SetDelayExposureTime(device_, 0, static_cast<DWORD>(exposure_time_), 0, base_time);
    }

    if (status != PCO_NOERROR)
      throw CameraException(name_, CameraException::CANT_SET_CONFIG);
  }
}