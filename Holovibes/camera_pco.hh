#ifndef CAMERA_PCO_HH
# define CAMERA_PCO_HH

# include "camera.hh"

# include <iostream>
# include <array>

# include <Windows.h>
# include <SC2_SDKStructures.h>
# include <SC2_CamExport.h>

namespace camera
{
  const SHORT NBUFFERS = 6;
  /*! This class contains common stuff for PCO cameras. */
  class CameraPCO : public Camera
  {
  public:
    CameraPCO(
      const std::string& ini_filepath,
      WORD camera_type);
    virtual ~CameraPCO();

    virtual void init_camera() override;
    virtual void start_acquisition() override;
    virtual void stop_acquisition() override;
    virtual void shutdown_camera() override;
    virtual void* get_frame() override;

  protected:
    HANDLE device_;
  private:
    int get_sensor_sizes();
    int allocate_buffers();
  private:
    WORD camera_type_;

    std::array<HANDLE, NBUFFERS> buffers_events_;
    std::array<unsigned short*, NBUFFERS> buffers_;
    std::array<DWORD, NBUFFERS> buffers_driver_status_;
    /*! Size of a buffer. */
    DWORD buffer_size_;

    WORD actual_res_x_;
    WORD actual_res_y_;
  };
}

#endif /* !CAMERA_PCO_HH */