#ifndef CAMERA_PIXELFLY_HH
# define CAMERA_PIXELFLY_HH

# include "camera.hh"

# include <Windows.h>
# include <SC2_SDKStructures.h>
# include <SC2_CamExport.h>

namespace camera
{
  class CameraPixelfly : public Camera
  {
  public:
    CameraPixelfly();

    virtual ~CameraPixelfly()
    {
      /* Ensure that the camera is closed in case of exception. */
      shutdown_camera();
    }

    virtual void init_camera() override;
    virtual void start_acquisition() override;
    virtual void stop_acquisition() override;
    virtual void shutdown_camera() override;
    virtual void* get_frame() override;

  private:
    virtual void load_default_params() override;
    virtual void load_ini_params() override;
    virtual void bind_params() override;

    void pco_set_size_parameters();
    void pco_fill_structures();
    void pco_get_sizes();
    void pco_allocate_buffer();

  private:
    HANDLE device_;
    HANDLE refresh_event_;
    WORD* buffer_;

    PCO_General      pco_general_;
    PCO_CameraType   pco_camtype_;
    PCO_Sensor       pco_sensor_;
    PCO_Description  pco_description_;
    PCO_Timing       pco_timing_;
    PCO_Storage      pco_storage_;
    PCO_Recording    pco_recording_;
  };
}

#endif /* !CAMERA_PIXELFLY */