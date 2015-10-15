#ifndef CAMERA_ADIMEC_HH
# define CAMERA_ADIMEC_HH

#include <BFType.h>

#include <camera.hh>
#include "camera_exception.hh"

namespace camera
{
  /* Adimec camera, used through a BitFlow CoaXPress frame grabber.
  ** The API used is the Ci API, because it allows for broad configuration
  ** of the used camera without dwelving into low-level details.
  ** See BitFlow's SDK for further details about the different APIs available
  ** for BitFlow products.
  */
  class CameraAdimec : public Camera
  {
  public:
    CameraAdimec();
    virtual ~CameraAdimec();

    // ICamera's main methods.
    virtual void init_camera() override;
    virtual void start_acquisition() override;
    virtual void stop_acquisition() override;
    virtual void shutdown_camera() override;
    virtual void* get_frame() override;

  private:
    // Enum that will decide what we have to close when an error
    // occurs. Binary masking is used here to add up flags.
    enum CloseFlag
    {
      NO_BOARD = 0x000, // Nothing to close
      BOARD = 0x00F,
      CAM = 0x0F0,
      BUFFER = 0xF00,
      ALL = 0xFFF
    };

    void err_check(BFRC status, std::string err_mess, CameraException cam_ex, int flag);

    // Camera's camera-specific methods, used while initializion takes place.
    virtual void load_ini_params() override;
    virtual void load_default_params() override;
    virtual void bind_params() override;

    // Handle to the opened BitFlow board.
    Bd board_;

    // Camera object, used to configure the camera.
    PBFCNF camera_;

    // Pointer to the allocated buffer used to receive images.
    PBFVOID buffer_;

    // QTabBank used by the camera.
    BFU8 quad_bank_;

    /* Configuration parameters
    */

    BFU32 exposure_time_;

    BFU32 roi_x_; // ROI origin's coordinates : (x, y)
    BFU32 roi_y_;
    BFU32 roi_width_; // ROI size
    BFU32 roi_height_;
  };
}

#endif /* !CAMERA_ADIMEC_HH */