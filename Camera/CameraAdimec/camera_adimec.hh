#ifndef CAMERA_ADIMEC_HH
# define CAMERA_ADIMEC_HH

#include <BFType.h>
#include <BiDef.h>

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
      BUFFER = 0xF00,
      ALL = 0xF0F
    };

    /* Contains the addresses of the Cyton board's registers
    ** that we manually set to override some configuration data.
    */
    enum RegAdress
    {
      FRAME_PERIOD = 0x8220,
      EXPOSURE_TIME = 0x8258
    };

    // Private error checking and reporting method. TODO: Use CameraUtils' services in it.
    void err_check(BFRC status, std::string err_mess, CameraException cam_ex, int flag);

    // Camera's camera-specific methods, used while initializion takes place.
    virtual void load_ini_params() override;
    virtual void load_default_params() override;
    virtual void bind_params() override;

    // Handle to the opened BitFlow board.
    Bd board_;

    // SDK-provided structure containing all kinds of data on acquisition over time.
    PBIBA info_;

    // Index of the last buffer that was read by Holovibes in the circular buffer set.
    BFU32 last_buf;

    // Camera object, used to configure the camera.
    PBFCNF camera_;

    // QTabBank used by the camera.
    BFU8 quad_bank_;

    /* Configuration parameters
    */

    BFU32 exposure_time_;

    BFU32 frame_period_;

    BFU32 roi_x_; // ROI's top-left coordinates : (x, y)
    BFU32 roi_y_;
    BFU32 roi_width_; // ROI size
    BFU32 roi_height_;
  };
}

#endif /* !CAMERA_ADIMEC_HH */