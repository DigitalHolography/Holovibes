#ifndef CAMERA_ADIMEC_HH
# define CAMERA_ADIMEC_HH

#include <BFType.h>

#include <camera.hh>

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
    // Camera's camera-specific methods, used while initializion takes place.
    virtual void load_ini_params() override;
    virtual void load_default_params() override;
    virtual void bind_params() override;

    // Handle to the opened BitFlow board.
    Bd board_;

    // Pointer to the allocated buffer used to receive images.
    PBFVOID buffer_;

    // QTabBank used by the camera.
    BFU8 quad_bank_;
  };
}

#endif /* !CAMERA_ADIMEC_HH */