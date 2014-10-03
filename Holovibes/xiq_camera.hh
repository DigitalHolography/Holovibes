#ifndef XIQ_HH
# define XIQ_HH

# include "camera.hh"

# include <Windows.h>
# include <xiApi.h>

namespace camera
{
  class XiqCamera : public Camera
  {
  public:
    XiqCamera();

    ~XiqCamera()
    {}

    virtual void init_camera() override;
    virtual void start_acquisition() override;
    virtual void stop_acquisition() override;
    virtual void shutdown_camera() override;
    virtual void* get_frame() override;

  private:
    virtual void load_default_params() override;
    virtual void load_ini_params() override;
    virtual void bind_params() override;

  private:
    HANDLE device_;
    XI_IMG frame_;

    /* Custom camera parameters. */
    /*! Downsampling rate
    ** 1: 1x1 sensor pixel  = 1 image pixel
    ** 2: 2x2 sensor pixels = 1 image pixel
    ** 4: 4x4 sensor pixels = 1 image pixel
    */
    int downsampling_rate_;
    /*! Downsampling type
    ** XI_BINNING  0: pixels are interpolated - better image
    ** XI_SKIPPING 1: pixels are skipped - higher frame rate
    */
    XI_DOWNSAMPLING_TYPE downsampling_type_;
    /*! Image format
    ** XI_MONO8, XI_MONO16, XI_RAW8, XI_RAW16
    */
    XI_IMG_FORMAT img_format_;
    /*! Buffer policy
    ** XI_BP_UNSAFE: User gets pointer to internally allocated circular
    ** buffer and data may be overwritten by device.
    ** XI_BP_SAFE: Data from device will be copied to user allocated buffer
    ** or xiApi allocated memory.
    */
    XI_BP buffer_policy_;
  };
}

#endif /* !XIQ_HH */
