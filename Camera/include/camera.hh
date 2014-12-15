#ifndef CAMERA_HH
# define CAMERA_HH

#ifdef CAMERA_EXPORTS
# define CAMERA_API __declspec(dllexport)
#else
# define CAMERA_API __declspec(dllimport)
#endif

namespace camera
{
  struct FrameDescriptor;

  static int FRAME_TIMEOUT = 10000;

  /*! Abstract Camera class. */
  class CAMERA_API Camera
  {
  public:
    virtual ~Camera();

    virtual const FrameDescriptor& get_frame_descriptor() const;
    virtual const char* get_name() const;
    virtual const char* get_ini_path() const;

    virtual void init_camera() = 0;
    virtual void start_acquisition() = 0;
    virtual void stop_acquisition() = 0;
    virtual void shutdown_camera() = 0;
    virtual void* get_frame() = 0;

  protected:
    Camera(const char* const ini_filepath);

    /*! Load default parameters. */
    virtual void load_default_params() = 0;
    /*! Load parameters from INI file. */
    virtual void load_ini_params() = 0;
    /*! Send current parameters to camera API. */
    virtual void bind_params() = 0;

  protected:
    class CameraPimpl;
    CameraPimpl* pimpl_;
    
  private:
    // Object is non copyable
    Camera& operator=(const Camera&) = delete;
    Camera(const Camera&) = delete;
  };

  extern "C" {
    CAMERA_API Camera* __cdecl InitializeCamera();
  }
}

#endif /* !CAMERA_HH */
