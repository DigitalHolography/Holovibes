#ifndef ICAMERA_HH
# define ICAMERA_HH

#ifdef CAMERA_EXPORTS
# define CAMERA_API __declspec(dllexport)
#else
# define CAMERA_API __declspec(dllimport)
#endif

namespace camera
{
  struct FrameDescriptor;

  /*! \defgroup CameraInterface Camera Interface
   * This small module is the entry point to using all existing cameras DLLs. It regroups :
   * * The ICamera interface.
   * * The new_camera_device hook function, used to grab a camera object from a DLL.
   * * A general timeout value for all cameras.
   * \{
   */

  /*! Timeout value for a camera to get a frame. */
  static int FRAME_TIMEOUT = 10000;

  //! Abstract interface for all cameras.
  /*! # Interface
   *
   * In C++, an interface is a class that must contains only pure virtual
   * methods. No data fields is allowed. Whatever the camera model, each must
   * implement the common interface.
   *
   * This is also the [C++ Mature Approach: Using an Abstract
   * Interface](http://www.codeproject.com/Articles/28969/HowTo-Export-C-classes-from-a-DLL)
   * to export a C++ class within a DLL.
   *
   * # How to use it ?
   *
   * Use only header files in Camera/include folder. */
  class ICamera
  {
  public:
    ICamera()
    {
    }

    virtual ~ICamera()
    {
    }

    /*! \brief Get the frame descriptor (frame format) */
    virtual const FrameDescriptor& get_frame_descriptor() const = 0;

    /*! \brief Get the name of the camera */
    virtual const char* get_name() const = 0;

    /*! \brief Get the default path of the INI configuration file. */
    virtual const char* get_ini_path() const = 0;

    /*! \brief Open the camera and initialize it.
     *
     * * Open the camera and retrieve the handler
     * * Call the bind parameters method to configure the camera API
     * * Do memory allocation and initialization (following what the user
     * needs)
     * * Retrieves informations about the camera model */
    virtual void init_camera() = 0;

    /*! \brief Set the camera in image acquisition mode.
     *
     * Depends on the camera API, but in most cases :
     *
     * * Arm the camera / Start the device
     * * Set the camera to record/acquisition state
     * * Load buffers to store images */
    virtual void start_acquisition() = 0;

    /*! \brief Stop the camera acquisition mode. */
    virtual void stop_acquisition() = 0;

    /*! \brief Shutdown the camera
     *
     * * Free ressources
     * * Close the device (handle)
     * * Ensure the camera is not in recording state anymore */
    virtual void shutdown_camera() = 0;
    /*! \brief Request the camera to get a frame
     *
     * Getting a frame should not be longer than FRAME_TIMEOUT */
    virtual void* get_frame() = 0;

  private:
    /*! \brief A camera object is non assignable. */
    ICamera& operator=(const ICamera&) = delete;

    /*! \brief A camera object is non copyable. */
    ICamera(const ICamera&) = delete;
  };

  /*! extern "C" is used to avoid C++ name mangling.
   * Without it, each camera DLL would provide a different symbol for the
   * new_camera_device function, thus preventing Holovibes from using it. */
  extern "C"
  {
    /*! \brief Ask the DLL to allocate a new camera, and get a handle to it.
     *
     * \return A pointer to the new camera object. */
    CAMERA_API ICamera* new_camera_device();
  }

  /** \} */ // End of Camera Interface group
}

#endif /* !ICAMERA_HH */