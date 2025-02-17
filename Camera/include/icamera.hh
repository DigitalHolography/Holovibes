/*! \file
 *
 * Interface for all cameras.*/
#pragma once

/*! \brief Contains all function and structure reference to camera usage */
namespace camera
{
struct FrameDescriptor;

/*! \struct CapturedFramesDescriptor
 *
 * \brief Simple struct to represent a single or mutiple frames located either in host memory or in device memory.
 */
struct CapturedFramesDescriptor
{
    CapturedFramesDescriptor(void* region1_, unsigned count1_, void* region2_, unsigned count2_, bool on_gpu_ = false)
        : region1(region1_)
        , count1(count1_)
        , region2(region2_)
        , count2(count2_)
        , on_gpu(on_gpu_)
    {
    }

    CapturedFramesDescriptor()
        : CapturedFramesDescriptor(nullptr, 0)
    {
    }
    CapturedFramesDescriptor(void* data)
        : CapturedFramesDescriptor(data, 1)
    {
    }

    CapturedFramesDescriptor(void* data, unsigned int count, bool on_gpu_ = false)
        : CapturedFramesDescriptor(data, count, nullptr, 0)
    {
    }

    /*! \brief Pointer to the first region containing data. */
    void* region1;

    /*! \brief The number of frames stored in region1. */
    unsigned int count1;

    /* 2 data regions are needed when getting data from a ringbuffer
     * the data can be located at the end and beginning of the buffer
     *  region2<-       ->region1
     * [XXXXXXX<-.......->XXXXXXX]
     */

    /*! \brief Pointer to the second region containing data. */
    void* region2 = nullptr;

    /*! \brief The number of frames stored in region2. */
    unsigned int count2 = 0;

    /*! \brief Whether data is located in host or device memory (GPU). */
    bool on_gpu;
};

/*! \defgroup CameraInterface Camera Interface
 *
 * This small module is the entry point to using all existing cameras DLLs. It
 * regroups :
 * * The ICamera interface.
 * * The new_camera_device hook function, used to grab a camera object from a DLL.
 * * A general timeout value for all cameras.
 * \{
 */

/*! \brief Timeout value in ms for a camera to get a frame. */
static int FRAME_TIMEOUT = 100000;

/*! \brief Abstract interface for all cameras.
 *
 * # Interface
 *
 * In C++, an interface is a class that must contains only pure virtual
 * methods. No data fields is allowed. Whatever the camera model, each must
 * implement the common interface.
 *
 * This is also the [C++ Mature Approach: Using an Abstract
 * Interface](http://www.codeproject.com/Articles/28969/HowTo-Export-C-classes-from-a-DLL)
 * to export a C++ class within a DLL.
 *
 * ## How to use it ?
 *
 * Use only header files in Camera/include folder.
 */
class ICamera
{
  public:
    ICamera() {}

    /*! \brief A camera object is non assignable. */
    ICamera& operator=(const ICamera&) = delete;

    /*! \brief A camera object is non copyable. */
    ICamera(const ICamera&) = delete;

    /*! \brief Destruct the ICamera object. */
    virtual ~ICamera() {}

    /*! \brief Get the frame descriptor (frame format) */
    virtual const FrameDescriptor& get_fd() const = 0;

    /*! \brief Get the pixel size */
    virtual const float get_pixel_size() const = 0;

    /*! \brief Get the name of the camera */
    virtual const char* get_name() const = 0;

    /*! \brief Get the default path of the INI configuration file. */
    virtual const char* get_ini_name() const = 0;

    /*! \brief Get the temperature if possible, else 0 */
    virtual int get_temperature() const = 0;

    /*! \brief Get the temperature if possible, else 0 */
    virtual int toggleFan() const = 0;

    /*! \brief Get the camera fps if possible, else -1 */
    virtual int get_camera_fps() const = 0;

    /*! \brief Open the camera and initialize it.
     *
     * * Open the camera and retrieve the handler
     * * Call the bind parameters method to configure the camera API
     * * Do memory allocation and initialization (following what the user needs)
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
    virtual CapturedFramesDescriptor get_frames() = 0;
};

/*! \brief extern "C" is used to avoid C++ name mangling.
 *
 * Without it, each camera DLL would provide a different symbol for the
 * new_camera_device function, thus preventing Holovibes from using it.
 */
extern "C"
{
    /*! \brief Ask the DLL to allocate a new camera, and get a handle to it.
     *
     * \return A pointer to the new camera object.
     */
    __declspec(dllexport) ICamera* new_camera_device();
}

/*! \} */ // End of Camera Interface group
} // namespace camera
