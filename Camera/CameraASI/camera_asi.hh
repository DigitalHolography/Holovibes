#pragma once

#include <ASICamera2.h>        // ASI SDK header provided by ZWOptical
#include "camera.hh"           // Base Camera interface
#include "camera_exception.hh" // Exception class for camera errors

namespace camera
{

/*!
 * \class CameraAsi
 * \brief A camera class for interfacing with ASI cameras.
 *
 * This class implements the Camera interface using the ASICamera2 SDK to operate
 * an ASI camera. It reads configuration parameters (such as camera ID, resolution,
 * pixel depth, gain, and exposure time) from an ini file, initializes the camera,
 * starts/stops video acquisition, and retrieves captured frames.
 */
class CameraAsi : public Camera
{
  public:
    /*!
     * \brief Constructor for the ASI camera.
     *
     * Reads the configuration from the ini file, loads the parameters, and initializes the camera.
     */
    CameraAsi();

    /*!
     * \brief Destructor for the ASI camera.
     *
     * Shuts down the camera and releases any allocated resources.
     */
    virtual ~CameraAsi();

    /*!
     * \brief Initializes the camera.
     *
     * Opens the camera, initializes it using the SDK, and configures the necessary settings.
     */
    virtual void init_camera() override;

    /*!
     * \brief Starts video acquisition.
     *
     * Begins capturing video frames from the camera.
     */
    virtual void start_acquisition() override;

    /*!
     * \brief Stops video acquisition.
     *
     * Stops capturing video frames from the camera.
     */
    virtual void stop_acquisition() override;

    /*!
     * \brief Shuts down the camera.
     *
     * Closes the camera and releases any resources allocated during operation.
     */
    virtual void shutdown_camera() override;

    /*!
     * \brief Retrieves captured frames.
     *
     * \return A descriptor containing the captured frame data.
     */
    virtual CapturedFramesDescriptor get_frames() override;

  private:
    /*!
     * \brief Loads configuration parameters from the ini file.
     *
     * Reads parameters such as camera ID, resolution, pixel depth, gain, and exposure time
     * from the ini file and stores them in the corresponding member variables.
     */
    virtual void load_ini_params() override;

    /*!
     * \brief Loads default parameters for the camera.
     *
     * Initializes the default values for the camera parameters.
     */
    virtual void load_default_params() override;

    /*!
     * \brief Binds internal parameters.
     *
     * Associates the configuration parameters with the internal variables as needed.
     */
    virtual void bind_params() override;

    int cameraID;            /*!< Unique camera ID (default: 0). */
    ASI_CAMERA_INFO camInfo; /*!< Structure containing camera properties from the SDK. */
    bool isInitialized;      /*!< Flag indicating if the camera has been successfully initialized. */
    int resolution_width_;   /*!< Camera resolution width. */
    int resolution_height_;  /*!< Camera resolution height. */
    int pixel_depth_value_;  /*!< Pixel depth value (e.g., 8 or 16 bits). */
    int gain_value_;         /*!< Gain value for the camera. */
    int exposure_time_;      /*!< Exposure time in microseconds. */
};

} // namespace camera
