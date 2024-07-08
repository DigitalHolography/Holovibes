/*! \file
 *
 * \brief Defines the CameraIds class for interfacing with IDS cameras.
 */
#pragma once

#include "camera.hh"

/* Disable warnings from uEye library */
#pragma warning(push, 0)
#include "uEye.h"
#pragma warning(pop)

namespace camera
{
/*! \class CameraIds
 *
 * \brief Encapsulates functionalities specific to IDS cameras.
 *
 * This class provides methods to initialize, start, stop, and shut down IDS cameras,
 * as well as to manage camera parameters and acquisition modes.
 */
class CameraIds : public Camera
{
  public:
    /*! \brief Constructor */
    CameraIds();

    /*! \brief Destructor */
    virtual ~CameraIds() {}

    /*! \brief Initializes the camera */
    virtual void init_camera() override;

    /*! \brief Starts the image acquisition process */
    virtual void start_acquisition() override;

    /*! \brief Stops the image acquisition process */
    virtual void stop_acquisition() override;

    /*! \brief Shuts down the camera */
    virtual void shutdown_camera() override;

    /*! \brief Retrieves the captured frames */
    virtual CapturedFramesDescriptor get_frames() override;

    /*! \brief Loads the default parameters for the camera */
    virtual void load_default_params() override;

    /*! \brief Loads the parameters from an INI file */
    virtual void load_ini_params() override;

    /*! \brief Binds the parameters to the camera settings */
    virtual void bind_params() override;

  private:
    /*! \brief Formats the gain value to ensure it is between 0 and 100.
     *
     * \return 0 if gain < 0 or gain > 100; otherwise returns the gain.
     */
    int format_gain() const;

    /*! \brief Retrieves the subsampling mode code from a string.
     *
     * \param ui User input string representing the subsampling mode.
     * \return The corresponding API-defined code, or the subsampling-disabling code if the value is invalid.
     */
    int get_subsampling_mode(const std::string ui) const;

    /*! \brief Retrieves the binning mode code from a string.
     *
     * \param ui User input string representing the binning mode.
     * \return The corresponding API-defined code, or the binning-disabling code if the value is invalid.
     */
    int get_binning_mode(const std::string ui);

    /*! \brief Retrieves the color mode code from a string.
     *
     * \param ui User input string representing the color mode.
     * \return The corresponding API-defined code, or the raw 8-bit format if the value is invalid.
     */
    int get_color_mode(const std::string ui);

    /*! \brief Retrieves the trigger mode code from a string.
     *
     * \param ui User input string representing the trigger mode.
     * \return The corresponding API-defined code, or the trigger-disabling code if the value is invalid.
     */
    int get_trigger_mode(const std::string ui) const;

  private:
    HIDS cam_; /*!< Camera handler */
    char* frame_; /*!< Pointer to the frame data */
    int frame_mem_pid_; /*!< Frame associated memory ID */
    unsigned int gain_; /*!< Camera gain */
    int subsampling_; /*!< Subsampling value (e.g., 2x2, 4x4) */
    int binning_; /*!< Binning value (e.g., 2x2, 4x4) */
    int color_mode_; /*!< Image format (also called color mode) */
    int aoi_x_; /*!< Area Of Interest (AOI) x-coordinate */
    int aoi_y_; /*!< Area Of Interest (AOI) y-coordinate */
    int aoi_width_; /*!< Area Of Interest (AOI) width */
    int aoi_height_; /*!< Area Of Interest (AOI) height */
    int trigger_mode_; /*!< Trigger mode */
};
} // namespace camera