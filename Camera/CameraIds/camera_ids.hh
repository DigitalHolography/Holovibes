/*! \file
 *
 * Camera IDS */
#pragma once

#include "camera.hh"

/* Disable warnings. */
#pragma warning(push, 0)
#include "uEye.h"
#pragma warning(pop)

namespace camera
{
/*! \class CameraIds
 *
 * \brief #TODO Add a description for this class
 */
class CameraIds : public Camera
{
  public:
    CameraIds();

    virtual ~CameraIds() {}

    virtual void init_camera() override;
    virtual void start_acquisition() override;
    virtual void stop_acquisition() override;
    virtual void shutdown_camera() override;

    virtual CapturedFramesDescriptor get_frames() override;

    virtual void load_default_params() override;
    virtual void load_ini_params() override;
    virtual void bind_params() override;

  private:
    /*! \brief Format gain, it should be between 0 and 100 as it is a coefficient.
     *
     * \return 0 if gain < 0 or gain > 100; else returns gain.
     */
    int format_gain() const;

    /*! \brief Retrieve subsampling mode code from a string.
     *
     * \return The corresponding API-defined code, or the subsampling-disabling code if the value is invalid.
     */
    int get_subsampling_mode(const std::string ui) const;

    /*! \brief Retrieve binning mode code from user input string.
     *
     * \return The corresponding API-defined code, or the binning-disabling code if the value is invalid.
     */
    int get_binning_mode(const std::string ui);

    /*! \brief Retrieve color mode code from user input string.
     *
     * \return The corresponding API-defined code, or the raw 8-bit format if the value is invalid.
     */
    int get_color_mode(const std::string ui);

    /*! \brief Retrieve trigger mode code from user input string.
     *
     * \return The corresponding API-defined code, or the trigger-disabling code if the value is invalid.
     */
    int get_trigger_mode(const std::string ui) const;

  private:
    /*! \brief camera handler */
    HIDS cam_;
    /*! \brief Frame pointer */
    char* frame_;
    /*! \brief Frame associated memory */
    int frame_mem_pid_;
    /*! \brief Gain */
    unsigned int gain_;
    /*! \brief Subsampling value (2x2, 4x4 ...) */
    int subsampling_;
    /*! \brief Binning value (2x2, 4x4 ...) */
    int binning_;
    /*! \brief Image format (also called color mode) */
    int color_mode_;
    /*! \brief Area Of Interest (AOI) x */
    int aoi_x_;
    /*! \brief Area Of Interest (AOI) y */
    int aoi_y_;
    /*! \brief Area Of Interest (AOI) width */
    int aoi_width_;
    /*! \brief Area Of Interest (AOI) height */
    int aoi_height_;
    /*! \brief Trigger mode */
    int trigger_mode_;
};
} // namespace camera
