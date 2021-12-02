/*! \file
 *
 * Camera Adimec Quartz-2A750 */
#pragma once

#include <BFType.h>
#include <BiDef.h>
#include "camera.hh"

#include "camera_exception.hh"

namespace camera
{
/*! \brief Adimec Quartz-2A750 camera, used through the BitFlow Cyton CXP-4 frame grabber.
 *
 * The API used is the Bi API, because it allows for broad configuration
 * of the used camera without dwelving into low-level details, like circular
 * buffer handling.
 *
 * See BitFlow's SDK for further details about the different APIs available
 * for BitFlow products.
 */
class CameraAdimec : public Camera
{
  public:
    CameraAdimec();

    // #FIXME Obsolote?
    virtual ~CameraAdimec()
    {
        // Make sure the camera is closed at program exit.
        // shutdown camera
        // BiBrdClose(board_);
    }

    virtual void init_camera() override;
    virtual void start_acquisition() override;
    virtual void stop_acquisition() override;
    virtual void shutdown_camera() override;
    virtual CapturedFramesDescriptor get_frames() override;

  private:
    virtual void load_ini_params() override;
    virtual void load_default_params() override;
    virtual void bind_params() override;

    /*! \brief Private error checking and reporting method.
     *
     * If status is different than BI_OK, print an error message and throw
     * an exception. Moreover, components designated by the flag will be
     * released.
     *
     * \param status The error status returned by the Bitflow function used.
     * \param err_mess Error messaege to be printed on standard error.
     * \param cam_ex A CameraException type.
     * \param flag See CloseFlag enum.
     */
    void err_check(const BFRC status, const std::string err_mess, const CameraException cam_ex, const int flag);

  private:
    /*! \brief Selecting a component to release.
     *
     * Enum that will decide what we have to close when an error
     * occurs. Binary masking is used here to add up flags.
     */
    enum CloseFlag
    {
        NO_BOARD = 0x00, /*!< Nothing to close */
        BUFFER = 0xF0,   /*!< Free allocated resources */
        BOARD = 0x0F,    /*!< Close the board */
        ALL = 0xFF       /*!< Release everything, in correct order */
    };

    /*! \brief Some of the board's registers.
     *
     * Contains the addresses of the Cyton board's registers
     * that we manually set to override some configuration data, at runtime.
     */
    enum RegAddress
    {
        ROI_WIDTH = 0x8118,
        ROI_HEIGHT = 0x811C,
        EXPOSURE_MODE = 0x8250,
        PIXEL_FORMAT = 0x8144,
        FRAME_PERIOD = 0x8220,
        EXPOSURE_TIME = 0x8258,
        START = 0x8204,
        STOP = 0x8208
    };

    Bd board_;            /*!< Handle to the opened BitFlow board. */
    PBIBA info_;          /*!< SDK-provided structure containing all kinds of data on acquisition over time. */
    BFU32 last_buf;       /*!< Index of the last buffer that was read by Holovibes in the circular buffer set. */
    BFU8 quad_bank_;      /*!< QTabBank used by the camera. */
    BFU32 queue_size_;    /*!< Queue size of bitflow frame grabber */
    BFU32 exposure_time_; /*!< Exposure time of the camera */
    BFU32 frame_period_;  /*!< Frame period of the camera */
    BFU32 roi_x_;         /*!< ROI top-left corner X-coordinate. */
    BFU32 roi_y_;         /*!< ROI top-left corner Y-coordinate. */
    BFU32 roi_width_;     /*!< ROI width in pixels. */
    BFU32 roi_height_;    /*!< ROI height in pixels. */
};
} // namespace camera
