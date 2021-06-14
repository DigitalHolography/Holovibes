#pragma once

#include <BFType.h>
#include <BiDef.h>
#include <camera.hh>

#include "camera_exception.hh"

namespace camera
{
class CameraPhantomBitflow : public Camera
{
  public:
    CameraPhantomBitflow();

    virtual ~CameraPhantomBitflow() {}

    virtual void init_camera() override;
    virtual void start_acquisition() override;
    virtual void stop_acquisition() override;
    virtual void shutdown_camera() override;
    virtual CapturedFramesDescriptor get_frames() override;

  private:
    virtual void load_ini_params() override;
    virtual void load_default_params() override;
    virtual void bind_params() override;

    // Camera register addresses
    enum RegAddress
    {
        ROI_WIDTH = 0x6000,
        ROI_HEIGHT = 0x6004,
        PIXEL_FORMAT = 0x6008,
        FRAME_RATE = 0x60C0,
        EXPOSURE_TIME = 0x60C8,
        START = 0x601C,
        STOP = 0x601C
    };

    enum PixelFormat
    {
        MONO_8 = 0x01080001,
        MONO_12 = 0x010C0047,
        MONO_16 = 0x01100007
    };

    enum CloseFlag
    {
        NO_BOARD = 0x00, //!< Nothing to close
        BUFFER = 0xF0,   //!< Free allocated resources
        BOARD = 0x0F,    //!< Close the board
        ALL = 0xFF       //!< Release everything, in correct order
    };

    Bd board_;       //!< Handle to the opened BitFlow board.
    PBIBA info_;     //!< SDK-provided structure containing all kinds of data on
                     //!< acquisition over time.
    BFU32 last_buf;  //!< Index of the last buffer that was read by Holovibes in
                     //!< the circular buffer set.
    BFU8 quad_bank_; //!< QTabBank used by the camera.
    BFU32 queue_size_;    //!< Queue size of bitflow frame grabber
    BFU32 exposure_time_; //!< Exposure time of the camera
    BFU32 frame_rate_;    //!< Frame period of the camera
    BFU32 roi_width_;     //!< ROI width in pixels.
    BFU32 roi_height_;    //!< ROI height in pixels.
    BFU32 pixel_format_;

    void err_check(const BFRC status,
                   const std::string err_mess,
                   const CameraException cam_ex,
                   const int flag);
};
} // namespace camera
