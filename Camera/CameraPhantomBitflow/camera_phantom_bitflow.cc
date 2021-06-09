#include <BiApi.h>
#include <iostream>
#include <cmath>
#include <cstdlib>

#include "camera_phantom_bitflow.hh"

namespace camera
{
void CameraPhantomBitflow::err_check(const BFRC status,
                                     const std::string err_mess,
                                     const CameraException cam_ex,
                                     const int flag)
{
    if (status != CI_OK)
    {
        std::cerr << "[CAMERA] " << err_mess << " : " << status << std::endl;
        if (flag & CloseFlag::BUFFER)
            BiBufferFree(board_, info_);
        if (flag & CloseFlag::BOARD)
            BiBrdClose(board_);
        throw cam_ex;
    }
}

CameraPhantomBitflow::CameraPhantomBitflow()
    : Camera("adimec.ini")
    , board_(nullptr)
    , info_(new BIBA())
    , last_buf(0)
    , quad_bank_(BFQTabBank0)
{
    name_ = "Phantom S710";

    fd_.width = 512;
    fd_.height = 512;
    fd_.depth = 2;
    fd_.byteEndian = Endianness::BigEndian;

    load_default_params();
    init_camera();
}

void CameraPhantomBitflow::init_camera()
{
    /* We don't want a specific type of board; there should not
     * be more than one anyway. */
    BFU32 type = BiTypeAny;
    BFU32 number = 0;

    err_check(BiBrdOpen(type, number, &board_),
              "Could not open board.",
              CameraException::NOT_INITIALIZED,
              CloseFlag::NO_BOARD);

    bind_params();
}

void CameraPhantomBitflow::start_acquisition()
{
    /* Asking the frame size (width * height * depth) to the board.
     * Such a method is more robust than hardcoding known values.*/
    BFU32 width;
    err_check(BiBrdInquire(board_, BiCamInqXSize, &width),
              "Could not get frame size",
              CameraException::CANT_START_ACQUISITION,
              CloseFlag::BOARD);
    BFU32 height;
    err_check(BiBrdInquire(board_, BiCamInqYSize0, &height),
              "Could not get frame size",
              CameraException::CANT_START_ACQUISITION,
              CloseFlag::BOARD);
    BFU32 depth;
    err_check(BiBrdInquire(board_, BiCamInqBitsPerPix, &depth),
              "Could not get frame depth",
              CameraException::CANT_START_ACQUISITION,
              CloseFlag::BOARD);

    // Aligned allocation ensures fast memory transfers.
    const BFSIZET alignment = 4096;
    err_check(BiBufferAllocAligned(board_,
                                   info_,
                                   width,
                                   height,
                                   depth,
                                   queue_size_,
                                   alignment),
              "Could not allocate buffer memory",
              CameraException::MEMORY_PROBLEM,
              CloseFlag::BOARD);

    /* If the board does not find any buffer marked AVAILABLE by the user,
     * it will overwrite them. */
    BFU32 error_handling = CirErIgnore;
    BFU32 options = BiAqEngJ;
    err_check(BiCircAqSetup(board_, info_, error_handling, options),
              "Could not setup board for acquisition",
              CameraException::CANT_START_ACQUISITION,
              CloseFlag::ALL);

    /* Acquisition is started without interruption. */
    options = BiWait;
    err_check(BiCirControl(board_, info_, BISTART, options),
              "Could not start acquisition",
              CameraException::CANT_START_ACQUISITION,
              CloseFlag::ALL);
}

void CameraPhantomBitflow::stop_acquisition()
{
    /* Free resources taken by BiCircAqSetup, in a single function call.
     * Allocated memory is freed separately, through BiBufferFree. */
    BiBufferFree(board_, info_);
    if (BiCircCleanUp(board_, info_) != BI_OK)
    {
        std::cerr << "[CAMERA] Could not stop acquisition cleanly."
                  << std::endl;
        shutdown_camera();
        throw CameraException(CameraException::CANT_STOP_ACQUISITION);
    }
}

void CameraPhantomBitflow::shutdown_camera()
{
    // Make sure the camera is closed at program end.
    BiBrdClose(board_);
}

CapturedFramesDescriptor CameraPhantomBitflow::get_frames()
{
    // Mark the previously read buffer as available for writing, for the board.
    BiCirBufferStatusSet(board_, info_, last_buf, BIAVAILABLE);

    // Wait for a freshly written image to be readable.
    BiCirHandle hd;
    BiCirWaitDoneFrame(board_,
                       info_,
                       static_cast<BFU32>(camera::FRAME_TIMEOUT),
                       &hd);

    BFU32 status;
    BiCirBufferStatusGet(board_, info_, hd.BufferNumber, &status);
    // Checking buffer status is correct. TODO : Log error in other cases.
    if (status == BINEW)
    {
        last_buf = hd.BufferNumber;
        BiCirBufferStatusSet(board_, info_, last_buf, BIHOLD);
    }

    if (hd.pBufData == reinterpret_cast<void*>(0xcccccccccccccccc))
        return get_frames();

    return CapturedFramesDescriptor(hd.pBufData);
}

void CameraPhantomBitflow::load_ini_params() {}

void CameraPhantomBitflow::load_default_params()
{
    pixel_size_ = 12;
    queue_size_ = 64;
    exposure_time_ = 100;
    frame_rate_ = 300;
    roi_width_ = 512;
    roi_height_ = 512;
}

void CameraPhantomBitflow::bind_params()
{
    /* We use a CoaXPress-specific register writing function to set parameters.
     * The register address parameter can be found in any .bfml configuration
     * file provided by Bitflow; here, it has been put into the RegAddress enum
     * for clarity.
     *
     * Whenever a parameter setting fails, setup fallbacks to default value. */

    /* Frame period should be set before exposure time, because the latter
     * depends of the former. */

    if (BFCXPWriteReg(board_, 0xFF, RegAddress::FRAME_RATE, frame_rate_) !=
        BF_OK)
        std::cerr << "[CAMERA] Could not set frame rate to " << frame_rate_
                  << std::endl;

    if (BFCXPWriteReg(board_,
                      0xFF,
                      RegAddress::EXPOSURE_TIME,
                      exposure_time_) != BF_OK)
        std::cerr << "[CAMERA] Could not set exposure time to "
                  << exposure_time_ << std::endl;

    /* After setting up the profile of the camera in SysReg, we read into
     * the registers of the camera to set width and height */
    if (BFCXPReadReg(board_, 0xFF, RegAddress::ROI_WIDTH, &roi_width_) != BF_OK)
        std::cerr << "[CAMERA] Cannot read the roi width of the registers of "
                     "the camera "
                  << std::endl;

    if (BFCXPReadReg(board_, 0xFF, RegAddress::ROI_HEIGHT, &roi_height_) !=
        BF_OK)
        std::cerr << "[CAMERA] Cannot read the roi height of the registers of "
                     "the camera "
                  << std::endl;

    fd_.width = roi_width_;
    fd_.height = roi_height_;
}

ICamera* new_camera_device() { return new CameraPhantomBitflow(); }
} // namespace camera
