#include <BiApi.h>
#include <iostream>
#include <cmath>
#include <cstdlib>

#include <spdlog/spdlog.h>

#include "camera_adimec.hh"
#include "camera_logger.hh"

namespace camera
{
CameraAdimec::CameraAdimec()
    : Camera("adimec.ini")
    , board_(nullptr)
    , info_(new BIBA())
    , last_buf(0)
    , quad_bank_(BFQTabBank0)
{
    name_ = "Adimec Quartz-2A750 M";
    /* Dimensions are initialized as there were no ROI; they will be updated
     * later if needed. */
    fd_.width = 1440;
    fd_.height = 1440;
    // Technically the camera is 11-bits, but each pixel value is encoded on 16
    // bits.
    fd_.depth = PixelDepth::Bits16;
    fd_.byteEndian = Endianness::LittleEndian;
    pixel_size_ = 12;

    load_default_params();
    if (ini_file_is_open())
    {
        load_ini_params();
        ini_file_.close();
    }

    init_camera();
}

void CameraAdimec::init_camera()
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

void CameraAdimec::start_acquisition()
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
    err_check(BiBufferAllocAligned(board_, info_, width, height, depth, queue_size_, alignment),
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

void CameraAdimec::stop_acquisition()
{
    /* Free resources taken by BiCircAqSetup, in a single function call.
     * Allocated memory is freed separately, through BiBufferFree. */
    BiBufferFree(board_, info_);
    if (BiCircCleanUp(board_, info_) != BI_OK)
    {
        Logger::camera()->error("Could not stop acquisition cleanly.");

        shutdown_camera();
        throw CameraException(CameraException::CANT_STOP_ACQUISITION);
    }
}

void CameraAdimec::shutdown_camera()
{
    // Make sure the camera is closed at program end.
    BiBrdClose(board_);
}

CapturedFramesDescriptor CameraAdimec::get_frames()
{
    // Mark the previously read buffer as available for writing, for the board.
    BiCirBufferStatusSet(board_, info_, last_buf, BIAVAILABLE);

    // Wait for a freshly written image to be readable.
    BiCirHandle hd;
    // TODO: Use timeout of global config
    BiCirWaitDoneFrame(board_, info_, static_cast<BFU32>(camera::FRAME_TIMEOUT), &hd);

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

void CameraAdimec::err_check(const BFRC status,
                             const std::string err_mess,
                             const CameraException cam_ex,
                             const int flag)
{
    if (status != CI_OK)
    {
        Logger::camera()->error("{} : {}", err_mess, status);
        if (flag & CloseFlag::BUFFER)
            BiBufferFree(board_, info_);
        if (flag & CloseFlag::BOARD)
            BiBrdClose(board_);
        throw cam_ex;
    }
}

void CameraAdimec::load_default_params()
{
    /* Values here are hardcoded to avoid being dependent on a default .bfml
     * file, which may be modified accidentally. When possible, these default
     * values were taken from the default mode for the Adimec-A2750 camera. */
    queue_size_ = 64;

    exposure_time_ = 0x0539;
    frame_period_ = 0x056C;

    roi_x_ = 0;
    roi_y_ = 0;
    roi_width_ = 1440;
    roi_height_ = 1440;
}

void CameraAdimec::load_ini_params()
{
    const boost::property_tree::ptree& pt = get_ini_pt();
    queue_size_ = pt.get<BFU32>("bitflow.queue_size", queue_size_);
    exposure_time_ = pt.get<BFU32>("adimec.exposure_time", exposure_time_);
    frame_period_ = pt.get<BFU32>("adimec.frame_period", frame_period_);
    roi_x_ = pt.get<BFU32>("adimec.roi_x", roi_x_);
    roi_y_ = pt.get<BFU32>("adimec.roi_y", roi_y_);
}

void CameraAdimec::bind_params()
{
    /* We use a CoaXPress-specific register writing function to set parameters.
     * The register address parameter can be found in any .bfml configuration
     * file provided by Bitflow; here, it has been put into the RegAddress enum
     * for clarity.
     *
     * Whenever a parameter setting fails, setup fallbacks to default value. */

    /* Frame period should be set before exposure time, because the latter
     * depends of the former. */

    if (BFCXPWriteReg(board_, 0xFF, RegAddress::FRAME_PERIOD, frame_period_) != BF_OK)
    {
        Logger::camera()->error("Could not set frame period to {}", frame_period_);
    }

    if (BFCXPWriteReg(board_, 0xFF, RegAddress::EXPOSURE_TIME, exposure_time_) != BF_OK)
    {
        Logger::camera()->error("Could not set exposure time to {}", exposure_time_);
    }

    /* After setting up the profile of the camera in SysReg, we read into the
     * registers of the camera to set width and height */
    if (BFCXPReadReg(board_, 0xFF, RegAddress::ROI_WIDTH, &roi_width_) != BF_OK)
    {
        Logger::camera()->error("Cannot read the roi width of the registers of the camera");
    }

    if (BFCXPReadReg(board_, 0xFF, RegAddress::ROI_HEIGHT, &roi_height_) != BF_OK)
    {
        Logger::camera()->error("Cannot read the roi height of the registers of the camera");
    }
    fd_.width = roi_width_;
    fd_.height = roi_height_;
}

ICamera* new_camera_device() { return new CameraAdimec(); }
} // namespace camera
