#include <BiApi.h>
#include <iostream>
#include <cmath>
#include <cstdlib>

#include "camera_phantom_bitflow.hh"

namespace camera
{
CameraPhantomBitflow::CameraPhantomBitflow()
    : Camera("phantom.ini")
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
    err_check(BiBrdOpen(BiTypeAny, number, &board_),
              "Could not open board.",
              CameraException::NOT_INITIALIZED,
              CloseFlag::NO_BOARD);

    bind_params();

    /* --- */

    BiBrdInquire(board_, BiCamInqFrameSize0, &BitmapSize);
    TotalMemorySize = NumBuffers * BitmapSize + PAGE_SIZE;

    /* Allocate memory for the array of buffer pointers */
    pMemArray = (PBFU32*)malloc(NumBuffers * sizeof(BFUPTR));
    if (pMemArray == NULL)
    {
        printf("Memory allocation error\n");
        throw CameraException::NOT_INITIALIZED;
    }

    /* Allocate memory for buffers */
    pMemory = (PBFU32)malloc(TotalMemorySize);
    if (pMemory == NULL)
    {
        printf("Memory allocation error\n");
        throw CameraException::NOT_INITIALIZED;
    }

    /* Create an array of the pointers to each buffer that has been allocated */
    for (BFU32 i = 0; i < NumBuffers; i++)
    {
        /* Div by sizeof(BFU32) because BitmapSize is in bytes */
        pMemArray[i] = pMemory + (i * (BitmapSize / sizeof(BFU32)));
    }

    /* Assign the array of pointers to buffin's array of pointers */
    RV = BiBufferAssign(board_, &BufArray, pMemArray, NumBuffers);
    if (RV != BI_OK)
    {
        BiErrorShow(board_, RV);
        throw CameraException::NOT_INITIALIZED;
    }
}

void CameraPhantomBitflow::start_acquisition()
{
    /* Setup for circular buffers */
    RV = BiCircAqSetup(board_, &BufArray, ErrorMode, CirSetupOptions);
    if (RV != BI_OK)
    {
        BiErrorShow(board_, RV);
        throw CameraException::CANT_START_ACQUISITION;
    }

    /* create signal for DMA done notification */
    if (CiSignalCreate(board_, CiIntTypeEOD, &EODSignal))
    {
        BFErrorShow(board_);
        throw CameraException::CANT_START_ACQUISITION;
    }

    /* Start acquisition of images */
    RV = BiCirControl(board_, &BufArray, BISTART, BiWait);
    if (RV != BI_OK)
    {
        if (RV < BI_WARNINGS)
        {
            throw CameraException::CANT_START_ACQUISITION;
        }
    }

    BFTick(&T0);
}

void CameraPhantomBitflow::stop_acquisition()
{
    /* Stop acquisition of images */
    RV = BiCirControl(board_, &BufArray, BISTOP, BiWait);
    if (RV != BI_OK)
        BiErrorShow(board_, RV);

    /* Clean things up */
    RV = BiCircCleanUp(board_, &BufArray);
    if (RV != BI_OK)
        BiErrorShow(board_, RV);

    /* cancel the signal */
    CiSignalCancel(board_, &EODSignal);

    /* Free memory */
    RV = BiBufferUnassign(board_, &BufArray);
    if (RV != BI_OK)
        BiErrorShow(board_, RV);
    free(pMemory);
    free(pMemArray);
}

void CameraPhantomBitflow::shutdown_camera()
{
    // Make sure the camera is closed at program end.
    BiBrdClose(board_);
}

CapturedFramesDescriptor CameraPhantomBitflow::get_frames()
{
    /* Get update on the total number of images  */
    CiSignalQueueSize(board_, &EODSignal, &Captured);

    BFU32 NbFrames = Captured - OldCaptured;
    if (Captured < OldCaptured)
    {
        NbFrames = 0xffffffff - OldCaptured + Captured;
    }
    if (NbFrames >= NumBuffers)
    {
        OldCaptured = Captured;
        return CapturedFramesDescriptor(nullptr, 0);
    }

    /* get loop time */
    Delta = BFTickDelta(&T0, BFTick(&T1));

    /* If 1 second has passed, update FPS */
    if (Delta > 1000)
    {
        FPS = (BFU32)((BFDOUBLE)(Captured - LastTime) /
                      ((BFDOUBLE)Delta / 1000.0));
        LastTime = Captured;
        BFTick(&T0);
        std::cout << "FPS: " << FPS << std::endl;
    }

    BFU32 idx = OldCaptured % NumBuffers;
    CapturedFramesDescriptor ret;
    ret.region1 = pMemArray[idx];
    ret.count1 = min(NbFrames, NumBuffers - idx);
    ret.region2 = pMemArray[0];
    ret.count2 = NbFrames - ret.count1;

    /* Keep old total */
    OldCaptured = Captured;

    return ret;
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
    if (BFCXPReadReg(board_, 0xFF, RegAddress::ROI_WIDTH, &roi_width_) != BF_OK)
        std::cerr << "[CAMERA] Cannot read the roi width of the registers of "
                     "the camera "
                  << std::endl;

    if (BFCXPReadReg(board_, 0xFF, RegAddress::ROI_HEIGHT, &roi_height_) !=
        BF_OK)
        std::cerr << "[CAMERA] Cannot read the roi height of the registers of "
                     "the camera "
                  << std::endl;

    if (BFCXPReadReg(board_, 0xFF, RegAddress::PIXEL_FORMAT, &pixel_format_) !=
        BF_OK)
        std::cerr
            << "[CAMERA] Cannot read the pixel format of the registers of "
               "the camera "
            << std::endl;

    fd_.width = roi_width_;
    fd_.height = roi_height_;
    if (pixel_format_ == PixelFormat::MONO_8)
    {
        fd_.depth = 1;
    }
    else if (pixel_format_ == PixelFormat::MONO_12 ||
             pixel_format_ == PixelFormat::MONO_16)
    {
        fd_.depth = 2;
    }
}

ICamera* new_camera_device() { return new CameraPhantomBitflow(); }
} // namespace camera
