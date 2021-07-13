#include <BiApi.h>
#include <iostream>
#include <cmath>
#include <cstdlib>

#include "camera_phantom_bitflow.hh"

namespace camera
{
static void print_BiError(Bd board, BFRC status)
{
    if (status == BI_OK)
    {
        return;
    }

    constexpr BFU32 const_error_text_size = 512;
    BFU32 error_text_size = const_error_text_size;
    char error_text[const_error_text_size];
    BiErrorTextGet(board, status, error_text, &error_text_size);
    fprintf(stderr, "%.*s\n", error_text_size, error_text);
    fflush(stderr);
}

CameraPhantomBitflow::CameraPhantomBitflow()
    : Camera("phantom.ini")
{
    name_ = "Phantom S710";
    load_default_params();
    init_camera();
}

void CameraPhantomBitflow::init_camera()
{
    RV = BiBrdOpen(BiTypeAny, 0, &board_);
    if (RV != BI_OK)
    {
        print_BiError(board_, RV);
        throw CameraException::NOT_INITIALIZED;
    }

    bind_params();

    BiBrdInquire(board_, BiCamInqFrameSize0, &bitmap_size);
    total_mem_size = nb_buffers * bitmap_size + PAGE_SIZE;

    /* Allocate memory for the array of buffer pointers */
    frames = (PBFU32*)malloc(nb_buffers * sizeof(BFUPTR));
    if (frames == NULL)
    {
        std::cerr << "Could not allocate pointers buffer" << std::endl;
        throw CameraException::NOT_INITIALIZED;
    }

    /* Allocate memory for buffers */
    data = (PBFU32)malloc(total_mem_size);
    if (data == NULL)
    {
        std::cerr << "Could not allocate data buffer" << std::endl;
        free(frames);
        throw CameraException::NOT_INITIALIZED;
    }

    /* Create an array of the pointers to each buffer that has been allocated */
    for (BFU32 i = 0; i < nb_buffers; i++)
    {
        /* Div by sizeof(BFU32) because bitmap_size is in bytes */
        frames[i] = data + (i * (bitmap_size / sizeof(BFU32)));
    }

    /* Assign the array of pointers to buffin's array of pointers */
    RV = BiBufferAssign(board_, &buf_array, frames, nb_buffers);
    if (RV != BI_OK)
    {
        print_BiError(board_, RV);
        free(data);
        free(frames);
        throw CameraException::NOT_INITIALIZED;
    }
}

void CameraPhantomBitflow::start_acquisition()
{
    /* Setup for circular buffers */
    BFU32 circ_setup_options = BiAqEngJ | NoResetOnError | HighFrameRateMode;
    BFU32 error_mode = CirErIgnore;
    RV = BiCircAqSetup(board_, &buf_array, error_mode, circ_setup_options);
    if (RV != BI_OK)
    {
        print_BiError(board_, RV);
        throw CameraException::CANT_START_ACQUISITION;
    }

    /* create signal for DMA done notification */
    RV = CiSignalCreate(board_, CiIntTypeEOD, &eod_signal);
    if (RV != BI_OK)
    {
        print_BiError(board_, RV);
        throw CameraException::CANT_START_ACQUISITION;
    }

    /* Start acquisition of images */
    RV = BiCirControl(board_, &buf_array, BISTART, BiWait);
    if (RV != BI_OK)
    {
        if (RV < BI_WARNINGS)
        {
            print_BiError(board_, RV);
            throw CameraException::CANT_START_ACQUISITION;
        }
    }
}

void CameraPhantomBitflow::stop_acquisition()
{
    /* Stop acquisition of images */
    RV = BiCirControl(board_, &buf_array, BISTOP, BiWait);
    if (RV != BI_OK)
    {
        print_BiError(board_, RV);
    }

    /* Clean things up */
    RV = BiCircCleanUp(board_, &buf_array);
    if (RV != BI_OK)
    {
        print_BiError(board_, RV);
    }

    /* cancel the signal */
    CiSignalCancel(board_, &eod_signal);

    /* Free memory */
    RV = BiBufferUnassign(board_, &buf_array);
    if (RV != BI_OK)
    {
        print_BiError(board_, RV);
    }
    free(data);
    free(frames);
}

void CameraPhantomBitflow::shutdown_camera()
{
    // Make sure the camera is closed at program end.
    BiBrdClose(board_);
}

CapturedFramesDescriptor CameraPhantomBitflow::get_frames()
{
    /* Get update on the total number of images  */
    CiSignalQueueSize(board_, &eod_signal, &captured);

    BFU32 nb_frames = captured - old_captured;
    if (captured < old_captured)
    {
        nb_frames = 0xffffffff - old_captured + captured;
    }
    if (nb_frames >= nb_buffers || nb_frames == 0)
    {
        old_captured = captured;
        return CapturedFramesDescriptor(nullptr, 0);
    }

    BFU32 idx = old_captured % nb_buffers;
    CapturedFramesDescriptor ret;
    ret.region1 = frames[idx];
    ret.count1 = min(nb_frames, nb_buffers - idx);
    ret.region2 = frames[0];
    ret.count2 = nb_frames - ret.count1;

    /* Keep old total */
    old_captured = captured;

    return ret;
}

void CameraPhantomBitflow::load_ini_params() {}

void CameraPhantomBitflow::load_default_params()
{
    fd_.byteEndian = Endianness::BigEndian;
}

void CameraPhantomBitflow::bind_params()
{
    BFU32 width = 0;
    BFU32 height = 0;
    BFU32 depth = 0;

    BFRC rc = BI_OK;
    rc |= BiBrdInquire(board_, BiCamInqXSize, &width);
    rc |= BiBrdInquire(board_, BiCamInqYSize0, &height);
    rc |= BiBrdInquire(board_, BiCamInqBytesPerPix, &depth);

    if (rc != BI_OK)
    {
        std::cerr << "Could not read frame description" << std::endl;
        throw CameraException::NOT_INITIALIZED;
    }

    fd_.width = width;
    fd_.height = height;
    fd_.depth = depth;
}

ICamera* new_camera_device() { return new CameraPhantomBitflow(); }
} // namespace camera
