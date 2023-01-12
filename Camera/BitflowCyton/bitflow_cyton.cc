#include <BiApi.h>
#include <BFErApi.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include "bitflow_cyton.hh"
#include "camera_logger.hh"

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

    Logger::camera()->error("{}", std::string(error_text, error_text_size));
}

CameraPhantomBitflow::CameraPhantomBitflow()
    : Camera("bitflow.ini")
{
    name_ = "Bitflow Cyton";

    if (ini_file_is_open())
    {
        load_ini_params();
        ini_file_.close();
    }
    else
    {
        Logger::camera()->error("Could not open bitflow.ini config file");
        throw CameraException(CameraException::NOT_INITIALIZED);
    }
    load_default_params();
    init_camera();
}

void CameraPhantomBitflow::init_camera()
{
    open_boards();
    bind_params();
    create_buffers();
}

void CameraPhantomBitflow::open_boards()
{
    for (size_t i = 0; i < nb_boards; ++i)
    {
        RV = BiBrdOpen(BiTypeAny, board_nums[i], &boards[i]);
        if (RV != BI_OK)
        {
            print_BiError(boards[i], RV);
            throw CameraException(CameraException::NOT_INITIALIZED);
        }
    }
}

void CameraPhantomBitflow::create_buffers()
{
    bitmap_size = fd_.get_frame_size();
    total_mem_size = nb_buffers * bitmap_size + PAGE_SIZE;

    /* Allocate memory for the array of buffer pointers */
    frames = (PBFU32*)malloc(nb_buffers * sizeof(BFUPTR));
    if (frames == NULL)
    {
        Logger::camera()->error("Could not allocate pointers buffer");
        throw CameraException(CameraException::NOT_INITIALIZED);
    }

    /* Allocate memory for buffers (in pinned memory) */
    cudaError_t status = cudaMallocHost(&data, total_mem_size);
    if (status != cudaSuccess || data == NULL)
    {
        Logger::camera()->error("Could not allocate data buffer");
        free(frames);
        throw CameraException(CameraException::NOT_INITIALIZED);
    }

    /* Create an array of pointers to each buffer that has been
       allocated */
    for (BFU32 i = 0; i < nb_buffers; i++)
    {
        /* Div by sizeof(BFU32) because bitmap_size is in bytes */
        frames[i] = data + (i * (bitmap_size / sizeof(BFU32)));
    }

    for (size_t i = 0; i < nb_boards; ++i)
    {
        /* Assign the array of pointers to buffin's array of pointers */
        RV = BiBufferAssign(boards[i], &buf_arrays[i], frames, nb_buffers);
        if (RV != BI_OK)
        {
            print_BiError(boards[i], RV);
            cudaFreeHost(data);
            free(frames);
            throw CameraException(CameraException::NOT_INITIALIZED);
        }
    }
}

BFU32 CameraPhantomBitflow::get_circ_options(size_t i)
{
    BFU32 default_options = BiAqEngJ | NoResetOnError | HighFrameRateMode;
    if (nb_boards == 1)
    {
        return default_options;
    }
    else if (nb_boards == 2)
    {
        return default_options | (i == 0 ? OnlyEvenLines : OnlyOddLines);
    }
    else if (nb_boards == 4)
    {
        return default_options | (FourHorizInterleavedChunks0 << i);
    }
    // Should not happen
    assert(false);
    return 0;
}

void CameraPhantomBitflow::start_acquisition()
{
    BFU32 error_mode = CirErIgnore;
    for (size_t i = 0; i < nb_boards; ++i)
    {
        /* Setup for circular buffers */
        RV = BiCircAqSetup(boards[i], &buf_arrays[i], error_mode, get_circ_options(i));
        if (RV != BI_OK)
        {
            print_BiError(boards[i], RV);
            throw CameraException(CameraException::CANT_START_ACQUISITION);
        }
    }

    /* create signal for DMA done notification */
    RV = CiSignalCreate(boards[0], CiIntTypeEOD, &eod_signal);
    if (RV != BI_OK)
    {
        print_BiError(boards[0], RV);
        throw CameraException(CameraException::CANT_START_ACQUISITION);
    }

    for (size_t i = 1; i < nb_boards; ++i)
    {
        /* Put in slave mode */
        BFRegRMW(boards[i], REG_CON8, 0x00800000, 0x00800000);
    }

    for (size_t i = 0; i < nb_boards; ++i)
    {
        /* Start acquisition of images */
        RV = BiCirControl(boards[i], &buf_arrays[i], BISTART, BiWait);
        if (RV != BI_OK)
        {
            if (RV < BI_WARNINGS)
            {
                print_BiError(boards[i], RV);
                throw CameraException(CameraException::CANT_START_ACQUISITION);
            }
        }
    }
}

void CameraPhantomBitflow::stop_acquisition()
{
    /* cancel the signal */
    CiSignalCancel(boards[0], &eod_signal);

    for (size_t i = 0; i < nb_boards; ++i)
    {
        /* Stop acquisition of images */
        RV = BiCirControl(boards[i], &buf_arrays[i], BISTOP, BiWait);
        if (RV != BI_OK)
        {
            print_BiError(boards[i], RV);
        }

        /* Clean things up */
        RV = BiCircCleanUp(boards[i], &buf_arrays[i]);
        if (RV != BI_OK)
        {
            print_BiError(boards[i], RV);
        }
    }
}

void CameraPhantomBitflow::shutdown_camera()
{
    for (size_t i = 0; i < nb_boards; ++i)
    {
        /* Free memory */
        RV = BiBufferUnassign(boards[i], &buf_arrays[i]);
        if (RV != BI_OK)
        {
            print_BiError(boards[i], RV);
        }
        BiBrdClose(boards[i]);
    }

    cudaFreeHost(data);
    free(frames);
}

// tour overflow
// captured = 1
// old_captured = 2
// nb_frames = 3
//

// tour normal
// captured 5
// old_captured 0
// nb_frames 5
// pas besoin de region 2


// tour circular buffer // nb buffer 10
// captured 5
// old_captured 8
// nb_frames



CapturedFramesDescriptor CameraPhantomBitflow::get_frames()
{
    /* Get update on the total number of images  */
    CiSignalQueueSize(boards[0], &eod_signal, &captured);

    BFU32 nb_frames = captured - old_captured;
    if (captured < old_captured) // nb_frames overflow
    {
        nb_frames = 0xffffffff - old_captured + captured;
    }
    if (nb_frames >= nb_buffers || nb_frames == 0) // si pas de frame on renvoit rien OU plus de frame que de buffer on give up
    {
        old_captured = captured;
        return CapturedFramesDescriptor(nullptr, 0);
    }

    BFU32 idx = old_captured % nb_buffers;
    CapturedFramesDescriptor ret;
    ret.region1 = frames[idx];
    ret.count1 = (nb_frames <= nb_buffers - idx) ? nb_frames : nb_buffers - idx;
    ret.region2 = frames[0];
    ret.count2 = nb_frames - ret.count1;

    /* Keep old total */
    old_captured = captured;

    return ret;
}

void CameraPhantomBitflow::load_ini_params()
{
    const boost::property_tree::ptree& pt = get_ini_pt();

    nb_boards = pt.get<int>("bitflow.number_of_boards", 0);

    if (nb_boards != 1 && nb_boards != 2 && nb_boards != 4)
    {
        Logger::camera()->error("bitflow.ini: number_of_boards should be 1, 2 or 4");
        throw CameraException(CameraException::NOT_INITIALIZED);
    }

    for (size_t i = 0; i < nb_boards; ++i)
    {
        board_nums[i] = pt.get<int>("bitflow.board" + std::to_string(i), -1);

        if (board_nums[i] == -1)
        {
            Logger::camera()->error("bitflow.ini: board {} has an invalid value", i);
            throw CameraException(CameraException::NOT_INITIALIZED);
        }
    }

    pixel_size_ = pt.get<float>("bitflow.pixel_size", 20.0f);
}

void CameraPhantomBitflow::load_default_params() { fd_.byteEndian = Endianness::LittleEndian; }

void CameraPhantomBitflow::bind_params()
{
    BFU32 width = 0;
    BFU32 height = 0;
    BFU32 depth = 0;

    BFRC rc = BI_OK;
    rc |= BiBrdInquire(boards[0], BiCamInqXSize, &width);
    rc |= BiBrdInquire(boards[0], BiCamInqYSize0, &height);
    rc |= BiBrdInquire(boards[0], BiCamInqBytesPerPix, &depth);

    if (rc != BI_OK)
    {
        Logger::camera()->error("Could not read frame description");
        throw CameraException(CameraException::NOT_INITIALIZED);
    }

    fd_.width = width;
    fd_.height = height * nb_boards;
    fd_.depth = depth;
}

ICamera* new_camera_device() { return new CameraPhantomBitflow(); }
} // namespace camera
