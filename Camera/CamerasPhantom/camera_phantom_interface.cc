#include "camera_phantom_interface.hh"

namespace camera
{
EHoloGrabber::EHoloGrabber(EGenTL& gentl, unsigned int buffer_part_count, std::string& pixel_format)
    : grabbers_(gentl)
    , buffer_part_count_(buffer_part_count)
    , nb_grabbers_(nb_grabbers)
{
    // Fetch the first grabber info to determine the width, height and depth
    // of the full image.
    // According to the requirements described above, we assume that the
    // full height is two times the height of the first grabber.

    depth_ = static_cast<PixelDepth>(gentl.imageGetBytesPerPixel(pixel_format));

    // The below loop will check which grabbers are available to use, i.e the ones which are connected to a camera
    // We don't use Euresys::EGrabberDiscovery because it doesn't allow us to detect when a frame grabber is
    // connected to something or not
    for (size_t ix = 0; ix < grabbers_.length(); ++ix)
    {
        try
        {
            // Try to query the remote device (the camera)
            grabbers_[ix]->getString<RemoteModule>("Banks");
        }
        catch (const Euresys::gentl_error&)
        {
            continue;
        }
        available_grabbers_.push_back(grabbers_[ix]);
    }

    //  set Error Error
    /*
        // Check if we have enough available frame grabbers
        if (available_grabbers_.size() < nb_grabbers_)
        {
            Logger::camera()->error("Not enough frame grabbers  connected to the camera, expected: {} but got: {}.",
                                    nb_grabbers_,
                                    available_grabbers_.size());
            throw CameraException(CameraException::CANT_SET_CONFIG);
        } // TODO Dont forget this !!!!!!!!
    */
}

EHoloGrabber::~EHoloGrabber()
{
    for (size_t i = 0; i < nb_grabbers_; i++)
        available_grabbers_[i]->reallocBuffers(0);

    cudaFreeHost(ptr_);
}
// magic nunmber for number max of frame grabber supported (can be less for some implementation)
#define NB_MAX_GRABBER 4

struct SetupParam
{
    unsigned int full_height;
    unsigned int width;
    unsigned int nb_grabbers;
    size_t stripe_height;
    std::string& stripe_arrangement;
    std::string& triggerSource;
    unsigned int block_height;
    unsigned int[NB_MAX_GRABBER] offsets;
    std::optionnal<std::string> trigger_mode;
    std::optionnal<std::string> trigger_selector;
    std::optionnal<unsigned int> AcquisitionFrameRate;
    unsigned int cycleMinimumPeriod;
    float exposureTime;
    std::string& gain_selector;
    float gain;
    std::string& balance_white_marker;
}

void EHoloGrabber::setup(const SetupParam& param)
{
    width_ = param.width;
    height_ = param.fullHeight;

    size_t pitch = param.width * gentl.imageGetBytesPerPixel(param.pixelFormat);
    size_t height = param.full_height / param.nb_grabbers;
    size_t stripePitch = param.stripe_height * param.nb_grabbers;

    for (size_t ix = 0; ix < param.nb_grabbers; ++ix)
    {
        available_grabbers_[ix]->setInteger<RemoteModule>("Width", static_cast<int64_t>(param.width));
        available_grabbers_[ix]->setInteger<RemoteModule>("Height", static_cast<int64_t>(height));
        available_grabbers_[ix]->setString<RemoteModule>("PixelFormat", param.pixelFormat);

        available_grabbers_[ix]->setString<StreamModule>("StripeArrangement", param.stripe_arrangement);
        available_grabbers_[ix]->setInteger<StreamModule>("LinePitch", pitch);
        available_grabbers_[ix]->setInteger<StreamModule>("LineWidth", pitch);
        available_grabbers_[ix]->setInteger<StreamModule>("StripeHeight", param.stripeHeight);
        available_grabbers_[ix]->setInteger<StreamModule>("StripePitch", stripePitch);
        available_grabbers_[ix]->setInteger<StreamModule>("BlockHeight", param.block_height);
        available_grabbers_[ix]->setString<StreamModule>("StatisticsSamplingSelector", "LastSecond");
        available_grabbers_[ix]->setString<StreamModule>("LUTConfiguration", "M_10x8");
    }

    for (size_t i = 0; i < param.nb_grabbers; i++)
        available_grabbers_[i]->setInteger<StreamModule>("StripeOffset", param.offsets[i]);

    available_grabbers_[0]->setString<RemoteModule>("TriggerSource", param.triggerSource); // source of trigger CXP
    if (param.trigger_mode)
        available_grabbers_[0]->setString<RemoteModule>("TriggerMode",
                                                        param.trigger_mode.value()); // camera in triggered mode

    std::string control_mode = triggerSource == "SWTRIGGER" ? "RC" : "EXTERNAL";
    available_grabbers_[0]->setString<DeviceModule>("CameraControlMethod",
                                                    control_mode); // tell grabber 0 to send trigger

    if (param.trigger_selector)
        available_grabbers_[0]->setString<RemoteModule>("TriggerSelector",
                                                        trigger_selector.value()); // source of trigger CXP

    if (triggerSource == "SWTRIGGER")
    {
        available_grabbers_[0]->setInteger<DeviceModule>("CycleMinimumPeriod",
                                                         param.cycleMinimumPeriod); // set the trigger rate to 250K Hz
        available_grabbers_[0]->setString<DeviceModule>("ExposureReadoutOverlap",
                                                        "True"); // camera needs 2 trigger to start
        available_grabbers_[0]->setString<DeviceModule>("ErrorSelector", "All");
        if (param.acquisition_frame_rate)
            available_grabbers_[0]->setInteger<RemoteModule>("AcquisitionFrameRate",
                                                             param.acquisitionFrameRate.value());
    }
    available_grabbers_[0]->setFloat<RemoteModule>("ExposureTime", exposureTime);
    available_grabbers_[0]->setString<RemoteModule>("BalanceWhiteMarker", balance_white_marker);

    available_grabbers_[0]->setFloat<RemoteModule>("Gain", gain);
    available_grabbers_[0]->setString<RemoteModule>("GainSelector", gain_selector);

    if (param.flat_field_correction)
        available_grabbers_[0]->setString<RemoteModule>("FlatFieldCorrection", param.flat_field_correction.value());
}

void EHoloGrabber::init(unsigned int nb_buffers)
{
    nb_buffers_ = nb_buffers;
    size_t frame_size = width_ * height_ * depth_;

    // Allocate buffers in pinned memory
    // Learn more about pinned memory:
    // https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/.

    uint8_t* device_ptr;

    cudaError_t alloc_res = cudaHostAlloc(&ptr_, frame_size * buffer_part_count_ * nb_buffers_, cudaHostAllocMapped);
    cudaError_t device_ptr_res = cudaHostGetDevicePointer(&device_ptr, ptr_, 0);
    if (alloc_res != cudaSuccess || device_ptr_res != cudaSuccess)
        Logger::camera()->error("Could not allocate buffers.");

    float prog = 0.0;
    for (size_t i = 0; i < nb_buffers; ++i)
    {
        // progress bar of the allocation of the ram buffers on the cpu.
        prog = (float)i / (nb_buffers - 1);
        int barWidth = 100;

        std::cout << "[";
        int pos = barWidth * prog;
        for (int i = 0; i < barWidth; ++i)
        {
            if (i < pos)
                std::cout << "=";
            else if (i == pos)
                std::cout << ">";
            else
                std::cout << " ";
        }
        std::cout << "] " << int(prog * 100.0) << " %\r";
        std::cout.flush();

        // The EGrabber API can handle directly buffers alocated in pinned
        // memory as we just have to use cudaHostAlloc and give each grabber
        // the host pointer and the associated pointer in device memory.

        size_t offset = i * frame_size * buffer_part_count_;

        for (size_t ix = 0; ix < nb_grabbers_; ix++)
        {
            available_grabbers_[ix]->announceAndQueue(
                UserMemory(ptr_ + offset, frame_size * buffer_part_count_, device_ptr + offset));
        }
    }
    std::cout << std::endl;
}

void EHoloGrabber::start()
{
    // Start each sub grabber in reverse order
    for (size_t i = 0; i < nb_grabbers_; i++)
    {
        available_grabbers_[nb_grabbers_ - 1 - i]->enableEvent<NewBufferData>();
        available_grabbers_[nb_grabbers_ - 1 - i]->start();
    }
}

void EHoloGrabber::stop()
{
    for (size_t i = 0; i < nb_grabbers_; i++)
        available_grabbers_[i]->stop();
}

} // namespace camera