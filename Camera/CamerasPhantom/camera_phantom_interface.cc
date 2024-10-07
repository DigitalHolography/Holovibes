#include "camera_phantom_interface.hh"

#include <cuda.h>
#include <cuda_runtime.h>

#include "camera_logger.hh"
#include "spdlog/spdlog.h"
#include "camera_exception.hh"

#include <EGrabber.h>
#include <EGrabbers.h>

#include <iostream>

namespace camera
{

EHoloGrabberInt::EHoloGrabberInt(Euresys::EGenTL& gentl,
                                 unsigned int buffer_part_count,
                                 std::string& pixel_format,
                                 unsigned int nb_grabbers)
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
            grabbers_[ix]->getString<Euresys::RemoteModule>("Banks");
        }
        catch (const Euresys::gentl_error&)
        {
            continue;
        }
        available_grabbers_.push_back(grabbers_[ix]);
    }
}

EHoloGrabberInt::~EHoloGrabberInt()
{
    for (size_t i = 0; i < nb_grabbers_; i++)
        available_grabbers_[i]->reallocBuffers(0);

    cudaFreeHost(ptr_);
}

void EHoloGrabberInt::setup(const SetupParam& param, Euresys::EGenTL& gentl)
{
    width_ = param.width;
    height_ = param.full_height;

    size_t pitch = param.width * gentl.imageGetBytesPerPixel(param.pixel_format);
    size_t height = param.full_height / param.nb_grabbers;
    size_t stripe_pitch = param.stripe_height * param.nb_grabbers;

    for (size_t ix = 0; ix < param.nb_grabbers; ++ix)
    {
        available_grabbers_[ix]->setInteger<Euresys::RemoteModule>("Width", static_cast<int64_t>(param.width));
        available_grabbers_[ix]->setInteger<Euresys::RemoteModule>("Height", static_cast<int64_t>(height));
        available_grabbers_[ix]->setString<Euresys::RemoteModule>("PixelFormat", param.pixel_format);

        available_grabbers_[ix]->setString<Euresys::StreamModule>("StripeArrangement", param.stripe_arrangement);
        available_grabbers_[ix]->setInteger<Euresys::StreamModule>("LinePitch", pitch);
        available_grabbers_[ix]->setInteger<Euresys::StreamModule>("LineWidth", pitch);
        available_grabbers_[ix]->setInteger<Euresys::StreamModule>("StripeHeight", param.stripe_height);
        available_grabbers_[ix]->setInteger<Euresys::StreamModule>("StripePitch", stripe_pitch);
        available_grabbers_[ix]->setInteger<Euresys::StreamModule>("BlockHeight", param.block_height);
        available_grabbers_[ix]->setString<Euresys::StreamModule>("StatisticsSamplingSelector", "LastSecond");
        available_grabbers_[ix]->setString<Euresys::StreamModule>("LUTConfiguration", "M_10x8");
    }

    for (size_t i = 0; i < param.nb_grabbers; ++i)
        available_grabbers_[i]->setInteger<Euresys::StreamModule>("StripeOffset", param.offsets[i]);

    available_grabbers_[0]->setString<Euresys::RemoteModule>("TriggerSource",
                                                             param.trigger_source); // source of trigger CXP
    if (param.trigger_mode)
        available_grabbers_[0]->setString<Euresys::RemoteModule>(
            "TriggerMode",
            param.trigger_mode.value()); // camera in triggered mode

    std::string control_mode = param.trigger_source == "SWTRIGGER" ? "RC" : "EXTERNAL";
    available_grabbers_[0]->setString<Euresys::DeviceModule>("CameraControlMethod",
                                                             control_mode); // tell grabber 0 to send trigger

    if (param.trigger_selector)
        available_grabbers_[0]->setString<Euresys::RemoteModule>(
            "TriggerSelector",
            param.trigger_selector.value()); // source of trigger CXP

    if (param.trigger_source == "SWTRIGGER")
    {
        available_grabbers_[0]->setInteger<Euresys::DeviceModule>(
            "CycleMinimumPeriod",
            param.cycle_minimum_period); // set the trigger rate to 250K Hz
        available_grabbers_[0]->setString<Euresys::DeviceModule>("ExposureReadoutOverlap",
                                                                 "True"); // camera needs 2 trigger to start
        available_grabbers_[0]->setString<Euresys::DeviceModule>("ErrorSelector", "All");
    }
    available_grabbers_[0]->setFloat<Euresys::RemoteModule>("ExposureTime", param.exposure_time);
    available_grabbers_[0]->setString<Euresys::RemoteModule>("BalanceWhiteMarker", param.balance_white_marker);

    available_grabbers_[0]->setFloat<Euresys::RemoteModule>("Gain", param.gain);
    available_grabbers_[0]->setString<Euresys::RemoteModule>("GainSelector", param.gain_selector);
}

void EHoloGrabberInt::init(unsigned int nb_buffers)
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

        for (size_t ix = 0; ix < nb_grabbers_; ++ix)
        {
            available_grabbers_[ix]->announceAndQueue(
                Euresys::UserMemory(ptr_ + offset, frame_size * buffer_part_count_, device_ptr + offset));
        }
    }
    std::cout << std::endl;
}

void EHoloGrabberInt::start()
{
    // Start each sub grabber in reverse order
    for (size_t i = 0; i < nb_grabbers_; ++i)
    {
        available_grabbers_[nb_grabbers_ - 1 - i]->enableEvent<Euresys::NewBufferData>();
        available_grabbers_[nb_grabbers_ - 1 - i]->start();
    }
}

void EHoloGrabberInt::stop()
{
    for (size_t i = 0; i < nb_grabbers_; i++)
        available_grabbers_[i]->stop();
}

// namespace
// {
// void dispatch_init(CameraPhantomInt* cam) { cam->init_camera(); }

// std::unique_ptr<EHoloGrabberInt> dispatch_make_holo_grabber(CameraPhantomInt* cam) { return cam->make_holo_grabber();
// } } // namespace std::unique_ptr<EHoloGrabberInt> CameraPhantomInt::call_make_holo_grabber() { return
// make_holo_grabber(); }

CameraPhantomInt::CameraPhantomInt(const std::string& ini_name, const std::string& ini_prefix)
    : Camera(ini_name)
    , ini_prefix_(ini_prefix)
{

    pixel_size_ = 20;

    if (ini_file_is_open())
    {
        load_ini_params();
        ini_file_.close();
    }

    gentl_ = std::make_unique<Euresys::EGenTL>();
    // grabber_ = call_make_holo_grabber();
    // std::make_unique<EHoloGrabberInt>(*gentl_, buffer_part_count_, pixel_format_, nb_grabbers_);

    // dispatch_init(this);
    // this->init_camera();
}
// std::unique_ptr<EHoloGrabberInt> CameraPhantomInt::make_holo_grabber() { return {}; }

// void CameraPhantomInt::init_camera() {}

void CameraPhantomInt::init_camera_(EHoloGrabberInt::SetupParam& param)
{
    grabber_->setup(param, *gentl_);
    grabber_->init(nb_buffers_);

    // Set frame descriptor according to grabber settings
    fd_.width = grabber_->width_;
    fd_.height = grabber_->height_;
    fd_.depth = grabber_->depth_;
    fd_.byteEndian = Endianness::LittleEndian;
}

void CameraPhantomInt::start_acquisition() { grabber_->start(); }

void CameraPhantomInt::stop_acquisition() { grabber_->stop(); }

void CameraPhantomInt::shutdown_camera() { return; }

CapturedFramesDescriptor CameraPhantomInt::get_frames()
{
    Euresys::ScopedBuffer buffer(*(grabber_->available_grabbers_[0]));

    for (int i = 1; i < nb_grabbers_; ++i)
        Euresys::ScopedBuffer stiching(*(grabber_->available_grabbers_[i]));

    // process available images
    size_t delivered = buffer.getInfo<size_t>(Euresys::ge::BUFFER_INFO_CUSTOM_NUM_DELIVERED_PARTS);

    CapturedFramesDescriptor ret;

    ret.on_gpu = true;
    ret.region1 = buffer.getUserPointer();
    ret.count1 = delivered;

    ret.region2 = nullptr;
    ret.count2 = 0;

    return ret;
}

void CameraPhantomInt::load_default_params() {}

void CameraPhantomInt::load_ini_params()
{
    const boost::property_tree::ptree& pt = get_ini_pt();
    std::string prefix = ini_prefix_ + ".";

    nb_buffers_ = pt.get<unsigned int>(prefix + "NbBuffers", nb_buffers_);
    buffer_part_count_ = pt.get<unsigned int>(prefix + "BufferPartCount", buffer_part_count_);
    nb_grabbers_ = pt.get<unsigned int>(prefix + "NbGrabbers", nb_grabbers_);
    full_height_ = pt.get<unsigned int>(prefix + "FullHeight", full_height_);
    width_ = pt.get<unsigned int>(prefix + "Width", width_);

    for (size_t i = 0; i < NB_MAX_GRABBER; ++i)
        stripe_offsets_[i] = pt.get<unsigned int>(prefix + "Offset" + std::to_string(i), stripe_offsets_[i]);

    trigger_source_ = pt.get<std::string>(prefix + "TriggerSource", trigger_source_);
    trigger_selector_ = pt.get<std::string>(prefix + "TriggerSelector", trigger_selector_);
    exposure_time_ = pt.get<float>(prefix + "ExposureTime", exposure_time_);
    cycle_minimum_period_ = pt.get<unsigned int>(prefix + "CycleMinimumPeriod", cycle_minimum_period_);
    pixel_format_ = pt.get<std::string>(prefix + "PixelFormat", pixel_format_);

    gain_selector_ = pt.get<std::string>(prefix + "GainSelector", gain_selector_);
    trigger_mode_ = pt.get<std::string>(prefix + "TriggerMode", trigger_mode_);
    gain_ = pt.get<float>(prefix + "Gain", gain_);
    balance_white_marker_ = pt.get<std::string>(prefix + "BalanceWhiteMarker", balance_white_marker_);
    fan_ctrl_ = pt.get<std::string>(prefix + "FanCtrl", fan_ctrl_);
    acquisition_frame_rate_ = pt.get<unsigned int>(prefix + "AcquisitionFrameRate", acquisition_frame_rate_);
}

void CameraPhantomInt::bind_params() { return; }

} // namespace camera