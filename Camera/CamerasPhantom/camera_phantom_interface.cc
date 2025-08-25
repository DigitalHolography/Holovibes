#include "camera_phantom_interface.hh"

#include <cuda.h>
#include <cuda_runtime.h>

#include <thread>
#include <chrono>

#include "camera_logger.hh"
#include "spdlog/spdlog.h"
#include "camera_exception.hh"

#include <EGrabber.h>

#include <iostream>

#include <windows.h>

static inline uint64_t qpc_now_us()
{
    LARGE_INTEGER qpc, freq;
    QueryPerformanceCounter(&qpc);
    QueryPerformanceFrequency(&freq);
    // Cast to long double to avoid overflow in the division
    long double us = (long double)qpc.QuadPart * 1e6L / (long double)freq.QuadPart;
    return (uint64_t)us;
}

struct SoftClockAlign
{
    // Minimal observed offset (host - camera)
    std::atomic<uint64_t> offset_min_us{UINT64_MAX};

    void observe(uint64_t t_cam_us, uint64_t t_host_us)
    {
        // t_host_us >= acquisition_time + latency >= t_cam_us + latency'
        // Therefore (t_host - t_cam) is an upper bound of the real offset.
        uint64_t cand = (t_host_us > t_cam_us) ? (t_host_us - t_cam_us) : 0;

        // Keep the minimum offset seen so far
        uint64_t cur = offset_min_us.load(std::memory_order_relaxed);
        while (cand < cur && !offset_min_us.compare_exchange_weak(cur, cand))
        { /* spin */
        }

        // [DEBUG] Log the current offset estimate
        // std::cout << "[SoftClockAlign] Candidate offset=" << cand
        //          << " us, current_min=" << offset_min_us.load(std::memory_order_relaxed) << " us" << std::endl;
    }

    uint64_t to_host_time(uint64_t t_cam_us) const
    {
        uint64_t off = offset_min_us.load(std::memory_order_relaxed);
        if (off == UINT64_MAX)
            return t_cam_us; // Not calibrated yet
        return t_cam_us + off;
    }
};

static SoftClockAlign g_align;

namespace
{
std::vector<Euresys::EGrabberInfo> find_grabbers(Euresys::EGenTL& gentl)
{
    using namespace Euresys;

    EGrabberDiscovery discovery(gentl);
    discovery.discover();
    std::vector<Euresys::EGrabberInfo> grabbers;
    for (int i = 0; i < discovery.egrabberCount(); ++i)
    {
        grabbers.push_back(discovery.egrabbers(i));
    }

    return grabbers;
}
} // namespace

namespace camera
{

EHoloGrabberInt::EHoloGrabberInt(Euresys::EGenTL& gentl,
                                 unsigned int buffer_part_count,
                                 std::string& pixel_format,
                                 unsigned int nb_grabbers)
    : grabbers_()
    , buffer_part_count_(buffer_part_count)
    , nb_grabbers_(nb_grabbers)
{
    // Get the available grabbers
    auto grabbers = find_grabbers(gentl);
    for (const auto& info : grabbers)
    {
        grabbers_.emplace_back(std::make_unique<Euresys::EGrabber<>>(info));
    }

    // Fetch the first grabber info to determine the width, height and depth
    // of the full image.
    // According to the requirements described above, we assume that the
    // full height is two times the height of the first grabber.

    depth_ = static_cast<PixelDepth>(gentl.imageGetBytesPerPixel(pixel_format));

    // The below loop will check which grabbers are available to use, i.e the ones which are connected to a camera
    // We don't use Euresys::EGrabberDiscovery because it doesn't allow us to detect when a frame grabber is
    // connected to something or not
    for (size_t ix = 0; ix < grabbers_.size(); ++ix)
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
        available_grabbers_.push_back(grabbers_[ix].get());
    }

    for (unsigned i = 0; i < available_grabbers_.size(); ++i)
        available_grabbers_[i]->setInteger<Euresys::StreamModule>("BufferPartCount", buffer_part_count_);
}

EHoloGrabberInt::~EHoloGrabberInt()
{
    for (size_t i = 0; i < available_grabbers_.size(); i++)
        available_grabbers_[i]->reallocBuffers(0);

    cudaFreeHost(ptr_);
}

void EHoloGrabberInt::setup(const CameraParamMap& params, Euresys::EGenTL& gentl)
{
    width_ = params.at<unsigned int>("Width");
    height_ = params.at<unsigned int>("FullHeight");

    size_t pitch = width_ * gentl.imageGetBytesPerPixel(params.at<std::string>("PixelFormat"));
    size_t height = height_ / nb_grabbers_;

    size_t stripe_pitch = params.at<unsigned int>("StripeHeight") * nb_grabbers_;

    for (size_t ix = 0; ix < nb_grabbers_; ++ix)
    {
        available_grabbers_[ix]->setInteger<Euresys::RemoteModule>("Width", static_cast<int64_t>(width_));
        available_grabbers_[ix]->setInteger<Euresys::RemoteModule>("Height", static_cast<int64_t>(height));
        available_grabbers_[ix]->setString<Euresys::RemoteModule>("PixelFormat", params.at<std::string>("PixelFormat"));

        available_grabbers_[ix]->setString<Euresys::StreamModule>("StripeArrangement",
                                                                  params.at<std::string>("StripeArrangement"));
        available_grabbers_[ix]->setInteger<Euresys::StreamModule>("LinePitch", pitch);
        available_grabbers_[ix]->setInteger<Euresys::StreamModule>("LineWidth", pitch);
        available_grabbers_[ix]->setInteger<Euresys::StreamModule>("StripeHeight",
                                                                   params.at<unsigned int>("StripeHeight"));
        available_grabbers_[ix]->setInteger<Euresys::StreamModule>("StripePitch", stripe_pitch);
        available_grabbers_[ix]->setInteger<Euresys::StreamModule>("BlockHeight",
                                                                   params.at<unsigned int>("BlockHeight"));
        available_grabbers_[ix]->setString<Euresys::StreamModule>("StatisticsSamplingSelector", "LastSecond");
        available_grabbers_[ix]->setString<Euresys::StreamModule>("LUTConfiguration", "M_10x8");
    }

    for (size_t i = 0; i < nb_grabbers_; ++i)
        available_grabbers_[i]->setInteger<Euresys::StreamModule>("StripeOffset",
                                                                  params.at<std::vector<unsigned int>>("Offset")[i]);

    if (params.has("TriggerMode"))
        available_grabbers_[0]->setString<Euresys::RemoteModule>("TriggerMode", params.at<std::string>("TriggerMode"));
    available_grabbers_[0]->setString<Euresys::RemoteModule>("TriggerSource", params.at<std::string>("TriggerSource"));

    std::string control_mode = params.at<std::string>("TriggerSource") == "SWTRIGGER" ? "RC" : "EXTERNAL";
    available_grabbers_[0]->setString<Euresys::DeviceModule>("CameraControlMethod", control_mode);

    if (params.has("TriggerSelector"))
        available_grabbers_[0]->setString<Euresys::RemoteModule>("TriggerSelector",
                                                                 params.at<std::string>("TriggerSelector"));

    if (params.at<std::string>("TriggerSource") == "SWTRIGGER")
    {
        available_grabbers_[0]->setInteger<Euresys::DeviceModule>("CycleMinimumPeriod",
                                                                  params.at<unsigned int>("CycleMinimumPeriod"));
        available_grabbers_[0]->setString<Euresys::DeviceModule>("ExposureReadoutOverlap", "True");
        available_grabbers_[0]->setString<Euresys::DeviceModule>("ErrorSelector", "All");
    }
    available_grabbers_[0]->setFloat<Euresys::RemoteModule>("ExposureTime", params.at<float>("ExposureTime"));
    available_grabbers_[0]->setString<Euresys::RemoteModule>("BalanceWhiteMarker",
                                                             params.at<std::string>("BalanceWhiteMarker"));

    available_grabbers_[0]->setFloat<Euresys::RemoteModule>("Gain", params.at<float>("Gain"));
    available_grabbers_[0]->setString<Euresys::RemoteModule>("GainSelector", params.at<std::string>("GainSelector"));
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

CameraPhantomInt::CameraPhantomInt(const std::string& ini_filename, const std::string& ini_prefix)
    : Camera(ini_filename)
    , ini_prefix_(ini_prefix)
    , params_{ini_prefix}
{

    pixel_size_ = 20;
    gentl_ = std::make_unique<Euresys::EGenTL>();
}

void CameraPhantomInt::init_camera()
{
    grabber_->setup(params_, *gentl_);
    grabber_->init(params_.at<unsigned int>("NbBuffers"));

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
    // 1) Host timestamp right BEFORE waiting (lower bound)
    uint64_t host_before_us = qpc_now_us();

    // 2) Blocking wait for a buffer
    auto buffer = Euresys::ScopedBuffer(*(grabber_->available_grabbers_[0]));

    // 3) Host timestamp right AFTER the buffer (upper bound, better for offset)
    uint64_t host_after_us = qpc_now_us();

    // Multi-grabber support
    unsigned int nb_grabbers = params_.at<unsigned int>("NbGrabbers");
    for (int i = 1; i < nb_grabbers; ++i)
        Euresys::ScopedBuffer stiching(*(grabber_->available_grabbers_[i]));

    // Number of delivered parts
    size_t delivered = buffer.getInfo<size_t>(Euresys::ge::BUFFER_INFO_CUSTOM_NUM_DELIVERED_PARTS);

    // 4) Camera (GenTL) timestamp for this frame
    uint64_t cam_ts_us = buffer.getInfo<uint64_t>(Euresys::gc::BUFFER_INFO_TIMESTAMP);

    // 5) Update offset estimator with the host-after marker
    g_align.observe(cam_ts_us, host_after_us);

    // 6) Host-synchronized timestamp for this frame
    uint64_t synced_ts_us = g_align.to_host_time(cam_ts_us);

    // 7) Nominal period (in µs)
    uint64_t period_us = 0;
    if (auto cycle_us_opt = params_.get<unsigned int>("CycleMinimumPeriod"))
        period_us = static_cast<uint64_t>(*cycle_us_opt);

    CapturedFramesDescriptor ret;
    ret.on_gpu = true;
    ret.region1 = buffer.getUserPointer();
    ret.count1 = delivered;
    ret.region2 = nullptr;
    ret.count2 = 0;

    ret.first_frame_timestamp_us = synced_ts_us; // Export host-synchronized timestamp
    ret.frame_period_us = period_us;
    ret.has_hw_timestamp = (cam_ts_us != 0);

    return ret;
}

void CameraPhantomInt::load_default_params()
{
    params_.set<unsigned int>("NbBuffers", 0);
    params_.set<unsigned int>("BufferPartCount", 0);
    params_.set<unsigned int>("NbGrabbers", 0);
    params_.set<unsigned int>("FullHeight", 0);
    params_.set<unsigned int>("Width", 0);
    params_.set<std::vector<unsigned int>>("Offset", std::vector<unsigned int>{0, 0, 0, 0});
    params_.set<unsigned int>("CycleMinimumPeriod", 0);
    params_.set<float>("Gain", 0);
    params_.set<float>("ExposureTime", 0);

    params_.set<std::string>("TriggerSource", "");
    params_.set<std::string>("PixelFormat", "");
    params_.set<std::string>("GainSelector", "");
    params_.set<std::string>("BalanceWhiteMarker", "");
}

void CameraPhantomInt::load_ini_params() { params_.set_from_ini(get_ini_pt()); }

void CameraPhantomInt::bind_params() { return; }

int CameraPhantomInt::get_temperature() const
{
    try
    {
        auto temperature = grabber_->available_grabbers_[0]->getInteger<Euresys::RemoteModule>("DeviceTemperature");
        return temperature;
    }
    catch (std::exception)
    {
        return 0;
    }
}

int CameraPhantomInt::get_camera_fps() const
{
    auto cycle = params_.get<unsigned int>("CycleMinimumPeriod");
    return cycle ? 1e6 / cycle.value() : -1;
}

} // namespace camera
