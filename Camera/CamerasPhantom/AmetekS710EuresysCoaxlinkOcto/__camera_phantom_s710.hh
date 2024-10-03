#pragma once

#include <EGrabber.h>
#include <EGrabbers.h>

#include "camera.hh"
#include "camera_exception.hh"

#include "spdlog/spdlog.h"
#include "camera_logger.hh"

#include <stdio.h>

namespace camera
{
using namespace Euresys;

/*! \class EHoloSubGrabber
 *
 * \brief Class to handle the different EHoloSubGrabber used to acquire images
 * from the Phantom S710 with a Coaxlink Octo frame grabber.
 *
 * By extending EGrabberCallbackOnDemand, the events and callbacks are handled
 * by the same thread calling get_frames (camera frame read worker) as we do not
 * need to capture multiple frames at once.
 */

// FIXME Delete this class
class EHoloSubGrabber : public EGrabberCallbackOnDemand
{
  public:
    EHoloSubGrabber(EGenTL& gentl,
                    int interfaceIndex,
                    int deviceIndex,
                    int dataStreamIndex,
                    gc::DEVICE_ACCESS_FLAGS deviceOpenFlags,
                    bool remoteRequired)
        : EGrabberCallbackOnDemand(gentl, interfaceIndex, deviceIndex, dataStreamIndex, deviceOpenFlags, remoteRequired)
    {
    }
};

/*! \class EHoloGrabber
 *
 *\brief Class to handle the different EHoloSubGrabber used to acquire images
 * from the Phantom S710 with a Coaxlink Octo frame grabber.
 *
 * This implementation supposes that the frame grabber has been configured
 * properly, through the GenICam API, so that:
 * 1. Only banks A and B are used.
 * 2. Each bank is responsible for capturing half of the full image height.
 *
 * For instance, to capture a frame of 1024*512, the first and second grabber
 * will acquire 1024*256 and stack both parts to create the full 1024*512 image.
 *
 * The documentation of the Euresys eGrabber Programmer Guide can be found at
 * https://documentation.euresys.com/Products/COAXLINK/COAXLINK_14_0/en-us/Content/00_Home/PDF_Guides.htm.
 */
class EHoloGrabberInt
{
    EHoloGrabber(EGenTL& gentl, unsigned int buffer_part_count, std::string& pixel_format)
        : grabbers_(gentl)
        , buffer_part_count_(buffer_part_count)
    {
        // Fetch the first grabber info to determine the width, height and depth
        // of the full image.
        // According to the requirements described above, we assume that the
        // full height is two times the height of the first grabber.

        depth_ = static_cast<PixelDepth>(gentl.imageGetBytesPerPixel(pixel_format));
        for (unsigned i = 0; i < grabbers_.length(); ++i)
            grabbers_[i]->setInteger<StreamModule>("BufferPartCount", buffer_part_count_);
    }

  public:
    virtual ~EHoloGrabber()
    {
        for (size_t i = 0; i < grabbers_.length(); i++)
            grabbers_[i]->reallocBuffers(0);

        cudaFreeHost(ptr_);
    }

    void setup(unsigned int fullHeight,
               unsigned int width,
               unsigned int nb_grabbers,
               unsigned int offset0,
               unsigned int offset1,
               unsigned int offset2,
               unsigned int offset3,
               std::string& triggerSource,
               float exposureTime,
               std::string& cycleMinimumPeriod,
               std::string& pixelFormat,
               std::string& trigger_mode,
               std::string& fan_ctrl,
               float gain,
               std::string& balance_white_marker,
               std::string& gain_selector,
               std::string& flat_field_correction,
               EGenTL& gentl)
    {
        width_ = width;
        height_ = fullHeight;

        // keep this
        // dynamic detection of the number of banks available.
        if (nb_grabbers == 0)
            nb_grabbers = grabbers_.length();

        if (nb_grabbers == 2)
        {
            grabbers_[0]->setString<RemoteModule>("Banks", "Banks_AB");
        }
        else if (nb_grabbers == 4)
        {
            grabbers_[0]->setString<RemoteModule>("Banks", "Banks_ABCD");
        } // else Error

        size_t pitch = width * gentl.imageGetBytesPerPixel(pixelFormat);
        size_t grabberCount = grabbers_.length();
        size_t height = fullHeight / grabberCount;
        size_t stripeHeight = 8;
        size_t stripePitch = stripeHeight * grabberCount;

        for (size_t ix = 0; ix < grabberCount; ++ix)
        {
            grabbers_[ix]->setInteger<RemoteModule>("Width", static_cast<int64_t>(width));
            grabbers_[ix]->setInteger<RemoteModule>("Height", static_cast<int64_t>(height));
            grabbers_[ix]->setString<RemoteModule>("PixelFormat", pixelFormat);

            grabbers_[ix]->setString<StreamModule>("StripeArrangement", "Geometry_1X_2YM");
            grabbers_[ix]->setInteger<StreamModule>("LinePitch", pitch);
            grabbers_[ix]->setInteger<StreamModule>("LineWidth", pitch);
            grabbers_[ix]->setInteger<StreamModule>("StripeHeight", stripeHeight);
            grabbers_[ix]->setInteger<StreamModule>("StripePitch", stripePitch);
            grabbers_[ix]->setInteger<StreamModule>("BlockHeight", 8);
            grabbers_[ix]->setString<StreamModule>("StatisticsSamplingSelector", "LastSecond");
            grabbers_[ix]->setString<StreamModule>("LUTConfiguration", "M_10x8");
            // grabbers_[ix]->setInteger<StreamModule>("StripeOffset", 8 * ix);
        }

        grabbers_[0]->setInteger<StreamModule>("StripeOffset", offset0);
        grabbers_[1]->setInteger<StreamModule>("StripeOffset", offset1);
        if (nb_grabbers == 4)
        {
            grabbers_[2]->setInteger<StreamModule>("StripeOffset", offset2);
            grabbers_[3]->setInteger<StreamModule>("StripeOffset", offset3);
        }

        grabbers_[0]->setString<RemoteModule>("TriggerMode", trigger_mode);    // camera in triggered mode
        grabbers_[0]->setString<RemoteModule>("TriggerSource", triggerSource); // source of trigger CXP
        std::string control_mode = triggerSource == "SWTRIGGER" ? "RC" : "EXTERNAL";
        grabbers_[0]->setString<DeviceModule>("CameraControlMethod", control_mode); // tell grabber 0 to send trigger

        /* 100 fps -> 10000us */
        // float factor = fps / 100;
        // float cycleMinimumPeriod = 10000 / factor;
        // float cycleMinimumPeriod = 1e6 / fps;
        // std::string CycleMinimumPeriod = std::to_string(cycleMinimumPeriod);
        if (triggerSource == "SWTRIGGER")
        {
            grabbers_[0]->setString<DeviceModule>("CycleMinimumPeriod",
                                                  cycleMinimumPeriod);               // set the trigger rate to 250K Hz
            grabbers_[0]->setString<DeviceModule>("ExposureReadoutOverlap", "True"); // camera needs 2 trigger to start
            grabbers_[0]->setString<DeviceModule>("ErrorSelector", "All");
            grabbers_[0]->setString<RemoteModule>("TimeStamp", "TSOff");
        }

        /* 100 fps -> 9000us */
        // float factor = fps / 100;
        // float Expvalue = 9000 / factor;
        grabbers_[0]->setFloat<RemoteModule>("ExposureTime", exposureTime);
        grabbers_[0]->setString<RemoteModule>("BalanceWhiteMarker", balance_white_marker);

        grabbers_[0]->setString<RemoteModule>("GainSelector", gain_selector);
        grabbers_[0]->setFloat<RemoteModule>("Gain", gain);
        grabbers_[0]->setString<RemoteModule>("FanCtrl", fan_ctrl); // TODO DO TO DO
        grabbers_[0]->setString<RemoteModule>("FlatFieldCorrection", flat_field_correction);
    }

    void init(unsigned int nb_buffers)
    {
        nb_buffers_ = nb_buffers;
        size_t grabber_count = grabbers_.length();
        size_t frame_size = width_ * height_ * depth_;

        // Allocate buffers in pinned memory
        // Learn more about pinned memory:
        // https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/.

        uint8_t* device_ptr;

        cudaError_t alloc_res =
            cudaHostAlloc(&ptr_, frame_size * buffer_part_count_ * nb_buffers_, cudaHostAllocMapped);
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

            for (size_t ix = 0; ix < grabber_count; ix++)
            {
                grabbers_[ix]->announceAndQueue(
                    UserMemory(ptr_ + offset, frame_size * buffer_part_count_, device_ptr + offset));
            }
        }
        std::cout << std::endl;
    }

    void start()
    {
        size_t grabber_count = grabbers_.length();

        // Start each sub grabber in reverse order
        for (size_t i = 0; i < grabber_count; i++)
        {
            grabbers_[grabber_count - 1 - i]->enableEvent<NewBufferData>();
            grabbers_[grabber_count - 1 - i]->start();
        }
    }

    void stop()
    {
        for (size_t i = 0; i < grabbers_.length(); i++)
            grabbers_[i]->stop();
    }

    /*! \brief The width of the acquired frames. */
    unsigned int width_;

    /*! \brief The total height of the acquired frames. */
    unsigned int height_;

    /*! \brief The depth of the acquired frames. */
    PixelDepth depth_;

    /*! \brief An EGrabbers instance composed of the two EHoloSubGrabber grabbers.  */
    EGrabbers<EHoloSubGrabber> grabbers_;

  private:
    /*! \brief The number of buffers used to store frames. It is equivalent to
     * the number of frames to store simultaneously.
     */
    unsigned int nb_buffers_;

    /*! \brief The number of images stored in each buffers.
     */
    unsigned int buffer_part_count_;

    /*! \brief A pointer the cuda memory allocated for the buffers.
     */
    uint8_t* ptr_;
};

class CameraPhantom : public Camera
{
  public:
    CameraPhantom(bool gpu = true);
    virtual ~CameraPhantom() {}

    virtual void init_camera() override;
    virtual void start_acquisition() override;
    virtual void stop_acquisition() override;
    virtual void shutdown_camera() override;
    virtual CapturedFramesDescriptor get_frames() override;

  private:
    virtual void load_ini_params() override;
    virtual void load_default_params() override;
    virtual void bind_params() override;

    std::unique_ptr<EGenTL> gentl_;
    std::unique_ptr<EHoloGrabber> grabber_;

    unsigned int nb_buffers_;
    unsigned int buffer_part_count_;
    unsigned int nb_grabbers_;
    unsigned int fullHeight_;
    unsigned int width_;

    unsigned int stripeOffset_grabber_0_;
    unsigned int stripeOffset_grabber_1_;
    unsigned int stripeOffset_grabber_2_;
    unsigned int stripeOffset_grabber_3_;

    std::string trigger_source_;
    std::string trigger_selector_;
    float exposure_time_;
    std::string cycle_minimum_period_;
    std::string pixel_format_;

    std::string trigger_mode_;
    std::string fan_ctrl_;
    float gain_;
    std::string balance_white_marker_;
    std::string gain_selector_;
    std::string flat_field_correction_;
};
} // namespace camera
