#pragma once

#include <EGrabber.h>
#include <EGrabbers.h>

#include "camera.hh"
#include "camera_exception.hh"

#include "spdlog/spdlog.h"
#include "camera_logger.hh"

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
class EHoloGrabber
{
  public:
    EHoloGrabber(EGenTL& gentl)
        : grabbers_(gentl)
    {
        // Fetch the first grabber info to determine the width, height and depth
        // of the full image.
        // According to the requirements described above, we assume that the
        // full height is two times the height of the first grabber.

        // Get pixel format and nb images
        std::string pixel_format = grabbers_[0]->getPixelFormat();
        buffer_part_count_ = grabbers_[0]->getInteger<StreamModule>("BufferPartCount");

        width_ = grabbers_[0]->getWidth();
        height_ = grabbers_[0]->getHeight() * grabbers_.length();
        depth_ = static_cast<PixelDepth>(gentl.imageGetBytesPerPixel(pixel_format));

        // for (unsigned i = 0; i < grabbers_.length(); ++i)
        //     grabbers_[i]->setInteger<StreamModule>("BufferPartCount", buffer_part_count);
    }

    virtual ~EHoloGrabber()
    {
        for (size_t i = 0; i < grabbers_.length(); i++)
            grabbers_[i]->reallocBuffers(0);

        cudaFreeHost(ptr_);
    }

    void setup()
    {
        grabbers_.root[0][1].reposition(0);
        grabbers_.root[0][1].reposition(1);

        // grabbers_[ix]->setInteger<StreamModule>("BlockHeight", 0);
        // grabbers_[ix]->setInteger<StreamModule>("StripeOffset", 4 * ix);

        size_t block_height = grabbers_.length() == 2 ? 0 : 8;
        size_t stripe_offset = grabbers_.length() == 2 ? 4 : 8;

        for (size_t ix = 0; ix < grabbers_.length(); ix++)
        {
            grabbers_[ix]->setInteger<StreamModule>("BlockHeight", block_height);
            grabbers_[ix]->setInteger<StreamModule>("StripeOffset", stripe_offset * ix);
        }
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

        for (size_t i = 0; i < nb_buffers; ++i)
        {
            // The EGrabber API can handle directly buffers alocated in pinned
            // memory as we just have to use cudaHostAlloc and give each grabber
            // the host pointer and the associated pointer in device memory.

            size_t offset = i * frame_size * buffer_part_count_;
            for (size_t ix = 0; ix < grabber_count; ix++)
                grabbers_[ix]->announceAndQueue(
                    UserMemory(ptr_ + offset, frame_size * buffer_part_count_, device_ptr + offset));
        }
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
    CameraPhantom();
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
