#pragma once

#include <EGrabber.h>
#include <EGrabbers.h>

#include "camera.hh"
#include "camera_exception.hh"

#include <spdlog/spdlog.h>

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

    /*! \brief Raw pointer to the last frame captured by onNewBufferEvent. */
    uint8_t* last_ptr_;

  private:
    virtual void onNewBufferEvent(const NewBufferData& data)
    {
        /* Using ScopedBuffer will tell the grabber that the buffer is available
         * to store new acquired frames at the end of this scope. This behavior
         * is not an issue as the next time onNewBufferEvent will be called, the
         * previous frame will have already been enqueued in Holovibes input
         * queue.
         */
        ScopedBuffer buffer(*this, data);
        last_ptr_ = static_cast<uint8_t*>(buffer.getUserPointer());
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

        width_ = grabbers_[0]->getWidth();
        height_ = grabbers_[0]->getHeight() * 2;
        std::string pixel_format = grabbers_[0]->getPixelFormat();
        depth_ = gentl.imageGetBytesPerPixel(pixel_format);
    }

    virtual ~EHoloGrabber()
    {
        for (size_t i = 0; i < grabbers_.length(); i++)
            grabbers_[i]->reallocBuffers(0);

        for (size_t i = 0; i < buffers_.size(); i++)
            cudaFreeHost(buffers_[i]);

        buffers_.clear();
    }

    void init(unsigned int nb_buffers)
    {
        nb_buffers_ = nb_buffers;
        size_t grabber_count = grabbers_.length();
        size_t frame_size = width_ * height_ * depth_;

        // Allocate buffers in pinned memory
        // Learn more about pinned memory:
        // https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/.

        buffers_.reserve(nb_buffers);
        while (buffers_.size() < nb_buffers)
        {
            // The EGrabber API can handle directly buffers alocated in pinned
            // memory as we just have to use cudaHostAlloc and give each grabber
            // the host pointer and the associated pointer in device memory.
            uint8_t *ptr, *device_ptr;
            cudaError_t alloc_res = cudaHostAlloc(&ptr, frame_size, cudaHostAllocMapped);
            cudaError_t device_ptr_res = cudaHostGetDevicePointer(&device_ptr, ptr, 0);

            if (alloc_res != cudaSuccess || device_ptr_res != cudaSuccess)
                spdlog::get("Setup")->error("Could not allocate buffers.");

            buffers_.push_back(ptr);
            for (size_t ix = 0; ix < grabber_count; ix++)
                grabbers_[ix]->announceAndQueue(UserMemory(ptr, frame_size, device_ptr));
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

    void* get_frame()
    {
        // For each grabber, if a new frame has been written into memory within
        // FRAME_TIMEOUT ms we call onNewBufferEvent. Otherwise, a timeout
        // exception will be thrown. This part of the code is thus blocking!
        for (size_t i = 0; i < grabbers_.length(); i++)
            grabbers_[i]->processEvent<NewBufferData>(FRAME_TIMEOUT);

        // The first and second grabber last_ptr_ is the identical.
        return grabbers_[0]->last_ptr_;
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
    unsigned int depth_;

  private:
    /*! \brief Unique ptr to the instance of the GenICam GenTL API. */
    std::unique_ptr<EGenTL> gentl_;

    /*! \brief An EGrabbers instance composed of the two EHoloSubGrabber grabbers.  */
    EGrabbers<EHoloSubGrabber> grabbers_;

    /*! \brief The number of buffers used to store frames. It is equivalent to
     * the number of frames to store simultaneously.
     */
    unsigned int nb_buffers_;

    /*! \brief A vector storing all host memory pointers allocated to later free them. */
    std::vector<uint8_t*> buffers_;
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
};
} // namespace camera
