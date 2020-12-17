/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

#include <EGrabber.h>
#include <EGrabbers.h>

#include "camera.hh"
#include "camera_exception.hh"

namespace camera
{
using namespace Euresys;

class EHoloSubGrabber : public EGrabberCallbackOnDemand
{
  public:
    EHoloSubGrabber(EGenTL& gentl,
                    int interfaceIndex,
                    int deviceIndex,
                    int dataStreamIndex,
                    gc::DEVICE_ACCESS_FLAGS deviceOpenFlags,
                    bool remoteRequired)
        : EGrabberCallbackOnDemand(gentl,
                                   interfaceIndex,
                                   deviceIndex,
                                   dataStreamIndex,
                                   deviceOpenFlags,
                                   remoteRequired)
    {
    }

    uint8_t* last_ptr_;

  private:
    virtual void onNewBufferEvent(const NewBufferData& data)
    {
        ScopedBuffer buffer(*this, data);
        last_ptr_ = static_cast<uint8_t*>(buffer.getUserPointer());
    }
};

class EHoloGrabber
{
  public:
    EHoloGrabber(EGenTL& gentl)
        : grabbers_(gentl)
    {
        std::string pixel_format = grabbers_[0]->getPixelFormat();

        width_ = grabbers_[0]->getWidth();
        height_ = grabbers_[0]->getHeight() * 2;
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

        // Allocate memory
        buffers_.reserve(nb_buffers);
        while (buffers_.size() < nb_buffers)
        {
            uint8_t *ptr, *device_ptr;
            cudaError_t alloc_res =
                cudaHostAlloc(&ptr, frame_size, cudaHostAllocMapped);
            cudaError_t device_ptr_res =
                cudaHostGetDevicePointer(&device_ptr, ptr, 0);

            if (alloc_res != cudaSuccess || device_ptr_res != cudaSuccess)
                std::cerr << "[CAMERA] Could not allocate buffers."
                          << std::endl;

            buffers_.push_back(ptr);
            for (size_t ix = 0; ix < grabber_count; ix++)
                grabbers_[ix]->announceAndQueue(
                    UserMemory(ptr, frame_size, device_ptr));
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
        for (size_t i = 0; i < grabbers_.length(); i++)
            grabbers_[i]->processEvent<NewBufferData>(FRAME_TIMEOUT);

        return grabbers_[0]->last_ptr_;
    }

    void stop()
    {
        for (size_t i = 0; i < grabbers_.length(); i++)
            grabbers_[i]->stop();
    }

    unsigned int width_;
    unsigned int height_;
    unsigned int depth_;

  private:
    std::unique_ptr<EGenTL> gentl_;
    EGrabbers<EHoloSubGrabber> grabbers_;

    unsigned int nb_buffers_;
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