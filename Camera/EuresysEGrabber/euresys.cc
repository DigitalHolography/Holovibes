#include <iostream>
#include <cmath>
#include <cstdlib>

#include <cuda.h>
#include <cuda_runtime.h>

#include "euresys.hh"
#include "camera_logger.hh"

namespace camera
{
CameraPhantom::CameraPhantom()
    : Camera("ametek_s710_euresys_coaxlink_octo.ini")
{
    name_ = "Euresys eGrabber";
    pixel_size_ = 6.75;

    gentl_ = std::make_unique<Euresys::EGenTL>();
    grabber_ = std::make_unique<EHoloGrabber>(*gentl_);

    init_camera();
}

void CameraPhantom::init_camera()
{
    nb_buffers_ = 64;

    grabber_->setup();

    grabber_->init(nb_buffers_);

    // Set frame descriptor according to grabber settings
    fd_.width = grabber_->width_;
    fd_.height = grabber_->height_;
    fd_.depth = grabber_->depth_;
    fd_.byteEndian = Endianness::LittleEndian;
}

void CameraPhantom::start_acquisition() { grabber_->start(); }

void CameraPhantom::stop_acquisition() { grabber_->stop(); }

void CameraPhantom::shutdown_camera() { return; }

CapturedFramesDescriptor CameraPhantom::get_frames()
{
    ScopedBuffer buffer(*(grabber_->grabbers_[0]));

    for (int i = 1; i < grabber_->grabbers_.length(); ++i)
        ScopedBuffer stiching(*(grabber_->grabbers_[i]));

    // process available images
    size_t delivered = buffer.getInfo<size_t>(ge::BUFFER_INFO_CUSTOM_NUM_DELIVERED_PARTS);

    CapturedFramesDescriptor ret;

    ret.on_gpu = true;
    ret.region1 = buffer.getUserPointer();
    ret.count1 = delivered;

    ret.region2 = nullptr;
    ret.count2 = 0;

    return ret;
}

void CameraPhantom::load_default_params() { return; }

void CameraPhantom::load_ini_params() { return; }

void CameraPhantom::bind_params() { return; }

ICamera* new_camera_device() { return new CameraPhantom(); }
} // namespace camera
