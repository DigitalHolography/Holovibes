#include <iostream>
#include <cmath>
#include <cstdlib>

#include <cuda.h>
#include <cuda_runtime.h>

#include "camera_phantom.hh"
#include "camera_logger.hh"

namespace camera
{
CameraPhantom::CameraPhantom()
    : Camera("phantom.ini")
{
    name_ = "Phantom S710";
    pixel_size_ = 20;

    load_default_params();
    if (ini_file_is_open())
    {
        load_ini_params();
        ini_file_.close();
    }

    gentl_ = std::make_unique<Euresys::EGenTL>();
    grabber_ = std::make_unique<EHoloGrabber>(*gentl_, nb_images_per_buffer_);

    init_camera();
}

void CameraPhantom::init_camera()
{
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
    uint8_t *bufferPtr = buffer.getInfo<uint8_t *>(gc::BUFFER_INFO_BASE);
    // size_t imageSize = buffer.getInfo<size_t>(ge::BUFFER_INFO_CUSTOM_PART_SIZE);

    // process available images
    size_t delivered = buffer.getInfo<size_t>(ge::BUFFER_INFO_CUSTOM_NUM_DELIVERED_PARTS);

    CapturedFramesDescriptor ret;

    ret.on_gpu = false;
    ret.region1 = buffer.getUserPointer();
    ret.count1 = delivered;
    ret.count2 = 0;

    return ret;
}


void CameraPhantom::load_default_params()
{
    nb_buffers_ = 16;
    nb_images_per_buffer_ = 16;
}

void CameraPhantom::load_ini_params()
{
    const boost::property_tree::ptree& pt = get_ini_pt();
    nb_buffers_ = pt.get<unsigned int>("phantom.nb_buffers", nb_buffers_);
    nb_images_per_buffer_ = pt.get<unsigned int>("phantom.nb_images_per_buffer", nb_images_per_buffer_);
}

void CameraPhantom::bind_params() { return; }

ICamera* new_camera_device() { return new CameraPhantom(); }
} // namespace camera
