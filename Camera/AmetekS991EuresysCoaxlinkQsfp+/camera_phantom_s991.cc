#include <iostream>
#include <cmath>
#include <cstdlib>

#include <cuda.h>
#include <cuda_runtime.h>

#include "camera_phantom_s991.hh"
#include "camera_logger.hh"

namespace camera
{
CameraPhantom::CameraPhantom()
    : Camera("phantom.ini")
{
    name_ = "Phantom S991";
    pixel_size_ = 20;

    load_default_params();
    if (ini_file_is_open())
    {
        load_ini_params();
        ini_file_.close();
    }

    gentl_ = std::make_unique<Euresys::EGenTL>();
    grabber_ = std::make_unique<EHoloGrabber>(*gentl_, nb_images_per_buffer_, pixel_format_);

    init_camera();
}

void CameraPhantom::init_camera()
{
    grabber_->setup(fullHeight_,
                    width_,
                    nb_grabbers_,
                    trigger_source_,
                    exposure_time_,
                    cycle_minimum_period_,
                    pixel_format_,
                    *gentl_);
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

void CameraPhantom::load_default_params()
{
    nb_buffers_ = 64;
    nb_images_per_buffer_ = 4;
    nb_grabbers_ = 2;
    fullHeight_ = 512;
    width_ = 512;
    trigger_source_ = "SWTRIGGER";
    trigger_selector_ = "ExposureStart";
    exposure_time_ = 9000;
    cycle_minimum_period_ = "10000.0";
    pixel_format_ = "Mono8";
}

void CameraPhantom::load_ini_params()
{
    const boost::property_tree::ptree& pt = get_ini_pt();
    nb_buffers_ = pt.get<unsigned int>("phantom.NbBuffers", nb_buffers_);
    nb_images_per_buffer_ = pt.get<unsigned int>("phantom.NbImagesPerBuffer", nb_images_per_buffer_);
    nb_grabbers_ = pt.get<unsigned int>("phantom.NbGrabbers", nb_grabbers_);
    fullHeight_ = pt.get<unsigned int>("phantom.FullHeight", fullHeight_);
    width_ = pt.get<unsigned int>("phantom.Width", width_);
    trigger_source_ = pt.get<std::string>("phantom.TriggerSource", trigger_source_);
    trigger_selector_ = pt.get<std::string>("phantom.TriggerSelector", trigger_selector_);
    exposure_time_ = pt.get<float>("phantom.ExposureTime", exposure_time_);
    cycle_minimum_period_ = pt.get<std::string>("phantom.CycleMinimumPeriod", cycle_minimum_period_);
    pixel_format_ = pt.get<std::string>("phantom.PixelFormat", pixel_format_);

    if (nb_grabbers_ != 4 && nb_grabbers_ != 2)
    {
        nb_grabbers_ = 4;
        Logger::camera()->warn("Invalid number of grabbers fallback to default value 4.");
    }
}

void CameraPhantom::bind_params() { return; }

ICamera* new_camera_device() { return new CameraPhantom(); }
} // namespace camera