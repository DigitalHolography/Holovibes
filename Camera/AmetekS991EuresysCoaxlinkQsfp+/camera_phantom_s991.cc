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
    : Camera("ametek_s991_euresys_coaxlink_qsfp+.ini")
{
    name_ = "Phantom S991";
    pixel_size_ = 20;

    if (ini_file_is_open())
    {
        load_ini_params();
        ini_file_.close();
    }

    gentl_ = std::make_unique<Euresys::EGenTL>();
    grabber_ = std::make_unique<EHoloGrabber>(*gentl_, buffer_part_count_, pixel_format_);

    init_camera();
}

void CameraPhantom::init_camera()
{
    grabber_->setup(fullHeight_,
                    width_,
                    nb_grabbers_,
                    stripeOffset_grabber_0_,
                    stripeOffset_grabber_1_,
                    trigger_source_,
                    exposure_time_,
                    cycle_minimum_period_,
                    acquisition_frame_rate_,
                    pixel_format_,
                    gain_selector_,
                    gain_,
                    balance_white_marker_,
                    trigger_mode_,
                    trigger_selector_,
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

    // ret.on_gpu = false;
    // ret.on_gpu = gpu_;
    ret.on_gpu = true;
    ret.region1 = buffer.getUserPointer();
    ret.count1 = delivered;

    ret.region2 = nullptr;
    ret.count2 = 0;

    return ret;
}

void CameraPhantom::load_default_params() {}

void CameraPhantom::load_ini_params()
{
    const boost::property_tree::ptree& pt = get_ini_pt();
    nb_buffers_ = pt.get<unsigned int>("s991.NbBuffers", nb_buffers_);
    buffer_part_count_ = pt.get<unsigned int>("s991.BufferPartCount", buffer_part_count_);
    nb_grabbers_ = pt.get<unsigned int>("s991.NbGrabbers", nb_grabbers_);
    fullHeight_ = pt.get<unsigned int>("s991.FullHeight", fullHeight_);
    width_ = pt.get<unsigned int>("s991.Width", width_);

    stripeOffset_grabber_0_ = pt.get<unsigned int>("s991.Offset0", stripeOffset_grabber_0_);
    stripeOffset_grabber_1_ = pt.get<unsigned int>("s991.Offset1", stripeOffset_grabber_1_);

    trigger_source_ = pt.get<std::string>("s991.TriggerSource", trigger_source_);
    trigger_selector_ = pt.get<std::string>("s991.TriggerSelector", trigger_selector_);
    exposure_time_ = pt.get<float>("s991.ExposureTime", exposure_time_);
    cycle_minimum_period_ = pt.get<unsigned int>("s991.CycleMinimumPeriod", cycle_minimum_period_);
    acquisition_frame_rate_ = pt.get<unsigned int>("s991.AcquisitionFrameRate", acquisition_frame_rate_);
    pixel_format_ = pt.get<std::string>("s991.PixelFormat", pixel_format_);

    gain_selector_ = pt.get<std::string>("s991.GainSelector", gain_selector_);
    trigger_mode_ = pt.get<std::string>("s991.TriggerMode", trigger_mode_);
    gain_ = pt.get<float>("s991.Gain", gain_);
    balance_white_marker_ = pt.get<std::string>("s991.BalanceWhiteMarker", balance_white_marker_);

    if (nb_grabbers_ != 4 && nb_grabbers_ != 2)
    {
        nb_grabbers_ = 4;
        Logger::camera()->warn("Invalid number of grabbers fallback to default value 4.");
    }
}

void CameraPhantom::bind_params() { return; }

ICamera* new_camera_device() { return new CameraPhantom(); }
} // namespace camera
