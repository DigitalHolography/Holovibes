#include <spdlog/spdlog.h>

#include "pco_edge5.5.hh"
#include "camera.hh"
#include "camera_exception.hh"
#include <camera.h>
#include "camera_logger.hh"
#include "defs.h"
#include "image.h"

namespace camera
{

CameraPCO_Edge5_5::CameraPCO_Edge5_5()
    : Camera("pco_edge_5_5.ini")
{
    try
    {
        pco::Description desc = pco_camera_.getDescription();
        pco::Configuration config = pco_camera_.getConfiguration();

        load_default_params();
        if (ini_file_is_open())
        {
            load_ini_params();
            ini_file_.close();
        }

        bind_params();
    }
    catch (const pco::CameraException e)
    {
        Logger::camera()->error("Failed to initialize PCO Edge 5.5 camera in constructor: {}", e.what());
        throw CameraException(CameraException::NOT_CONNECTED);
    }
}

CapturedFramesDescriptor CameraPCO_Edge5_5::get_frames()
{
    try
    {
        pco_camera_.waitForNewImage(true, FRAME_TIMEOUT / 1000.0);
        pco_camera_.image(current_image_,
                          0,
                          (fd_.depth == PixelDepth::Bits8) ? pco::DataFormat::Mono8 : pco::DataFormat::Mono16);

        static auto last_frame_time = std::chrono::steady_clock::now();
        auto current_time = std::chrono::steady_clock::now();
        auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_frame_time).count();

        last_frame_time = current_time;

        auto now = std::chrono::steady_clock::now();
        uint64_t system_timestamp_us =
            std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();

        uint64_t camera_timestamp_us = 0;
        bool has_hw_timestamp = false;
        if (current_image_.getTimestampPtr() != nullptr)
        {
            camera_timestamp_us = *reinterpret_cast<const uint64_t*>(current_image_.getTimestampPtr());
            has_hw_timestamp = true;
        }

        CapturedFramesDescriptor frames;
        frames.region1 = current_image_.data().first;
        frames.count1 = 1;
        frames.on_gpu = false;
        frames.first_frame_timestamp_us = has_hw_timestamp ? camera_timestamp_us : system_timestamp_us;
        frames.frame_period_us = static_cast<uint64_t>(frame_period_ * 1e6);
        frames.camera_timestamp_us = camera_timestamp_us;
        frames.has_hw_timestamp = has_hw_timestamp;

        return frames;
    }
    catch (const pco::CameraException& e)
    {
        Logger::camera()->error("Failed to get frame: {}", e.what());
        throw CameraException(CameraException::CANT_GET_FRAME);
    }
}

void CameraPCO_Edge5_5::start_acquisition()
{
    try
    {
        pco_camera_.record(100, pco::RecordMode::fifo);
        Logger::camera()->info("PCO Edge 5.5 acquisition started in fifo mode (10 frame buffer)");
    }
    catch (const pco::CameraException& e)
    {
        Logger::camera()->error("Failed to start PCO Edge 5.5 acquisition: {}", e.what());
        throw CameraException(CameraException::CANT_START_ACQUISITION);
    }
}

void CameraPCO_Edge5_5::stop_acquisition()
{
    try
    {
        pco_camera_.stop();
        Logger::camera()->info("PCO Edge 5.5 acquisition stopped");
    }
    catch (const pco::CameraException& e)
    {
        Logger::camera()->error("Failed to stop PCO Edge 5.5 acquisition: {}", e.what());
        throw CameraException(CameraException::CANT_STOP_ACQUISITION);
    }
}

const char* CameraPCO_Edge5_5::get_name() const { return "PCO_Edge5_5"; }

const char* CameraPCO_Edge5_5::get_ini_name() const { return "pco_edge5_5.ini"; }

int CameraPCO_Edge5_5::get_temperature() const { return 0; }

const float CameraPCO_Edge5_5::get_pixel_size() const { return 6.5; }

int CameraPCO_Edge5_5::get_camera_fps() const
{
    unsigned int width = roi_width_;
    unsigned int height = roi_height_;
    
    // PCO Edge 5.5 USB frame rate table (RS/GR mode) - from datasheet
    // Vertical resolution reduction (2560 width)
    if (width == 2560 && height == 2160) return 30;
    if (width == 2560 && height == 1024) return 63;
    if (width == 2560 && height == 512) return 126;
    if (width == 2560 && height == 256) return 248;
    if (width == 2560 && height == 128) return 481;
    
    // Typical resolutions from datasheet
    if (width == 1920 && height == 1080) return 60;
    if (width == 1600 && height == 1200) return 54;
    if (width == 1280 && height == 1024) return 63;
    if (width == 640 && height == 480) return 134;
    if (width == 320 && height == 240) return 264;
    
    // For 2560px width with other heights, CORRECT height-based scaling
    if (width == 2560)
    {
        double base_fps = 30.0;          // Base FPS at full height (2160)
        double height_ratio = 2160.0 / height;
        int estimated_fps = static_cast<int>(base_fps * height_ratio);
        
        // Log warning for non-standard heights
        if (height != 2160 && height != 1024 && height != 512 && 
            height != 256 && height != 128) {
            Logger::camera()->debug("Estimated FPS for {}x{}: {} (height-based scaling)", 
                                    width, height, estimated_fps);
        }
        
        return estimated_fps;
    }
    
    // For other widths with standard heights from datasheet
    // These follow similar height-scaling patterns
    if (width == 1920 && height <= 1080) {
        double base_fps = 60.0;  // At 1080 height
        double height_ratio = 1080.0 / height;
        return static_cast<int>(base_fps * height_ratio);
    }
    if (width == 1280 && height <= 1024) {
        double base_fps = 63.0;  // At 1024 height
        double height_ratio = 1024.0 / height;
        return static_cast<int>(base_fps * height_ratio);
    }
    
    // Fallback: conservative pixel-based estimation (with warning)
    Logger::camera()->warn("Using approximate FPS estimation for {}x{} ROI", width, height);
    double reference_pixels = 2560.0 * 2160.0;
    double current_pixels = width * height;
    double pixel_ratio = reference_pixels / current_pixels;
    return static_cast<int>(30.0 * pixel_ratio);
}

void CameraPCO_Edge5_5::init_camera() {}

CameraPCO_Edge5_5::~CameraPCO_Edge5_5()
{
    try
    {
        shutdown_camera();
    }
    catch (const CameraException& e)
    {
        Logger::camera()->error("Error during PCO Edge 5.5 shutdown: {}", e.what());
    }
    Logger::camera()->debug("PCO Edge 5.5 destroyed");
}

void CameraPCO_Edge5_5::shutdown_camera()
{
    try
    {
        pco_camera_.stop();
    }
    catch (...)
    {
    }
}

void CameraPCO_Edge5_5::load_default_params()
{
    pco::Configuration config = pco_camera_.getConfiguration();
    pco::Description desc = pco_camera_.getDescription();

    fd_.width = config.roi.width();
    fd_.height = config.roi.height();
    fd_.depth = PixelDepth::Bits16;
    fd_.byteEndian = Endianness::LittleEndian;
    
    pixel_size_ = get_pixel_size();
    exposure_time_ = config.exposure_time_s;
    
    // Validate exposure time for Edge 5.5 (500 µs - 2 s for RS mode)
    if (exposure_time_ < 0.0005) exposure_time_ = 0.0005;  // 500 µs minimum
    if (exposure_time_ > 2.0) exposure_time_ = 2.0;        // 2 seconds maximum

    // ROI params - adjust to meet Edge 5.5 constraints
    // Horizontal: steps of 4 columns (min. 64)
    // Vertical: steps of 1 row (min. 16)
    roi_x_ = config.roi.x0 - 1;
    roi_y_ = config.roi.y0 - 1;
    roi_width_ = config.roi.width();
    roi_height_ = config.roi.height();
    
    // Apply Edge 5.5 constraints
    if (roi_width_ < 64) roi_width_ = 64;
    roi_width_ = (roi_width_ / 4) * 4;  // Round down to nearest multiple of 4
    
    if (roi_height_ < 16) roi_height_ = 16;

    // Other params
    frame_period_ = 1.0 / get_camera_fps();
    trigger_mode_ = config.trigger_mode;
    pixel_rate_ = config.pixelrate;  // Will be 172 or 320 MPixel/s depending on shutter mode

    Logger::camera()->debug("Loaded camera defaults: {}x{} ROI, {}s exposure",
                           roi_width_, roi_height_, exposure_time_);
}

void CameraPCO_Edge5_5::load_ini_params()
{
    const boost::property_tree::ptree& pt = get_ini_pt();

    exposure_time_ = pt.get<double>("pco_edge.exposure_time", exposure_time_);
    
    // Validate exposure time range for Edge 5.5
    if (exposure_time_ < 0.0005) exposure_time_ = 0.0005;  // 500 µs minimum
    if (exposure_time_ > 2.0) exposure_time_ = 2.0;        // 2 seconds maximum
    
    frame_period_ = pt.get<double>("pco_edge.frame_period", frame_period_);

    // ROI params
    roi_x_ = pt.get<unsigned int>("pco_edge.roi_x", roi_x_);
    roi_y_ = pt.get<unsigned int>("pco_edge.roi_y", roi_y_);
    roi_width_ = pt.get<unsigned int>("pco_edge.roi_width", roi_width_);
    roi_height_ = pt.get<unsigned int>("pco_edge.roi_height", roi_height_);
    
    // Apply Edge 5.5 constraints
    if (roi_width_ < 64) roi_width_ = 64;
    roi_width_ = (roi_width_ / 4) * 4;  // Round down to nearest multiple of 4
    
    if (roi_height_ < 16) roi_height_ = 16;

    // FD params
    pixel_format_ = pt.get<std::string>("pco_edge.pixel_format", "Mono16");
    fd_.width = roi_width_;
    fd_.height = roi_height_;
    fd_.depth = (pixel_format_ == "Mono8") ? PixelDepth::Bits8 : PixelDepth::Bits16;

    // Other params
    trigger_mode_ = pt.get<unsigned int>("pco_edge.trigger_mode", trigger_mode_);
    pixel_rate_ = pt.get<unsigned int>("pco_edge.pixel_rate", pixel_rate_);

    Logger::camera()->debug("Loaded INI params: {}x{} ROI at ({},{}), {}s exposure",
                            roi_width_,
                            roi_height_,
                            roi_x_,
                            roi_y_,
                            exposure_time_);
}

void CameraPCO_Edge5_5::bind_params()
{
    try
    {
        pco::Configuration config = pco_camera_.getConfiguration();

        // Set exposure time with Edge 5.5 validation
        if (exposure_time_ < 0.0005) exposure_time_ = 0.0005;  // 500 µs minimum
        if (exposure_time_ > 2.0) exposure_time_ = 2.0;        // 2 seconds maximum
        config.exposure_time_s = exposure_time_;
        Logger::camera()->info("Setting exposure time to: {}s", exposure_time_);

        // Apply ROI with Edge 5.5 constraints
        // Ensure width is multiple of 4 and at least 64
        if (roi_width_ < 64) roi_width_ = 64;
        roi_width_ = (roi_width_ / 4) * 4;
        
        // Ensure height is at least 16
        if (roi_height_ < 16) roi_height_ = 16;
        
        pco::Roi desired_roi{roi_x_ + 1, roi_y_ + 1, roi_x_ + roi_width_, roi_y_ + roi_height_};
        config.roi = desired_roi;
        Logger::camera()->info("Setting ROI to: {}x{} at ({},{})", roi_width_, roi_height_, roi_x_, roi_y_);

        pco_camera_.setConfiguration(config);

        // Verify configuration
        config = pco_camera_.getConfiguration();
        fd_.width = config.roi.width();
        fd_.height = config.roi.height();
        exposure_time_ = config.exposure_time_s;

        roi_x_ = config.roi.x0 - 1;
        roi_y_ = config.roi.y0 - 1;
        roi_width_ = config.roi.width();
        roi_height_ = config.roi.height();

        // Calculate expected frame rate
        double expected_fps = 1.0 / (exposure_time_ + 0.001); // Add readout time
        Logger::camera()->info("PCO Edge 5.5 Camera configured: {}x{}, {}s exposure, Expected FPS: {:.1f}",
                               fd_.width, fd_.height, exposure_time_, expected_fps);

    }
    catch (const pco::CameraException e)
    {
        Logger::camera()->error("Failed to configure PCO Edge 5.5 camera: {}", e.what());
        throw CameraException(CameraException::CANT_SET_CONFIG);
    }
}

ICamera* new_camera_device() 
{
    return new CameraPCO_Edge5_5(); 
}
} // namespace camera