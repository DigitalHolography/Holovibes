#include <spdlog/spdlog.h>

#include "pco_edge4.2lt.hh"
#include "camera.hh"
#include "camera_exception.hh"
#include <camera.h>
#include "camera_logger.hh"
#include "defs.h"
#include "image.h"

namespace camera
{

CameraPCO_Edge4_2lt::CameraPCO_Edge4_2lt()
    : Camera("pco_edge_4.2lt.ini")
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
        Logger::camera()->error("Failed to initialize PCO camera in constructor: {}", e.what());
        throw CameraException(CameraException::NOT_CONNECTED);
    }
}
CapturedFramesDescriptor CameraPCO_Edge4_2lt::get_frames()
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

void CameraPCO_Edge4_2lt::start_acquisition()
{
    try
    {

        pco_camera_.record(100, pco::RecordMode::fifo);
        Logger::camera()->info("PCO Edge 4.2 LT acquisition started in fifo mode (10 frame buffer)");
    }
    catch (const pco::CameraException& e)
    {
        Logger::camera()->error("Failed to start PCO Edge 4.2 LT acquisition: {}", e.what());
        throw CameraException(CameraException::CANT_START_ACQUISITION);
    }
}
void CameraPCO_Edge4_2lt::stop_acquisition()
{
    try
    {
        pco_camera_.stop();
        Logger::camera()->info("PCO Edge 4.2 LT acquisition stopped");
    }
    catch (const pco::CameraException& e)
    {
        Logger::camera()->error("Failed to stop PCO Edge 4.2 LT acquisition: {}", e.what());
        throw CameraException(CameraException::CANT_STOP_ACQUISITION);
    }
}
const char* CameraPCO_Edge4_2lt::get_name() const { return "PCO_Edge4_2lt"; }

const char* CameraPCO_Edge4_2lt::get_ini_name() const { return "pco_edge4_2lt.ini"; }

int CameraPCO_Edge4_2lt::get_temperature() const { return 0; }

const float CameraPCO_Edge4_2lt::get_pixel_size() const { return 6.5; }

int CameraPCO_Edge4_2lt::get_camera_fps() const
{
    unsigned int width = roi_width_;
    unsigned int height = roi_height_;

    // Table de résolutions typiques avec leurs FPS
    // Priorité aux résolutions exactes de la table
    if (width == 2048 && height == 2048)
        return 40.0;
    if (width == 2048 && height == 1024)
        return 80.0;
    if (width == 2048 && height == 512)
        return 160.0;
    if (width == 2048 && height == 256)
        return 315.0;
    if (width == 2048 && height == 128)
        return 610.0;

    // Résolutions typiques
    if (width == 1920 && height == 1080)
        return 76.0;
    if (width == 1600 && height == 1200)
        return 69.0;
    if (width == 1280 && height == 1024)
        return 80.0;
    if (width == 640 && height == 480)
        return 170.0;
    if (width == 320 && height == 240)
        return 335.0;

    // Pour les autres résolutions, estimation basée sur la hauteur
    if (width == 2048)
    {
        double base_fps = 40.0;
        double ratio = 2048.0 / height;
        return base_fps * ratio;
    }

    double reference_pixels = 2048.0 * 2048.0;
    double current_pixels = width * height;
    double pixel_ratio = reference_pixels / current_pixels;

    return 40.0 * pixel_ratio;
}

void CameraPCO_Edge4_2lt::init_camera() {}

CameraPCO_Edge4_2lt::~CameraPCO_Edge4_2lt()
{
    try
    {
        shutdown_camera();
    }
    catch (const CameraException& e)
    {
        Logger::camera()->error("Error during PCO Edge 4.2 LT shutdown: {}", e.what());
    }
    Logger::camera()->debug("PCO Edge 4.2 LT destroyed");
}

void CameraPCO_Edge4_2lt::shutdown_camera()
{
    try
    {
        pco_camera_.stop();
    }
    catch (...)
    {
    }
}

void CameraPCO_Edge4_2lt::load_default_params()
{
    pco::Configuration config = pco_camera_.getConfiguration();
    pco::Description desc = pco_camera_.getDescription();

    fd_.width = config.roi.width();
    fd_.height = config.roi.height();
    fd_.depth = PixelDepth::Bits16;
    fd_.byteEndian = Endianness::LittleEndian;
    
    pixel_size_ = get_pixel_size();
    exposure_time_ = config.exposure_time_s;

    // roi params
    roi_x_ = config.roi.x0 - 1;
    roi_y_ = config.roi.y0 - 1;
    roi_width_ = config.roi.width();
    roi_height_ = config.roi.height();

    // other params
    frame_period_ = 1.0 / get_camera_fps();
    trigger_mode_ = config.trigger_mode;
    pixel_rate_ = config.pixelrate;

    Logger::camera()->debug("Loaded camera defaults: {}x{} ROI, {}s exposure",
                           roi_width_, roi_height_, exposure_time_);
}

void CameraPCO_Edge4_2lt::load_ini_params()
{
    const boost::property_tree::ptree& pt = get_ini_pt();

    exposure_time_ = pt.get<double>("pco_edge.exposure_time", exposure_time_);
    frame_period_ = pt.get<double>("pco_edge.frame_period", frame_period_);

    // roi params
    roi_x_ = pt.get<unsigned int>("pco_edge.roi_x", roi_x_);
    roi_y_ = pt.get<unsigned int>("pco_edge.roi_y", roi_y_);
    roi_width_ = pt.get<unsigned int>("pco_edge.roi_width", roi_width_);
    roi_height_ = pt.get<unsigned int>("pco_edge.roi_height", roi_height_);

    // fd params
    pixel_format_ = pt.get<std::string>("pco_edge.pixel_format", "Mono16");
    fd_.width = roi_width_;
    fd_.height = roi_height_;
    fd_.depth = (pixel_format_ == "Mono8") ? PixelDepth::Bits8 : PixelDepth::Bits16;

    // other params
    trigger_mode_ = pt.get<unsigned int>("pco_edge.trigger_mode", trigger_mode_);
    pixel_rate_ = pt.get<unsigned int>("pco_edge.pixel_rate", pixel_rate_);

    Logger::camera()->debug("Loaded INI params: {}x{} ROI at ({},{}), {}s exposure",
                            roi_width_,
                            roi_height_,
                            roi_x_,
                            roi_y_,
                            exposure_time_);
}

void CameraPCO_Edge4_2lt::bind_params()
{
    try
    {
        pco::Configuration config = pco_camera_.getConfiguration();

        // Set exposure time
        config.exposure_time_s = exposure_time_;
        Logger::camera()->info("Setting exposure time to: {}s", exposure_time_);

        // Apply ROI
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
        Logger::camera()->info("PCO Camera configured: {}x{}, {}s exposure, Expected FPS: {:.1f}",
                               fd_.width, fd_.height, exposure_time_, expected_fps);

    }
    catch (const pco::CameraException e)
    {
        Logger::camera()->error("Failed to configure PCO camera: {}", e.what());
        throw CameraException(CameraException::CANT_SET_CONFIG);
    }
}
ICamera* new_camera_device() 
{
    return new CameraPCO_Edge4_2lt(); 
}
} // namespace camera