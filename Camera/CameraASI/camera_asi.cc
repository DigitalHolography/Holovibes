#include "camera_asi.hh"
#include "camera_logger.hh"
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <string>
#include <boost/property_tree/ptree.hpp>

// For the ASI interface, include the SDK header
#include <ASICamera2.h>

namespace camera
{

/*!
 * \brief Constructor for CameraAsi.
 *
 * Reads the ini file, loads parameters, and initializes the camera.
 */
CameraAsi::CameraAsi()
    : Camera("asi.ini")
    , cameraID(0)
    , isInitialized(false)
{
    name_ = "ASI Camera";

    if (ini_file_is_open())
    {
        load_ini_params();
        ini_file_.close();
    }
    else
    {
        Logger::camera()->error("Unable to open the configuration file asi.ini");
        throw CameraException(CameraException::NOT_INITIALIZED);
    }
    load_default_params();
    init_camera();
}

/*!
 * \brief Destructor for CameraAsi.
 *
 * Closes the camera if it is open.
 */
CameraAsi::~CameraAsi() { shutdown_camera(); }

/*!
 * \brief Initializes the ASI camera.
 *
 * Configures the resolution, pixel depth, ROI, gain, and exposure time.
 * Throws a CameraException if initialization fails.
 */
void CameraAsi::init_camera()
{
    fd_.width = resolution_width_;
    fd_.height = resolution_height_;
    if (pixel_depth_value_ == 8)
        fd_.depth = PixelDepth::Bits8;
    else if (pixel_depth_value_ == 16)
        fd_.depth = PixelDepth::Bits16;
    else
    {
        Logger::camera()->error("Unsupported pixel depth: {}", pixel_depth_value_);
        throw CameraException(CameraException::NOT_INITIALIZED);
    }
    fd_.byteEndian = Endianness::LittleEndian;

    int nbCameras = ASIGetNumOfConnectedCameras();
    if (cameraID >= nbCameras)
    {
        Logger::camera()->error("Invalid Camera ID {}. Only {} cameras detected.", cameraID, nbCameras);
        throw CameraException(CameraException::NOT_INITIALIZED);
    }

    ASI_ERROR_CODE ret = ASIGetCameraProperty(&camInfo, cameraID);
    if (ret != ASI_SUCCESS)
    {
        Logger::camera()->error("Unable to retrieve properties of the ASI camera (ID: {})", cameraID);
        throw CameraException(CameraException::NOT_INITIALIZED);
    }

    ret = ASIOpenCamera(cameraID);
    if (ret != ASI_SUCCESS)
    {
        Logger::camera()->error("Failed to open the ASI camera (ID: {})", cameraID);
        throw CameraException(CameraException::NOT_INITIALIZED);
    }

    ret = ASIInitCamera(cameraID);
    if (ret != ASI_SUCCESS)
    {
        Logger::camera()->error("Failed to initialize the ASI camera (ID: {})", cameraID);
        throw CameraException(CameraException::NOT_INITIALIZED);
    }

    if (camInfo.IsTriggerCam == ASI_TRUE)
    {
        ret = ASISetCameraMode(cameraID, ASI_MODE_NORMAL);
        if (ret != ASI_SUCCESS)
        {
            Logger::camera()->error("Failed to set the ASI camera (ID: {}) to normal mode", cameraID);
            throw CameraException(CameraException::NOT_INITIALIZED);
        }
    }

    // Setting up the ROI using the resolution read from the ini file
    ASI_IMG_TYPE type = (pixel_depth_value_ == 8) ? ASI_IMG_RAW8 : ASI_IMG_RAW16;
    ret = ASISetROIFormat(cameraID, fd_.width, fd_.height, 1, type);
    if (ret != ASI_SUCCESS)
    {
        Logger::camera()->error("Failed to configure the ROI for the ASI camera (ID: {})", cameraID);
        throw CameraException(CameraException::NOT_INITIALIZED);
    }
    ret = ASISetStartPos(cameraID, 0, 0);
    if (ret != ASI_SUCCESS)
    {
        Logger::camera()->error("Failed to set the start position for the ASI camera (ID: {})", cameraID);
        throw CameraException(CameraException::NOT_INITIALIZED);
    }

    // Setting the gain
    ret = ASISetControlValue(cameraID, ASI_GAIN, gain_value_, ASI_FALSE);
    if (ret != ASI_SUCCESS)
    {
        Logger::camera()->error("Failed to set the gain for the ASI camera (ID: {})", cameraID);
        throw CameraException(CameraException::NOT_INITIALIZED);
    }

    // Setting the exposure time
    ret = ASISetControlValue(cameraID, ASI_EXPOSURE, exposure_time_, ASI_FALSE);
    if (ret != ASI_SUCCESS)
    {
        Logger::camera()->error("Failed to set the exposure time for the ASI camera (ID: {})", cameraID);
        throw CameraException(CameraException::NOT_INITIALIZED);
    }

    isInitialized = true;
    Logger::camera()->info("ASI Camera (ID: {}) successfully initialized", cameraID);
}

/*!
 * \brief Starts video acquisition.
 *
 * Begins capturing video frames from the camera.
 * Throws a CameraException if acquisition cannot be started.
 */
void CameraAsi::start_acquisition()
{
    if (!isInitialized)
    {
        throw CameraException(CameraException::CANT_START_ACQUISITION);
    }

    ASI_ERROR_CODE ret = ASIStartVideoCapture(cameraID);
    if (ret != ASI_SUCCESS)
    {
        Logger::camera()->error("Failed to start video acquisition for the ASI camera (ID: {})", cameraID);
        throw CameraException(CameraException::CANT_START_ACQUISITION);
    }
    Logger::camera()->info("Video acquisition started for the ASI camera (ID: {})", cameraID);
}

/*!
 * \brief Stops video acquisition.
 *
 * Ends the video capture process.
 * Throws a CameraException if the acquisition cannot be stopped.
 */
void CameraAsi::stop_acquisition()
{
    if (!isInitialized)
    {
        throw CameraException(CameraException::CANT_START_ACQUISITION);
    }

    ASI_ERROR_CODE ret = ASIStopVideoCapture(cameraID);
    if (ret != ASI_SUCCESS)
    {
        Logger::camera()->error("Failed to stop video acquisition for the ASI camera (ID: {})", cameraID);
        // Optionally, do not throw an exception here; just log the error.
    }
    Logger::camera()->info("Video acquisition stopped for the ASI camera (ID: {})", cameraID);
}

/*!
 * \brief Retrieves a captured frame.
 *
 * Captures and returns a single frame from the camera.
 *
 * \return CapturedFramesDescriptor A descriptor containing the captured frame data.
 * \throws CameraException if video data retrieval fails.
 */
CapturedFramesDescriptor CameraAsi::get_frames()
{
    if (!isInitialized)
    {
        throw CameraException(CameraException::CANT_START_ACQUISITION);
    }

    // Retrieving the current ROI parameters
    int width = 0, height = 0, bin = 0;
    ASI_IMG_TYPE imgType;
    ASI_ERROR_CODE ret = ASIGetROIFormat(cameraID, &width, &height, &bin, &imgType);
    if (ret != ASI_SUCCESS)
    {
        Logger::camera()->error("Failed to read the ROI for the ASI camera (ID: {})", cameraID);
        throw CameraException(CameraException::CANT_START_ACQUISITION);
    }

    // Calculating the buffer size based on the image format
    long bufferSize = 0;
    if (imgType == ASI_IMG_RAW8 || imgType == ASI_IMG_Y8)
    {
        bufferSize = width * height;
    }
    else if (imgType == ASI_IMG_RGB24)
    {
        bufferSize = width * height * 3;
    }
    else if (imgType == ASI_IMG_RAW16)
    {
        bufferSize = width * height * 2;
    }
    else
    {
        Logger::camera()->error("Unsupported image format for the ASI camera (ID: {})", cameraID);
        throw CameraException(CameraException::CANT_START_ACQUISITION);
    }

    // Allocating a buffer to store the frame
    unsigned char* buffer = new unsigned char[bufferSize];
    ret = ASIGetVideoData(cameraID, buffer, bufferSize, 1000); // wait time up to 1000 ms
    if (ret != ASI_SUCCESS)
    {
        delete[] buffer;
        Logger::camera()->error("Failed to retrieve video data for the ASI camera (ID: {})", cameraID);
        throw CameraException(CameraException::CANT_START_ACQUISITION);
    }

    // Constructing the frame descriptor (for this example, only one region is used)
    CapturedFramesDescriptor desc;
    desc.region1 = buffer;
    desc.count1 = 1; // single frame
    desc.region2 = nullptr;
    desc.count2 = 0;

    return desc;
}

/*!
 * \brief Shuts down the camera.
 *
 * Closes the camera and releases resources.
 */
void CameraAsi::shutdown_camera()
{
    if (isInitialized)
    {
        ASI_ERROR_CODE ret = ASICloseCamera(cameraID);
        if (ret != ASI_SUCCESS)
        {
            Logger::camera()->error("Failed to close the ASI camera (ID: {})", cameraID);
        }
        else
        {
            Logger::camera()->info("ASI Camera (ID: {}) closed successfully", cameraID);
        }
        isInitialized = false;
    }
}

/*!
 * \brief Loads configuration parameters from the ini file.
 *
 * Reads parameters such as camera ID, resolution, pixel depth, gain, and exposure time from the ini file.
 */
void CameraAsi::load_ini_params()
{
    const boost::property_tree::ptree& pt = get_ini_pt();

    cameraID = pt.get<int>("asi.camera_id", 0);
    resolution_width_ = pt.get<int>("asi.resolution_width", 3840);
    resolution_height_ = pt.get<int>("asi.resolution_height", 2160);
    pixel_depth_value_ = pt.get<int>("asi.pixel_depth", 8);
    gain_value_ = pt.get<int>("asi.gain", 50);
    exposure_time_ = pt.get<int>("asi.exposure_time", 10000);

    Logger::camera()->info(
        ".ini loaded : camera_id = {}, resolution = {}x{}, pixel_depth = {}, gain = {}, exposure_time = {}",
        cameraID,
        resolution_width_,
        resolution_height_,
        pixel_depth_value_,
        gain_value_,
        exposure_time_);
}

/*!
 * \brief Loads default parameters.
 *
 * Initializes default values for camera parameters.
 */
void CameraAsi::load_default_params()
{
    // Default initialization
    std::memset(&camInfo, 0, sizeof(ASI_CAMERA_INFO));
    isInitialized = false;
}

/*!
 * \brief Binds internal parameters.
 *
 * Example: logs the camera's maximum resolution.
 */
void CameraAsi::bind_params()
{
    Logger::camera()->info("Camera maximum resolution: {}x{}", camInfo.MaxWidth, camInfo.MaxHeight);
}

/*!
 * \brief Factory function to create a new CameraAsi object.
 *
 * \return ICamera* A pointer to a new CameraAsi instance.
 */
ICamera* new_camera_device() { return new CameraAsi(); }

} // namespace camera
