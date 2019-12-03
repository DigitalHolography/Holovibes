/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */


#include "camera_pixelink.hh"

#include <memory>
#include <iostream>
#include <limits>
#include <string>

#include "camera_exception.hh"

namespace camera
{
    CameraPixelink::CameraPixelink()
        : Camera("pixelink.ini")
    {
        name_ = "MISSING NAME";

        load_default_params();

        if (ini_file_is_open()) {
            load_ini_params();
            ini_file_.close();
        }
    }

    CameraPixelink::~CameraPixelink()
    {
        try
        {
            shutdown_camera();
        }
        catch (CameraException&)
        {}
    }

    void CameraPixelink::init_camera()
    {
        U32 nb_cameras;
        //Query the number of connected cameras
        auto err_code = PxLGetNumberCamerasEx(nullptr, &nb_cameras);

        if (!API_SUCCESS(err_code) || nb_cameras < 1)
        {
            throw CameraException(CameraException::NOT_CONNECTED);
        }

        std::cout << "Found " << nb_cameras << " connected Pixelink cameras" << std::endl;

        auto cameras_ids = std::make_unique<CAMERA_ID_INFO[]>(nb_cameras);
        
        std::memset(static_cast<void *>(cameras_ids.get()), 0, sizeof(cameras_ids) * nb_cameras);
        cameras_ids[0].StructSize = sizeof(CAMERA_ID_INFO); //Weird habit camera SDKs have
        //Query the infos of the connected cameras
        err_code = PxLGetNumberCamerasEx(cameras_ids.get(), &nb_cameras);

        if (!API_SUCCESS(err_code))
        {
            throw CameraException(CameraException::NOT_CONNECTED);
        }

        HANDLE device;
        for (int i = 0; i < nb_cameras; ++i)
        {
            auto cam_id = cameras_ids[i];
            U32 serial_number = cam_id.CameraSerialNum;
            err_code = PxLInitializeEx(serial_number, &device, 0); //You could also give 0 as the serial number for the driver to select any available camera

            if (!API_SUCCESS(err_code))
            {
                continue;
            }

            device_ = device;
            query_camera_name();

            std::cout << "Connected to camera " << name_ << std::endl;

            return;
        }

        std::cout << "Could not setup any camera" << std::endl;
        throw CameraException(CameraException::NOT_INITIALIZED);
    }

    void CameraPixelink::allocate_data_buffer()
    {
        auto size = desc_.width * desc_.height * desc_.depth * 12;
        buffer_size_ = size;
        std::cout << "Allocation :" << std::endl
                  << "Width: " << desc_.width << std::endl
                  << "Height: " << desc_.height << std::endl
                  << "Depth: " << desc_.depth << std::endl
                  << "Size: " << size << std::endl;
        data_buffer_.reset(new unsigned short[size]);
    }

    void CameraPixelink::release_data_buffer()
    {
        data_buffer_.reset();
    }

    void CameraPixelink::start_acquisition()
    {
        auto err_code = PxLSetStreamState(device_, START_STREAM);
        if (!API_SUCCESS(err_code))
        {
            throw CameraException(CameraException::CANT_START_ACQUISITION);
        }
        allocate_data_buffer();
    }

    void CameraPixelink::stop_acquisition()
    {
        auto err_code = PxLSetStreamState(device_, STOP_STREAM);
        if (!API_SUCCESS(err_code))
        {
            throw CameraException(CameraException::CANT_STOP_ACQUISITION);
        }
        release_data_buffer();
    }

    void CameraPixelink::shutdown_camera()
    {
        try
        {
            stop_acquisition();
        }
        catch (CameraException&)
        {}

        auto err_code = PxLUninitialize(device_);
        if (!API_SUCCESS(err_code))
        {
            throw CameraException(CameraException::CANT_SHUTDOWN);
        }
    }

    void* CameraPixelink::get_frame()
    {
        static int count = 0;
        std::cout << "Count: " << count++ << std::endl;
        std::cout << "MARKER 1" << std::endl;
        std::cout << "buffer_size_ = " << buffer_size_ << std::endl;
        auto err_code = PxLGetNextFrame(device_, buffer_size_, static_cast<void*>(data_buffer_.get()), &PxL_fd_);
        while (err_code == 0x9000000c)//Timeout
        {
            std::cout << "Timed out" << std::endl;
            err_code = PxLGetNextFrame(device_, buffer_size_, static_cast<void*>(data_buffer_.get()), &PxL_fd_);
        }
        std::cout << "MARKER 2" << std::endl;
        if (!API_SUCCESS(err_code))
        {
            std::cout << "MARKER 2.5" << std::endl;
            std::cout << "err: 0x" << std::hex << err_code << std::endl;
            throw CameraException(CameraException::CANT_GET_FRAME);
        }
        std::cout << "MARKER 3" << std::endl;

        return static_cast<void *>(data_buffer_.get());
    }

    void CameraPixelink::load_default_params()
    {
        desc_.width = 1280;
        desc_.height = 1024;
        desc_.depth = 2;
        desc_.byteEndian = Endianness::LittleEndian;

        pixel_size_ = 6.5f;

        exposure_time_ = 50000;

        f_roi[0] = 0.f;//Left border of auto roi in pixels (must be integer)
        f_roi[1] = 0.f;//Top border of auto roi in pixels (must be integer)
        f_roi[2] = static_cast<float>(desc_.width);//Width of auto roi in pixels (must be integer)
        f_roi[3] = static_cast<float>(desc_.height);//Height of auto roi in pixels (must be integer)

        f_brightness = 0.5f;//A percentage

        f_frame_rate = 24.f;//Frame per second

        f_pixel_addressing[0] = 1.f;//"Scale"
        f_pixel_addressing[1] = 2.f;//Binning

        f_pixel_format = static_cast<float>(PIXEL_FORMAT_BAYER16);

        f_shutter = exposure_time_ / 1000000.f;

        f_trigger = 0.f;
    }

    void CameraPixelink::load_ini_params()
    {
        const boost::property_tree::ptree& pt = get_ini_pt();
        desc_.width = pt.get<unsigned short>("pixelink.roi_width", 1280);
        desc_.height = pt.get<unsigned short>("pixelink.roi_height", 1024);

        f_roi[0] = static_cast<float>(pt.get<long>("pixelink.roi_startx", 0));
        f_roi[1] = static_cast<float>(pt.get<long>("pixelink.roi_starty", 0));
        f_roi[2] = static_cast<float>(desc_.width);
        f_roi[3] = static_cast<float>(desc_.height);

        f_brightness = pt.get<float>("pixelink.brightness", 0.5f);

        f_frame_rate = pt.get<float>("pixelink.frame_rate", 24.0);

        f_pixel_addressing[0] = static_cast<float>(pt.get<int>("pixelink.pixel_addressing_value", 1));
        f_pixel_addressing[1] = static_cast<float>(pt.get<int>("pixelink.pixel_addressing_mode", 2));

        std::string pixel_format = pt.get<std::string>("pixelink.pixel_format");
        if (pixel_format == "YUV422")
        {
            f_pixel_format = PIXEL_FORMAT_YUV422;
            desc_.depth = 1;
        }
        else if (pixel_format == "BAYER8")
        {
            f_pixel_format = PIXEL_FORMAT_BAYER8;
            desc_.depth = 1;
        }
        else if (pixel_format == "BAYER16")
        {
            f_pixel_format = PIXEL_FORMAT_BAYER16;
            desc_.depth = 2;
        }
        else
        {
            std::cout << "Unsupported pixel format " << pixel_format << std::endl;
        }

        exposure_time_ = static_cast<float>(pt.get<long>("pixelink.exposure_time", 50000));
        f_shutter = exposure_time_ / 1000000;

        f_trigger = static_cast<float>(pt.get<int>("pixeling.trigger", 0));
    }

    void CameraPixelink::bind_params()
    {
        std::memset(&PxL_fd_, 0, sizeof(FRAME_DESC));
        PxL_fd_.uSize = sizeof(FRAME_DESC);
        
        auto err_code = PxLSetFeature(device_, FEATURE_ROI, FEATURE_FLAG_MANUAL, 4, f_roi);
        if (!API_SUCCESS(err_code))
        {
            std::cout << "Failed to set camera feature FEATURE_ROI" << std::endl;
        }

        err_code = PxLSetFeature(device_, FEATURE_BRIGHTNESS, FEATURE_FLAG_MANUAL, 1, &f_brightness);
        if (!API_SUCCESS(err_code))
        {
            std::cout << "Failed to set camera feature FEATURE_BRIGHTNESS" << std::endl;
        }

        f_frame_rate = 200.f;
        err_code = PxLSetFeature(device_, FEATURE_FRAME_RATE, FEATURE_FLAG_MANUAL, 1, &f_frame_rate);
        if (!API_SUCCESS(err_code))
        {
            std::cout << "Failed to set camera feature FEATURE_FRAME_RATE" << std::endl;
        }

        err_code = PxLSetFeature(device_, FEATURE_PIXEL_ADDRESSING, FEATURE_FLAG_MANUAL, 2, f_pixel_addressing);
        if (!API_SUCCESS(err_code))
        {
            std::cout << "Failed to set camera feature FEATURE_PIXEL_ADDRESSING" << std::endl;
        }

        err_code = PxLSetFeature(device_, FEATURE_PIXEL_FORMAT, FEATURE_FLAG_MANUAL, 1, &f_pixel_format);
        if (!API_SUCCESS(err_code))
        {
            std::cout << "Failed to set camera feature FEATURE_PIXEL_FORMAT" << std::endl;
        }

        err_code = PxLSetFeature(device_, FEATURE_SHUTTER, FEATURE_FLAG_MANUAL, 1, &f_shutter);
        if (!API_SUCCESS(err_code))
        {
            std::cout << "Failed to set camera feature FEATURE_SHUTTER" << std::endl;
        }

        err_code = PxLSetFeature(device_, FEATURE_TRIGGER, FEATURE_FLAG_MANUAL, 1, &f_trigger);
        if (!API_SUCCESS(err_code))
        {
            std::cout << "Failed to set camera feature FEATURE_TRIGGER" << std::endl;
        }
    }

    void CameraPixelink::query_camera_name()
    {
        CAMERA_INFO info;
        auto err_code = PxLGetCameraInfoEx(device_, &info, sizeof(CAMERA_INFO));
        if (API_SUCCESS(err_code))
        {
            name_ = info.ModelName; //See if CameraName field would really be better
        }
    }

    ICamera *new_camera_device()
    {
        return new CameraPixelink();
    }
}
