#include <iostream>
#include "pco_edge4.2lt.hh"
#include "pco_edge5.5.hh"
#include "camera_exception.hh"
#include "camera_logger.hh"

namespace camera
{

std::string detect_pco_camera_model()
{
    try
    {
        pco::Camera temp_camera;

        std::string model = temp_camera.getName();
        Logger::camera()->info("Detected PCO camera model: {}", model);
        return model;
    }
    catch (const pco::CameraException& e)
    {
        Logger::camera()->error("No PCO camera detected: {}", e.what());
        return "";
    }
}
ICamera* new_camera_device()
{
    try
    {
        std::string model = detect_pco_camera_model();

        if (model == "pco.edge 4.2m LT_ rolling shutter")
        {
            return new CameraPCO_Edge4_2lt();
        }
        else if (model.find("5.5") !=
                 std::string::npos) // this is touchy but we dont have access to the exact camera and thus its
                                    // associated model string, this needs to be modified if we ever do
        {
            return new CameraPCO_Edge5_5();
        }
        else
        {
            Logger::camera()->error("Unsupported PCO model: {}", model);
            return nullptr;
        }
    }
    catch (...)
    {
        Logger::camera()->error("No PCO camera detected");
        return nullptr;
    }
}
} // namespace camera