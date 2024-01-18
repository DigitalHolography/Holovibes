#include "camera_dll.hh"

#include <stdexcept>
#include <memory>
#include <string>

#include <windows.h>

#include "API.hh"
#include "logger.hh"

namespace camera
{
std::shared_ptr<ICamera> CameraDLL::load_camera(const std::string& dll_filepath)
{
    LOG_FUNC(dll_filepath);
    HINSTANCE dll_handle = LoadLibraryA(dll_filepath.c_str());

    save_camera(dll_filepath);

    if (!dll_handle)
        throw std::runtime_error("unable to load DLL camera");

    save_camera(dll_filepath);

    FnInit init = nullptr;
    init = reinterpret_cast<FnInit>(GetProcAddress(dll_handle, "new_camera_device"));

    if (!init)
        throw std::runtime_error("unable to retrieve the 'new_camera_device' function");

    return std::shared_ptr<ICamera>(init(), DeleterDLL(dll_handle));
}

CameraDLL::DeleterDLL::DeleterDLL(HINSTANCE dll_handle)
    : dll_handle_(dll_handle)
{
}

void CameraDLL::DeleterDLL::operator()(ICamera* camera)
{
    delete camera;
    FreeLibrary(dll_handle_);
}

void CameraDLL::save_camera(const std::string& dll_filepath)
{
    auto path = holovibes::settings::user_settings_filepath;
    std::ifstream input_file(path);
    json j_us = json::parse(input_file);

    j_us["camera"]["dll"] = dll_filepath;

    std::ofstream output_file(path);
    output_file << j_us.dump(1);
}

} // namespace camera
