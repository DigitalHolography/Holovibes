#include "camera_utils.hh"
#include "camera_logger.hh"

namespace camera
{

std::string get_exe_dir()
{
#ifdef UNICODE
    wchar_t path[MAX_PATH];
#else
    char path[MAX_PATH];
#endif
    HMODULE hmodule = GetModuleHandle(NULL);
    if (hmodule != NULL)
    {
        GetModuleFileName(hmodule, path, (sizeof(path)));
        std::filesystem::path p(path);
        return p.parent_path().string();
    }

    Logger::camera()->error("Failed to find executable dir");
    throw std::runtime_error("Failed to find executable dir");
}

} // namespace camera
