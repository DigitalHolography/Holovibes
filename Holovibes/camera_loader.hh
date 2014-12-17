#ifndef CAMERA_LOADER_HH
# define CAMERA_LOADER_HH

# include <Windows.h>
#undef min
# include <icamera.hh>
# include <string>
# include <exception>
# include <memory>

namespace camera
{
  class CameraLoader
  {
    using FnInit = ICamera* (*)();
  public:
    CameraLoader()
      : dll_handle_(nullptr)
      , camera_(nullptr)
    {}

    ~CameraLoader()
    {
      unload_camera();
    }

    std::unique_ptr<ICamera>& load_camera(const std::string& dll_filepath)
    {
      dll_handle_ = LoadLibrary(dll_filepath.c_str());
      if (!dll_handle_)
        throw std::runtime_error("unable to load DLL camera");

      FnInit init = nullptr;
      init = reinterpret_cast<FnInit>(GetProcAddress(dll_handle_, "new_camera_device"));

      if (!init)
        throw std::runtime_error("unable to retrieve the 'new_camera_device' function");

      camera_.reset(init());
      return camera_;
    }

    void unload_camera()
    {
      camera_.reset(nullptr);

      if (dll_handle_)
        FreeLibrary(dll_handle_);
    }

    const std::unique_ptr<ICamera>& get_camera() const
    {
      return camera_;
    }

    std::unique_ptr<ICamera>& get_camera()
    {
      return camera_;
    }

  private:
    HINSTANCE dll_handle_;
    std::unique_ptr<ICamera> camera_;
  };
}

#endif /* !CAMERA_LOADER_HH */