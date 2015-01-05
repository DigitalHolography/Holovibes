#ifndef CAMERA_DLL
# define CAMERA_DLL

# include <Windows.h>
# undef min
# include <icamera.hh>
# include <string>
# include <exception>
# include <memory>

namespace camera
{
  class CameraDLL
  {
  public:
    static std::shared_ptr<ICamera> load_camera(const std::string& dll_filepath);
  private:
    /* Custom deleter that will delete the camera and the DLL handle. */
    class DeleterDLL
    {
    public:
      DeleterDLL(HINSTANCE dll_handle);
      void operator()(ICamera* camera);
    private:
      HINSTANCE dll_handle_;
    };

  private:
    using FnInit = ICamera* (*)();
  };
}

#endif /* !CAMERA_DLL */