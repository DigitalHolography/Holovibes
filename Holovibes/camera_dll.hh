#pragma once

# include <Windows.h>
# undef min
# include <icamera.hh>
# include <string>
# include <exception>
# include <memory>

namespace camera
{
  /*! \brief Encapsulate a camera DLL ressource.
   *
   * Use a custom deleter class (functor) to automatically free the DLL
   * ressource when the ICamera object is destroyed. */
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