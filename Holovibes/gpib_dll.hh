#pragma once

# include <Windows.h>
# undef min
# include "../GPIB/IVisaInterface.hh"
# include <string>
# include <exception>
# include <memory>

namespace gpib
{
  /*! \brief Encapsulate a GPIB DLL ressource.

   * Very similar to camera_dll.hh as it is basically just a copy
   * paste of it with 2 seds. This surely can be improved but due the lack
   * of time, quick'n dirty will do.
   */
  class GpibDLL
  {
  public:
    static std::shared_ptr<IVisaInterface> load_gpib(const std::string& dll_filepath, const std::string gpib_path);
  private:
    class DeleterDLL
    {
    public:
      DeleterDLL(HINSTANCE dll_handle);
      void operator()(IVisaInterface* Gpib);
    private:
      HINSTANCE dll_handle_;
    };

  private:
    using FnInit = IVisaInterface* (*)(const std::string path);
  };
}
