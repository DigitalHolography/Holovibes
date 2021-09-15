/*! \file
 *
 * \brief Encapsulate a GPIB DLL ressource
 */
#pragma once

#include <memory>

#include <windows.h>

#undef min
#include "../GPIB/IVisaInterface.hh"

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
    /*! \brief  Load the dll from the GPIB when needed
     *
     * \param dll_filepath Path of the .dll file generated by HoloVibes
     */
    static std::shared_ptr<IVisaInterface>
    load_gpib(const std::string& dll_filepath);

  private:
    /*! \brief Custom deleter the dll and the class. */
    class DeleterDLL
    {
      public:
        DeleterDLL(HINSTANCE dll_handle);
        void operator()(IVisaInterface* Gpib);

      private:
        HINSTANCE dll_handle_;
    };

  private:
    using FnInit = IVisaInterface* (*)();
};
} // namespace gpib
