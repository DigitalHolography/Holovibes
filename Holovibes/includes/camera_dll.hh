/*! \file
 *
 * \brief Encapsulate a camera DLL ressource.
 */
#pragma once

#include <memory>
#include <string>

#include <windows.h>

#undef min
#include "icamera.hh"

/*! \brief Namespace of Camera class declaration */
namespace camera
{
/*! \class CameraDLL
 *
 * \brief Encapsulate a camera DLL ressource.
 *
 * Use a custom deleter class (functor) to automatically free the DLL
 * ressource when the ICamera object is destroyed.
 */
class CameraDLL
{
  public:
    /*! \brief Return an specialize instance of ICamera contained in dll file.
     *
     * \param dll_filepath Path to the dll file.
     * \return shared_ptr on ICamera who FreeLibrary on reset().
     */
    static std::shared_ptr<ICamera> load_camera(const std::string& dll_filepath);

  private:
    /*! \class DeleterDLL
     *
     * \brief Custom deleter that will delete the camera and the DLL handle.
     */
    class DeleterDLL
    {
      public:
        DeleterDLL(HINSTANCE dll_handle);
        /*! \brief Free camera and dll_handle_
         *
         * It will be call when releasing shared_ptr return by load_camera.
         */
        void operator()(ICamera* camera);

      private:
        HINSTANCE dll_handle_;
    };

  private:
    using FnInit = ICamera* (*)();
};
} // namespace camera
