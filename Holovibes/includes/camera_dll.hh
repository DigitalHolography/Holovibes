/*! \file
 *
 * \brief Encapsulates a camera DLL resource.
 */
#pragma once

#include <memory>
#include <string>

#include <windows.h>

#undef min
#include "icamera.hh"

/*! \brief Namespace for camera handling */
namespace camera
{
/*! \class CameraDLL
 *
 * \brief Encapsulates a camera DLL resource.
 *
 * This class uses a custom deleter class (functor) to automatically free the DLL
 * resource when the ICamera object is destroyed.
 */
class CameraDLL
{
  public:
    /*! \brief Returns a specialized instance of ICamera contained in a DLL file.
     *
     * \param dll_filepath Path to the DLL file.
     * \return shared_ptr to ICamera which calls FreeLibrary on reset().
     */
    static std::shared_ptr<ICamera> load_camera(const std::string& dll_filepath);

  private:
    /*! \class DeleterDLL
     *
     * \brief Custom deleter that deletes the camera instance and the DLL handle.
     */
    class DeleterDLL
    {
      public:
        /*! \brief Constructor that accepts a DLL handle
         *
         * \param dll_handle Handle to the loaded DLL.
         */
        DeleterDLL(HINSTANCE dll_handle);

        /*! \brief Frees the camera instance and the DLL handle
         *
         * This is called when the shared_ptr returned by load_camera is released.
         *
         * \param camera Pointer to the ICamera instance to be deleted.
         */
        void operator()(ICamera* camera);

      private:
        HINSTANCE dll_handle_; /*!< Handle to the DLL */
    };

    /*! \brief Typedef for the function pointer to initialize the camera */
    using FnInit = ICamera* (*)();
};
} // namespace camera