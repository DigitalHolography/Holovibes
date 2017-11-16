/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

/*! \file
 *
 * Encapsulate a camera DLL ressource. */
#pragma once

# undef min
# include "icamera.hh"

namespace camera
{
  /*! \brief Encapsulate a camera DLL ressource.
   *
   * Use a custom deleter class (functor) to automatically free the DLL
   * ressource when the ICamera object is destroyed. */
  class CameraDLL
  {
  public:
    /*! \brief Return an specialize instance of ICamera contained in dll file.
     *  \param dll_filepath Path to the dll file.
     *  \return shared_ptr on ICamera who FreeLibrary on reset().
     */
    static std::shared_ptr<ICamera> load_camera(const std::string& dll_filepath);
  private:
    /*! \brief Custom deleter that will delete the camera and the DLL handle. */
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
}