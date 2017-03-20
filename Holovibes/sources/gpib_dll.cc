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

#include "gpib_dll.hh"
#include <iostream>

namespace gpib
{
  std::shared_ptr<IVisaInterface> GpibDLL::load_gpib(const std::string& dll_filepath, const std::string gpib_path)
  {
    HINSTANCE dll_handle = nullptr;

    dll_handle = LoadLibrary(dll_filepath.c_str());
    if (!dll_handle)
      throw std::runtime_error("unable to load DLL gpib");

    FnInit init = nullptr;
    init = reinterpret_cast<FnInit>(GetProcAddress(dll_handle, "new_gpib_controller"));

    if (!init)
      throw std::runtime_error("unable to retrieve the 'new_gpib_controller' function");

    return std::shared_ptr<IVisaInterface>(init(gpib_path), DeleterDLL(dll_handle));
  }

  GpibDLL::DeleterDLL::DeleterDLL(HINSTANCE dll_handle)
    : dll_handle_(dll_handle)
  {}

  void GpibDLL::DeleterDLL::operator()(IVisaInterface* Gpib)
  {
    delete Gpib;
    FreeLibrary(dll_handle_);
  }
}
