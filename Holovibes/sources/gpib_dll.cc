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
