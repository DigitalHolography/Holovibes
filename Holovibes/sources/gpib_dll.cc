/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#include <stdexcept>
#include <memory>

#include "gpib_dll.hh"

namespace gpib
{
std::shared_ptr<IVisaInterface>
GpibDLL::load_gpib(const std::string& dll_filepath)
{
    HINSTANCE dll_handle = nullptr;

    dll_handle = LoadLibrary(dll_filepath.c_str());
    if (!dll_handle)
        throw std::runtime_error("Unable to load DLL gpib");

    FnInit init = nullptr;
    init = reinterpret_cast<FnInit>(
        GetProcAddress(dll_handle, "new_gpib_controller"));

    if (!init)
        throw std::runtime_error(
            "Unable to retrieve the 'new_gpib_controller' function");

    return std::shared_ptr<IVisaInterface>(init(), DeleterDLL(dll_handle));
}

GpibDLL::DeleterDLL::DeleterDLL(HINSTANCE dll_handle)
    : dll_handle_(dll_handle)
{
}

void GpibDLL::DeleterDLL::operator()(IVisaInterface* Gpib)
{
    delete Gpib;
    FreeLibrary(dll_handle_);
}
} // namespace gpib
