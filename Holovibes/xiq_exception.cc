#include "stdafx.h"
#include "xiq_exception.hh"

/*! Disable 'conditional expression is constant' warning.
**  This warning comes from the 'while (false)' in macro.
*/
#pragma warning (push)
#pragma warning (disable:4127)

namespace camera
{
  const char* XiqException::what() const
  {
    const std::string msg = camera_name_ + " " + match_error(code_);
    return msg.c_str();
  }

  std::string XiqException::match_error(XI_RETURN code) const
  {

#define MATCH_ERROR(XiRet, Str)                         \
    do                                                  \
    {                                                   \
      if (code == (XiRet))                              \
        return (Str);                                   \
    }                                                   \
    while (false)

    MATCH_ERROR(XI_OK, "Function call succeeded");
    MATCH_ERROR(XI_INVALID_HANDLE, "Invalid handle");
    MATCH_ERROR(XI_READREG, "Register read error");
    MATCH_ERROR(XI_WRITEREG, "Register write error");
    MATCH_ERROR(XI_FREE_RESOURCES, "Freeing resiurces error");
    MATCH_ERROR(XI_FREE_CHANNEL, "Freeing channel error");
    MATCH_ERROR(XI_FREE_BANDWIDTH, "Freeing bandwith error");
    MATCH_ERROR(XI_READBLK, "Read block error");
    MATCH_ERROR(XI_WRITEBLK, "Write block error");
    MATCH_ERROR(XI_NO_IMAGE, "No image");
    MATCH_ERROR(XI_TIMEOUT, "Timeout");
    MATCH_ERROR(XI_INVALID_ARG, "Invalid arguments supplied");
    MATCH_ERROR(XI_NOT_SUPPORTED, "Not supported");
    MATCH_ERROR(XI_ISOCH_ATTACH_BUFFERS, "Attach buffers error");
    MATCH_ERROR(XI_GET_OVERLAPPED_RESULT, "Overlapped result");
    MATCH_ERROR(XI_MEMORY_ALLOCATION, "Memory allocation error");
    MATCH_ERROR(XI_DLLCONTEXTISNULL, "DLL context is NULL");
    MATCH_ERROR(XI_DLLCONTEXTISNONZERO, "DLL context is non zero");
    MATCH_ERROR(XI_DLLCONTEXTEXIST, "DLL context exists");
    MATCH_ERROR(XI_TOOMANYDEVICES, "Too many devices connected");
    MATCH_ERROR(XI_ERRORCAMCONTEXT, "Camera context error");
    MATCH_ERROR(XI_UNKNOWN_HARDWARE, "Unknown hardware");
    MATCH_ERROR(XI_INVALID_TM_FILE, "Invalid TM file");
    MATCH_ERROR(XI_INVALID_TM_TAG, "Invalid TM tag");
    MATCH_ERROR(XI_INCOMPLETE_TM, "Incomplete TM");
    MATCH_ERROR(XI_BUS_RESET_FAILED, "Bus reset error");
    MATCH_ERROR(XI_NOT_IMPLEMENTED, "Not implemented");
    MATCH_ERROR(XI_SHADING_TOOBRIGHT, "Shading too bright");
    MATCH_ERROR(XI_SHADING_TOODARK, "Shading too dark");
    MATCH_ERROR(XI_TOO_LOW_GAIN, "Gain is too low");
    MATCH_ERROR(XI_INVALID_BPL, "Invalid bad pixel list");
    MATCH_ERROR(XI_BPL_REALLOC, "Bad pixel list realloc error");
    MATCH_ERROR(XI_INVALID_PIXEL_LIST, "Invalid pixel list");
    MATCH_ERROR(XI_INVALID_FFS, "Invalid Flash File System");
    MATCH_ERROR(XI_INVALID_PROFILE, "Invalid profile");
    MATCH_ERROR(XI_INVALID_CALIBRATION, "Invalid calibration");
    MATCH_ERROR(XI_INVALID_BUFFER, "Invalid buffer");
    MATCH_ERROR(XI_INVALID_DATA, "Invalid data");
    MATCH_ERROR(XI_TGBUSY, "Timing generator is busy");
    MATCH_ERROR(XI_IO_WRONG, "Wrong operation open/write/read/close");
    MATCH_ERROR(XI_ACQUISITION_ALREADY_UP, "Acquisition already started");
    MATCH_ERROR(XI_OLD_DRIVER_VERSION, "Old version of device driver installed to the system.");
    MATCH_ERROR(XI_GET_LAST_ERROR, "To get error code please call GetLastError function.");
    MATCH_ERROR(XI_CANT_PROCESS, "Data can't be processed");
    MATCH_ERROR(XI_ACQUISITION_STOPED, "Acquisition has been stopped. It should be started before GetImage.");
    MATCH_ERROR(XI_ACQUISITION_STOPED_WERR, "Acquisition has been stoped with error.");
    MATCH_ERROR(XI_INVALID_INPUT_ICC_PROFILE, "Input ICC profile missed or corrupted");
    MATCH_ERROR(XI_INVALID_OUTPUT_ICC_PROFILE, "Output ICC profile missed or corrupted");
    MATCH_ERROR(XI_DEVICE_NOT_READY, "Device not ready to operate");
    MATCH_ERROR(XI_SHADING_TOOCONTRAST, "Shading too contrast");
    MATCH_ERROR(XI_ALREADY_INITIALIZED, "Modile already initialized");
    MATCH_ERROR(XI_NOT_ENOUGH_PRIVILEGES, "Application doesn't enough privileges(one or more app");
    MATCH_ERROR(XI_NOT_COMPATIBLE_DRIVER, "Installed driver not compatible with current software");
    MATCH_ERROR(XI_TM_INVALID_RESOURCE, "TM file was not loaded successfully from resources");
    MATCH_ERROR(XI_DEVICE_HAS_BEEN_RESETED, "Device has been reseted, abnormal initial state");
    MATCH_ERROR(XI_NO_DEVICES_FOUND, "No Devices Found");
    MATCH_ERROR(XI_RESOURCE_OR_FUNCTION_LOCKED, "Resource(device) or function locked by mutex");
    MATCH_ERROR(XI_BUFFER_SIZE_TOO_SMALL, "Buffer provided by user is too small");
    MATCH_ERROR(XI_COULDNT_INIT_PROCESSOR, "Couldn't initialize processor.");
    MATCH_ERROR(XI_UNKNOWN_PARAM, "Unknown parameter");
    MATCH_ERROR(XI_WRONG_PARAM_VALUE, "Wrong parameter value");
    MATCH_ERROR(XI_WRONG_PARAM_TYPE, "Wrong parameter type");
    MATCH_ERROR(XI_WRONG_PARAM_SIZE, "Wrong parameter size");
    MATCH_ERROR(XI_BUFFER_TOO_SMALL, "Input buffer too small");
    MATCH_ERROR(XI_NOT_SUPPORTED_PARAM, "Parameter info not supported");
    MATCH_ERROR(XI_NOT_SUPPORTED_PARAM_INFO, "Parameter info not supported");
    MATCH_ERROR(XI_NOT_SUPPORTED_DATA_FORMAT, "Data format not supported");
    MATCH_ERROR(XI_READ_ONLY_PARAM, "Read only parameter");
    MATCH_ERROR(XI_BANDWIDTH_NOT_SUPPORTED, "This camera does not support currently available bandwidth");
    MATCH_ERROR(XI_INVALID_FFS_FILE_NAME, "FFS file selector is invalid or NULL");
    MATCH_ERROR(XI_FFS_FILE_NOT_FOUND, "FFS file not found");
    MATCH_ERROR(XI_PROC_OTHER_ERROR, "Processing error - other");
    MATCH_ERROR(XI_PROC_PROCESSING_ERROR, "Error while image processing.");
    MATCH_ERROR(XI_PROC_INPUT_FORMAT_UNSUPPORTED, "Input format is not supported for processing.");
    MATCH_ERROR(XI_PROC_OUTPUT_FORMAT_UNSUPPORTED, "Output format is not supported for processing.");

#undef MATCH_ERROR
    return "Unknown error";
  }
}

#pragma warning (pop)