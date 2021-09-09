
			
#ifndef _ERROR_CODES_H_
#define _ERROR_CODES_H_

//
typedef int MM40_RETURN;

/** @name Error codes
         Definitions of the error codes used in API

         @note Most functions return MM40_OK on success, an error code otherwise
   */
   /** @{ */
#define MM40_OK                            0 //!< Function call succeeded
#define MM40_INVALID_HANDLE                1 //!< Invalid handle
#define MM40_READREG                       2 //!< Register read error
#define MM40_WRITEREG                      3 //!< Register write error
#define MM40_FREE_RESOURCES                4 //!< Freeing resources error
#define MM40_FREE_CHANNEL                  5 //!< Freeing channel error
#define MM40_FREE_BANDWIDTH                6 //!< Freeing bandwith error
#define MM40_READBLK                       7 //!< Read block error
#define MM40_WRITEBLK                      8 //!< Write block error
#define MM40_NO_IMAGE                      9 //!< No image
#define MM40_TIMEOUT                      10 //!< Timeout
#define MM40_INVALID_ARG                  11 //!< Invalid arguments supplied
#define MM40_NOT_SUPPORTED                12 //!< Not supported
#define MM40_ISOCH_ATTACH_BUFFERS         13 //!< Attach buffers error
#define MM40_GET_OVERLAPPED_RESULT        14 //!< Overlapped result
#define MM40_MEMORY_ALLOCATION            15 //!< Memory allocation error
#define MM40_DLLCONTEXTISNULL             16 //!< DLL context is NULL
#define MM40_DLLCONTEXTISNONZERO          17 //!< DLL context is non zero
#define MM40_DLLCONTEXTEXIST              18 //!< DLL context exists
#define MM40_TOOMANYDEVICES               19 //!< Too many devices connected
#define MM40_ERRORCAMCONTEXT              20 //!< Camera context error
#define MM40_UNKNOWN_HARDWARE             21 //!< Unknown hardware
#define MM40_INVALID_TM_FILE              22 //!< Invalid TM file
#define MM40_INVALID_TM_TAG               23 //!< Invalid TM tag
#define MM40_INCOMPLETE_TM                24 //!< Incomplete TM
#define MM40_BUS_RESET_FAILED             25 //!< Bus reset error
#define MM40_NOT_IMPLEMENTED              26 //!< Not implemented
#define MM40_SHADING_TOOBRIGHT            27 //!< Shading is too bright
#define MM40_SHADING_TOODARK              28 //!< Shading is too dark
#define MM40_TOO_LOW_GAIN                 29 //!< Gain is too low
#define MM40_INVALID_BPL                  30 //!< Invalid sensor defect correction list
#define MM40_BPL_REALLOC                  31 //!< Error while sensor defect correction list reallocation
#define MM40_INVALID_PIXEL_LIST           32 //!< Invalid pixel list
#define MM40_INVALID_FFS                  33 //!< Invalid Flash File System
#define MM40_INVALID_PROFILE              34 //!< Invalid profile
#define MM40_INVALID_CALIBRATION          35 //!< Invalid calibration
#define MM40_INVALID_BUFFER               36 //!< Invalid buffer
#define MM40_INVALID_DATA                 38 //!< Invalid data
#define MM40_TGBUSY                       39 //!< Timing generator is busy
#define MM40_IO_WRONG                     40 //!< Wrong operation open/write/read/close
#define MM40_ACQUISITION_ALREADY_UP       41 //!< Acquisition already started
#define MM40_OLD_DRIVER_VERSION           42 //!< Old version of device driver installed to the system.
#define MM40_GET_LAST_ERROR               43 //!< To get error code please call GetLastError function.
#define MM40_CANT_PROCESS                 44 //!< Data cannot be processed
#define MM40_ACQUISITION_STOPED           45 //!< Acquisition is stopped. It needs to be started to perform operation.
#define MM40_ACQUISITION_STOPED_WERR      46 //!< Acquisition has been stopped with an error.
#define MM40_INVALID_INPUT_ICC_PROFILE    47 //!< Input ICC profile missing or corrupted
#define MM40_INVALID_OUTPUT_ICC_PROFILE   48 //!< Output ICC profile missing or corrupted
#define MM40_DEVICE_NOT_READY             49 //!< Device not ready to operate
#define MM40_SHADING_TOOCONTRAST          50 //!< Shading is too contrast
#define MM40_ALREADY_INITIALIZED          51 //!< Module already initialized
#define MM40_NOT_ENOUGH_PRIVILEGES        52 //!< Application does not have enough privileges (one or more app)
#define MM40_NOT_COMPATIBLE_DRIVER        53 //!< Installed driver is not compatible with current software
#define MM40_TM_INVALID_RESOURCE          54 //!< TM file was not loaded successfully from resources
#define MM40_DEVICE_HAS_BEEN_RESETED      55 //!< Device has been reset, abnormal initial state
#define MM40_NO_DEVICES_FOUND             56 //!< No Devices Found
#define MM40_RESOURCE_OR_FUNCTION_LOCKED  57 //!< Resource (device) or function locked by mutex
#define MM40_BUFFER_SIZE_TOO_SMALL        58 //!< Buffer provided by user is too small
#define MM40_COULDNT_INIT_PROCESSOR       59 //!< Could not initialize processor.
#define MM40_NOT_INITIALIZED              60 //!< The object/module/procedure/process being referred to has not been started.
#define MM40_RESOURCE_NOT_FOUND           61 //!< Resource not found(could be processor, file, item...).
/** @} */
#endif // _ERROR_CODES_H_
