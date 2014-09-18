
			
//-------------------------------------------------------------------------------------------------------------------
// xiApi header file
#ifndef __XIAPI_H
#define __XIAPI_H

#ifdef WIN32
#include <windows.h>
#else
// linux
#include "wintypedefs.h"
#endif
#include "m3Identify.h"
#ifdef XIAPI_EXPORTS
#define XIAPI __declspec(dllexport)
#else
#define XIAPI __declspec(dllimport)
#endif

typedef int XI_RETURN;

#ifdef __cplusplus
extern "C" {
#endif

//-------------------------------------------------------------------------------------------------------------------
// xiApi parameters

// 
#define  XI_PRM_DEVICE_NAME                     "device_name"             // Return device name 
#define  XI_PRM_DEVICE_TYPE                     "device_type"             // Return device type 
#define  XI_PRM_DEVICE_MODEL_ID                 "device_model_id"         // Return device model id 
#define  XI_PRM_DEVICE_SN                       "device_sn"               // Return device serial number 
#define  XI_PRM_DEVICE_SENS_SN                  "device_sens_sn"          // Return sensor serial number 
#define  XI_PRM_DEVICE_INSTANCE_PATH            "device_inst_path"        // Return device system instance path. 
#define  XI_PRM_DEVICE_USER_ID                  "device_user_id"          // Return custom ID of camera. 
// 
#define  XI_PRM_EXPOSURE                        "exposure"                // Exposure time in microseconds 
#define  XI_PRM_GAIN                            "gain"                    // Gain in dB 
#define  XI_PRM_DOWNSAMPLING                    "downsampling"            // Change image resolution by binning or skipping. 
#define  XI_PRM_DOWNSAMPLING_TYPE               "downsampling_type"       // Change image downsampling type. XI_DOWNSAMPLING_TYPE
#define  XI_PRM_SHUTTER_TYPE                    "shutter_type"            // Change sensor shutter type(CMOS sensor). XI_SHUTTER_TYPE
#define  XI_PRM_IMAGE_DATA_FORMAT               "imgdataformat"           // Output data format. XI_IMG_FORMAT
#define  XI_PRM_TRANSPORT_PIXEL_FORMAT          "transport_pixel_format"  // Current format of pixels on transport layer. XI_GenTL_Image_Format_e
#define  XI_PRM_SENSOR_TAPS                     "sensor_taps"             // Number of taps 
#define  XI_PRM_SENSOR_PIXEL_CLOCK_FREQ_HZ      "sensor_pixel_clock_freq_hz"// Sensor pixel clock frequency in Hz. 
#define  XI_PRM_SENSOR_PIXEL_CLOCK_FREQ_INDEX   "sensor_pixel_clock_freq_index"// Sensor pixel clock frequency. Selects frequency index for getter of XI_PRM_SENSOR_PIXEL_CLOCK_FREQ_HZ parameter. 
#define  XI_PRM_SENSOR_DATA_BIT_DEPTH           "sensor_bit_depth"        // Sensor output data bit depth. 
#define  XI_PRM_OUTPUT_DATA_BIT_DEPTH           "output_bit_depth"        // Device output data bit depth. 
#define  XI_PRM_OUTPUT_DATA_PACKING             "output_bit_packing"      // Device output data packing (or grouping) enabled. Packing could be enabled if output_data_bit_depth > 8 and packing is available. XI_SWITCH
#define  XI_PRM_FRAMERATE                       "framerate"               // Define framerate in Hz 
#define  XI_PRM_ACQ_TIMING_MODE                 "acq_timing_mode"         // Type of sensor frames timing. XI_ACQ_TIMING_MODE
#define  XI_PRM_AVAILABLE_BANDWIDTH             "available_bandwidth"     // Calculate and return available interface bandwidth(int Megabits) 
#define  XI_PRM_LIMIT_BANDWIDTH                 "limit_bandwidth"         // Set/get bandwidth(datarate)(in Megabits) 
#define  XI_PRM_BUFFER_POLICY                   "buffer_policy"           // Data move policy XI_BP
#define  XI_PRM_WIDTH                           "width"                   // Width of the Image provided by the device (in pixels). 
#define  XI_PRM_HEIGHT                          "height"                  // Height of the Image provided by the device (in pixels). 
#define  XI_PRM_OFFSET_X                        "offsetX"                 // Horizontal offset from the origin to the area of interest (in pixels). 
#define  XI_PRM_OFFSET_Y                        "offsetY"                 // Vertical offset from the origin to the area of interest (in pixels). 
#define  XI_PRM_LUT_EN                          "LUTEnable"               // Activates LUT. XI_SWITCH
#define  XI_PRM_LUT_INDEX                       "LUTIndex"                // Control the index (offset) of the coefficient to access in the LUT. 
#define  XI_PRM_LUT_VALUE                       "LUTValue"                // Value at entry LUTIndex of the LUT 
#define  XI_PRM_TRG_SOURCE                      "trigger_source"          // Defines source of trigger. XI_TRG_SOURCE
#define  XI_PRM_TRG_SELECTOR                    "trigger_selector"        // Selects the type of trigger. XI_TRG_SELECTOR
#define  XI_PRM_TRG_SOFTWARE                    "trigger_software"        // Generates an internal trigger. XI_PRM_TRG_SOURCE must be set to TRG_SOFTWARE. 
#define  XI_PRM_TRG_DELAY                       "trigger_delay"           // Specifies the delay in microseconds (us) to apply after the trigger reception before activating it. 
#define  XI_PRM_GPI_SELECTOR                    "gpi_selector"            // Selects GPI 
#define  XI_PRM_GPI_MODE                        "gpi_mode"                // Defines GPI functionality XI_GPI_MODE
#define  XI_PRM_GPI_LEVEL                       "gpi_level"               // GPI level 
#define  XI_PRM_GPO_SELECTOR                    "gpo_selector"            // Selects GPO 
#define  XI_PRM_GPO_MODE                        "gpo_mode"                // Defines GPO functionality XI_GPO_MODE
#define  XI_PRM_LED_SELECTOR                    "led_selector"            // Selects LED 
#define  XI_PRM_LED_MODE                        "led_mode"                // Defines LED functionality XI_LED_MODE
#define  XI_PRM_ACQ_FRAME_BURST_COUNT           "acq_frame_burst_count"   // Sets number of frames acquired by burst. This burst is used only if trigger is set to FrameBurstStart 
// 
#define  XI_PRM_IS_DEVICE_EXIST                 "isexist"                 // Returns 1 if camera connected and works properly. XI_SWITCH
#define  XI_PRM_ACQ_BUFFER_SIZE                 "acq_buffer_size"         // Acquisition buffer size in bytes 
#define  XI_PRM_ACQ_TRANSPORT_BUFFER_SIZE       "acq_transport_buffer_size"// Acquisition transport buffer size in bytes 
#define  XI_PRM_BUFFERS_QUEUE_SIZE              "buffers_queue_size"      // Queue of field/frame buffers 
#define  XI_PRM_RECENT_FRAME                    "recent_frame"            // GetImage returns most recent frame XI_SWITCH
// 
#define  XI_PRM_CMS                             "cms"                     // Mode of color management system. XI_CMS_MODE
#define  XI_PRM_APPLY_CMS                       "apply_cms"               // Enable applying of CMS profiles to xiGetImage (see XI_PRM_INPUT_CMS_PROFILE, XI_PRM_OUTPUT_CMS_PROFILE). XI_SWITCH
#define  XI_PRM_INPUT_CMS_PROFILE               "input_cms_profile"       // Filename for input cms profile (e.g. input.icc) 
#define  XI_PRM_OUTPUT_CMS_PROFILE              "output_cms_profile"      // Filename for output cms profile (e.g. input.icc) 
#define  XI_PRM_IMAGE_IS_COLOR                  "iscolor"                 // Returns 1 for color cameras. XI_SWITCH
#define  XI_PRM_COLOR_FILTER_ARRAY              "cfa"                     // Returns color filter array type of RAW data. XI_COLOR_FILTER_ARRAY
#define  XI_PRM_WB_KR                           "wb_kr"                   // White balance red coefficient 
#define  XI_PRM_WB_KG                           "wb_kg"                   // White balance green coefficient 
#define  XI_PRM_WB_KB                           "wb_kb"                   // White balance blue coefficient 
#define  XI_PRM_MANUAL_WB                       "manual_wb"               // Calculates White Balance(xiGetImage function must be called) 
#define  XI_PRM_AUTO_WB                         "auto_wb"                 // Automatic white balance XI_SWITCH
#define  XI_PRM_GAMMAY                          "gammaY"                  // Luminosity gamma 
#define  XI_PRM_GAMMAC                          "gammaC"                  // Chromaticity gamma 
#define  XI_PRM_SHARPNESS                       "sharpness"               // Sharpness Strenght 
#define  XI_PRM_CC_MATRIX_00                    "ccMTX00"                 // Color Correction Matrix element [0][0] 
#define  XI_PRM_CC_MATRIX_01                    "ccMTX01"                 // Color Correction Matrix element [0][1] 
#define  XI_PRM_CC_MATRIX_02                    "ccMTX02"                 // Color Correction Matrix element [0][2] 
#define  XI_PRM_CC_MATRIX_03                    "ccMTX03"                 // Color Correction Matrix element [0][3] 
#define  XI_PRM_CC_MATRIX_10                    "ccMTX10"                 // Color Correction Matrix element [1][0] 
#define  XI_PRM_CC_MATRIX_11                    "ccMTX11"                 // Color Correction Matrix element [1][1] 
#define  XI_PRM_CC_MATRIX_12                    "ccMTX12"                 // Color Correction Matrix element [1][2] 
#define  XI_PRM_CC_MATRIX_13                    "ccMTX13"                 // Color Correction Matrix element [1][3] 
#define  XI_PRM_CC_MATRIX_20                    "ccMTX20"                 // Color Correction Matrix element [2][0] 
#define  XI_PRM_CC_MATRIX_21                    "ccMTX21"                 // Color Correction Matrix element [2][1] 
#define  XI_PRM_CC_MATRIX_22                    "ccMTX22"                 // Color Correction Matrix element [2][2] 
#define  XI_PRM_CC_MATRIX_23                    "ccMTX23"                 // Color Correction Matrix element [2][3] 
#define  XI_PRM_CC_MATRIX_30                    "ccMTX30"                 // Color Correction Matrix element [3][0] 
#define  XI_PRM_CC_MATRIX_31                    "ccMTX31"                 // Color Correction Matrix element [3][1] 
#define  XI_PRM_CC_MATRIX_32                    "ccMTX32"                 // Color Correction Matrix element [3][2] 
#define  XI_PRM_CC_MATRIX_33                    "ccMTX33"                 // Color Correction Matrix element [3][3] 
#define  XI_PRM_DEFAULT_CC_MATRIX               "defccMTX"                // Set default Color Correction Matrix 
// 
#define  XI_PRM_AEAG                            "aeag"                    // Automatic exposure/gain XI_SWITCH
#define  XI_PRM_AEAG_ROI_OFFSET_X               "aeag_roi_offset_x"       // Automatic exposure/gain ROI offset X 
#define  XI_PRM_AEAG_ROI_OFFSET_Y               "aeag_roi_offset_y"       // Automatic exposure/gain ROI offset Y 
#define  XI_PRM_AEAG_ROI_WIDTH                  "aeag_roi_width"          // Automatic exposure/gain ROI Width 
#define  XI_PRM_AEAG_ROI_HEIGHT                 "aeag_roi_height"         // Automatic exposure/gain ROI Height 
#define  XI_PRM_EXP_PRIORITY                    "exp_priority"            // Exposure priority (0.5 - exposure 50%, gain 50%). 
#define  XI_PRM_AE_MAX_LIMIT                    "ae_max_limit"            // Maximum limit of exposure in AEAG procedure 
#define  XI_PRM_AG_MAX_LIMIT                    "ag_max_limit"            // Maximum limit of gain in AEAG procedure 
#define  XI_PRM_AEAG_LEVEL                      "aeag_level"              // Average intensity of output signal AEAG should achieve(in %) 
// 
#define  XI_PRM_BPC                             "bpc"                     // Correction of bad pixels XI_SWITCH
// 
#define  XI_PRM_DEBOUNCE_EN                     "dbnc_en"                 // Enable/Disable debounce to selected GPI XI_SWITCH
#define  XI_PRM_DEBOUNCE_T0                     "dbnc_t0"                 // Debounce time (x * 10us) 
#define  XI_PRM_DEBOUNCE_T1                     "dbnc_t1"                 // Debounce time (x * 10us) 
#define  XI_PRM_DEBOUNCE_POL                    "dbnc_pol"                // Debounce polarity (pol = 1 t0 - falling edge, t1 - rising edge) 
// 
#define  XI_PRM_IS_COOLED                       "iscooled"                // Returns 1 for cameras that support cooling. 
#define  XI_PRM_COOLING                         "cooling"                 // Start camera cooling. XI_SWITCH
#define  XI_PRM_TARGET_TEMP                     "target_temp"             // Set sensor target temperature for cooling. 
#define  XI_PRM_CHIP_TEMP                       "chip_temp"               // Camera sensor temperature 
#define  XI_PRM_HOUS_TEMP                       "hous_temp"               // Camera housing tepmerature 
// 
#define  XI_PRM_SENSOR_MODE                     "sensor_mode"             // Current sensor mode. Allows to select sensor mode by one integer. Setting of this parameter affects: image dimensions and downsampling. 
#define  XI_PRM_HDR                             "hdr"                     // Enable High Dynamic Range feature. XI_SWITCH
#define  XI_PRM_HDR_KNEEPOINT_COUNT             "hdr_kneepoint_count"     // The number of kneepoints in the PWLR. 
#define  XI_PRM_HDR_T1                          "hdr_t1"                  // position of first kneepoint(in % of XI_PRM_EXPOSURE) 
#define  XI_PRM_HDR_T2                          "hdr_t2"                  // position of second kneepoint (in % of XI_PRM_EXPOSURE) 
#define  XI_PRM_KNEEPOINT1                      "hdr_kneepoint1"          // value of first kneepoint (% of sensor saturation) 
#define  XI_PRM_KNEEPOINT2                      "hdr_kneepoint2"          // value of second kneepoint (% of sensor saturation) 
#define  XI_PRM_IMAGE_BLACK_LEVEL               "image_black_level"       // Last image black level counts. Can be used for Offline processing to recall it. 
// 
#define  XI_PRM_API_VERSION                     "api_version"             // Returns version of API. 
#define  XI_PRM_DRV_VERSION                     "drv_version"             // Returns version of current device driver. 
#define  XI_PRM_MCU1_VERSION                    "version_mcu1"            // Returns version of MCU1 firmware. 
#define  XI_PRM_MCU2_VERSION                    "version_mcu2"            // Returns version of MCU2 firmware. 
#define  XI_PRM_FPGA1_VERSION                   "version_fpga1"           // Returns version of FPGA1 firmware. 
// 
#define  XI_PRM_DEBUG_LEVEL                     "debug_level"             // Set debug level XI_DEBUG_LEVEL
#define  XI_PRM_AUTO_BANDWIDTH_CALCULATION      "auto_bandwidth_calculation"// Automatic bandwidth calculation, XI_SWITCH
// 
#define  XI_PRM_READ_FILE_FFS                   "read_file_ffs"           // Read file from camera flash filesystem. 
#define  XI_PRM_WRITE_FILE_FFS                  "write_file_ffs"          // Write file to camera flash filesystem. 
#define  XI_PRM_FFS_FILE_NAME                   "ffs_file_name"           // Set name of file to be written/read from camera FFS. 
#define  XI_PRM_FREE_FFS_SIZE                   "free_ffs_size"           // Size of free camera FFS. 
#define  XI_PRM_USED_FFS_SIZE                   "used_ffs_size"           // Size of used camera FFS. 
// 
#define  XI_PRM_API_CONTEXT_LIST                "xiapi_context_list"      // List of current parameters settings context - parameters with values. Used for offline processing. 

//-------------------------------------------------------------------------------------------------------------------
// Error codes xiApi
typedef enum
{	
	XI_OK                             = 0, // Function call succeeded
	XI_INVALID_HANDLE                 = 1, // Invalid handle
	XI_READREG                        = 2, // Register read error
	XI_WRITEREG                       = 3, // Register write error
	XI_FREE_RESOURCES                 = 4, // Freeing resiurces error
	XI_FREE_CHANNEL                   = 5, // Freeing channel error
	XI_FREE_BANDWIDTH                 = 6, // Freeing bandwith error
	XI_READBLK                        = 7, // Read block error
	XI_WRITEBLK                       = 8, // Write block error
	XI_NO_IMAGE                       = 9, // No image
	XI_TIMEOUT                        =10, // Timeout
	XI_INVALID_ARG                    =11, // Invalid arguments supplied
	XI_NOT_SUPPORTED                  =12, // Not supported
	XI_ISOCH_ATTACH_BUFFERS           =13, // Attach buffers error
	XI_GET_OVERLAPPED_RESULT          =14, // Overlapped result
	XI_MEMORY_ALLOCATION              =15, // Memory allocation error
	XI_DLLCONTEXTISNULL               =16, // DLL context is NULL
	XI_DLLCONTEXTISNONZERO            =17, // DLL context is non zero
	XI_DLLCONTEXTEXIST                =18, // DLL context exists
	XI_TOOMANYDEVICES                 =19, // Too many devices connected
	XI_ERRORCAMCONTEXT                =20, // Camera context error
	XI_UNKNOWN_HARDWARE               =21, // Unknown hardware
	XI_INVALID_TM_FILE                =22, // Invalid TM file
	XI_INVALID_TM_TAG                 =23, // Invalid TM tag
	XI_INCOMPLETE_TM                  =24, // Incomplete TM
	XI_BUS_RESET_FAILED               =25, // Bus reset error
	XI_NOT_IMPLEMENTED                =26, // Not implemented
	XI_SHADING_TOOBRIGHT              =27, // Shading too bright
	XI_SHADING_TOODARK                =28, // Shading too dark
	XI_TOO_LOW_GAIN                   =29, // Gain is too low
	XI_INVALID_BPL                    =30, // Invalid bad pixel list
	XI_BPL_REALLOC                    =31, // Bad pixel list realloc error
	XI_INVALID_PIXEL_LIST             =32, // Invalid pixel list
	XI_INVALID_FFS                    =33, // Invalid Flash File System
	XI_INVALID_PROFILE                =34, // Invalid profile
	XI_INVALID_CALIBRATION            =35, // Invalid calibration
	XI_INVALID_BUFFER                 =36, // Invalid buffer
	XI_INVALID_DATA                   =38, // Invalid data
	XI_TGBUSY                         =39, // Timing generator is busy
	XI_IO_WRONG                       =40, // Wrong operation open/write/read/close
	XI_ACQUISITION_ALREADY_UP         =41, // Acquisition already started
	XI_OLD_DRIVER_VERSION             =42, // Old version of device driver installed to the system.
	XI_GET_LAST_ERROR                 =43, // To get error code please call GetLastError function.
	XI_CANT_PROCESS                   =44, // Data can't be processed
	XI_ACQUISITION_STOPED             =45, // Acquisition has been stopped. It should be started before GetImage.
	XI_ACQUISITION_STOPED_WERR        =46, // Acquisition has been stoped with error.
	XI_INVALID_INPUT_ICC_PROFILE      =47, // Input ICC profile missed or corrupted
	XI_INVALID_OUTPUT_ICC_PROFILE     =48, // Output ICC profile missed or corrupted
	XI_DEVICE_NOT_READY               =49, // Device not ready to operate
	XI_SHADING_TOOCONTRAST            =50, // Shading too contrast
	XI_ALREADY_INITIALIZED            =51, // Modile already initialized
	XI_NOT_ENOUGH_PRIVILEGES          =52, // Application doesn't enough privileges(one or more app
	XI_NOT_COMPATIBLE_DRIVER          =53, // Installed driver not compatible with current software
	XI_TM_INVALID_RESOURCE            =54, // TM file was not loaded successfully from resources
	XI_DEVICE_HAS_BEEN_RESETED        =55, // Device has been reseted, abnormal initial state
	XI_NO_DEVICES_FOUND               =56, // No Devices Found
	XI_RESOURCE_OR_FUNCTION_LOCKED    =57, // Resource(device) or function locked by mutex
	XI_BUFFER_SIZE_TOO_SMALL          =58, // Buffer provided by user is too small
	XI_COULDNT_INIT_PROCESSOR         =59, // Couldn't initialize processor.
	XI_UNKNOWN_PARAM                  =100, // Unknown parameter
	XI_WRONG_PARAM_VALUE              =101, // Wrong parameter value
	XI_WRONG_PARAM_TYPE               =103, // Wrong parameter type
	XI_WRONG_PARAM_SIZE               =104, // Wrong parameter size
	XI_BUFFER_TOO_SMALL               =105, // Input buffer too small
	XI_NOT_SUPPORTED_PARAM            =106, // Parameter info not supported
	XI_NOT_SUPPORTED_PARAM_INFO       =107, // Parameter info not supported
	XI_NOT_SUPPORTED_DATA_FORMAT      =108, // Data format not supported
	XI_READ_ONLY_PARAM                =109, // Read only parameter
	XI_BANDWIDTH_NOT_SUPPORTED        =111, // This camera does not support currently available bandwidth
	XI_INVALID_FFS_FILE_NAME          =112, // FFS file selector is invalid or NULL
	XI_FFS_FILE_NOT_FOUND             =113, // FFS file not found
	XI_PROC_OTHER_ERROR               =201, // Processing error - other
	XI_PROC_PROCESSING_ERROR          =202, // Error while image processing.
	XI_PROC_INPUT_FORMAT_UNSUPPORTED  =203, // Input format is not supported for processing.
	XI_PROC_OUTPUT_FORMAT_UNSUPPORTED =204, // Output format is not supported for processing.
	
}XI_RET;

//-------------------------------------------------------------------------------------------------------------------
// xiAPI enumerators
// Debug level enumerator.
typedef enum
{
	XI_DL_DETAIL                 =0, // Same as trace plus locking resources
	XI_DL_TRACE                  =1, // Information level.
	XI_DL_WARNING                =2, // Warning level.
	XI_DL_ERROR                  =3, // Error level.
	XI_DL_FATAL                  =4, // Fatal error level.
	XI_DL_DISABLED               =100, // Print no errors at all.
	
} XI_DEBUG_LEVEL;

// structure containing information about output image format
typedef enum
{
	XI_MONO8                     =0, // 8 bits per pixel
	XI_MONO16                    =1, // 16 bits per pixel
	XI_RGB24                     =2, // RGB data format
	XI_RGB32                     =3, // RGBA data format
	XI_RGB_PLANAR                =4, // RGB planar data format
	XI_RAW8                      =5, // 8 bits per pixel raw data from sensor
	XI_RAW16                     =6, // 16 bits per pixel raw data from sensor
	XI_FRM_TRANSPORT_DATA        =7, // Data from transport layer (e.g. packed). Format see XI_PRM_TRANSPORT_PIXEL_FORMAT
	
} XI_IMG_FORMAT;

// structure containing information about bayer color matrix
typedef enum
{
	XI_CFA_NONE                  =0, //  B/W sensors
	XI_CFA_BAYER_RGGB            =1, // Regular RGGB
	XI_CFA_CMYG                  =2, // AK Sony sens
	XI_CFA_RGR                   =3, // 2R+G readout
	XI_CFA_BAYER_BGGR            =4, // BGGR readout
	XI_CFA_BAYER_GRBG            =5, // GRBG readout
	XI_CFA_BAYER_GBRG            =6, // GBRG readout
	
} XI_COLOR_FILTER_ARRAY;

// structure containing information about buffer policy(can be safe, data will be copied to user/app buffer or unsafe, user will get internally allocated buffer without data copy).
typedef enum
{
	XI_BP_UNSAFE                 =0, // User gets pointer to internally allocated circle buffer and data may be overwritten by device.
	XI_BP_SAFE                   =1, // Data from device will be copied to user allocated buffer or xiApi allocated memory.
	
} XI_BP;

// structure containing information about trigger source
typedef enum
{
	XI_TRG_OFF                   =0, // Camera works in free run mode.
	XI_TRG_EDGE_RISING           =1, // External trigger (rising edge).
	XI_TRG_EDGE_FALLING          =2, // External trigger (falling edge).
	XI_TRG_SOFTWARE              =3, // Software(manual) trigger.
	
} XI_TRG_SOURCE;

// structure containing information about trigger functionality
typedef enum
{
	XI_TRG_SEL_FRAME_START       =0, // Selects a trigger starting the capture of one frame
	XI_TRG_SEL_EXPOSURE_ACTIVE   =1, // Selects a trigger controlling the duration of one frame.
	XI_TRG_SEL_FRAME_BURST_START =2, // Selects a trigger starting the capture of the bursts of frames in an acquisition.
	XI_TRG_SEL_FRAME_BURST_ACTIVE=3, // Selects a trigger controlling the duration of the capture of the bursts of frames in an acquisition.
	XI_TRG_SEL_MULTIPLE_EXPOSURES=4, // Selects a trigger which when first trigger starts exposure and consequent pulses are gating exposure(active HI)
	
} XI_TRG_SELECTOR;

// structure containing information about acqisition timing modes
typedef enum
{
	XI_ACQ_TIMING_MODE_FREE_RUN  =0, // Selects a mode when sensor timing is given by fastest framerate possible (by exposure time and readout).
	XI_ACQ_TIMING_MODE_FRAME_RATE=1, // Selects a mode when sensor frame acquisition start is given by frame rate.
	
} XI_ACQ_TIMING_MODE;

// structure containing information about GPI functionality
typedef enum
{
	XI_GPI_OFF                   =0, // Input off. In this mode the input level can be get using parameter XI_PRM_GPI_LEVEL.
	XI_GPI_TRIGGER               =1, // Trigger input
	XI_GPI_EXT_EVENT             =2, // External signal input. It is not implemented yet.
	
} XI_GPI_MODE;

// structure containing information about GPO functionality
typedef enum
{
	XI_GPO_OFF                   =0, // Output off
	XI_GPO_ON                    =1, // Logical level.
	XI_GPO_FRAME_ACTIVE          =2, // On from exposure started until read out finished.
	XI_GPO_FRAME_ACTIVE_NEG      =3, // Off from exposure started until read out finished.
	XI_GPO_EXPOSURE_ACTIVE       =4, // On during exposure(integration) time
	XI_GPO_EXPOSURE_ACTIVE_NEG   =5, // Off during exposure(integration) time
	XI_GPO_FRAME_TRIGGER_WAIT    =6, // On when sensor is ready for next trigger edge.
	XI_GPO_FRAME_TRIGGER_WAIT_NEG=7, // Off when sensor is ready for next trigger edge.
	XI_GPO_EXPOSURE_PULSE        =8, // Short On/Off pulse on start of each exposure.
	XI_GPO_EXPOSURE_PULSE_NEG    =9, // Short Off/On pulse on start of each exposure.
	XI_GPO_BUSY                  =10, // ON when camera is busy (trigger mode - starts with trigger reception and ends with end of frame transfer from sensor; freerun - active when acq active)
	XI_GPO_BUSY_NEG              =11, // OFF when camera is busy (trigger mode  - starts with trigger reception and ends with end of frame transfer from sensor; freerun - active when acq active)
	
} XI_GPO_MODE;

// structure containing information about LED functionality
typedef enum
{
	XI_LED_HEARTBEAT             =0, // set led to blink if link is ok, (led 1), heartbeat (led 2)
	XI_LED_TRIGGER_ACTIVE        =1, // set led to blink if trigger detected
	XI_LED_EXT_EVENT_ACTIVE      =2, // set led to blink if external signal detected
	XI_LED_LINK                  =3, // set led to blink if link is ok
	XI_LED_ACQUISITION           =4, // set led to blink if data streaming
	XI_LED_EXPOSURE_ACTIVE       =5, // set led to blink if sensor integration time
	XI_LED_FRAME_ACTIVE          =6, // set led to blink if device busy/not busy
	XI_LED_OFF                   =7, // set led to zero
	XI_LED_ON                    =8, // set led to one
	XI_LED_BLINK                 =9, // set led to ~1Hz blink
	
} XI_LED_MODE;

// structure containing information about parameters type
typedef enum
{
	xiTypeInteger                =0, // integer parameter type
	xiTypeFloat                  =1, // float parameter type
	xiTypeString                 =2, // string parameter type
	
} XI_PRM_TYPE;

// Turn parameter On/Off
typedef enum
{
	XI_OFF                       =0, // Turn parameter off
	XI_ON                        =1, // Turn parameter on
	
} XI_SWITCH;

// Downsampling types
typedef enum
{
	XI_BINNING                   =0, // Downsampling is using  binning
	XI_SKIPPING                  =1, // Downsampling is using  skipping
	
} XI_DOWNSAMPLING_TYPE;

// Shutter mode types
typedef enum
{
	XI_SHUTTER_GLOBAL            =0, // Sensor Global Shutter(CMOS sensor)
	XI_SHUTTER_ROLLING           =1, // Sensor Electronic Rolling Shutter(CMOS sensor)
	XI_SHUTTER_GLOBAL_RESET_RELEASE=2, // Sensor Global Reset Release Shutter(CMOS sensor)
	
} XI_SHUTTER_TYPE;

// structure containing information about CMS functionality
typedef enum
{
	XI_CMS_DIS                   =0, // CMS disable
	XI_CMS_EN                    =1, // CMS enable
	XI_CMS_EN_FAST               =2, // CMS enable(fast)
	
} XI_CMS_MODE;

// structure containing information about options for selection of camera before onening
typedef enum
{
	XI_OPEN_BY_INST_PATH         =0, // Open camera by its hardware path
	XI_OPEN_BY_SN                =1, // Open camera by its serial number
	XI_OPEN_BY_USER_ID           =2, // open camera by its custom user ID
	
} XI_OPEN_BY;

//-------------------------------------------------------------------------------------------------------------------
// xiAPI structures
// structure containing information about incoming image.
typedef struct
{
	DWORD         size;      // Size of current structure on application side. When xiGetImage is called and size>=SIZE_XI_IMG_V2 then GPI_level, tsSec and tsUSec are filled.
	LPVOID        bp;        // pointer to data. If NULL, xiApi allocates new buffer.
	DWORD         bp_size;   // Filled buffer size. When buffer policy is set to XI_BP_SAFE, xiGetImage will fill this field with current size of image data received.
	XI_IMG_FORMAT frm;       // format of incoming data.
	DWORD         width;     // width of incoming image.
	DWORD         height;    // height of incoming image.
	DWORD         nframe;    // frame number(reset by exposure, gain, downsampling change).
	DWORD         tsSec;     // TimeStamp in seconds
	DWORD         tsUSec;    // TimeStamp in microseconds
	DWORD         GPI_level; // Input level
	DWORD         black_level;// Black level of image (ONLY for MONO and RAW formats)
	DWORD         padding_x; // Number of extra bytes provided at the end of each line to facilitate image alignment in buffers.
	DWORD         AbsoluteOffsetX;// Horizontal offset of origin of sensor and buffer image first pixel.
	DWORD         AbsoluteOffsetY;// Vertical offset of origin of sensor and buffer image first pixel.
	
}XI_IMG, *LPXI_IMG;

//-------------------------------------------------------------------------------------------------------------------
// Global definitions

#define SIZE_XI_IMG_V1               28                   // structure size default
#define SIZE_XI_IMG_V2               40                   // structure size with timestamp and GPI level information
#define SIZE_XI_IMG_V3               44                   // structure size with black level information
#define SIZE_XI_IMG_V4               48                   // structure size with horizontal buffer padding information padding_x
#define SIZE_XI_IMG_V5               56                   // structure size with AbsoluteOffsetX, AbsoluteOffsetY
#define XI_PRM_INFO_MIN              ":min"               // Parameter minimum
#define XI_PRM_INFO_MAX              ":max"               // Parameter maximum
#define XI_PRM_INFO_INCREMENT        ":inc"               // Parameter increment
#define XI_PRM_INFO                  ":info"              // Parameter value
#define XI_PRMM_DIRECT_UPDATE        ":direct_update"     // Parameter modifier for direct update without stopping the streaming. E.g. XI_PRM_EXPOSURE XI_PRMM_DIRECT_UPDATE can be used with this modifier
#define XI_MQ_LED_STATUS1            1                    // MQ Status 1 LED selection value.
#define XI_MQ_LED_STATUS2            2                    // MQ Status 2 LED selection value.
#define XI_MQ_LED_POWER              3                    // MQ Power LED selection value.
#define XI_MS_LED_STATUS1            1                    // CURRERA-R LED 1 selection value.
#define XI_MS_LED_STATUS2            2                    // CURRERA-R LED 2 selection value.
/*************************************************************************************/

#ifdef XIAPI_AS_APPLICATION
#undef XIAPI
#define XIAPI
#endif // XIAPI_AS_APPLICATION

/*************************************************************************************/
/**
   \brief Return number of discovered devices
   
   Returns the pointer to the number of all discovered devices.

   @param[out] pNumberDevices			number of discovered devices
   @return XI_OK on success, error value otherwise.
 */
XIAPI XI_RETURN __cdecl xiGetNumberDevices(OUT PDWORD pNumberDevices);
/**
   \brief Get device parameter
   
   Allows the user to get the current device state and information.
  Parameters can be used:XI_PRM_DEVICE_SN, XI_PRM_DEVICE_INSTANCE_PATH, XI_PRM_DEVICE_TYPE, XI_PRM_DEVICE_NAME

   @param[in] DevId						index of the device
   @param[in] prm						parameter name string.
   @param[in] val						pointer to parameter set value.
   @param[in] size						pointer to integer.
   @param[in] type						pointer to type container.
   @return XI_OK on success, error value otherwise.
 */
XIAPI XI_RETURN __cdecl xiGetDeviceInfo(IN DWORD DevId, const char* prm, void* val, DWORD * size, XI_PRM_TYPE * type);
/**
   \brief Initialize device
   
   This function prepares the camera's software for work.
   It populates structures, runs initializing procedures, allocates resources - prepares the camera for work.

	\note Function creates and returns handle of the specified device. To de-initialize the camera and destroy the handler xiCloseDevice should be called.	

   @param[in] DevId						index of the device
   @param[out] hDevice					handle to device
   @return XI_OK on success, error value otherwise.
 */
XIAPI XI_RETURN __cdecl xiOpenDevice(IN DWORD DevId, OUT PHANDLE hDevice);
/**   
	\brief Initialize selected device      
	
	This function prepares the camera's software for work. Camera is selected by using appropriate enumerator and input parameters. 
	It populates structures, runs initializing procedures, allocates resources - prepares the camera for work.	
	
	\note Function creates and returns handle of the specified device. To de-initialize the camera and destroy the handler xiCloseDevice should be called.	  
	
	@param[in]  sel                     select method to be used for camera selection
	@param[in]  prm                     input string to be used during camera selection
	@param[out] hDevice					handle to device   @return XI_OK on success, error value otherwise. 
	*/
XIAPI XI_RETURN __cdecl xiOpenDeviceBy(IN XI_OPEN_BY sel, IN const char* prm, OUT PHANDLE hDevice);
/**
   \brief Uninitialize device
   
   Closes camera handle and releases allocated resources.

   @param[in] hDevice					handle to device
   @return XI_OK on success, error value otherwise.
 */
XIAPI XI_RETURN __cdecl xiCloseDevice(IN HANDLE hDevice);
/**
   \brief Start image acquisition
   
   Begins the work cycle and starts data acquisition from the camera.

   @param[in] hDevice					handle to device
   @return XI_OK on success, error value otherwise.
 */
XIAPI XI_RETURN __cdecl xiStartAcquisition(IN HANDLE hDevice);
/**
   \brief Stop image acquisition
   
   Ends the work cycle of the camera, stops data acquisition and deallocates internal image buffers.

   @param[in] hDevice					handle to device
   @return XI_OK on success, error value otherwise.
 */
XIAPI XI_RETURN __cdecl xiStopAcquisition(IN HANDLE hDevice);
/**
   \brief Return pointer to image structure
   
   Allows the user to retrieve the frame into LPXI_IMG structure.

   @param[in] hDevice					handle to device
   @param[in] timeout					time interval required to wait for the image (in milliseconds).
   @param[out] img						pointer to image info structure
   @return XI_OK on success, error value otherwise.
 */
XIAPI XI_RETURN __cdecl xiGetImage(IN HANDLE hDevice, IN DWORD timeout, OUT LPXI_IMG img);
/**
   \brief Set device parameter
   
   Allows the user to control device.

   @param[in] hDevice					handle to device
   @param[in] prm						parameter name string.
   @param[in] val						pointer to parameter set value.
   @param[in] size						size of val.
   @param[in] type						val data type.
   @return XI_OK on success, error value otherwise.
 */
XIAPI XI_RETURN __cdecl xiSetParam(IN HANDLE hDevice, const char* prm, void* val, DWORD size, XI_PRM_TYPE type);
/**
   \brief Get device parameter
   
   Allows the user to get the current device state and information.

   @param[in] hDevice					handle to device
   @param[in] prm						parameter name string.
   @param[in] val						pointer to parameter set value.
   @param[in] size						pointer to integer.
   @param[in] type						pointer to type container.
   @return XI_OK on success, error value otherwise.
 */
XIAPI XI_RETURN __cdecl xiGetParam(IN HANDLE hDevice, const char* prm, void* val, DWORD * size, XI_PRM_TYPE * type);

/*-----------------------------------------------------------------------------------*/
//Set device parameter
XIAPI XI_RETURN __cdecl xiSetParamInt(IN HANDLE hDevice, const char* prm, const int val);
XIAPI XI_RETURN __cdecl xiSetParamFloat(IN HANDLE hDevice, const char* prm, const float val);
XIAPI XI_RETURN __cdecl xiSetParamString(IN HANDLE hDevice, const char* prm, void* val, DWORD size);
/*-----------------------------------------------------------------------------------*/
//Get device parameter
XIAPI XI_RETURN __cdecl xiGetParamInt(IN HANDLE hDevice, const char* prm, int* val);
XIAPI XI_RETURN __cdecl xiGetParamFloat(IN HANDLE hDevice, const char* prm, float* val);
XIAPI XI_RETURN __cdecl xiGetParamString(IN HANDLE hDevice, const char* prm, void* val, DWORD size);
/*-----------------------------------------------------------------------------------*/
//Get device info
XIAPI XI_RETURN __cdecl xiGetDeviceInfoInt(IN DWORD DevId, const char* prm, int* value);
XIAPI XI_RETURN __cdecl xiGetDeviceInfoString(IN DWORD DevId, const char* prm, char* value, DWORD value_size);
/*-----------------------------------------------------------------------------------*/


/*************************************************************************************/
// XIMEA Offline Processing Interface
// All functions can be called independently on camera device
/*************************************************************************************/

/*-----------------------------------------------------------------------------------*/
// Workflow:
//
// xiProcessingHandle_t proc;
// xiProcOpen(proc)
// get cam_context (string) previously stored in acquisition time
// xiProcSetParam(proc, XI_PRM_API_CONTEXT_LIST, cam_context, strlen(cam_context), xiTypeString)
// while (in_image is available)
// {
//    xiProcPushImage(proc, in_image)
//    xiProcPullImage(proc, out_image)
//    use processed image (out_image)
// }
// xiProcClose(proc)
/*-----------------------------------------------------------------------------------*/

typedef void* xiProcessingHandle_t;

/**
* OpenProcessing
* Opens new instance for Image Processing entity
* @param[out] processing_handle New processing handle - valid on success
*/
XIAPI XI_RETURN __cdecl xiProcOpen(xiProcessingHandle_t* processing_handle);

/**
* xiSetProcParam
* Sets the selected parameter to processing
* @param[in] processing_handle			Handle for processing
* @param[in] prm						parameter name string.
* @param[in] val						pointer to parameter set value.
* @param[in] size						size of val.
* @param[in] type						val data type.
* @return XI_OK on success, error value otherwise.
*/
XIAPI XI_RETURN __cdecl xiProcSetParam(xiProcessingHandle_t processing_handle, const char* prm, void* val, DWORD size, XI_PRM_TYPE type);

/**
* xiPushImage
* Set unprocessed image to processing chain
* @param[in] processing_handle Processing handle
* @param[out] fist_pixel First byte of first pixel of image to be processed
*/
XIAPI XI_RETURN __cdecl xiProcPushImage(xiProcessingHandle_t processing_handle, unsigned char* first_pixel);

/**
* xiPullImage
* Gets processed image from processing chain
* @param[in] processing_handle Processing handle
* @param[in] timeout_ms Processing handle
*/
XIAPI XI_RETURN __cdecl xiProcPullImage(xiProcessingHandle_t processing_handle, int timeout_ms, XI_IMG* new_image);

/**
* CloseProcessing
* Closes instance for Image Processing entity
* @param processing_handle[out] Processing handle to be closed
*/
XIAPI XI_RETURN __cdecl xiProcClose(xiProcessingHandle_t processing_handle);

/*************************************************************************************/

#ifdef __cplusplus
}
#endif

#endif /* __XIAPI_H */
