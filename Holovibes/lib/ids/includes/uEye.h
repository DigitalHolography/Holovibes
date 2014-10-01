/*****************************************************************************/
/*! \file    ueye.h
*   \author  (c) 2004-2014 by Imaging Development Systems GmbH
*   \date    Date: 2014/03/10
*   \version PRODUCTVERSION: 4.40
*
*   \brief   Library interface for IDS uEye - camera family.
*            definition of exported API functions and constants
*//***************************************************************************/

#ifndef __IDS_HEADER__ 
#define __IDS_HEADER__

#pragma pack(push, 8)

#ifdef __cplusplus
extern "C" {
#endif  /* __cplusplus */

/* make sure __LINUX__ is defined */
#if defined __linux__
#   if !defined __LINUX__
#       define __LINUX__
#   endif /* !defined __LINUX__ */
#endif /* defined __linux__ */


// ----------------------------------------------------------------------------
// Version definition
// ----------------------------------------------------------------------------
#ifndef UEYE_VERSION
#   define UEYE_VERSION(major, minor, build)    (((major)<<24) + ((minor)<<16) + (build))
#endif

#ifndef UEYE_VERSION_CODE
#   define UEYE_VERSION_CODE   UEYE_VERSION(4, 40, 0)
#endif


// ----------------------------------------------------------------------------
// Driver name
// ----------------------------------------------------------------------------
#if defined(__linux__) || defined(__LINUX__)
#   define DRIVER_DLL_NAME                  "libueye_api.so"
#elif defined (_WIN64)
#   define DRIVER_DLL_NAME                  "ueye_api_64.dll"
#else
#   define DRIVER_DLL_NAME                  "ueye_api.dll"
#endif


// ----------------------------------------------------------------------------
// Color modes
// ----------------------------------------------------------------------------
#define IS_COLORMODE_INVALID                0
#define IS_COLORMODE_MONOCHROME             1
#define IS_COLORMODE_BAYER                  2
#define IS_COLORMODE_CBYCRY                 4
#define IS_COLORMODE_JPEG                   8

// ----------------------------------------------------------------------------
//  Sensor Types
// ----------------------------------------------------------------------------
#define IS_SENSOR_INVALID           0x0000

// CMOS Sensors
#define IS_SENSOR_UI141X_M          0x0001      // VGA rolling shutter, monochrome
#define IS_SENSOR_UI141X_C          0x0002      // VGA rolling shutter, color
#define IS_SENSOR_UI144X_M          0x0003      // SXGA rolling shutter, monochrome
#define IS_SENSOR_UI144X_C          0x0004      // SXGA rolling shutter, SXGA color

#define IS_SENSOR_UI154X_M          0x0030      // SXGA rolling shutter, monochrome
#define IS_SENSOR_UI154X_C          0x0031      // SXGA rolling shutter, color
#define IS_SENSOR_UI145X_C          0x0008      // UXGA rolling shutter, color

#define IS_SENSOR_UI146X_C          0x000a      // QXGA rolling shutter, color
#define IS_SENSOR_UI148X_M          0x000b      // 5MP rolling shutter, monochrome
#define IS_SENSOR_UI148X_C          0x000c      // 5MP rolling shutter, color

#define IS_SENSOR_UI121X_M          0x0010      // VGA global shutter, monochrome
#define IS_SENSOR_UI121X_C          0x0011      // VGA global shutter, VGA color
#define IS_SENSOR_UI122X_M          0x0012      // WVGA global shutter, monochrome
#define IS_SENSOR_UI122X_C          0x0013      // WVGA global shutter, color

#define IS_SENSOR_UI164X_C          0x0015      // SXGA rolling shutter, color

#define IS_SENSOR_UI155X_C          0x0017      // UXGA rolling shutter, color

#define IS_SENSOR_UI1223_M          0x0018      // WVGA global shutter, monochrome
#define IS_SENSOR_UI1223_C          0x0019      // WVGA global shutter, color

#define IS_SENSOR_UI149X_M          0x003E      // 10MP rolling shutter, monochrome
#define IS_SENSOR_UI149X_C          0x003F      // 10MP rolling shutter, color

#define IS_SENSOR_UI1225_M          0x0022      // WVGA global shutter, monochrome, LE model
#define IS_SENSOR_UI1225_C          0x0023      // WVGA global shutter, color, LE model

#define IS_SENSOR_UI1645_C          0x0025      // SXGA rolling shutter, color, LE model
#define IS_SENSOR_UI1555_C          0x0027      // UXGA rolling shutter, color, LE model
#define IS_SENSOR_UI1545_M          0x0028      // SXGA rolling shutter, monochrome, LE model
#define IS_SENSOR_UI1545_C          0x0029      // SXGA rolling shutter, color, LE model
#define IS_SENSOR_UI1455_C          0x002B      // UXGA rolling shutter, color, LE model
#define IS_SENSOR_UI1465_C          0x002D      // QXGA rolling shutter, color, LE model
#define IS_SENSOR_UI1485_M          0x002E      // 5MP rolling shutter, monochrome, LE model
#define IS_SENSOR_UI1485_C          0x002F      // 5MP rolling shutter, color, LE model
#define IS_SENSOR_UI1495_M          0x0040      // 10MP rolling shutter, monochrome, LE model
#define IS_SENSOR_UI1495_C          0x0041      // 10MP rolling shutter, color, LE model

#define IS_SENSOR_UI112X_M          0x004A      // 0768x576, HDR sensor, monochrome
#define IS_SENSOR_UI112X_C          0x004B      // 0768x576, HDR sensor, color

#define IS_SENSOR_UI1008_M          0x004C
#define IS_SENSOR_UI1008_C          0x004D

#define IS_SENSOR_UIF005_M          0x0076 
#define IS_SENSOR_UIF005_C          0x0077 

#define IS_SENSOR_UI1005_M          0x020A 
#define IS_SENSOR_UI1005_C          0x020B

#define IS_SENSOR_UI1240_M          0x0050      // SXGA global shutter, monochrome
#define IS_SENSOR_UI1240_C          0x0051      // SXGA global shutter, color
#define IS_SENSOR_UI1240_NIR        0x0062      // SXGA global shutter, NIR

#define IS_SENSOR_UI1240LE_M        0x0054      // SXGA global shutter, monochrome, single board
#define IS_SENSOR_UI1240LE_C        0x0055      // SXGA global shutter, color, single board
#define IS_SENSOR_UI1240LE_NIR      0x0064      // SXGA global shutter, NIR, single board

#define IS_SENSOR_UI1240ML_M        0x0066      // SXGA global shutter, monochrome, single board
#define IS_SENSOR_UI1240ML_C        0x0067      // SXGA global shutter, color, single board
#define IS_SENSOR_UI1240ML_NIR      0x0200      // SXGA global shutter, NIR, single board

#define IS_SENSOR_UI1243_M_SMI      0x0078
#define IS_SENSOR_UI1243_C_SMI      0x0079

#define IS_SENSOR_UI1543_M          0x0032      // SXGA rolling shutter, monochrome, single board
#define IS_SENSOR_UI1543_C          0x0033      // SXGA rolling shutter, color, single board

#define IS_SENSOR_UI1544_M          0x003A      // SXGA rolling shutter, monochrome, single board
#define IS_SENSOR_UI1544_C          0x003B      // SXGA rolling shutter, color, single board
#define IS_SENSOR_UI1543_M_WO       0x003C      // SXGA rolling shutter, monochrome, single board
#define IS_SENSOR_UI1543_C_WO       0x003D      // SXGA rolling shutter, color, single board
#define IS_SENSOR_UI1453_C          0x0035      // UXGA rolling shutter, color, single board
#define IS_SENSOR_UI1463_C          0x0037      // QXGA rolling shutter, color, single board
#define IS_SENSOR_UI1483_M          0x0038      // QSXG rolling shutter, monochrome, single board
#define IS_SENSOR_UI1483_C          0x0039      // QSXG rolling shutter, color, single board
#define IS_SENSOR_UI1493_M          0x004E      // 10Mp rolling shutter, monochrome, single board
#define IS_SENSOR_UI1493_C          0x004F      // 10MP rolling shutter, color, single board

#define IS_SENSOR_UI1463_M_WO       0x0044      // QXGA rolling shutter, monochrome, single board
#define IS_SENSOR_UI1463_C_WO       0x0045      // QXGA rolling shutter, color, single board

#define IS_SENSOR_UI1553_C_WN       0x0047      // UXGA rolling shutter, color, single board
#define IS_SENSOR_UI1483_M_WO       0x0048      // QSXGA rolling shutter, monochrome, single board
#define IS_SENSOR_UI1483_C_WO       0x0049      // QSXGA rolling shutter, color, single board

#define IS_SENSOR_UI1580_M          0x005A      // 5MP rolling shutter, monochrome
#define IS_SENSOR_UI1580_C          0x005B      // 5MP rolling shutter, color
#define IS_SENSOR_UI1580LE_M        0x0060      // 5MP rolling shutter, monochrome, single board
#define IS_SENSOR_UI1580LE_C        0x0061      // 5MP rolling shutter, color, single board

#define IS_SENSOR_UI1360M           0x0068      // 2.2MP global shutter, monochrome
#define IS_SENSOR_UI1360C           0x0069      // 2.2MP global shutter, color
#define IS_SENSOR_UI1360NIR         0x0212      // 2.2MP global shutter, NIR

#define IS_SENSOR_UI1370M           0x006A      // 4.2MP global shutter, monochrome
#define IS_SENSOR_UI1370C           0x006B      // 4.2MP global shutter, color
#define IS_SENSOR_UI1370NIR         0x0214      // 4.2MP global shutter, NIR

#define IS_SENSOR_UI1250_M          0x006C      // 2MP global shutter, monochrome
#define IS_SENSOR_UI1250_C          0x006D      // 2MP global shutter, color
#define IS_SENSOR_UI1250_NIR        0x006E      // 2MP global shutter, NIR

#define IS_SENSOR_UI1250LE_M        0x0070      // 2MP global shutter, monochrome, single board
#define IS_SENSOR_UI1250LE_C        0x0071      // 2MP global shutter, color, single board
#define IS_SENSOR_UI1250LE_NIR      0x0072      // 2MP global shutter, NIR, single board

#define IS_SENSOR_UI1250ML_M        0x0074      // 2MP global shutter, monochrome, single board
#define IS_SENSOR_UI1250ML_C        0x0075      // 2MP global shutter, color, single board
#define IS_SENSOR_UI1250ML_NIR      0x0202      // 2MP global shutter, NIR, single board

#define IS_SENSOR_XS                0x020B      // 5MP rolling shutter, color

#define IS_SENSOR_UI1493_M_AR       0x0204
#define IS_SENSOR_UI1493_C_AR       0x0205


// CCD Sensors
#define IS_SENSOR_UI223X_M          0x0080      // Sony CCD sensor - XGA monochrome
#define IS_SENSOR_UI223X_C          0x0081      // Sony CCD sensor - XGA color

#define IS_SENSOR_UI241X_M          0x0082      // Sony CCD sensor - VGA monochrome
#define IS_SENSOR_UI241X_C          0x0083      // Sony CCD sensor - VGA color

#define IS_SENSOR_UI234X_M          0x0084      // Sony CCD sensor - SXGA monochrome
#define IS_SENSOR_UI234X_C          0x0085      // Sony CCD sensor - SXGA color

#define IS_SENSOR_UI221X_M          0x0088      // Sony CCD sensor - VGA monochrome
#define IS_SENSOR_UI221X_C          0x0089      // Sony CCD sensor - VGA color

#define IS_SENSOR_UI231X_M          0x0090      // Sony CCD sensor - VGA monochrome
#define IS_SENSOR_UI231X_C          0x0091      // Sony CCD sensor - VGA color

#define IS_SENSOR_UI222X_M          0x0092      // Sony CCD sensor - CCIR / PAL monochrome
#define IS_SENSOR_UI222X_C          0x0093      // Sony CCD sensor - CCIR / PAL color

#define IS_SENSOR_UI224X_M          0x0096      // Sony CCD sensor - SXGA monochrome
#define IS_SENSOR_UI224X_C          0x0097      // Sony CCD sensor - SXGA color

#define IS_SENSOR_UI225X_M          0x0098      // Sony CCD sensor - UXGA monochrome
#define IS_SENSOR_UI225X_C          0x0099      // Sony CCD sensor - UXGA color

#define IS_SENSOR_UI214X_M          0x009A      // Sony CCD sensor - SXGA monochrome
#define IS_SENSOR_UI214X_C          0x009B      // Sony CCD sensor - SXGA color

#define IS_SENSOR_UI228X_M          0x009C      // Sony CCD sensor - QXGA monochrome
#define IS_SENSOR_UI228X_C          0x009D      // Sony CCD sensor - QXGA color

#define IS_SENSOR_UI241X_M_R2       0x0182      // Sony CCD sensor - VGA monochrome
#define IS_SENSOR_UI251X_M          0x0182      // Sony CCD sensor - VGA monochrome
#define IS_SENSOR_UI241X_C_R2       0x0183      // Sony CCD sensor - VGA color
#define IS_SENSOR_UI251X_C          0x0183      // Sony CCD sensor - VGA color

#define IS_SENSOR_UI2130_M          0x019E      // Sony CCD sensor - WXGA monochrome
#define IS_SENSOR_UI2130_C          0x019F      // Sony CCD sensor - WXGA color

// ----------------------------------------------------------------------------
// Error codes
// ----------------------------------------------------------------------------
#define IS_NO_SUCCESS                        -1   // function call failed
#define IS_SUCCESS                            0   // function call succeeded
#define IS_INVALID_CAMERA_HANDLE              1   // camera handle is not valid or zero
#define IS_INVALID_HANDLE                     1   // a handle other than the camera handle is invalid

#define IS_IO_REQUEST_FAILED                  2   // an io request to the driver failed
#define IS_CANT_OPEN_DEVICE                   3   // returned by is_InitCamera
#define IS_CANT_CLOSE_DEVICE                  4
#define IS_CANT_SETUP_MEMORY                  5
#define IS_NO_HWND_FOR_ERROR_REPORT           6
#define IS_ERROR_MESSAGE_NOT_CREATED          7
#define IS_ERROR_STRING_NOT_FOUND             8
#define IS_HOOK_NOT_CREATED                   9
#define IS_TIMER_NOT_CREATED                 10
#define IS_CANT_OPEN_REGISTRY                11
#define IS_CANT_READ_REGISTRY                12
#define IS_CANT_VALIDATE_BOARD               13
#define IS_CANT_GIVE_BOARD_ACCESS            14
#define IS_NO_IMAGE_MEM_ALLOCATED            15
#define IS_CANT_CLEANUP_MEMORY               16
#define IS_CANT_COMMUNICATE_WITH_DRIVER      17
#define IS_FUNCTION_NOT_SUPPORTED_YET        18
#define IS_OPERATING_SYSTEM_NOT_SUPPORTED    19

#define IS_INVALID_VIDEO_IN                  20
#define IS_INVALID_IMG_SIZE                  21
#define IS_INVALID_ADDRESS                   22
#define IS_INVALID_VIDEO_MODE                23
#define IS_INVALID_AGC_MODE                  24
#define IS_INVALID_GAMMA_MODE                25
#define IS_INVALID_SYNC_LEVEL                26
#define IS_INVALID_CBARS_MODE                27
#define IS_INVALID_COLOR_MODE                28
#define IS_INVALID_SCALE_FACTOR              29
#define IS_INVALID_IMAGE_SIZE                30
#define IS_INVALID_IMAGE_POS                 31
#define IS_INVALID_CAPTURE_MODE              32
#define IS_INVALID_RISC_PROGRAM              33
#define IS_INVALID_BRIGHTNESS                34
#define IS_INVALID_CONTRAST                  35
#define IS_INVALID_SATURATION_U              36
#define IS_INVALID_SATURATION_V              37
#define IS_INVALID_HUE                       38
#define IS_INVALID_HOR_FILTER_STEP           39
#define IS_INVALID_VERT_FILTER_STEP          40
#define IS_INVALID_EEPROM_READ_ADDRESS       41
#define IS_INVALID_EEPROM_WRITE_ADDRESS      42
#define IS_INVALID_EEPROM_READ_LENGTH        43
#define IS_INVALID_EEPROM_WRITE_LENGTH       44
#define IS_INVALID_BOARD_INFO_POINTER        45
#define IS_INVALID_DISPLAY_MODE              46
#define IS_INVALID_ERR_REP_MODE              47
#define IS_INVALID_BITS_PIXEL                48
#define IS_INVALID_MEMORY_POINTER            49

#define IS_FILE_WRITE_OPEN_ERROR             50
#define IS_FILE_READ_OPEN_ERROR              51
#define IS_FILE_READ_INVALID_BMP_ID          52
#define IS_FILE_READ_INVALID_BMP_SIZE        53
#define IS_FILE_READ_INVALID_BIT_COUNT       54
#define IS_WRONG_KERNEL_VERSION              55

#define IS_RISC_INVALID_XLENGTH              60
#define IS_RISC_INVALID_YLENGTH              61
#define IS_RISC_EXCEED_IMG_SIZE              62

// DirectDraw Mode errors
#define IS_DD_MAIN_FAILED                    70
#define IS_DD_PRIMSURFACE_FAILED             71
#define IS_DD_SCRN_SIZE_NOT_SUPPORTED        72
#define IS_DD_CLIPPER_FAILED                 73
#define IS_DD_CLIPPER_HWND_FAILED            74
#define IS_DD_CLIPPER_CONNECT_FAILED         75
#define IS_DD_BACKSURFACE_FAILED             76
#define IS_DD_BACKSURFACE_IN_SYSMEM          77
#define IS_DD_MDL_MALLOC_ERR                 78
#define IS_DD_MDL_SIZE_ERR                   79
#define IS_DD_CLIP_NO_CHANGE                 80
#define IS_DD_PRIMMEM_NULL                   81
#define IS_DD_BACKMEM_NULL                   82
#define IS_DD_BACKOVLMEM_NULL                83
#define IS_DD_OVERLAYSURFACE_FAILED          84
#define IS_DD_OVERLAYSURFACE_IN_SYSMEM       85
#define IS_DD_OVERLAY_NOT_ALLOWED            86
#define IS_DD_OVERLAY_COLKEY_ERR             87
#define IS_DD_OVERLAY_NOT_ENABLED            88
#define IS_DD_GET_DC_ERROR                   89
#define IS_DD_DDRAW_DLL_NOT_LOADED           90
#define IS_DD_THREAD_NOT_CREATED             91
#define IS_DD_CANT_GET_CAPS                  92
#define IS_DD_NO_OVERLAYSURFACE              93
#define IS_DD_NO_OVERLAYSTRETCH              94
#define IS_DD_CANT_CREATE_OVERLAYSURFACE     95
#define IS_DD_CANT_UPDATE_OVERLAYSURFACE     96
#define IS_DD_INVALID_STRETCH                97

#define IS_EV_INVALID_EVENT_NUMBER          100
#define IS_INVALID_MODE                     101
#define IS_CANT_FIND_FALCHOOK               102
#define IS_CANT_FIND_HOOK                   102
#define IS_CANT_GET_HOOK_PROC_ADDR          103
#define IS_CANT_CHAIN_HOOK_PROC             104
#define IS_CANT_SETUP_WND_PROC              105
#define IS_HWND_NULL                        106
#define IS_INVALID_UPDATE_MODE              107
#define IS_NO_ACTIVE_IMG_MEM                108
#define IS_CANT_INIT_EVENT                  109
#define IS_FUNC_NOT_AVAIL_IN_OS             110
#define IS_CAMERA_NOT_CONNECTED             111
#define IS_SEQUENCE_LIST_EMPTY              112
#define IS_CANT_ADD_TO_SEQUENCE             113
#define IS_LOW_OF_SEQUENCE_RISC_MEM         114
#define IS_IMGMEM2FREE_USED_IN_SEQ          115
#define IS_IMGMEM_NOT_IN_SEQUENCE_LIST      116
#define IS_SEQUENCE_BUF_ALREADY_LOCKED      117
#define IS_INVALID_DEVICE_ID                118
#define IS_INVALID_BOARD_ID                 119
#define IS_ALL_DEVICES_BUSY                 120
#define IS_HOOK_BUSY                        121
#define IS_TIMED_OUT                        122
#define IS_NULL_POINTER                     123
#define IS_WRONG_HOOK_VERSION               124
#define IS_INVALID_PARAMETER                125   // a parameter specified was invalid
#define IS_NOT_ALLOWED                      126
#define IS_OUT_OF_MEMORY                    127
#define IS_INVALID_WHILE_LIVE               128
#define IS_ACCESS_VIOLATION                 129   // an internal exception occurred
#define IS_UNKNOWN_ROP_EFFECT               130
#define IS_INVALID_RENDER_MODE              131
#define IS_INVALID_THREAD_CONTEXT           132
#define IS_NO_HARDWARE_INSTALLED            133
#define IS_INVALID_WATCHDOG_TIME            134
#define IS_INVALID_WATCHDOG_MODE            135
#define IS_INVALID_PASSTHROUGH_IN           136
#define IS_ERROR_SETTING_PASSTHROUGH_IN     137
#define IS_FAILURE_ON_SETTING_WATCHDOG      138
#define IS_NO_USB20                         139   // the usb port doesnt support usb 2.0
#define IS_CAPTURE_RUNNING                  140   // there is already a capture running

#define IS_MEMORY_BOARD_ACTIVATED           141   // operation could not execute while mboard is enabled
#define IS_MEMORY_BOARD_DEACTIVATED         142   // operation could not execute while mboard is disabled
#define IS_NO_MEMORY_BOARD_CONNECTED        143   // no memory board connected
#define IS_TOO_LESS_MEMORY                  144   // image size is above memory capacity
#define IS_IMAGE_NOT_PRESENT                145   // requested image is no longer present in the camera
#define IS_MEMORY_MODE_RUNNING              146
#define IS_MEMORYBOARD_DISABLED             147

#define IS_TRIGGER_ACTIVATED                148   // operation could not execute while trigger is enabled
#define IS_WRONG_KEY                        150
#define IS_CRC_ERROR                        151
#define IS_NOT_YET_RELEASED                 152   // this feature is not available yet
#define IS_NOT_CALIBRATED                   153   // the camera is not calibrated
#define IS_WAITING_FOR_KERNEL               154   // a request to the kernel exceeded
#define IS_NOT_SUPPORTED                    155   // operation mode is not supported
#define IS_TRIGGER_NOT_ACTIVATED            156   // operation could not execute while trigger is disabled
#define IS_OPERATION_ABORTED                157
#define IS_BAD_STRUCTURE_SIZE               158
#define IS_INVALID_BUFFER_SIZE              159
#define IS_INVALID_PIXEL_CLOCK              160
#define IS_INVALID_EXPOSURE_TIME            161
#define IS_AUTO_EXPOSURE_RUNNING            162
#define IS_CANNOT_CREATE_BB_SURF            163   // error creating backbuffer surface  
#define IS_CANNOT_CREATE_BB_MIX             164   // backbuffer mixer surfaces can not be created
#define IS_BB_OVLMEM_NULL                   165   // backbuffer overlay mem could not be locked  
#define IS_CANNOT_CREATE_BB_OVL             166   // backbuffer overlay mem could not be created  
#define IS_NOT_SUPP_IN_OVL_SURF_MODE        167   // function not supported in overlay surface mode  
#define IS_INVALID_SURFACE                  168   // surface invalid
#define IS_SURFACE_LOST                     169   // surface has been lost  
#define IS_RELEASE_BB_OVL_DC                170   // error releasing backbuffer overlay DC  
#define IS_BB_TIMER_NOT_CREATED             171   // backbuffer timer could not be created  
#define IS_BB_OVL_NOT_EN                    172   // backbuffer overlay has not been enabled  
#define IS_ONLY_IN_BB_MODE                  173   // only possible in backbuffer mode 
#define IS_INVALID_COLOR_FORMAT             174   // invalid color format
#define IS_INVALID_WB_BINNING_MODE          175   // invalid binning mode for AWB 
#define IS_INVALID_I2C_DEVICE_ADDRESS       176   // invalid I2C device address
#define IS_COULD_NOT_CONVERT                177   // current image couldn't be converted
#define IS_TRANSFER_ERROR                   178   // transfer failed
#define IS_PARAMETER_SET_NOT_PRESENT        179   // the parameter set is not present
#define IS_INVALID_CAMERA_TYPE              180   // the camera type in the ini file doesn't match
#define IS_INVALID_HOST_IP_HIBYTE           181   // HIBYTE of host address is invalid
#define IS_CM_NOT_SUPP_IN_CURR_DISPLAYMODE  182   // color mode is not supported in the current display mode
#define IS_NO_IR_FILTER                     183
#define IS_STARTER_FW_UPLOAD_NEEDED         184   // device starter firmware is not compatible    

#define IS_DR_LIBRARY_NOT_FOUND                     185   // the DirectRender library could not be found
#define IS_DR_DEVICE_OUT_OF_MEMORY                  186   // insufficient graphics adapter video memory
#define IS_DR_CANNOT_CREATE_SURFACE                 187   // the image or overlay surface could not be created
#define IS_DR_CANNOT_CREATE_VERTEX_BUFFER           188   // the vertex buffer could not be created
#define IS_DR_CANNOT_CREATE_TEXTURE                 189   // the texture could not be created  
#define IS_DR_CANNOT_LOCK_OVERLAY_SURFACE           190   // the overlay surface could not be locked
#define IS_DR_CANNOT_UNLOCK_OVERLAY_SURFACE         191   // the overlay surface could not be unlocked
#define IS_DR_CANNOT_GET_OVERLAY_DC                 192   // cannot get the overlay surface DC 
#define IS_DR_CANNOT_RELEASE_OVERLAY_DC             193   // cannot release the overlay surface DC
#define IS_DR_DEVICE_CAPS_INSUFFICIENT              194   // insufficient graphics adapter capabilities
#define IS_INCOMPATIBLE_SETTING                     195   // Operation is not possible because of another incompatible setting
#define IS_DR_NOT_ALLOWED_WHILE_DC_IS_ACTIVE        196   // user App still has DC handle.
#define IS_DEVICE_ALREADY_PAIRED                    197   // The device is already paired
#define IS_SUBNETMASK_MISMATCH                      198   // The subnetmasks of the device and the adapter differ
#define IS_SUBNET_MISMATCH                          199   // The subnets of the device and the adapter differ
#define IS_INVALID_IP_CONFIGURATION                 200   // The IP configuation of the device is invalid
#define IS_DEVICE_NOT_COMPATIBLE                    201   // The device is incompatible to the driver
#define IS_NETWORK_FRAME_SIZE_INCOMPATIBLE          202   // The frame size settings of the device and the network adapter are incompatible
#define IS_NETWORK_CONFIGURATION_INVALID            203   // The network adapter configuration is invalid
#define IS_ERROR_CPU_IDLE_STATES_CONFIGURATION      204   // The setting of the CPU idle state configuration failed
#define IS_DEVICE_BUSY                              205   // The device is busy. The operation must be executed again later.
#define IS_SENSOR_INITIALIZATION_FAILED             206   // The sensor initialization failed


// ----------------------------------------------------------------------------
// common definitions
// ----------------------------------------------------------------------------
#define IS_OFF                              0
#define IS_ON                               1
#define IS_IGNORE_PARAMETER                 -1


// ----------------------------------------------------------------------------
//  device enumeration
// ----------------------------------------------------------------------------
#define IS_USE_DEVICE_ID                    0x8000L
#define IS_ALLOW_STARTER_FW_UPLOAD          0x10000L


// ----------------------------------------------------------------------------
// AutoExit enable/disable
// ----------------------------------------------------------------------------
#define IS_GET_AUTO_EXIT_ENABLED            0x8000
#define IS_DISABLE_AUTO_EXIT                0
#define IS_ENABLE_AUTO_EXIT                 1


// ----------------------------------------------------------------------------
// live/freeze parameters
// ----------------------------------------------------------------------------
#define IS_GET_LIVE                         0x8000

#define IS_WAIT                             0x0001
#define IS_DONT_WAIT                        0x0000
#define IS_FORCE_VIDEO_STOP                 0x4000
#define IS_FORCE_VIDEO_START                0x4000
#define IS_USE_NEXT_MEM                     0x8000


// ----------------------------------------------------------------------------
// video finish constants
// ----------------------------------------------------------------------------
#define IS_VIDEO_NOT_FINISH                 0
#define IS_VIDEO_FINISH                     1


// ----------------------------------------------------------------------------
// bitmap render modes
// ----------------------------------------------------------------------------
#define IS_GET_RENDER_MODE                  0x8000

#define IS_RENDER_DISABLED                  0x0000
#define IS_RENDER_NORMAL                    0x0001
#define IS_RENDER_FIT_TO_WINDOW             0x0002
#define IS_RENDER_DOWNSCALE_1_2             0x0004
#define IS_RENDER_MIRROR_UPDOWN             0x0010

#define IS_RENDER_PLANAR_COLOR_RED          0x0080
#define IS_RENDER_PLANAR_COLOR_GREEN        0x0100
#define IS_RENDER_PLANAR_COLOR_BLUE         0x0200

#define IS_RENDER_PLANAR_MONO_RED           0x0400
#define IS_RENDER_PLANAR_MONO_GREEN         0x0800
#define IS_RENDER_PLANAR_MONO_BLUE          0x1000

#define IS_USE_AS_DC_STRUCTURE              0x4000
#define IS_USE_AS_DC_HANDLE                 0x8000


// ----------------------------------------------------------------------------
// Trigger modes
// ----------------------------------------------------------------------------
#define IS_GET_EXTERNALTRIGGER              0x8000
#define IS_GET_TRIGGER_STATUS               0x8001
#define IS_GET_TRIGGER_MASK                 0x8002
#define IS_GET_TRIGGER_INPUTS               0x8003
#define IS_GET_SUPPORTED_TRIGGER_MODE       0x8004
#define IS_GET_TRIGGER_COUNTER              0x8000

#define IS_SET_TRIGGER_MASK                 0x0100
#define IS_SET_TRIGGER_CONTINUOUS           0x1000
#define IS_SET_TRIGGER_OFF                  0x0000
#define IS_SET_TRIGGER_HI_LO                (IS_SET_TRIGGER_CONTINUOUS | 0x0001) 
#define IS_SET_TRIGGER_LO_HI                (IS_SET_TRIGGER_CONTINUOUS | 0x0002) 
#define IS_SET_TRIGGER_SOFTWARE             (IS_SET_TRIGGER_CONTINUOUS | 0x0008) 
#define IS_SET_TRIGGER_HI_LO_SYNC           0x0010
#define IS_SET_TRIGGER_LO_HI_SYNC           0x0020
#define IS_SET_TRIGGER_PRE_HI_LO            (IS_SET_TRIGGER_CONTINUOUS | 0x0040)
#define IS_SET_TRIGGER_PRE_LO_HI            (IS_SET_TRIGGER_CONTINUOUS | 0x0080)

#define IS_GET_TRIGGER_DELAY                0x8000
#define IS_GET_MIN_TRIGGER_DELAY            0x8001
#define IS_GET_MAX_TRIGGER_DELAY            0x8002
#define IS_GET_TRIGGER_DELAY_GRANULARITY    0x8003


// ----------------------------------------------------------------------------
// Timing
// ----------------------------------------------------------------------------

// Pixelclock
#define IS_GET_PIXEL_CLOCK                  0x8000
#define IS_GET_DEFAULT_PIXEL_CLK            0x8001
#define IS_GET_PIXEL_CLOCK_INC              0x8005

// Frame rate
#define IS_GET_FRAMERATE                    0x8000
#define IS_GET_DEFAULT_FRAMERATE            0x8001


// ----------------------------------------------------------------------------
// Gain definitions
// ----------------------------------------------------------------------------
#define IS_GET_MASTER_GAIN                  0x8000
#define IS_GET_RED_GAIN                     0x8001
#define IS_GET_GREEN_GAIN                   0x8002
#define IS_GET_BLUE_GAIN                    0x8003
#define IS_GET_DEFAULT_MASTER               0x8004
#define IS_GET_DEFAULT_RED                  0x8005
#define IS_GET_DEFAULT_GREEN                0x8006
#define IS_GET_DEFAULT_BLUE                 0x8007
#define IS_GET_GAINBOOST                    0x8008
#define IS_SET_GAINBOOST_ON                 0x0001
#define IS_SET_GAINBOOST_OFF                0x0000
#define IS_GET_SUPPORTED_GAINBOOST          0x0002
#define IS_MIN_GAIN                         0
#define IS_MAX_GAIN                         100


// ----------------------------------------------------------------------------
// Gain factor definitions
// ----------------------------------------------------------------------------
#define IS_GET_MASTER_GAIN_FACTOR           0x8000
#define IS_GET_RED_GAIN_FACTOR              0x8001
#define IS_GET_GREEN_GAIN_FACTOR            0x8002
#define IS_GET_BLUE_GAIN_FACTOR             0x8003
#define IS_SET_MASTER_GAIN_FACTOR           0x8004
#define IS_SET_RED_GAIN_FACTOR              0x8005
#define IS_SET_GREEN_GAIN_FACTOR            0x8006
#define IS_SET_BLUE_GAIN_FACTOR             0x8007
#define IS_GET_DEFAULT_MASTER_GAIN_FACTOR   0x8008
#define IS_GET_DEFAULT_RED_GAIN_FACTOR      0x8009
#define IS_GET_DEFAULT_GREEN_GAIN_FACTOR    0x800a
#define IS_GET_DEFAULT_BLUE_GAIN_FACTOR     0x800b
#define IS_INQUIRE_MASTER_GAIN_FACTOR       0x800c
#define IS_INQUIRE_RED_GAIN_FACTOR          0x800d
#define IS_INQUIRE_GREEN_GAIN_FACTOR        0x800e
#define IS_INQUIRE_BLUE_GAIN_FACTOR         0x800f


// ----------------------------------------------------------------------------
// Global Shutter definitions
// ----------------------------------------------------------------------------
#define IS_SET_GLOBAL_SHUTTER_ON            0x0001
#define IS_SET_GLOBAL_SHUTTER_OFF           0x0000
#define IS_GET_GLOBAL_SHUTTER               0x0010
#define IS_GET_SUPPORTED_GLOBAL_SHUTTER     0x0020


// ----------------------------------------------------------------------------
// Black level definitions
// ----------------------------------------------------------------------------
#define IS_GET_BL_COMPENSATION              0x8000
#define IS_GET_BL_OFFSET                    0x8001
#define IS_GET_BL_DEFAULT_MODE              0x8002
#define IS_GET_BL_DEFAULT_OFFSET            0x8003
#define IS_GET_BL_SUPPORTED_MODE            0x8004

#define IS_BL_COMPENSATION_DISABLE          0
#define IS_BL_COMPENSATION_ENABLE           1
#define IS_BL_COMPENSATION_OFFSET           32

#define IS_MIN_BL_OFFSET                    0
#define IS_MAX_BL_OFFSET                    255


// ----------------------------------------------------------------------------
// Hardware gamma definitions
// ----------------------------------------------------------------------------
#define IS_GET_HW_GAMMA                     0x8000
#define IS_GET_HW_SUPPORTED_GAMMA           0x8001
#define IS_SET_HW_GAMMA_OFF                 0x0000
#define IS_SET_HW_GAMMA_ON                  0x0001


// ----------------------------------------------------------------------------
// Image parameters
// ----------------------------------------------------------------------------

// Saturation
#define IS_GET_SATURATION_U                 0x8000
#define IS_MIN_SATURATION_U                 0
#define IS_MAX_SATURATION_U                 200
#define IS_DEFAULT_SATURATION_U             100
#define IS_GET_SATURATION_V                 0x8001
#define IS_MIN_SATURATION_V                 0
#define IS_MAX_SATURATION_V                 200
#define IS_DEFAULT_SATURATION_V             100


// ----------------------------------------------------------------------------
// Image position and size
// ----------------------------------------------------------------------------

/* Image */
#define IS_AOI_IMAGE_SET_AOI                0x0001
#define IS_AOI_IMAGE_GET_AOI                0x0002
#define IS_AOI_IMAGE_SET_POS                0x0003
#define IS_AOI_IMAGE_GET_POS                0x0004
#define IS_AOI_IMAGE_SET_SIZE               0x0005
#define IS_AOI_IMAGE_GET_SIZE               0x0006
#define IS_AOI_IMAGE_GET_POS_MIN            0x0007
#define IS_AOI_IMAGE_GET_SIZE_MIN           0x0008
#define IS_AOI_IMAGE_GET_POS_MAX            0x0009
#define IS_AOI_IMAGE_GET_SIZE_MAX           0x0010
#define IS_AOI_IMAGE_GET_POS_INC            0x0011
#define IS_AOI_IMAGE_GET_SIZE_INC           0x0012
#define IS_AOI_IMAGE_GET_POS_X_ABS          0x0013
#define IS_AOI_IMAGE_GET_POS_Y_ABS          0x0014
#define IS_AOI_IMAGE_GET_ORIGINAL_AOI       0x0015

/* Absolute position */
#define IS_AOI_IMAGE_POS_ABSOLUTE           0x10000000
 
/* Fast move */
#define IS_AOI_IMAGE_SET_POS_FAST           0x0020
#define IS_AOI_IMAGE_GET_POS_FAST_SUPPORTED 0x0021

/* Auto features */
#define IS_AOI_AUTO_BRIGHTNESS_SET_AOI      0x0030
#define IS_AOI_AUTO_BRIGHTNESS_GET_AOI      0x0031
#define IS_AOI_AUTO_WHITEBALANCE_SET_AOI    0x0032
#define IS_AOI_AUTO_WHITEBALANCE_GET_AOI    0x0033

/* Multi AOI */
#define IS_AOI_MULTI_GET_SUPPORTED_MODES    0x0100
#define IS_AOI_MULTI_SET_AOI                0x0200
#define IS_AOI_MULTI_GET_AOI                0x0400
#define IS_AOI_MULTI_DISABLE_AOI            0x0800
#define IS_AOI_MULTI_MODE_X_Y_AXES          0x0001
#define IS_AOI_MULTI_MODE_Y_AXES            0x0002

/* AOI sequence */
#define IS_AOI_SEQUENCE_GET_SUPPORTED       0x0050
#define IS_AOI_SEQUENCE_SET_PARAMS          0x0051
#define IS_AOI_SEQUENCE_GET_PARAMS          0x0052
#define IS_AOI_SEQUENCE_SET_ENABLE          0x0053
#define IS_AOI_SEQUENCE_GET_ENABLE          0x0054

#define IS_AOI_SEQUENCE_INDEX_AOI_1         0
#define IS_AOI_SEQUENCE_INDEX_AOI_2         1
#define IS_AOI_SEQUENCE_INDEX_AOI_3         2
#define IS_AOI_SEQUENCE_INDEX_AOI_4         4


// ----------------------------------------------------------------------------
// ROP effect constants
// ----------------------------------------------------------------------------
#define IS_GET_ROP_EFFECT                   0x8000
#define IS_GET_SUPPORTED_ROP_EFFECT         0x8001

#define IS_SET_ROP_NONE                     0
#define IS_SET_ROP_MIRROR_UPDOWN            8
#define IS_SET_ROP_MIRROR_UPDOWN_ODD        16
#define IS_SET_ROP_MIRROR_UPDOWN_EVEN       32
#define IS_SET_ROP_MIRROR_LEFTRIGHT         64


// ----------------------------------------------------------------------------
// Subsampling
// ----------------------------------------------------------------------------
#define IS_GET_SUBSAMPLING                      0x8000
#define IS_GET_SUPPORTED_SUBSAMPLING            0x8001
#define IS_GET_SUBSAMPLING_TYPE                 0x8002
#define IS_GET_SUBSAMPLING_FACTOR_HORIZONTAL    0x8004
#define IS_GET_SUBSAMPLING_FACTOR_VERTICAL      0x8008

#define IS_SUBSAMPLING_DISABLE                  0x00

#define IS_SUBSAMPLING_2X_VERTICAL              0x0001
#define IS_SUBSAMPLING_2X_HORIZONTAL            0x0002
#define IS_SUBSAMPLING_4X_VERTICAL              0x0004
#define IS_SUBSAMPLING_4X_HORIZONTAL            0x0008
#define IS_SUBSAMPLING_3X_VERTICAL              0x0010
#define IS_SUBSAMPLING_3X_HORIZONTAL            0x0020
#define IS_SUBSAMPLING_5X_VERTICAL              0x0040
#define IS_SUBSAMPLING_5X_HORIZONTAL            0x0080
#define IS_SUBSAMPLING_6X_VERTICAL              0x0100
#define IS_SUBSAMPLING_6X_HORIZONTAL            0x0200
#define IS_SUBSAMPLING_8X_VERTICAL              0x0400
#define IS_SUBSAMPLING_8X_HORIZONTAL            0x0800
#define IS_SUBSAMPLING_16X_VERTICAL             0x1000
#define IS_SUBSAMPLING_16X_HORIZONTAL           0x2000

#define IS_SUBSAMPLING_COLOR                    0x01
#define IS_SUBSAMPLING_MONO                     0x02

#define IS_SUBSAMPLING_MASK_VERTICAL            (IS_SUBSAMPLING_2X_VERTICAL | IS_SUBSAMPLING_4X_VERTICAL | IS_SUBSAMPLING_3X_VERTICAL | IS_SUBSAMPLING_5X_VERTICAL | IS_SUBSAMPLING_6X_VERTICAL | IS_SUBSAMPLING_8X_VERTICAL | IS_SUBSAMPLING_16X_VERTICAL)
#define IS_SUBSAMPLING_MASK_HORIZONTAL          (IS_SUBSAMPLING_2X_HORIZONTAL | IS_SUBSAMPLING_4X_HORIZONTAL | IS_SUBSAMPLING_3X_HORIZONTAL | IS_SUBSAMPLING_5X_HORIZONTAL | IS_SUBSAMPLING_6X_HORIZONTAL | IS_SUBSAMPLING_8X_HORIZONTAL | IS_SUBSAMPLING_16X_HORIZONTAL)


// ----------------------------------------------------------------------------
// Binning
// ----------------------------------------------------------------------------
#define IS_GET_BINNING                      0x8000
#define IS_GET_SUPPORTED_BINNING            0x8001
#define IS_GET_BINNING_TYPE                 0x8002
#define IS_GET_BINNING_FACTOR_HORIZONTAL    0x8004
#define IS_GET_BINNING_FACTOR_VERTICAL      0x8008

#define IS_BINNING_DISABLE                  0x00

#define IS_BINNING_2X_VERTICAL              0x0001
#define IS_BINNING_2X_HORIZONTAL            0x0002
#define IS_BINNING_4X_VERTICAL              0x0004
#define IS_BINNING_4X_HORIZONTAL            0x0008
#define IS_BINNING_3X_VERTICAL              0x0010
#define IS_BINNING_3X_HORIZONTAL            0x0020
#define IS_BINNING_5X_VERTICAL              0x0040
#define IS_BINNING_5X_HORIZONTAL            0x0080
#define IS_BINNING_6X_VERTICAL              0x0100
#define IS_BINNING_6X_HORIZONTAL            0x0200
#define IS_BINNING_8X_VERTICAL              0x0400
#define IS_BINNING_8X_HORIZONTAL            0x0800
#define IS_BINNING_16X_VERTICAL             0x1000
#define IS_BINNING_16X_HORIZONTAL           0x2000

#define IS_BINNING_COLOR                    0x01
#define IS_BINNING_MONO                     0x02

#define IS_BINNING_MASK_VERTICAL            (IS_BINNING_2X_VERTICAL | IS_BINNING_3X_VERTICAL | IS_BINNING_4X_VERTICAL | IS_BINNING_5X_VERTICAL | IS_BINNING_6X_VERTICAL | IS_BINNING_8X_VERTICAL | IS_BINNING_16X_VERTICAL)
#define IS_BINNING_MASK_HORIZONTAL          (IS_BINNING_2X_HORIZONTAL | IS_BINNING_3X_HORIZONTAL | IS_BINNING_4X_HORIZONTAL | IS_BINNING_5X_HORIZONTAL | IS_BINNING_6X_HORIZONTAL | IS_BINNING_8X_HORIZONTAL | IS_BINNING_16X_HORIZONTAL)


// ----------------------------------------------------------------------------
// Auto Control Parameter
// ----------------------------------------------------------------------------
#define IS_SET_ENABLE_AUTO_GAIN                     0x8800
#define IS_GET_ENABLE_AUTO_GAIN                     0x8801
#define IS_SET_ENABLE_AUTO_SHUTTER                  0x8802
#define IS_GET_ENABLE_AUTO_SHUTTER                  0x8803
#define IS_SET_ENABLE_AUTO_WHITEBALANCE             0x8804
#define IS_GET_ENABLE_AUTO_WHITEBALANCE             0x8805
#define IS_SET_ENABLE_AUTO_FRAMERATE                0x8806
#define IS_GET_ENABLE_AUTO_FRAMERATE                0x8807
#define IS_SET_ENABLE_AUTO_SENSOR_GAIN              0x8808
#define IS_GET_ENABLE_AUTO_SENSOR_GAIN              0x8809
#define IS_SET_ENABLE_AUTO_SENSOR_SHUTTER           0x8810
#define IS_GET_ENABLE_AUTO_SENSOR_SHUTTER           0x8811
#define IS_SET_ENABLE_AUTO_SENSOR_GAIN_SHUTTER      0x8812
#define IS_GET_ENABLE_AUTO_SENSOR_GAIN_SHUTTER      0x8813
#define IS_SET_ENABLE_AUTO_SENSOR_FRAMERATE         0x8814
#define IS_GET_ENABLE_AUTO_SENSOR_FRAMERATE         0x8815
#define IS_SET_ENABLE_AUTO_SENSOR_WHITEBALANCE      0x8816
#define IS_GET_ENABLE_AUTO_SENSOR_WHITEBALANCE      0x8817

#define IS_SET_AUTO_REFERENCE                       0x8000
#define IS_GET_AUTO_REFERENCE                       0x8001
#define IS_SET_AUTO_GAIN_MAX                        0x8002
#define IS_GET_AUTO_GAIN_MAX                        0x8003
#define IS_SET_AUTO_SHUTTER_MAX                     0x8004
#define IS_GET_AUTO_SHUTTER_MAX                     0x8005
#define IS_SET_AUTO_SPEED                           0x8006
#define IS_GET_AUTO_SPEED                           0x8007
#define IS_SET_AUTO_WB_OFFSET                       0x8008
#define IS_GET_AUTO_WB_OFFSET                       0x8009
#define IS_SET_AUTO_WB_GAIN_RANGE                   0x800A
#define IS_GET_AUTO_WB_GAIN_RANGE                   0x800B
#define IS_SET_AUTO_WB_SPEED                        0x800C
#define IS_GET_AUTO_WB_SPEED                        0x800D
#define IS_SET_AUTO_WB_ONCE                         0x800E
#define IS_GET_AUTO_WB_ONCE                         0x800F
#define IS_SET_AUTO_BRIGHTNESS_ONCE                 0x8010
#define IS_GET_AUTO_BRIGHTNESS_ONCE                 0x8011
#define IS_SET_AUTO_HYSTERESIS                      0x8012
#define IS_GET_AUTO_HYSTERESIS                      0x8013
#define IS_GET_AUTO_HYSTERESIS_RANGE                0x8014
#define IS_SET_AUTO_WB_HYSTERESIS                   0x8015
#define IS_GET_AUTO_WB_HYSTERESIS                   0x8016
#define IS_GET_AUTO_WB_HYSTERESIS_RANGE             0x8017
#define IS_SET_AUTO_SKIPFRAMES                      0x8018
#define IS_GET_AUTO_SKIPFRAMES                      0x8019
#define IS_GET_AUTO_SKIPFRAMES_RANGE                0x801A
#define IS_SET_AUTO_WB_SKIPFRAMES                   0x801B
#define IS_GET_AUTO_WB_SKIPFRAMES                   0x801C
#define IS_GET_AUTO_WB_SKIPFRAMES_RANGE             0x801D
#define IS_SET_SENS_AUTO_SHUTTER_PHOTOM             0x801E
#define IS_SET_SENS_AUTO_GAIN_PHOTOM                0x801F
#define IS_GET_SENS_AUTO_SHUTTER_PHOTOM             0x8020
#define IS_GET_SENS_AUTO_GAIN_PHOTOM                0x8021
#define IS_GET_SENS_AUTO_SHUTTER_PHOTOM_DEF         0x8022
#define IS_GET_SENS_AUTO_GAIN_PHOTOM_DEF            0x8023
#define IS_SET_SENS_AUTO_CONTRAST_CORRECTION        0x8024
#define IS_GET_SENS_AUTO_CONTRAST_CORRECTION        0x8025
#define IS_GET_SENS_AUTO_CONTRAST_CORRECTION_RANGE  0x8026
#define IS_GET_SENS_AUTO_CONTRAST_CORRECTION_INC    0x8027
#define IS_GET_SENS_AUTO_CONTRAST_CORRECTION_DEF    0x8028
#define IS_SET_SENS_AUTO_CONTRAST_FDT_AOI_ENABLE    0x8029
#define IS_GET_SENS_AUTO_CONTRAST_FDT_AOI_ENABLE    0x8030
#define IS_SET_SENS_AUTO_BACKLIGHT_COMP             0x8031
#define IS_GET_SENS_AUTO_BACKLIGHT_COMP             0x8032
#define IS_GET_SENS_AUTO_BACKLIGHT_COMP_RANGE       0x8033
#define IS_GET_SENS_AUTO_BACKLIGHT_COMP_INC         0x8034
#define IS_GET_SENS_AUTO_BACKLIGHT_COMP_DEF         0x8035
#define IS_SET_ANTI_FLICKER_MODE                    0x8036
#define IS_GET_ANTI_FLICKER_MODE                    0x8037
#define IS_GET_ANTI_FLICKER_MODE_DEF                0x8038
#define IS_GET_AUTO_REFERENCE_DEF                   0x8039
#define IS_GET_AUTO_WB_OFFSET_DEF                   0x803A
#define IS_GET_AUTO_WB_OFFSET_MIN                   0x803B
#define IS_GET_AUTO_WB_OFFSET_MAX                   0x803C

// ----------------------------------------------------------------------------
// Auto Control definitions
// ----------------------------------------------------------------------------
#define IS_MIN_AUTO_BRIGHT_REFERENCE          0
#define IS_MAX_AUTO_BRIGHT_REFERENCE        255
#define IS_DEFAULT_AUTO_BRIGHT_REFERENCE    128
#define IS_MIN_AUTO_SPEED                     0
#define IS_MAX_AUTO_SPEED                   100
#define IS_DEFAULT_AUTO_SPEED                50

#define IS_DEFAULT_AUTO_WB_OFFSET             0
#define IS_MIN_AUTO_WB_OFFSET               -50
#define IS_MAX_AUTO_WB_OFFSET                50
#define IS_DEFAULT_AUTO_WB_SPEED             50
#define IS_MIN_AUTO_WB_SPEED                  0
#define IS_MAX_AUTO_WB_SPEED                100
#define IS_MIN_AUTO_WB_REFERENCE              0
#define IS_MAX_AUTO_WB_REFERENCE            255


// ----------------------------------------------------------------------------
// AOI types to set/get
// ----------------------------------------------------------------------------
#define IS_SET_AUTO_BRIGHT_AOI              0x8000
#define IS_GET_AUTO_BRIGHT_AOI              0x8001
#define IS_SET_IMAGE_AOI                    0x8002
#define IS_GET_IMAGE_AOI                    0x8003
#define IS_SET_AUTO_WB_AOI                  0x8004
#define IS_GET_AUTO_WB_AOI                  0x8005


// ----------------------------------------------------------------------------
// pixel formats
// ----------------------------------------------------------------------------

/*! \brief Read current color format in function is_SetColorMode, \ref is_SetColorMode */
#define IS_GET_COLOR_MODE                   0x8000

/*! \brief Planar vs packed format */
#define IS_CM_FORMAT_PLANAR                 0x2000
#define IS_CM_FORMAT_MASK                   0x2000

/*! \brief BGR vs. RGB order */
#define IS_CM_ORDER_BGR                     0x0000
#define IS_CM_ORDER_RGB                     0x0080
#define IS_CM_ORDER_MASK                    0x0080 

/*! \brief This flag indicates whether a packed source pixelformat should be used (also for the debayered pixel formats) */
#define IS_CM_PREFER_PACKED_SOURCE_FORMAT   0x4000

/*! 
 * \brief Enumeration of pixel formats supported by the function is_SetColorMode, \ref is_SetColorMode. 
 */

/*! \brief Raw sensor data, occupies 8 bits */
#define IS_CM_SENSOR_RAW8           11

/*! \brief Raw sensor data, occupies 16 bits */
#define IS_CM_SENSOR_RAW10          33

/*! \brief Raw sensor data, occupies 16 bits */
#define IS_CM_SENSOR_RAW12          27

/*! \brief Raw sensor data, occupies 16 bits */
#define IS_CM_SENSOR_RAW16          29

/*! \brief Mono, occupies 8 bits */
#define IS_CM_MONO8                 6

/*! \brief Mono, occupies 16 bits */
#define IS_CM_MONO10                34

/*! \brief Mono, occupies 16 bits */
#define IS_CM_MONO12                26

/*! \brief Mono, occupies 16 bits */
#define IS_CM_MONO16                28

/*! \brief BGR (5 5 5 1), 1 bit not used, occupies 16 bits */
#define IS_CM_BGR5_PACKED           (3  | IS_CM_ORDER_BGR)

/*! \brief BGR (5 6 5), occupies 16 bits */
#define IS_CM_BGR565_PACKED         (2  | IS_CM_ORDER_BGR)

/*! \brief BGR and RGB (8 8 8), occupies 24 bits */
#define IS_CM_RGB8_PACKED           (1  | IS_CM_ORDER_RGB)
#define IS_CM_BGR8_PACKED           (1  | IS_CM_ORDER_BGR)

/*! \brief BGRA and RGBA (8 8 8 8), alpha not used, occupies 32 bits */
#define IS_CM_RGBA8_PACKED          (0  | IS_CM_ORDER_RGB)
#define IS_CM_BGRA8_PACKED          (0  | IS_CM_ORDER_BGR)

/*! \brief BGRY and RGBY (8 8 8 8), occupies 32 bits */
#define IS_CM_RGBY8_PACKED          (24 | IS_CM_ORDER_RGB)
#define IS_CM_BGRY8_PACKED          (24 | IS_CM_ORDER_BGR)

/*! \brief BGR and RGB (10 10 10 2), 2 bits not used, occupies 32 bits, debayering is done from 12 bit raw */ 
#define IS_CM_RGB10_PACKED          (25 | IS_CM_ORDER_RGB)
#define IS_CM_BGR10_PACKED          (25 | IS_CM_ORDER_BGR)

/*! \brief BGR and RGB (10(16) 10(16) 10(16)), 6 MSB bits not used respectively, occupies 48 bits */
#define IS_CM_RGB10_UNPACKED        (35 | IS_CM_ORDER_RGB)
#define IS_CM_BGR10_UNPACKED        (35 | IS_CM_ORDER_BGR)

/*! \brief BGR and RGB (12(16) 12(16) 12(16)), 4 MSB bits not used respectively, occupies 48 bits */
#define IS_CM_RGB12_UNPACKED        (30 | IS_CM_ORDER_RGB)
#define IS_CM_BGR12_UNPACKED        (30 | IS_CM_ORDER_BGR)

/*! \brief BGRA and RGBA (12(16) 12(16) 12(16) 16), 4 MSB bits not used respectively, alpha not used, occupies 64 bits */
#define IS_CM_RGBA12_UNPACKED       (31 | IS_CM_ORDER_RGB)
#define IS_CM_BGRA12_UNPACKED       (31 | IS_CM_ORDER_BGR)

#define IS_CM_JPEG                  32

/*! \brief YUV422 (8 8), occupies 16 bits */
#define IS_CM_UYVY_PACKED           12
#define IS_CM_UYVY_MONO_PACKED      13
#define IS_CM_UYVY_BAYER_PACKED     14

/*! \brief YCbCr422 (8 8), occupies 16 bits */
#define IS_CM_CBYCRY_PACKED         23

/*! \brief RGB planar (8 8 8), occupies 24 bits */
#define IS_CM_RGB8_PLANAR           (1 | IS_CM_ORDER_RGB | IS_CM_FORMAT_PLANAR)

#define IS_CM_RGB12_PLANAR          //no compliant version
#define IS_CM_RGB16_PLANAR          //no compliant version

#define IS_CM_ALL_POSSIBLE          0xFFFF
#define IS_CM_MODE_MASK             0x007F


// ----------------------------------------------------------------------------
// Hotpixel correction
// ----------------------------------------------------------------------------
#define IS_HOTPIXEL_DISABLE_CORRECTION                  0x0000
#define IS_HOTPIXEL_ENABLE_SENSOR_CORRECTION            0x0001
#define IS_HOTPIXEL_ENABLE_CAMERA_CORRECTION            0x0002
#define IS_HOTPIXEL_ENABLE_SOFTWARE_USER_CORRECTION     0x0004
#define IS_HOTPIXEL_DISABLE_SENSOR_CORRECTION           0x0008

#define IS_HOTPIXEL_GET_CORRECTION_MODE                 0x8000
#define IS_HOTPIXEL_GET_SUPPORTED_CORRECTION_MODES      0x8001

#define IS_HOTPIXEL_GET_SOFTWARE_USER_LIST_EXISTS       0x8100
#define IS_HOTPIXEL_GET_SOFTWARE_USER_LIST_NUMBER       0x8101
#define IS_HOTPIXEL_GET_SOFTWARE_USER_LIST              0x8102
#define IS_HOTPIXEL_SET_SOFTWARE_USER_LIST              0x8103
#define IS_HOTPIXEL_SAVE_SOFTWARE_USER_LIST             0x8104
#define IS_HOTPIXEL_LOAD_SOFTWARE_USER_LIST             0x8105

#define IS_HOTPIXEL_GET_CAMERA_FACTORY_LIST_EXISTS      0x8106
#define IS_HOTPIXEL_GET_CAMERA_FACTORY_LIST_NUMBER      0x8107
#define IS_HOTPIXEL_GET_CAMERA_FACTORY_LIST             0x8108

#define IS_HOTPIXEL_GET_CAMERA_USER_LIST_EXISTS         0x8109
#define IS_HOTPIXEL_GET_CAMERA_USER_LIST_NUMBER         0x810A
#define IS_HOTPIXEL_GET_CAMERA_USER_LIST                0x810B
#define IS_HOTPIXEL_SET_CAMERA_USER_LIST                0x810C
#define IS_HOTPIXEL_GET_CAMERA_USER_LIST_MAX_NUMBER     0x810D
#define IS_HOTPIXEL_DELETE_CAMERA_USER_LIST             0x810E

#define IS_HOTPIXEL_GET_MERGED_CAMERA_LIST_NUMBER       0x810F
#define IS_HOTPIXEL_GET_MERGED_CAMERA_LIST              0x8110

#define IS_HOTPIXEL_SAVE_SOFTWARE_USER_LIST_UNICODE     0x8111
#define IS_HOTPIXEL_LOAD_SOFTWARE_USER_LIST_UNICODE     0x8112

// ----------------------------------------------------------------------------
// color correction definitions
// ----------------------------------------------------------------------------
#define IS_GET_CCOR_MODE                    0x8000
#define IS_GET_SUPPORTED_CCOR_MODE          0x8001
#define IS_GET_DEFAULT_CCOR_MODE            0x8002
#define IS_GET_CCOR_FACTOR                  0x8003
#define IS_GET_CCOR_FACTOR_MIN              0x8004
#define IS_GET_CCOR_FACTOR_MAX              0x8005
#define IS_GET_CCOR_FACTOR_DEFAULT          0x8006

#define IS_CCOR_DISABLE                     0x0000
#define IS_CCOR_ENABLE                      0x0001
#define IS_CCOR_ENABLE_NORMAL               IS_CCOR_ENABLE
#define IS_CCOR_ENABLE_BG40_ENHANCED        0x0002
#define IS_CCOR_ENABLE_HQ_ENHANCED          0x0004
#define IS_CCOR_SET_IR_AUTOMATIC            0x0080
#define IS_CCOR_FACTOR                      0x0100

#define IS_CCOR_ENABLE_MASK             (IS_CCOR_ENABLE_NORMAL | IS_CCOR_ENABLE_BG40_ENHANCED | IS_CCOR_ENABLE_HQ_ENHANCED)


// ----------------------------------------------------------------------------
// bayer algorithm modes
// ----------------------------------------------------------------------------
#define IS_GET_BAYER_CV_MODE                0x8000

#define IS_SET_BAYER_CV_NORMAL              0x0000
#define IS_SET_BAYER_CV_BETTER              0x0001
#define IS_SET_BAYER_CV_BEST                0x0002


// ----------------------------------------------------------------------------
// color converter modes
// ----------------------------------------------------------------------------
#define IS_CONV_MODE_NONE                   0x0000
#define IS_CONV_MODE_SOFTWARE               0x0001
#define IS_CONV_MODE_SOFTWARE_3X3           0x0002
#define IS_CONV_MODE_SOFTWARE_5X5           0x0004
#define IS_CONV_MODE_HARDWARE_3X3           0x0008
#define IS_CONV_MODE_OPENCL_3X3             0x0020
#define IS_CONV_MODE_OPENCL_5X5             0x0040

#define IS_CONV_MODE_JPEG                   0x0100



// ----------------------------------------------------------------------------
// Edge enhancement
// ----------------------------------------------------------------------------
#define IS_GET_EDGE_ENHANCEMENT             0x8000

#define IS_EDGE_EN_DISABLE                  0
#define IS_EDGE_EN_STRONG                   1
#define IS_EDGE_EN_WEAK                     2


// ----------------------------------------------------------------------------
// white balance modes
// ----------------------------------------------------------------------------
#define IS_GET_WB_MODE                      0x8000

#define IS_SET_WB_DISABLE                   0x0000
#define IS_SET_WB_USER                      0x0001
#define IS_SET_WB_AUTO_ENABLE               0x0002
#define IS_SET_WB_AUTO_ENABLE_ONCE          0x0004

#define IS_SET_WB_DAYLIGHT_65               0x0101
#define IS_SET_WB_COOL_WHITE                0x0102
#define IS_SET_WB_U30                       0x0103
#define IS_SET_WB_ILLUMINANT_A              0x0104
#define IS_SET_WB_HORIZON                   0x0105


// ----------------------------------------------------------------------------
// EEPROM defines
// ----------------------------------------------------------------------------
#define IS_EEPROM_MIN_USER_ADDRESS          0
#define IS_EEPROM_MAX_USER_ADDRESS          63
#define IS_EEPROM_MAX_USER_SPACE            64


// ----------------------------------------------------------------------------
// Error report modes
// ----------------------------------------------------------------------------
#define IS_GET_ERR_REP_MODE                 0x8000
#define IS_ENABLE_ERR_REP                   1
#define IS_DISABLE_ERR_REP                  0


// ----------------------------------------------------------------------------
// Display mode selectors
// ----------------------------------------------------------------------------
#define IS_GET_DISPLAY_MODE                 0x8000

#define IS_SET_DM_DIB                       1
#define IS_SET_DM_DIRECT3D                  4
#define IS_SET_DM_OPENGL                    8

#define IS_SET_DM_MONO                      0x800
#define IS_SET_DM_BAYER                     0x1000
#define IS_SET_DM_YCBCR                     0x4000


// ----------------------------------------------------------------------------
// DirectRenderer commands
// ----------------------------------------------------------------------------
#define DR_GET_OVERLAY_DC                       1
#define DR_GET_MAX_OVERLAY_SIZE                 2
#define DR_GET_OVERLAY_KEY_COLOR                3
#define DR_RELEASE_OVERLAY_DC                   4
#define DR_SHOW_OVERLAY                         5
#define DR_HIDE_OVERLAY                         6
#define DR_SET_OVERLAY_SIZE                     7
#define DR_SET_OVERLAY_POSITION                 8
#define DR_SET_OVERLAY_KEY_COLOR                9
#define DR_SET_HWND                             10
#define DR_ENABLE_SCALING                       11
#define DR_DISABLE_SCALING                      12
#define DR_CLEAR_OVERLAY                        13
#define DR_ENABLE_SEMI_TRANSPARENT_OVERLAY      14
#define DR_DISABLE_SEMI_TRANSPARENT_OVERLAY     15
#define DR_CHECK_COMPATIBILITY                  16
#define DR_SET_VSYNC_OFF                        17
#define DR_SET_VSYNC_AUTO                       18
#define DR_SET_USER_SYNC                        19
#define DR_GET_USER_SYNC_POSITION_RANGE         20
#define DR_LOAD_OVERLAY_FROM_FILE               21
#define DR_STEAL_NEXT_FRAME                     22
#define DR_SET_STEAL_FORMAT                     23
#define DR_GET_STEAL_FORMAT                     24
#define DR_ENABLE_IMAGE_SCALING                 25
#define DR_GET_OVERLAY_SIZE                     26
#define DR_CHECK_COLOR_MODE_SUPPORT             27
#define DR_GET_OVERLAY_DATA                     28
#define DR_UPDATE_OVERLAY_DATA                  29
#define DR_GET_SUPPORTED                        30

// ----------------------------------------------------------------------------
// save options
// ----------------------------------------------------------------------------
#define IS_SAVE_USE_ACTUAL_IMAGE_SIZE       0x00010000


// ----------------------------------------------------------------------------
// renumeration modes
// ----------------------------------------------------------------------------
#define IS_RENUM_BY_CAMERA                  0
#define IS_RENUM_BY_HOST                    1


// ----------------------------------------------------------------------------
// event constants
// ----------------------------------------------------------------------------
#define IS_SET_EVENT_ODD                        0
#define IS_SET_EVENT_EVEN                       1
#define IS_SET_EVENT_FRAME                      2
#define IS_SET_EVENT_EXTTRIG                    3
#define IS_SET_EVENT_VSYNC                      4
#define IS_SET_EVENT_SEQ                        5
#define IS_SET_EVENT_STEAL                      6
#define IS_SET_EVENT_VPRES                      7
#define IS_SET_EVENT_CAPTURE_STATUS             8
#define IS_SET_EVENT_TRANSFER_FAILED            IS_SET_EVENT_CAPTURE_STATUS
#define IS_SET_EVENT_DEVICE_RECONNECTED         9
#define IS_SET_EVENT_MEMORY_MODE_FINISH         10
#define IS_SET_EVENT_FRAME_RECEIVED             11
#define IS_SET_EVENT_WB_FINISHED                12
#define IS_SET_EVENT_AUTOBRIGHTNESS_FINISHED    13
#define IS_SET_EVENT_OVERLAY_DATA_LOST          16
#define IS_SET_EVENT_CAMERA_MEMORY              17
#define IS_SET_EVENT_CONNECTIONSPEED_CHANGED    18
#define IS_SET_EVENT_AUTOFOCUS_FINISHED         19
#define IS_SET_EVENT_FIRST_PACKET_RECEIVED      20

#define IS_SET_EVENT_REMOVE                 128
#define IS_SET_EVENT_REMOVAL                129
#define IS_SET_EVENT_NEW_DEVICE             130
#define IS_SET_EVENT_STATUS_CHANGED         131


// ----------------------------------------------------------------------------
// Window message defines
// ----------------------------------------------------------------------------
#define IS_UEYE_MESSAGE                     (WM_USER + 0x0100) 
  #define IS_FRAME                          0x0000
  #define IS_SEQUENCE                       0x0001
  #define IS_TRIGGER                        0x0002
  #define IS_CAPTURE_STATUS                 0x0003
  #define IS_TRANSFER_FAILED                IS_CAPTURE_STATUS
  #define IS_DEVICE_RECONNECTED             0x0004
  #define IS_MEMORY_MODE_FINISH             0x0005
  #define IS_FRAME_RECEIVED                 0x0006
  #define IS_GENERIC_ERROR                  0x0007
  #define IS_STEAL_VIDEO                    0x0008
  #define IS_WB_FINISHED                    0x0009
  #define IS_AUTOBRIGHTNESS_FINISHED        0x000A
  #define IS_OVERLAY_DATA_LOST              0x000B
  #define IS_CAMERA_MEMORY                  0x000C
  #define IS_CONNECTIONSPEED_CHANGED        0x000D
  #define IS_AUTOFOCUS_FINISHED             0x000E
  #define IS_FIRST_PACKET_RECEIVED          0x000F

  #define IS_DEVICE_REMOVED                 0x1000
  #define IS_DEVICE_REMOVAL                 0x1001
  #define IS_NEW_DEVICE                     0x1002
  #define IS_DEVICE_STATUS_CHANGED          0x1003


// ----------------------------------------------------------------------------
// Camera id constants
// ----------------------------------------------------------------------------
#define IS_GET_CAMERA_ID                    0x8000


// ----------------------------------------------------------------------------
// Camera info constants
// ----------------------------------------------------------------------------
#define IS_GET_STATUS                       0x8000

#define IS_EXT_TRIGGER_EVENT_CNT            0
#define IS_FIFO_OVR_CNT                     1
#define IS_SEQUENCE_CNT                     2
#define IS_LAST_FRAME_FIFO_OVR              3
#define IS_SEQUENCE_SIZE                    4
#define IS_VIDEO_PRESENT                    5
#define IS_STEAL_FINISHED                   6
#define IS_STORE_FILE_PATH                  7
#define IS_LUMA_BANDWIDTH_FILTER            8
#define IS_BOARD_REVISION                   9
#define IS_MIRROR_BITMAP_UPDOWN             10
#define IS_BUS_OVR_CNT                      11
#define IS_STEAL_ERROR_CNT                  12
#define IS_LOW_COLOR_REMOVAL                13
#define IS_CHROMA_COMB_FILTER               14
#define IS_CHROMA_AGC                       15
#define IS_WATCHDOG_ON_BOARD                16
#define IS_PASSTHROUGH_ON_BOARD             17
#define IS_EXTERNAL_VREF_MODE               18
#define IS_WAIT_TIMEOUT                     19
#define IS_TRIGGER_MISSED                   20
#define IS_LAST_CAPTURE_ERROR               21
#define IS_PARAMETER_SET_1                  22
#define IS_PARAMETER_SET_2                  23
#define IS_STANDBY                          24
#define IS_STANDBY_SUPPORTED                25
#define IS_QUEUED_IMAGE_EVENT_CNT           26
#define IS_PARAMETER_EXT                    27


// ----------------------------------------------------------------------------
// Interface type defines
// ----------------------------------------------------------------------------
#define IS_INTERFACE_TYPE_USB               0x40
#define IS_INTERFACE_TYPE_USB3              0x60
#define IS_INTERFACE_TYPE_ETH               0x80


// ----------------------------------------------------------------------------
// Board type defines
// ----------------------------------------------------------------------------
#define IS_BOARD_TYPE_UEYE_USB              (IS_INTERFACE_TYPE_USB + 0)     // 0x40
#define IS_BOARD_TYPE_UEYE_USB_SE           IS_BOARD_TYPE_UEYE_USB          // 0x40
#define IS_BOARD_TYPE_UEYE_USB_RE           IS_BOARD_TYPE_UEYE_USB          // 0x40
#define IS_BOARD_TYPE_UEYE_USB_ME           (IS_INTERFACE_TYPE_USB + 0x01)  // 0x41
#define IS_BOARD_TYPE_UEYE_USB_LE           (IS_INTERFACE_TYPE_USB + 0x02)  // 0x42
#define IS_BOARD_TYPE_UEYE_USB_XS           (IS_INTERFACE_TYPE_USB + 0x03)  // 0x43
#define IS_BOARD_TYPE_UEYE_USB_ML           (IS_INTERFACE_TYPE_USB + 0x05)  // 0x45

#define IS_BOARD_TYPE_UEYE_USB3_LE          (IS_INTERFACE_TYPE_USB3 + 0x02) // 0x62
#define IS_BOARD_TYPE_UEYE_USB3_CP          (IS_INTERFACE_TYPE_USB3 + 0x04) // 0x64
#define IS_BOARD_TYPE_UEYE_USB3_ML          (IS_INTERFACE_TYPE_USB3 + 0x05) // 0x65

#define IS_BOARD_TYPE_UEYE_ETH              IS_INTERFACE_TYPE_ETH           // 0x80
#define IS_BOARD_TYPE_UEYE_ETH_HE           IS_BOARD_TYPE_UEYE_ETH          // 0x80
#define IS_BOARD_TYPE_UEYE_ETH_SE           (IS_INTERFACE_TYPE_ETH + 0x01)  // 0x81
#define IS_BOARD_TYPE_UEYE_ETH_RE           IS_BOARD_TYPE_UEYE_ETH_SE       // 0x81
#define IS_BOARD_TYPE_UEYE_ETH_LE           (IS_INTERFACE_TYPE_ETH + 0x02)  // 0x82
#define IS_BOARD_TYPE_UEYE_ETH_CP           (IS_INTERFACE_TYPE_ETH + 0x04)  // 0x84
#define IS_BOARD_TYPE_UEYE_ETH_SEP          (IS_INTERFACE_TYPE_ETH + 0x06)  // 0x86
#define IS_BOARD_TYPE_UEYE_ETH_REP          IS_BOARD_TYPE_UEYE_ETH_SEP      // 0x86
#define IS_BOARD_TYPE_UEYE_ETH_LEET         (IS_INTERFACE_TYPE_ETH + 0x07)  // 0x87
#define IS_BOARD_TYPE_UEYE_ETH_TE           (IS_INTERFACE_TYPE_ETH + 0x08)  // 0x88

// ----------------------------------------------------------------------------
// Camera type defines
// ----------------------------------------------------------------------------
#define IS_CAMERA_TYPE_UEYE_USB         IS_BOARD_TYPE_UEYE_USB_SE
#define IS_CAMERA_TYPE_UEYE_USB_SE      IS_BOARD_TYPE_UEYE_USB_SE
#define IS_CAMERA_TYPE_UEYE_USB_RE      IS_BOARD_TYPE_UEYE_USB_RE
#define IS_CAMERA_TYPE_UEYE_USB_ME      IS_BOARD_TYPE_UEYE_USB_ME
#define IS_CAMERA_TYPE_UEYE_USB_LE      IS_BOARD_TYPE_UEYE_USB_LE
#define IS_CAMERA_TYPE_UEYE_USB_ML      IS_BOARD_TYPE_UEYE_USB_ML

#define IS_CAMERA_TYPE_UEYE_USB3_LE     IS_BOARD_TYPE_UEYE_USB3_LE
#define IS_CAMERA_TYPE_UEYE_USB3_CP     IS_BOARD_TYPE_UEYE_USB3_CP
#define IS_CAMERA_TYPE_UEYE_USB3_ML     IS_BOARD_TYPE_UEYE_USB3_ML

#define IS_CAMERA_TYPE_UEYE_ETH         IS_BOARD_TYPE_UEYE_ETH_HE
#define IS_CAMERA_TYPE_UEYE_ETH_HE      IS_BOARD_TYPE_UEYE_ETH_HE
#define IS_CAMERA_TYPE_UEYE_ETH_SE      IS_BOARD_TYPE_UEYE_ETH_SE
#define IS_CAMERA_TYPE_UEYE_ETH_RE      IS_BOARD_TYPE_UEYE_ETH_RE
#define IS_CAMERA_TYPE_UEYE_ETH_LE      IS_BOARD_TYPE_UEYE_ETH_LE
#define IS_CAMERA_TYPE_UEYE_ETH_CP      IS_BOARD_TYPE_UEYE_ETH_CP
#define IS_CAMERA_TYPE_UEYE_ETH_SEP     IS_BOARD_TYPE_UEYE_ETH_SEP
#define IS_CAMERA_TYPE_UEYE_ETH_REP     IS_BOARD_TYPE_UEYE_ETH_REP
#define IS_CAMERA_TYPE_UEYE_ETH_LEET    IS_BOARD_TYPE_UEYE_ETH_LEET
#define IS_CAMERA_TYPE_UEYE_ETH_TE      IS_BOARD_TYPE_UEYE_ETH_TE


// ----------------------------------------------------------------------------
// Readable operation system defines
// ----------------------------------------------------------------------------
#define IS_OS_UNDETERMINED                  0
#define IS_OS_WIN95                         1
#define IS_OS_WINNT40                       2
#define IS_OS_WIN98                         3
#define IS_OS_WIN2000                       4
#define IS_OS_WINXP                         5
#define IS_OS_WINME                         6
#define IS_OS_WINNET                        7
#define IS_OS_WINSERVER2003                 8
#define IS_OS_WINVISTA                      9
#define IS_OS_LINUX24                       10
#define IS_OS_LINUX26                       11
#define IS_OS_WIN7                          12
#define IS_OS_WIN8                          13
#define IS_OS_WIN8SERVER                    14
#define IS_OS_GREATER_THAN_WIN8             15


// ----------------------------------------------------------------------------
// Bus speed
// ----------------------------------------------------------------------------
#define IS_USB_10                           0x0001 //  1,5 Mb/s
#define IS_USB_11                           0x0002 //   12 Mb/s
#define IS_USB_20                           0x0004 //  480 Mb/s
#define IS_USB_30                           0x0008 // 4000 Mb/s
#define IS_ETHERNET_10                      0x0080 //   10 Mb/s
#define IS_ETHERNET_100                     0x0100 //  100 Mb/s
#define IS_ETHERNET_1000                    0x0200 // 1000 Mb/s
#define IS_ETHERNET_10000                   0x0400 //10000 Mb/s

#define IS_USB_LOW_SPEED                    1
#define IS_USB_FULL_SPEED                   12
#define IS_USB_HIGH_SPEED                   480
#define IS_USB_SUPER_SPEED                  4000
#define IS_ETHERNET_10Base                  10
#define IS_ETHERNET_100Base                 100
#define IS_ETHERNET_1000Base                1000
#define IS_ETHERNET_10GBase                 10000


// ----------------------------------------------------------------------------
// HDR
// ----------------------------------------------------------------------------
#define IS_HDR_NOT_SUPPORTED                0
#define IS_HDR_KNEEPOINTS                   1
#define IS_DISABLE_HDR                      0
#define IS_ENABLE_HDR                       1


// ----------------------------------------------------------------------------
// Test images
// ----------------------------------------------------------------------------
#define IS_TEST_IMAGE_NONE                          0x00000000
#define IS_TEST_IMAGE_WHITE                         0x00000001
#define IS_TEST_IMAGE_BLACK                         0x00000002
#define IS_TEST_IMAGE_HORIZONTAL_GREYSCALE          0x00000004
#define IS_TEST_IMAGE_VERTICAL_GREYSCALE            0x00000008
#define IS_TEST_IMAGE_DIAGONAL_GREYSCALE            0x00000010
#define IS_TEST_IMAGE_WEDGE_GRAY                    0x00000020
#define IS_TEST_IMAGE_WEDGE_COLOR                   0x00000040
#define IS_TEST_IMAGE_ANIMATED_WEDGE_GRAY           0x00000080

#define IS_TEST_IMAGE_ANIMATED_WEDGE_COLOR          0x00000100
#define IS_TEST_IMAGE_MONO_BARS                     0x00000200
#define IS_TEST_IMAGE_COLOR_BARS1                   0x00000400
#define IS_TEST_IMAGE_COLOR_BARS2                   0x00000800
#define IS_TEST_IMAGE_GREYSCALE1                    0x00001000
#define IS_TEST_IMAGE_GREY_AND_COLOR_BARS           0x00002000
#define IS_TEST_IMAGE_MOVING_GREY_AND_COLOR_BARS    0x00004000
#define IS_TEST_IMAGE_ANIMATED_LINE                 0x00008000

#define IS_TEST_IMAGE_ALTERNATE_PATTERN             0x00010000
#define IS_TEST_IMAGE_VARIABLE_GREY                 0x00020000
#define IS_TEST_IMAGE_MONOCHROME_HORIZONTAL_BARS    0x00040000
#define IS_TEST_IMAGE_MONOCHROME_VERTICAL_BARS      0x00080000
#define IS_TEST_IMAGE_CURSOR_H                      0x00100000
#define IS_TEST_IMAGE_CURSOR_V                      0x00200000
#define IS_TEST_IMAGE_COLDPIXEL_GRID                0x00400000
#define IS_TEST_IMAGE_HOTPIXEL_GRID                 0x00800000

#define IS_TEST_IMAGE_VARIABLE_RED_PART             0x01000000
#define IS_TEST_IMAGE_VARIABLE_GREEN_PART           0x02000000
#define IS_TEST_IMAGE_VARIABLE_BLUE_PART            0x04000000
#define IS_TEST_IMAGE_SHADING_IMAGE                 0x08000000
#define IS_TEST_IMAGE_WEDGE_GRAY_SENSOR             0x10000000
#define IS_TEST_IMAGE_ANIMATED_WEDGE_GRAY_SENSOR    0x20000000
#define IS_TEST_IMAGE_RAMPING_PATTERN               0x40000000
#define IS_TEST_IMAGE_CHESS_PATTERN                 0x80000000


// ----------------------------------------------------------------------------
// Sensor scaler
// ----------------------------------------------------------------------------
#define IS_ENABLE_SENSOR_SCALER             1
#define IS_ENABLE_ANTI_ALIASING             2


// ----------------------------------------------------------------------------
// Timeouts
// ----------------------------------------------------------------------------
#define IS_TRIGGER_TIMEOUT                  0


// ----------------------------------------------------------------------------
// Auto pixel clock modes
// ----------------------------------------------------------------------------
#define IS_BEST_PCLK_RUN_ONCE               0


// ----------------------------------------------------------------------------
// Sequence flags
// ----------------------------------------------------------------------------
#define IS_LOCK_LAST_BUFFER                 0x8002
#define IS_GET_ALLOC_ID_OF_THIS_BUF         0x8004
#define IS_GET_ALLOC_ID_OF_LAST_BUF         0x8008
#define IS_USE_ALLOC_ID                     0x8000
#define IS_USE_CURRENT_IMG_SIZE             0xC000

// ------------------------------------------
// Memory information flags
// ------------------------------------------
#define IS_GET_D3D_MEM                  0x8000


// ----------------------------------------------------------------------------
// Image files types
// ----------------------------------------------------------------------------
#define IS_IMG_BMP                          0
#define IS_IMG_JPG                          1
#define IS_IMG_PNG                          2
#define IS_IMG_RAW                          4
#define IS_IMG_TIF                          8


// ----------------------------------------------------------------------------
// I2C defines

// nRegisterAddr | IS_I2C_16_BIT_REGISTER
// nRegisterAddr | IS_I2C_0_BIT_REGISTER
// ----------------------------------------------------------------------------
#define IS_I2C_16_BIT_REGISTER          0x10000000
#define IS_I2C_0_BIT_REGISTER	        0x20000000

// nDeviceAddr | IS_I2C_DONT_WAIT
#define IS_I2C_DONT_WAIT                0x00800000


// ----------------------------------------------------------------------------
// Gamma modes
// ----------------------------------------------------------------------------
#define IS_GET_GAMMA_MODE                   0x8000
#define IS_SET_GAMMA_OFF                    0
#define IS_SET_GAMMA_ON                     1


// ----------------------------------------------------------------------------
// Capture modes   (Falcon)
// ----------------------------------------------------------------------------
#define IS_GET_CAPTURE_MODE                 0x8000

#define IS_SET_CM_ODD                       0x0001
#define IS_SET_CM_EVEN                      0x0002
#define IS_SET_CM_FRAME                     0x0004
#define IS_SET_CM_NONINTERLACED             0x0008
#define IS_SET_CM_NEXT_FRAME                0x0010
#define IS_SET_CM_NEXT_FIELD                0x0020
#define IS_SET_CM_BOTHFIELDS            (IS_SET_CM_ODD | IS_SET_CM_EVEN | IS_SET_CM_NONINTERLACED)
#define IS_SET_CM_FRAME_STEREO              0x2004


// ----------------------------------------------------------------------------
// Typedefs
// ----------------------------------------------------------------------------
#ifdef __LINUX__
        #define FORCEINLINE         inline
        #define USHORT              IS_U16

		#include <unistd.h>
        #include <wchar.h>

		#define Sleep(n)       usleep(n)

		#include <stdint.h>

        // aliases for common Win32 types
        typedef int32_t           BOOLEAN;
        typedef int32_t           BOOL;
        typedef int32_t           INT;
        typedef uint32_t          UINT;
        typedef int32_t           LONG;
        typedef void              VOID;
        typedef void*             LPVOID;
        typedef uint32_t          ULONG;

        typedef uint64_t          UINT64;
        typedef int64_t           __int64;
        typedef int64_t           LONGLONG;
        typedef uint32_t          DWORD;
        typedef uint16_t          WORD;

        typedef unsigned char     BYTE;
        typedef char              CHAR;
        typedef char              TCHAR;
        typedef unsigned char     UCHAR;

        typedef int8_t*           LPTSTR;
        typedef const int8_t*     LPCTSTR;
        typedef const int8_t*     LPCSTR;
        typedef uint32_t          WPARAM;
        typedef uint32_t          LPARAM;
        typedef uint32_t          LRESULT;
        typedef uint32_t          HRESULT;

        typedef void*             HWND;
        typedef void*             HGLOBAL;
        typedef void*             HINSTANCE;
        typedef void*             HDC;
        typedef void*             HMODULE;
        typedef void*             HKEY;
        typedef void*             HANDLE;

        typedef BYTE*             LPBYTE;
        typedef DWORD*            PDWORD;
        typedef VOID*             PVOID;
        typedef CHAR*             PCHAR;

				

    #ifndef FALSE
        #define FALSE 0
    #endif
    #ifndef TRUE
        #define TRUE 1
    #endif

    #ifndef HIBYTE
        #define HIBYTE(_x_)    ( (_x_)>>8 )
    #endif

    #ifndef LOBYTE
        #define LOBYTE(_x_)    ( (_x_)&0x00ff )
    #endif

    #ifndef HIWORD
        #define HIWORD(_x_)    ( (_x_)>>16 )
    #endif

    #ifndef LOWORD
        #define LOWORD(_x_)    ( (_x_)&0x0000ffff )
    #endif

    #ifndef _min_
        #define _min_( _x_, _y_ ) ( (_x_)<(_y_) ? (_x_) : (_y_) )
    #endif

    #ifndef _max_
        #define _max_( _x_, _y_ ) ( (_x_)>(_y_) ? (_x_) : (_y_) )
    #endif

    #ifndef WM_USER
        #define WM_USER        0x400
    #endif

        // unknown stuff for Linux
        #define WINAPI
        #define CALLBACK
        #undef  UNICODE
        #define __stdcall
        #define __cdecl
#if defined __i386__
        #define IDSEXP    __attribute__((cdecl)) INT
        #define IDSEXPUL  __attribute__((cdecl)) ULONG
#else
        #define IDSEXP    INT
        #define IDSEXPUL  ULONG
#endif

        typedef long (*WNDPROC) (HWND, UINT, WPARAM, LPARAM);
#if 0
        typedef union _LARGE_INTEGER
        {
            struct
            {
                DWORD LowPart;
                LONG HighPart;
            };
            struct
            {
                DWORD LowPart;
                LONG HighPart;
            } u;
            LONGLONG QuadPart;
        } LARGE_INTEGER, *PLARGE_INTEGER;

        //useful structs that were in windows.h
        typedef struct tagRECT
        {
            long    left;
            long    top;
            long    right;
            long    bottom;
        } RECT, *PRECT, *LPRECT;

        typedef struct tagRGNDATAHEADER
        {
            DWORD   dwSize;
            DWORD   iType;
            DWORD   nCount;
            DWORD   nRgnSize;
            RECT    rcBound;
        } RGNDATAHEADER, *PRGNDATAHEADER;

        typedef struct tagRGNDATA
        {
            RGNDATAHEADER   rdh;
            char            Buffer[1];
        } RGNDATA, *PRGNDATA, *LPRGNDATA;


        typedef struct tagBITMAPINFOHEADER
        {
                DWORD      biSize;
                long       biWidth;
                long       biHeight;
                WORD       biPlanes;
                WORD       biBitCount;
                DWORD      biCompression;
                DWORD      biSizeImage;
                long       biXPelsPerMeter;
                long       biYPelsPerMeter;
                DWORD      biClrUsed;
                DWORD      biClrImportant;
        } BITMAPINFOHEADER, *PBITMAPINFOHEADER, *LPBITMAPINFOHEADER;

        typedef struct tagRGBQUAD
        {
                BYTE    rgbBlue;
                BYTE    rgbGreen;
                BYTE    rgbRed;
                BYTE    rgbReserved;
        } RGBQUAD;

        typedef struct tagBITMAPINFO
        {
            BITMAPINFOHEADER    bmiHeader;
            RGBQUAD             bmiColors[1];
        } BITMAPINFO, *PBITMAPINFO, *LPBITMAPINFO;
#endif /* 0 */

    #define ZeroMemory(a,b)      memset((a), 0, (b))
    #define OutputDebugString(s) fprintf(stderr, s)


    #define INFINITE    -1
#else

#include <windows.h>

#if defined (_MSC_VER) || defined (__BORLANDC__) || defined (_WIN32_WCE)
  #if defined (_PURE_C) && !defined (_IDS_EXPORT) && !defined (_FALC_EXPORT)
    #define IDSEXP     extern  __declspec(dllimport) INT __cdecl
    #define IDSEXPUL   extern  __declspec(dllimport) ULONG __cdecl
  #elif defined (__STDC__) && !defined (_IDS_EXPORT) && !defined (_FALC_EXPORT)
    #define IDSEXP   extern  __declspec(dllimport) INT __cdecl
    #define IDSEXPUL extern  __declspec(dllimport) ULONG __cdecl
  #elif !defined (_IDS_EXPORT) && !defined (_FALC_EXPORT)   // using the DLL, not creating one
    #define IDSEXP   extern "C" __declspec(dllimport) INT __cdecl
    #define IDSEXPUL extern "C" __declspec(dllimport) ULONG __cdecl
  #elif defined (_IDS_VBSTD) || defined (_FALC_VBSTD)     // for creating stdcall dll
    #define IDSEXP    extern __declspec(dllexport) INT __stdcall
    #define IDSEXPUL  extern __declspec(dllexport) ULONG __stdcall
  #else            // for creating cdecl dll
    #define IDSEXP    extern  __declspec(dllexport) INT __cdecl
    #define IDSEXPUL  extern  __declspec(dllexport) ULONG __cdecl
  #endif
#elif !defined (_IDS_EXPORT) && !defined (_FALC_EXPORT)  // using the DLL, not creating one
  #define IDSEXP    extern  __declspec(dllimport) INT __cdecl
  #define IDSEXPUL  extern  __declspec(dllimport) ULONG __cdecl
#endif

typedef int     INT;

#endif    // Linux

#ifdef _WIN32_WCE
  typedef TCHAR IS_CHAR;
#else
  typedef char IS_CHAR;
#endif


// ----------------------------------------------------------------------------
// Typedefs
// ----------------------------------------------------------------------------
typedef DWORD   HIDS;
#define HIDS_DEFINED

typedef DWORD   HCAM;
#define HCAM_DEFINED

typedef DWORD   HFALC;
#define HFALC_DEFINED


// ----------------------------------------------------------------------------
// Invalid values for device handles
// ----------------------------------------------------------------------------
#define IS_INVALID_HIDS (HIDS)0
#define IS_INVALID_HCAM (HIDS)0
#define IS_INVALID_HFALC (HIDS)0


// ----------------------------------------------------------------------------
// Info struct
// ----------------------------------------------------------------------------
#define FALCINFO   BOARDINFO
#define PFALCINFO  PBOARDINFO
#define CAMINFO    BOARDINFO
#define PCAMINFO   PBOARDINFO

typedef struct
{
  char          SerNo[12];          // e.g. "1234512345"  (11 char)
  char          ID[20];             // e.g. "IDS GmbH"
  char          Version[10];        // e.g. "V2.10"  (9 char)
  char          Date[12];           // e.g. "24.01.2006" (11 char)
  unsigned char Select;             // contains board select number for multi board support
  unsigned char Type;               // e.g. IS_BOARD_TYPE_UEYE_USB
  char          Reserved[8];        // (7 char)
} BOARDINFO, *PBOARDINFO;


typedef struct _SENSORINFO
{
  WORD          SensorID;               // e.g. IS_SENSOR_UI224X_C
  IS_CHAR       strSensorName[32];      // e.g. "UI-224X-C"
  char          nColorMode;             // e.g. IS_COLORMODE_BAYER
  DWORD         nMaxWidth;              // e.g. 1280
  DWORD         nMaxHeight;             // e.g. 1024
  BOOL          bMasterGain;            // e.g. TRUE
  BOOL          bRGain;                 // e.g. TRUE
  BOOL          bGGain;                 // e.g. TRUE
  BOOL          bBGain;                 // e.g. TRUE
  BOOL          bGlobShutter;           // e.g. TRUE
  WORD          wPixelSize;             // e.g. 465 = 4.65 um
  char          nUpperLeftBayerPixel;   // e.g. BAYER_PIXEL_RED (value = 0)  
  char          Reserved[13];           // not used
} SENSORINFO, *PSENSORINFO; 


typedef enum  _BAYER_PIXEL
{
    BAYER_PIXEL_RED   = 0,
    BAYER_PIXEL_GREEN = 1,
    BAYER_PIXEL_BLUE  = 2

} BAYER_PIXEL; 


typedef struct _REVISIONINFO
{
    WORD  size;                     // 2
    WORD  Sensor;                   // 2
    WORD  Cypress;                  // 2
    DWORD Blackfin;                 // 4
    WORD  DspFirmware;              // 2
                                    // --12
    WORD  USB_Board;                // 2
    WORD  Sensor_Board;             // 2
    WORD  Processing_Board;         // 2
    WORD  Memory_Board;             // 2
    WORD  Housing;                  // 2
    WORD  Filter;                   // 2
    WORD  Timing_Board;             // 2
    WORD  Product;                  // 2
    WORD  Power_Board;              // 2
    WORD  Logic_Board;              // 2
    WORD  FX3;                      // 2
    WORD  FPGA;                     // 2
                                    // --36
    BYTE reserved[92];              // --128
} REVISIONINFO, *PREVISIONINFO;


// ----------------------------------------------------------------------------
// Capture status
// ----------------------------------------------------------------------------
typedef enum _UEYE_CAPTURE_STATUS
{
    IS_CAP_STATUS_API_NO_DEST_MEM       =   0xa2,
    IS_CAP_STATUS_API_CONVERSION_FAILED =   0xa3,
    IS_CAP_STATUS_API_IMAGE_LOCKED      =   0xa5,
    
    IS_CAP_STATUS_DRV_OUT_OF_BUFFERS    =   0xb2,
    IS_CAP_STATUS_DRV_DEVICE_NOT_READY  =   0xb4,

    IS_CAP_STATUS_USB_TRANSFER_FAILED   =   0xc7,

    IS_CAP_STATUS_DEV_TIMEOUT           =   0xd6,

    IS_CAP_STATUS_ETH_BUFFER_OVERRUN    =   0xe4,
    IS_CAP_STATUS_ETH_MISSED_IMAGES     =   0xe5

} UEYE_CAPTURE_STATUS;


typedef struct _UEYE_CAPTURE_STATUS_INFO
{
    DWORD dwCapStatusCnt_Total;
    BYTE  reserved[60];
    DWORD adwCapStatusCnt_Detail[256]; // access via UEYE_CAPTURE_STATUS

} UEYE_CAPTURE_STATUS_INFO;


typedef enum E_CAPTURE_STATUS_CMD
{
    IS_CAPTURE_STATUS_INFO_CMD_RESET = 1,
    IS_CAPTURE_STATUS_INFO_CMD_GET   = 2

} CAPTURE_STATUS_CMD;


IDSEXP is_CaptureStatus(HIDS hCam, UINT nCommand, void *pParam, UINT nSizeOfParam);


#ifndef UEYE_CAMERA_INFO_STRUCT
#define UEYE_CAMERA_INFO_STRUCT
typedef struct _UEYE_CAMERA_INFO
{
  DWORD     dwCameraID;             /*<! \brief this is the user definable camera ID */
  DWORD     dwDeviceID;             /*<! \brief this is the systems enumeration ID */
  DWORD     dwSensorID;             /*<! \brief this is the sensor ID e.g. IS_SENSOR_UI141X_M */
  DWORD     dwInUse;                /*<! \brief flag, indicates whether the camera is in use or not */
  IS_CHAR   SerNo[16];              /*<! \brief zero terminated serial number string */
  IS_CHAR   Model[16];              /*<! \brief zero terminated short model name string */
  DWORD     dwStatus;               /*<! \brief various flags with camera status */
  DWORD     dwReserved[2];          /*<! \brief reserved */
  IS_CHAR   FullModelName[32];      /*<! \brief zero terminated full model name string,
                                         \note Use this string for display purpose only!
                                               Do not depend on the contents of this string! */
  DWORD     dwReserved2[5];         /*<! \brief reserved */
}UEYE_CAMERA_INFO, *PUEYE_CAMERA_INFO;
#endif //UEYE_CAMERA_INFO_STRUCT

// usage of the list:
// 1. call the DLL with .dwCount = 0
// 2. DLL returns .dwCount = N  (N = number of available cameras)
// 3. call DLL with .dwCount = N and a pointer to UEYE_CAMERA_LIST with
//    and array of UEYE_CAMERA_INFO[N]
// 4. DLL will fill in the array with the camera infos and
//    will update the .dwCount member with the actual number of cameras
//    because there may be a change in number of cameras between step 2 and 3
// 5. check if there's a difference in actual .dwCount and formerly
//    reported value of N and call DLL again with an updated array size
#ifndef UEYE_CAMERA_LIST_STRUCT
#define UEYE_CAMERA_LIST_STRUCT
typedef struct _UEYE_CAMERA_LIST
{
  ULONG     dwCount;
  UEYE_CAMERA_INFO uci[1];
}UEYE_CAMERA_LIST, *PUEYE_CAMERA_LIST;
#endif //UEYE_CAMERA_LIST_STRUCT

// ----------------------------------------------------------------------------
// the  following defines are the status bits of the dwStatus member of
// the UEYE_CAMERA_INFO structure
#define FIRMWARE_DOWNLOAD_NOT_SUPPORTED                 0x00000001
#define INTERFACE_SPEED_NOT_SUPPORTED                   0x00000002
#define INVALID_SENSOR_DETECTED                         0x00000004
#define AUTHORIZATION_FAILED                            0x00000008
#define DEVSTS_INCLUDED_STARTER_FIRMWARE_INCOMPATIBLE   0x00000010

// the following macro determines the availability of the camera based
// on the status flags
#define IS_CAMERA_AVAILABLE(_s_)     ( (((_s_) & FIRMWARE_DOWNLOAD_NOT_SUPPORTED) == 0) && \
                                       (((_s_) & INTERFACE_SPEED_NOT_SUPPORTED)   == 0) && \
                                       (((_s_) & INVALID_SENSOR_DETECTED)         == 0) && \
                                       (((_s_) & AUTHORIZATION_FAILED)            == 0) )


// ----------------------------------------------------------------------------
// Auto feature structs and definitions
// ----------------------------------------------------------------------------
#define AC_SHUTTER                          0x00000001
#define AC_GAIN                             0x00000002
#define AC_WHITEBAL                         0x00000004
#define AC_WB_RED_CHANNEL                   0x00000008
#define AC_WB_GREEN_CHANNEL                 0x00000010
#define AC_WB_BLUE_CHANNEL                  0x00000020
#define AC_FRAMERATE                        0x00000040
#define AC_SENSOR_SHUTTER                   0x00000080
#define AC_SENSOR_GAIN                      0x00000100
#define AC_SENSOR_GAIN_SHUTTER              0x00000200
#define AC_SENSOR_FRAMERATE                 0x00000400
#define AC_SENSOR_WB                        0x00000800
#define AC_SENSOR_AUTO_REFERENCE            0x00001000
#define AC_SENSOR_AUTO_SPEED                0x00002000
#define AC_SENSOR_AUTO_HYSTERESIS           0x00004000
#define AC_SENSOR_AUTO_SKIPFRAMES           0x00008000
#define AC_SENSOR_AUTO_CONTRAST_CORRECTION  0x00010000
#define AC_SENSOR_AUTO_CONTRAST_FDT_AOI     0x00020000
#define AC_SENSOR_AUTO_BACKLIGHT_COMP       0x00040000

#define ACS_ADJUSTING 0x00000001
#define ACS_FINISHED  0x00000002
#define ACS_DISABLED  0x00000004


typedef struct _AUTO_BRIGHT_STATUS
{
    DWORD curValue;             // current average greylevel
    long curError;              // current auto brightness error
    DWORD curController;        // current active brightness controller -> AC_x
    DWORD curCtrlStatus;        // current control status -> ACS_x
} AUTO_BRIGHT_STATUS, *PAUTO_BRIGHT_STATUS;



typedef struct _AUTO_WB_CHANNNEL_STATUS
{
    DWORD curValue;             // current average greylevel
    long curError;              // current auto wb error
    DWORD curCtrlStatus;        // current control status -> ACS_x
} AUTO_WB_CHANNNEL_STATUS, *PAUTO_WB_CHANNNEL_STATUS;

typedef struct _AUTO_WB_STATUS
{
    AUTO_WB_CHANNNEL_STATUS RedChannel;
    AUTO_WB_CHANNNEL_STATUS GreenChannel;
    AUTO_WB_CHANNNEL_STATUS BlueChannel;
    DWORD curController;        // current active wb controller -> AC_x
} AUTO_WB_STATUS, *PAUTO_WB_STATUS;

// auto shutter photometry capabilities
typedef enum E_AUTO_SHUTTER_PHOTOM
{
    AS_PM_NONE                  = 0,
    AS_PM_SENS_CENTER_WEIGHTED  = 0x00000001, // sensor auto shutter: center weighted
    AS_PM_SENS_CENTER_SPOT      = 0x00000002, // sensor auto shutter: center spot
    AS_PM_SENS_PORTRAIT         = 0x00000004, // sensor auto shutter: portrait
    AS_PM_SENS_LANDSCAPE        = 0x00000008  // sensor auto shutter: landscape

}AUTO_SHUTTER_PHOTOM;

// auto gain photometry capabilities
typedef enum E_AUTO_GAIN_PHOTOM
{
    AG_PM_NONE                  = 0,
    AG_PM_SENS_CENTER_WEIGHTED  = 0x00000001, // sensor auto gain: center weighted
    AG_PM_SENS_CENTER_SPOT      = 0x00000002, // sensor auto gain: center spot
    AG_PM_SENS_PORTRAIT         = 0x00000004, // sensor auto gain: portrait
    AG_PM_SENS_LANDSCAPE        = 0x00000008  // sensor auto gain: landscape
}AUTO_GAIN_PHOTOM;

// anti flicker modes
typedef enum E_ANTI_FLICKER_MODE
{
    ANTIFLCK_MODE_OFF             = 0,
    ANTIFLCK_MODE_SENS_AUTO       = 0x00000001,
    ANTIFLCK_MODE_SENS_50_FIXED   = 0x00000002,
    ANTIFLCK_MODE_SENS_60_FIXED   = 0x00000004
}ANTI_FLICKER_MODE;

// whitebalance  modes
typedef enum E_WHITEBALANCE_MODE
{
    WB_MODE_DISABLE                 = 0,
    WB_MODE_AUTO                    = 0x00000001,
    WB_MODE_ALL_PULLIN              = 0x00000002,
    WB_MODE_INCANDESCENT_LAMP       = 0x00000004,
    WB_MODE_FLUORESCENT_DL          = 0x00000008,
    WB_MODE_OUTDOOR_CLEAR_SKY       = 0x00000010,
    WB_MODE_OUTDOOR_CLOUDY          = 0x00000020
}WHITEBALANCE_MODE;

typedef struct _UEYE_AUTO_INFO
{
    DWORD               AutoAbility;                // auto control ability
    AUTO_BRIGHT_STATUS  sBrightCtrlStatus;          // brightness auto control status
    AUTO_WB_STATUS      sWBCtrlStatus;              // white balance auto control status
    DWORD               AShutterPhotomCaps;         // auto shutter photometry capabilities(AUTO_SHUTTER_PHOTOM)
    DWORD               AGainPhotomCaps;            // auto gain photometry capabilities (AUTO_GAIN_PHOTOM)
    DWORD               AAntiFlickerCaps;           // auto anti-flicker capabilities
    DWORD               SensorWBModeCaps;           // white balance mode capabilities
    DWORD               reserved[8];
} UEYE_AUTO_INFO, *PUEYE_AUTO_INFO;

typedef struct _DC_INFO
{
    unsigned int        nSize;      // size of this structure (16  bytes)
    HDC                 hDC;        // handle to a DC
    unsigned int        nCx;        // destination width to use for rendering
    unsigned int        nCy;        // destination height to use for rendering
}DC_INFO, *PDC_INFO;


// ----------------------------------------------------------------------------
// function exports
// ----------------------------------------------------------------------------
#ifdef __LINUX__
    IDSEXP is_WaitEvent             (HIDS hCam, INT which, INT nTimeout);
#endif


  IDSEXP   is_SetSaturation          (HIDS hCam, INT ChromU, INT ChromV);
  IDSEXP   is_PrepareStealVideo      (HIDS hCam, int Mode, ULONG StealColorMode);
  IDSEXP   is_GetNumberOfDevices     (void);
  

// ----------------------------------------------------------------------------
// common function
// ----------------------------------------------------------------------------
  IDSEXP   is_StopLiveVideo          (HIDS hCam, INT Wait);
  IDSEXP   is_FreezeVideo            (HIDS hCam, INT Wait);
  IDSEXP   is_CaptureVideo           (HIDS hCam, INT Wait);
  IDSEXP   is_IsVideoFinish          (HIDS hCam, INT* pValue);
  IDSEXP   is_HasVideoStarted        (HIDS hCam, BOOL* pbo);
  
  IDSEXP   is_AllocImageMem          (HIDS hCam, INT width, INT height, INT bitspixel, char** ppcImgMem, int* pid);
  IDSEXP   is_SetImageMem            (HIDS hCam, char* pcMem, int id);
  IDSEXP   is_FreeImageMem           (HIDS hCam, char* pcMem, int id);
  IDSEXP   is_GetImageMem            (HIDS hCam, VOID** pMem);
  IDSEXP   is_GetActiveImageMem      (HIDS hCam, char** ppcMem, int* pnID);
  IDSEXP   is_InquireImageMem        (HIDS hCam, char* pcMem, int nID, int* pnX, int* pnY, int* pnBits, int* pnPitch);
  IDSEXP   is_GetImageMemPitch       (HIDS hCam, INT* pPitch);

  IDSEXP   is_SetAllocatedImageMem   (HIDS hCam, INT width, INT height, INT bitspixel, char* pcImgMem, int* pid);
  IDSEXP   is_CopyImageMem           (HIDS hCam, char* pcSource, int nID, char* pcDest);
  IDSEXP   is_CopyImageMemLines      (HIDS hCam, char* pcSource, int nID, int nLines, char* pcDest);

  IDSEXP   is_AddToSequence          (HIDS hCam, char* pcMem, INT nID);
  IDSEXP   is_ClearSequence          (HIDS hCam);
  IDSEXP   is_GetActSeqBuf           (HIDS hCam, INT* pnNum, char** ppcMem, char** ppcMemLast);
  IDSEXP   is_LockSeqBuf             (HIDS hCam, INT nNum, char* pcMem);
  IDSEXP   is_UnlockSeqBuf           (HIDS hCam, INT nNum, char* pcMem);

  IDSEXP   is_GetError               (HIDS hCam, INT* pErr, IS_CHAR** ppcErr);
  IDSEXP   is_SetErrorReport         (HIDS hCam, INT Mode);

  IDSEXP   is_ReadEEPROM             (HIDS hCam, INT Adr, char* pcString, INT Count);
  IDSEXP   is_WriteEEPROM            (HIDS hCam, INT Adr, char* pcString, INT Count);
  
  IDSEXP   is_SetColorMode           (HIDS hCam, INT Mode);
  IDSEXP   is_GetColorDepth          (HIDS hCam, INT* pnCol, INT* pnColMode);
  
  // Bitmap display function
  IDSEXP   is_RenderBitmap           (HIDS hCam, INT nMemID, HWND hwnd, INT nMode);

  IDSEXP   is_SetDisplayMode         (HIDS hCam, INT Mode);
  IDSEXP   is_SetDisplayPos          (HIDS hCam, INT x, INT y);
  
  IDSEXP   is_SetHwnd                (HIDS hCam, HWND hwnd);

  IDSEXP   is_GetVsyncCount          (HIDS hCam, long* pIntr, long* pActIntr);
  
  // Version information
  IDSEXP   is_GetDLLVersion          (void);

  IDSEXP   is_InitEvent              (HIDS hCam, HANDLE hEv, INT which);
  IDSEXP   is_ExitEvent              (HIDS hCam, INT which);
  IDSEXP   is_EnableEvent            (HIDS hCam, INT which);
  IDSEXP   is_DisableEvent           (HIDS hCam, INT which);

  IDSEXP   is_SetExternalTrigger     (HIDS hCam, INT nTriggerMode);
  IDSEXP   is_SetTriggerCounter      (HIDS hCam, INT nValue);
  IDSEXP   is_SetRopEffect           (HIDS hCam, INT effect, INT param, INT reserved);


  // Camera functions
  IDSEXP is_InitCamera                  (HIDS* phCam, HWND hWnd);
  IDSEXP is_ExitCamera                  (HIDS hCam);
  IDSEXP is_GetCameraInfo               (HIDS hCam, PCAMINFO pInfo);
  IDSEXPUL is_CameraStatus              (HIDS hCam, INT nInfo, ULONG ulValue);
  IDSEXP is_GetCameraType               (HIDS hCam);
  IDSEXP is_GetNumberOfCameras          (INT* pnNumCams);

  IDSEXP is_GetUsedBandwidth            (HIDS hCam);
  
  // Set/Get Frame rate
  IDSEXP is_GetFrameTimeRange           (HIDS hCam, double *min, double *max, double *intervall);
  IDSEXP is_SetFrameRate                (HIDS hCam, double FPS, double* newFPS);
    
  // Get frames per second
  IDSEXP is_GetFramesPerSecond          (HIDS hCam, double *dblFPS);

  // Get Sensor info
  IDSEXP is_GetSensorInfo               (HIDS hCam, PSENSORINFO pInfo);

  // Get RevisionInfo
  IDSEXP is_GetRevisionInfo             (HIDS hCam, PREVISIONINFO prevInfo);

  // enable/disable AutoExit after device remove
  IDSEXP is_EnableAutoExit              (HIDS hCam, INT nMode);

  // Message
  IDSEXP is_EnableMessage               (HIDS hCam, INT which, HWND hWnd);

  // hardware gain settings
  IDSEXP is_SetHardwareGain             (HIDS hCam, INT nMaster, INT nRed, INT nGreen, INT nBlue);

  // enable/disable WhiteBalance
  IDSEXP is_SetWhiteBalance             (HIDS hCam, INT nMode);
  IDSEXP is_SetWhiteBalanceMultipliers  (HIDS hCam, double dblRed, double dblGreen, double dblBlue);
  IDSEXP is_GetWhiteBalanceMultipliers  (HIDS hCam, double *pdblRed, double *pdblGreen, double *pdblBlue);

  // Sensor features
  IDSEXP is_SetColorCorrection          (HIDS hCam, INT nEnable, double *factors);
  
  IDSEXP is_SetSubSampling              (HIDS hCam, INT mode);
  IDSEXP is_ForceTrigger                (HIDS hCam);

  // new with driver version 1.12.0006
  IDSEXP is_GetBusSpeed                 (HIDS hCam);

  // new with driver version 1.12.0015
  IDSEXP is_SetBinning                  (HIDS hCam, INT mode);

  // new with driver version 1.12.0017
  IDSEXP is_ResetToDefault              (HIDS hCam);
  
  // new with driver version 1.14.0008
  IDSEXP is_SetCameraID                 (HIDS hCam, INT nID);
  IDSEXP is_SetBayerConversion          (HIDS hCam, INT nMode);
    
  // new with driver version 1.14.0009
  IDSEXP is_SetHardwareGamma            (HIDS hCam, INT nMode);

  // new with driver version 2.00.0001
  IDSEXP is_GetCameraList               (PUEYE_CAMERA_LIST pucl);

  // new with driver version 2.00.0011
  IDSEXP is_SetAutoParameter            (HIDS hCam, INT param, double *pval1, double *pval2);
  IDSEXP is_GetAutoInfo                 (HIDS hCam, UEYE_AUTO_INFO *pInfo);
    
  IDSEXP is_GetImageHistogram           (HIDS hCam, int nID, INT ColorMode, DWORD* pHistoMem);
  IDSEXP is_SetTriggerDelay             (HIDS hCam, INT nTriggerDelay);

  // new with driver version 2.21.0000
  IDSEXP is_SetGainBoost                (HIDS hCam, INT mode);
  
  IDSEXP is_SetGlobalShutter            (HIDS hCam, INT mode);
  IDSEXP is_SetExtendedRegister         (HIDS hCam, INT index,WORD value);
  IDSEXP is_GetExtendedRegister         (HIDS hCam, INT index, WORD *pwValue);

  // new with driver version 2.22.0002
  IDSEXP is_SetHWGainFactor             (HIDS hCam, INT nMode, INT nFactor);

  // camera renumeration
  IDSEXP is_Renumerate                  (HIDS hCam, INT nMode);

  // Read / Write I2C
  IDSEXP is_WriteI2C                    (HIDS hCam, INT nDeviceAddr, INT nRegisterAddr, BYTE* pbData, INT nLen);
  IDSEXP is_ReadI2C                     (HIDS hCam, INT nDeviceAddr, INT nRegisterAddr, BYTE* pbData, INT nLen);


  // new with driver version 3.10.0000
  typedef struct _KNEEPOINT
  {
      double x;
      double y;
  } KNEEPOINT, *PKNEEPOINT;


  typedef struct _KNEEPOINTARRAY
  {
      INT NumberOfUsedKneepoints;
      KNEEPOINT Kneepoint[10];
  } KNEEPOINTARRAY, *PKNEEPOINTARRAY;


  typedef struct _KNEEPOINTINFO
  {
      INT NumberOfSupportedKneepoints;
      INT NumberOfUsedKneepoints;
      double MinValueX;
      double MaxValueX;
      double MinValueY;
      double MaxValueY;
      KNEEPOINT DefaultKneepoint[10];
      INT Reserved[10];
  } KNEEPOINTINFO, *PKNEEPOINTINFO;


  // HDR functions
  IDSEXP is_GetHdrMode                  (HIDS hCam, INT *Mode);
  IDSEXP is_EnableHdr                   (HIDS hCam, INT Enable);
  IDSEXP is_SetHdrKneepoints            (HIDS hCam, KNEEPOINTARRAY *KneepointArray, INT KneepointArraySize);
  IDSEXP is_GetHdrKneepoints            (HIDS hCam, KNEEPOINTARRAY *KneepointArray, INT KneepointArraySize);
  IDSEXP is_GetHdrKneepointInfo         (HIDS hCam, KNEEPOINTINFO *KneepointInfo, INT KneepointInfoSize);

  IDSEXP is_SetOptimalCameraTiming      (HIDS hCam, INT Mode, INT Timeout, INT *pMaxPxlClk, double *pMaxFrameRate);

  IDSEXP is_GetSupportedTestImages      (HIDS hCam, INT *SupportedTestImages);
  IDSEXP is_GetTestImageValueRange      (HIDS hCam, INT TestImage, INT *TestImageValueMin, INT *TestImageValueMax);
  IDSEXP is_SetSensorTestImage          (HIDS hCam, INT Param1, INT Param2);

  IDSEXP is_GetColorConverter           (HIDS hCam, INT ColorMode, INT *pCurrentConvertMode, INT *pDefaultConvertMode, INT *pSupportedConvertModes);
  IDSEXP is_SetColorConverter           (HIDS hCam, INT ColorMode, INT ConvertMode);

  IDSEXP is_WaitForNextImage            (HIDS hCam, UINT timeout, char **ppcMem, INT *imageID);
  IDSEXP is_InitImageQueue              (HIDS hCam, INT nMode);
  IDSEXP is_ExitImageQueue              (HIDS hCam);

  IDSEXP is_SetTimeout                  (HIDS hCam, UINT nMode, UINT Timeout);
  IDSEXP is_GetTimeout                  (HIDS hCam, UINT nMode, UINT *pTimeout);


  typedef enum  eUEYE_GET_ESTIMATED_TIME_MODE
  {
      IS_SE_STARTER_FW_UPLOAD =   0x00000001, /*!< get estimated duration of GigE SE starter firmware upload in milliseconds */
      IS_CP_STARTER_FW_UPLOAD =   0x00000002, /*!< get estimated duration of GigE CP starter firmware upload in milliseconds */
      IS_STARTER_FW_UPLOAD    =   0x00000004  /*!< get estimated duration of starter firmware upload in milliseconds using hCam to */ 
  } UEYE_GET_ESTIMATED_TIME_MODE;    
  
  
  IDSEXP is_GetDuration                 (HIDS hCam, UINT nMode, INT* pnTime);


  // new with driver version 3.40.0000
  typedef struct _SENSORSCALERINFO
  {
      INT       nCurrMode;
      INT       nNumberOfSteps;
      double    dblFactorIncrement;
      double    dblMinFactor;
      double    dblMaxFactor;
      double    dblCurrFactor;
      INT       nSupportedModes;
      BYTE      bReserved[84];
  } SENSORSCALERINFO;


  IDSEXP is_GetSensorScalerInfo (HIDS hCam, SENSORSCALERINFO *pSensorScalerInfo, INT nSensorScalerInfoSize);
  IDSEXP is_SetSensorScaler      (HIDS hCam, UINT nMode, double dblFactor); 

  typedef struct _UEYETIME
  {
      WORD      wYear;
      WORD      wMonth;
      WORD      wDay;
      WORD      wHour;
      WORD      wMinute;
      WORD      wSecond;
      WORD      wMilliseconds;
      BYTE      byReserved[10];
  } UEYETIME;


  typedef struct _UEYEIMAGEINFO
  {
      DWORD                 dwFlags;
      BYTE                  byReserved1[4];
      UINT64                u64TimestampDevice;
      UEYETIME              TimestampSystem;
      DWORD                 dwIoStatus;
      WORD                  wAOIIndex;
      WORD                  wAOICycle;
      UINT64                u64FrameNumber;
      DWORD                 dwImageBuffers;
      DWORD                 dwImageBuffersInUse;
      DWORD                 dwReserved3;
      DWORD                 dwImageHeight;
      DWORD                 dwImageWidth;
      DWORD                 dwHostProcessTime; /* Time spend processing this image in micro seconds */
  } UEYEIMAGEINFO;


    IDSEXP is_GetImageInfo (HIDS hCam, INT nImageBufferID, UEYEIMAGEINFO *pImageInfo, INT nImageInfoSize);


    // New functions and defines for 3.52 (uEye XS)
    IDSEXP is_ImageFormat (HIDS hCam, UINT nCommand, void *pParam, UINT nSizeOfParam);


    typedef enum E_IMAGE_FORMAT_CMD
    {
       IMGFRMT_CMD_GET_NUM_ENTRIES              = 1,  /* Get the number of supported image formats.
                                                         pParam hast to be a Pointer to IS_U32. If  -1 is reported, the device
                                                         supports continuous AOI settings (maybe with fixed increments)         */
       IMGFRMT_CMD_GET_LIST                     = 2,  /* Get a array of IMAGE_FORMAT_ELEMENTs.                                  */
       IMGFRMT_CMD_SET_FORMAT                   = 3,  /* Select a image format                                                  */
       IMGFRMT_CMD_GET_ARBITRARY_AOI_SUPPORTED  = 4,  /* Does the device supports the setting of an arbitrary AOI.              */
       IMGFRMT_CMD_GET_FORMAT_INFO              = 5   /* Get IMAGE_FORMAT_INFO for a given formatID                             */

    } IMAGE_FORMAT_CMD;


    typedef enum E_CAPTUREMODE
    {

        // no trigger
        CAPTMODE_FREERUN                    = 0x00000001,
        CAPTMODE_SINGLE                     = 0x00000002,

        // software trigger modes
        CAPTMODE_TRIGGER_SOFT_SINGLE        = 0x00000010,
        CAPTMODE_TRIGGER_SOFT_CONTINUOUS    = 0x00000020,

        // hardware trigger modes
        CAPTMODE_TRIGGER_HW_SINGLE          = 0x00000100,
        CAPTMODE_TRIGGER_HW_CONTINUOUS      = 0x00000200

    } CAPTUREMODE;


    typedef struct S_IMAGE_FORMAT_INFO
    {
        INT  nFormatID;                     /* Image format ID.                                                         */
        UINT nWidth;                        /* Width.                                                                   */
        UINT nHeight;                       /* Height.                                                                  */
        INT nX0;                            /* Start position x.                                                        */
        INT nY0;                            /* Start position Y.                                                        */
        UINT nSupportedCaptureModes;        /* Flag field with supported capture modes for this resolution (Bitmask of
                                               of CAPTUREMODE enumeration).                                             */
        UINT nBinningMode;                  /* Binning mode.                                                            */
        UINT nSubsamplingMode;              /* Subsampling mode.                                                        */
        IS_CHAR strFormatName[64];          /* Format name.                                                             */
        double dSensorScalerFactor;         /* Sensor scaler factor.                                                    */
        UINT nReserved[22];
    } IMAGE_FORMAT_INFO;


    typedef struct S_IMAGE_FORMAT_LIST
    {
        UINT nSizeOfListEntry;              /* Size of one list entry in byte. */
        UINT nNumListElements;              /* Number of list entries.         */
        UINT nReserved[4];                  /* Reserved for future use.        */
        IMAGE_FORMAT_INFO FormatInfo[1];    /* First format entry.             */
    } IMAGE_FORMAT_LIST;


IDSEXP is_FaceDetection (HIDS hCam, UINT nCommand, void *pParam, UINT nSizeOfParam);


    typedef enum E_FDT_CAPABILITY_FLAGS
    {
        FDT_CAP_INVALID             = 0,
        FDT_CAP_SUPPORTED           = 0x00000001, /* Face detection supported.                                      */
        FDT_CAP_SEARCH_ANGLE        = 0x00000002, /* Search angle.                                                  */
        FDT_CAP_SEARCH_AOI          = 0x00000004, /* Search AOI.                                                    */
        FDT_CAP_INFO_POSX           = 0x00000010, /* Query horizontal position (center) of detected face.           */
        FDT_CAP_INFO_POSY           = 0x00000020, /* Query vertical position(center) of detected face.              */
        FDT_CAP_INFO_WIDTH          = 0x00000040, /* Query width of detected face.                                  */
        FDT_CAP_INFO_HEIGHT         = 0x00000080, /* Query height of detected face.                                 */
        FDT_CAP_INFO_ANGLE          = 0x00000100, /* Query angle of detected face.                                  */
        FDT_CAP_INFO_POSTURE        = 0x00000200, /* Query posture of detected face.                                */
        FDT_CAP_INFO_FACENUMBER     = 0x00000400, /* Query number of detected faces.                                */
        FDT_CAP_INFO_OVL            = 0x00000800, /* Overlay: Mark the detected face in the image.                  */
        FDT_CAP_INFO_NUM_OVL        = 0x00001000, /* Overlay: Limit the maximum number of overlays in one image.    */
        FDT_CAP_INFO_OVL_LINEWIDTH  = 0x00002000  /* Overlay line width.                                            */
    } FDT_CAPABILITY_FLAGS;


    typedef struct S_FDT_INFO_EL
    {
        INT nFacePosX;              /* Start X position.                                                                */
        INT nFacePosY;              /* Start Y position.                                                                */
        INT nFaceWidth;             /* Face width.                                                                      */
        INT nFaceHeight;            /* Face height.                                                                     */
        INT nAngle;                 /* Face Angle (0...360° clockwise, 0° at twelve o'clock position. -1: undefined ).  */
        UINT nPosture;              /* Face posture.                                                                    */
        UEYETIME TimestampSystem;   /* System time stamp (device query time) .                                          */
        UINT64 nReserved;           /* Reserved for future use.                                                         */
        UINT nReserved2[4];         /* Reserved for future use.                                                         */

    } FDT_INFO_EL;


    typedef struct S_FDT_INFO_LIST
    {
        UINT nSizeOfListEntry;      /* Size of one list entry in byte(in).  */
        UINT nNumDetectedFaces;     /* Number of detected faces(out).       */
        UINT nNumListElements;      /* Number of list elements(in).         */
        UINT nReserved[4];          /* reserved for future use(out).        */
        FDT_INFO_EL FaceEntry[1];   /* First face entry.                    */
    } FDT_INFO_LIST;


    typedef enum E_FDT_CMD
    {
        FDT_CMD_GET_CAPABILITIES            = 0,    /* Get the capabilities for face detection.                     */
        FDT_CMD_SET_DISABLE                 = 1,    /* Disable face detection.                                      */
        FDT_CMD_SET_ENABLE                  = 2,    /* Enable face detection.                                       */
        FDT_CMD_SET_SEARCH_ANGLE            = 3,    /* Set the search angle.                                        */
        FDT_CMD_GET_SEARCH_ANGLE            = 4,    /* Get the search angle parameter.                              */
        FDT_CMD_SET_SEARCH_ANGLE_ENABLE     = 5,    /* Enable search angle.                                         */
        FDT_CMD_SET_SEARCH_ANGLE_DISABLE    = 6,    /* Enable search angle.                                         */
        FDT_CMD_GET_SEARCH_ANGLE_ENABLE     = 7,    /* Get the current setting of search angle enable.              */
        FDT_CMD_SET_SEARCH_AOI              = 8,    /* Set the search AOI.                                          */
        FDT_CMD_GET_SEARCH_AOI              = 9,    /* Get the search AOI.                                          */
        FDT_CMD_GET_FACE_LIST               = 10,   /* Get a list with detected faces.                              */
        FDT_CMD_GET_NUMBER_FACES            = 11,   /* Get the number of detected faces.                            */
        FDT_CMD_SET_SUSPEND                 = 12,   /* Keep the face detection result of that moment.               */
        FDT_CMD_SET_RESUME                  = 13,   /* Continue with the face detection.                            */
        FDT_CMD_GET_MAX_NUM_FACES           = 14,   /* Get the maximum number of faces that can be detected once.   */
        FDT_CMD_SET_INFO_MAX_NUM_OVL        = 15,   /* Set the maximum number of overlays displayed.                */
        FDT_CMD_GET_INFO_MAX_NUM_OVL        = 16,   /* Get the setting 'maximum number of overlays displayed'.      */
        FDT_CMD_SET_INFO_OVL_LINE_WIDTH     = 17,   /* Set the overlay line width.                                  */
        FDT_CMD_GET_INFO_OVL_LINE_WIDTH     = 18,   /* Get the overlay line width.                                  */
        FDT_CMD_GET_ENABLE                  = 19,   /* Face detection enabled?.                                     */
        FDT_CMD_GET_SUSPEND                 = 20,   /* Face detection suspended?.                                   */
        FDT_CMD_GET_HORIZONTAL_RESOLUTION   = 21,   /* Horizontal resolution of face detection.                     */
        FDT_CMD_GET_VERTICAL_RESOLUTION     = 22    /* Vertical resolution of face detection.                       */

    } FDT_CMD;


    IDSEXP is_Focus (HIDS hCam, UINT nCommand, void *pParam, UINT nSizeOfParam);


    typedef enum E_FOCUS_CAPABILITY_FLAGS
    {
        FOC_CAP_INVALID             = 0,
        FOC_CAP_AUTOFOCUS_SUPPORTED = 0x00000001,   /* Auto focus supported.                                    */
        FOC_CAP_MANUAL_SUPPORTED    = 0x00000002,   /* Manual focus supported.                                  */
        FOC_CAP_GET_DISTANCE        = 0x00000004,   /* Support for query the distance of the focused object.    */
        FOC_CAP_SET_AUTOFOCUS_RANGE = 0x00000008,   /* Support for setting focus ranges.                        */
        FOC_CAP_AUTOFOCUS_FDT_AOI   = 0x00000010,   /* Use of face detection AOI for autofocus supported.       */
        FOC_CAP_AUTOFOCUS_ZONE      = 0x00000020
    } FOCUS_CAPABILITY_FLAGS;


    typedef enum E_FOCUS_RANGE
    {
        FOC_RANGE_NORMAL            = 0x00000001,   /* Normal focus range(without Macro).   */
        FOC_RANGE_ALLRANGE          = 0x00000002,   /* Allrange (macro to Infinity).        */
        FOC_RANGE_MACRO             = 0x00000004    /* Macro (only macro).                  */
    } FOCUS_RANGE;

    typedef enum E_FOCUS_STATUS
    {
        FOC_STATUS_ERROR            = 0x00000001,   /* Focus error has occurred */
        FOC_STATUS_FOCUSED          = 0x00000002,   /* Lens has been focused */
        FOC_STATUS_FOCUSING         = 0x00000004,   /* Lens is being focused */
        FOC_STATUS_TIMEOUT          = 0x00000008,   /* Focus timout has occurred */
        FOC_STATUS_CANCEL           = 0x00000010    /* Focus has been canceled */
    } FOCUS_STATUS;

    typedef enum E_FOCUS_ZONE_WEIGHT
    {
        FOC_ZONE_WEIGHT_DISABLE     = 0,
        FOC_ZONE_WEIGHT_WEAK        = 0x0021,
        FOC_ZONE_WEIGHT_MIDDLE      = 0x0032,
        FOC_ZONE_WEIGHT_STRONG      = 0x0042,

    } FOCUS_ZONE_WEIGHT;

    typedef enum E_FOCUS_CMD
    {
        FOC_CMD_GET_CAPABILITIES                    = 0,    /* Get focus capabilities.                                              */
        FOC_CMD_SET_DISABLE_AUTOFOCUS               = 1,    /* Disable autofocus.                                                   */
        FOC_CMD_SET_ENABLE_AUTOFOCUS                = 2,    /* Enable autofocus.                                                    */
        FOC_CMD_GET_AUTOFOCUS_ENABLE                = 3,    /* Autofocus enabled?.                                                  */
        FOC_CMD_SET_AUTOFOCUS_RANGE                 = 4,    /* Preset autofocus range.                                              */
        FOC_CMD_GET_AUTOFOCUS_RANGE                 = 5,    /* Get preset of autofocus range.                                       */
        FOC_CMD_GET_DISTANCE                        = 6,    /* Get distance to focused object.                                      */
        FOC_CMD_SET_MANUAL_FOCUS                    = 7,    /* Set manual focus.                                                    */
        FOC_CMD_GET_MANUAL_FOCUS                    = 8,    /* Get the value for manual focus.                                      */
        FOC_CMD_GET_MANUAL_FOCUS_MIN                = 9,    /* Get the minimum manual focus value.                                  */
        FOC_CMD_GET_MANUAL_FOCUS_MAX                = 10,   /* Get the maximum manual focus value.                                  */
        FOC_CMD_GET_MANUAL_FOCUS_INC                = 11,   /* Get the increment of the manual focus value.                         */
        FOC_CMD_SET_ENABLE_AF_FDT_AOI               = 12,   /* Enable face detection AOI use for autofocus.                         */
        FOC_CMD_SET_DISABLE_AF_FDT_AOI              = 13,   /* Disable face detection AOI use for autofocus                         */
        FOC_CMD_GET_AF_FDT_AOI_ENABLE               = 14,   /* Use autofocus FDT AOI?                                               */
        FOC_CMD_SET_ENABLE_AUTOFOCUS_ONCE           = 15,   /* Enable autofocus once                                                */
        FOC_CMD_GET_AUTOFOCUS_STATUS                = 16,   /* Get the autofocus status                                             */
        FOC_CMD_SET_AUTOFOCUS_ZONE_AOI              = 17,   /* Set the focus measurement window                                     */
        FOC_CMD_GET_AUTOFOCUS_ZONE_AOI              = 18,   /* Get the focus measurement window                                     */
        FOC_CMD_GET_AUTOFOCUS_ZONE_AOI_DEFAULT      = 19,   /* Get the default focus measurement window                             */
        FOC_CMD_GET_AUTOFOCUS_ZONE_POS_MIN          = 20,   /* Get the minimal position of the measurement window                   */
        FOC_CMD_GET_AUTOFOCUS_ZONE_POS_MAX          = 21,   /* Get the maximal position of the measurement window                   */
        FOC_CMD_GET_AUTOFOCUS_ZONE_POS_INC          = 22,   /* Get the incrementation for the positions of the measurement window   */
        FOC_CMD_GET_AUTOFOCUS_ZONE_SIZE_MIN         = 23,   /* Get the minimal size of the measurement window                       */
        FOC_CMD_GET_AUTOFOCUS_ZONE_SIZE_MAX         = 24,   /* Get the maxiaml size of the measurement window                       */
        FOC_CMD_GET_AUTOFOCUS_ZONE_SIZE_INC         = 25,   /* Get the incrementation for the size of the measurement window        */
        FOC_CMD_SET_AUTOFOCUS_ZONE_WEIGHT           = 26,   /* Set the weight for the different zones                               */
        FOC_CMD_GET_AUTOFOCUS_ZONE_WEIGHT           = 27,   /* Get the weight for the different zones                               */
        FOC_CMD_GET_AUTOFOCUS_ZONE_WEIGHT_COUNT     = 28,   /* Get the zone count                                                   */
        FOC_CMD_GET_AUTOFOCUS_ZONE_WEIGHT_DEFAULT   = 29,   /* Get the default weight for the different zones                       */
        

    } FOCUS_CMD;


    IDSEXP is_ImageStabilization(HIDS hCam, UINT nCommand, void *pParam, UINT nSizeOfParam);

    // image stabilization capability flags
    typedef enum E_IMGSTAB_CAPABILITY_FLAGS
    {
        IMGSTAB_CAP_INVALID                         = 0,
        IMGSTAB_CAP_IMAGE_STABILIZATION_SUPPORTED   = 0x00000001    /* Image stabilization supported. */
    } IMGSTAB_CAPABILITY_FLAGS;


    typedef enum E_IMGSTAB_CMD
    {
        IMGSTAB_CMD_GET_CAPABILITIES        = 0,    /* Get the capabilities for image stabilization.    */
        IMGSTAB_CMD_SET_DISABLE             = 1,    /* Disable image stabilization.                     */
        IMGSTAB_CMD_SET_ENABLE              = 2,    /* Enable image stabilization.                      */
        IMGSTAB_CMD_GET_ENABLE              = 3     /* Image stabilization enabled?                     */

    } IMGSTAB_CMD;

    // New functions and defines for 3.61 (uEye XS)
    IDSEXP is_ScenePreset(HIDS hCam, UINT nCommand, void *pParam, UINT nSizeOfParam);

    typedef enum E_SCENE_CMD
     {
        SCENE_CMD_GET_SUPPORTED_PRESETS = 1,/* Get the supported scene presets      */
        SCENE_CMD_SET_PRESET            = 2,/* Set the scene preset                 */
        SCENE_CMD_GET_PRESET            = 3,/* Get the current sensor scene preset  */
        SCENE_CMD_GET_DEFAULT_PRESET    = 4 /* Get the default sensor scene preset  */
    } SCENE_CMD;



    typedef enum E_SCENE_PRESET
    {
      SCENE_INVALID             = 0,
      SCENE_SENSOR_AUTOMATIC    = 0x00000001,
      SCENE_SENSOR_PORTRAIT     = 0x00000002,
      SCENE_SENSOR_SUNNY        = 0x00000004,
      SCENE_SENSOR_ENTERTAINMENT= 0x00000008,
      SCENE_SENSOR_NIGHT        = 0x00000010,
      SCENE_SENSOR_SPORTS       = 0x00000040,
      SCENE_SENSOR_LANDSCAPE    = 0x00000080
     } SCENE_PRESET;


IDSEXP is_Zoom (HIDS hCam, UINT nCommand, void *pParam, UINT nSizeOfParam);

typedef enum E_ZOOM_CMD
{
    ZOOM_CMD_GET_CAPABILITIES               = 0, /* Get the zoom capabilities. */
    ZOOM_CMD_DIGITAL_GET_NUM_LIST_ENTRIES   = 1, /* Get the number of list entries. */
    ZOOM_CMD_DIGITAL_GET_LIST               = 2, /* Get a list of supported zoom factors. */
    ZOOM_CMD_DIGITAL_SET_VALUE              = 3, /* Set the digital zoom factor zoom factors. */
    ZOOM_CMD_DIGITAL_GET_VALUE              = 4, /* Get a current digital zoom factor. */
    ZOOM_CMD_DIGITAL_GET_VALUE_RANGE        = 5, /* Get the digital zoom factor range. */
    ZOOM_CMD_DIGITAL_GET_VALUE_DEFAULT      = 6  /* Get the default digital zoom factor. */

} ZOOM_CMD;


typedef enum E_ZOOM_CAPABILITY_FLAGS
{
    ZOOM_CAP_INVALID        = 0,
    ZOOM_CAP_DIGITAL_ZOOM   = 0x00001
} ZOOM_CAPABILITY_FLAGS;


IDSEXP is_Sharpness (HIDS hCam, UINT nCommand, void *pParam, UINT nSizeOfParam);


typedef enum E_SHARPNESS_CMD
{
    SHARPNESS_CMD_GET_CAPABILITIES          = 0, /* Get the sharpness capabilities */
    SHARPNESS_CMD_GET_VALUE                 = 1, /* Get the current sharpness value */
    SHARPNESS_CMD_GET_MIN_VALUE             = 2, /* Get the minimum sharpness value */
    SHARPNESS_CMD_GET_MAX_VALUE             = 3, /* Get the maximum sharpness value */
    SHARPNESS_CMD_GET_INCREMENT             = 4, /* Get the sharpness increment */
    SHARPNESS_CMD_GET_DEFAULT_VALUE         = 5, /* Get the default sharpness value */
    SHARPNESS_CMD_SET_VALUE                 = 6  /* Set the sharpness value */

} SHARPNESS_CMD;


typedef enum E_SHARPNESS_CAPABILITY_FLAGS
{
    SHARPNESS_CAP_INVALID                   = 0x0000,
    SHARPNESS_CAP_SHARPNESS_SUPPORTED       = 0x0001

} SHARPNESS_CAPABILITY_FLAGS;


IDSEXP is_Saturation (HIDS hCam, UINT nCommand, void *pParam, UINT nSizeOfParam);


typedef enum E_SATURATION_CMD
{
    SATURATION_CMD_GET_CAPABILITIES         = 0, /* Get the saturation capabilities */
    SATURATION_CMD_GET_VALUE                = 1, /* Get the current saturation value */
    SATURATION_CMD_GET_MIN_VALUE            = 2, /* Get the minimum saturation value */
    SATURATION_CMD_GET_MAX_VALUE            = 3, /* Get the maximum saturation value */
    SATURATION_CMD_GET_INCREMENT            = 4, /* Get the saturation increment */
    SATURATION_CMD_GET_DEFAULT_VALUE        = 5, /* Get the default saturation value */
    SATURATION_CMD_SET_VALUE                = 6  /* Set the saturation value */

} SATURATION_CMD;


typedef enum E_SATURATION_CAPABILITY_FLAGS
{
    SATURATION_CAP_INVALID                  = 0x0000,
    SATURATION_CAP_SATURATION_SUPPORTED     = 0x0001

} SATURATION_CAPABILITY_FLAGS;


// Trigger debounce modes
typedef enum E_TRIGGER_DEBOUNCE_MODE
{
    TRIGGER_DEBOUNCE_MODE_NONE              = 0x0000,
    TRIGGER_DEBOUNCE_MODE_FALLING_EDGE      = 0x0001,   
    TRIGGER_DEBOUNCE_MODE_RISING_EDGE       = 0x0002,
    TRIGGER_DEBOUNCE_MODE_BOTH_EDGES        = 0x0004,
    TRIGGER_DEBOUNCE_MODE_AUTOMATIC         = 0x0008

} TRIGGER_DEBOUNCE_MODE;


// Trigger debounce commands
typedef enum E_TRIGGER_DEBOUNCE_CMD
{
    TRIGGER_DEBOUNCE_CMD_SET_MODE                   = 0, /* Set a new trigger debounce mode */
    TRIGGER_DEBOUNCE_CMD_SET_DELAY_TIME             = 1, /* Set a new trigger debounce delay time */
    TRIGGER_DEBOUNCE_CMD_GET_SUPPORTED_MODES        = 2, /* Get the supported modes */
    TRIGGER_DEBOUNCE_CMD_GET_MODE                   = 3, /* Get the current trigger debounce mode */
    TRIGGER_DEBOUNCE_CMD_GET_DELAY_TIME             = 4, /* Get the current trigger debounce delay time */
    TRIGGER_DEBOUNCE_CMD_GET_DELAY_TIME_MIN         = 5, /* Get the minimum value for the trigger debounce delay time */
    TRIGGER_DEBOUNCE_CMD_GET_DELAY_TIME_MAX         = 6, /* Get the maximum value for the trigger debounce delay time */
    TRIGGER_DEBOUNCE_CMD_GET_DELAY_TIME_INC         = 7, /* Get the increment of the trigger debounce delay time */
    TRIGGER_DEBOUNCE_CMD_GET_MODE_DEFAULT           = 8, /* Get the default trigger debounce mode */
    TRIGGER_DEBOUNCE_CMD_GET_DELAY_TIME_DEFAULT     = 9  /* Get the default trigger debounce delay time */

} TRIGGER_DEBOUNCE_CMD;


IDSEXP is_TriggerDebounce(HIDS hCam, UINT nCommand, void *pParam, UINT nSizeOfParam);


typedef enum E_RGB_COLOR_MODELS
{
    RGB_COLOR_MODEL_SRGB_D50        = 0x0001,
    RGB_COLOR_MODEL_SRGB_D65        = 0x0002,
    RGB_COLOR_MODEL_CIE_RGB_E       = 0x0004,
    RGB_COLOR_MODEL_ECI_RGB_D50     = 0x0008,
    RGB_COLOR_MODEL_ADOBE_RGB_D65   = 0x0010,

} RGB_COLOR_MODELS;

typedef enum E_LENS_SHADING_MODELS
{
    LSC_MODEL_AGL   = 0x0001,
    LSC_MODEL_TL84  = 0x0002,
    LSC_MODEL_D50   = 0x0004,
    LSC_MODEL_D65   = 0x0008,

} LENS_SHADING_MODELS;


typedef enum E_COLOR_TEMPERATURE_CMD
{
    COLOR_TEMPERATURE_CMD_SET_TEMPERATURE                   = 0,  /* Set a new color temperature */
    COLOR_TEMPERATURE_CMD_SET_RGB_COLOR_MODEL               = 1,  /* Set a new RGB color model */
    COLOR_TEMPERATURE_CMD_GET_SUPPORTED_RGB_COLOR_MODELS    = 2,  /* Get the supported RGB color models */
    COLOR_TEMPERATURE_CMD_GET_TEMPERATURE                   = 3,  /* Get the current color temperature */
    COLOR_TEMPERATURE_CMD_GET_RGB_COLOR_MODEL               = 4,  /* Get the current RGB color model */
    COLOR_TEMPERATURE_CMD_GET_TEMPERATURE_MIN               = 5,  /* Get the minimum value for the color temperature */
    COLOR_TEMPERATURE_CMD_GET_TEMPERATURE_MAX               = 6,  /* Get the maximum value for the color temperature */
    COLOR_TEMPERATURE_CMD_GET_TEMPERATURE_INC               = 7,  /* Get the increment of the color temperature */
    COLOR_TEMPERATURE_CMD_GET_TEMPERATURE_DEFAULT           = 8,  /* Get the default color temperature */
    COLOR_TEMPERATURE_CMD_GET_RGB_COLOR_MODEL_DEFAULT       = 9,  /* Get the default RGB color model */
    COLOR_TEMPERATURE_CMD_SET_LENS_SHADING_MODEL            = 10, /* Set a new lens shading model */
    COLOR_TEMPERATURE_CMD_GET_LENS_SHADING_MODEL            = 11, /* Get the current lens shading model */ 
    COLOR_TEMPERATURE_CMD_GET_LENS_SHADING_MODEL_SUPPORTED  = 12, /* Get the supported lens shading models */
    COLOR_TEMPERATURE_CMD_GET_LENS_SHADING_MODEL_DEFAULT    = 13, /* Get the default lens shading model */ 

} COLOR_TEMPERATURE_CMD;

IDSEXP is_ColorTemperature(HIDS hCam, UINT nCommand, void *pParam, UINT nSizeOfParam);


typedef struct _OPENGL_DISPLAY
{
	int 	nWindowID;
	void 	*pDisplay;
} OPENGL_DISPLAY;

IDSEXP is_DirectRenderer(HIDS hCam, UINT nMode, void *pParam, UINT SizeOfParam);

IDSEXP is_HotPixel(HIDS hCam, UINT nMode, void *pParam, UINT SizeOfParam);


typedef struct 
{
    INT s32X;
    INT s32Y;
} IS_POINT_2D;

typedef struct 
{
    INT s32Width;
    INT s32Height;
} IS_SIZE_2D;

typedef struct
{
    INT s32X;
    INT s32Y;
    INT s32Width;
    INT s32Height;
} IS_RECT;


/*!
 * \brief Parameters of an AOI used in the AOI sequence mode
 */
typedef struct
{
    INT     s32AOIIndex;
    INT     s32NumberOfCycleRepetitions;
    INT     s32X;
    INT     s32Y;
    double  dblExposure;
    INT     s32Gain;
    INT     s32BinningMode;
    INT     s32SubsamplingMode;
    INT     s32DetachImageParameters;
    double  dblScalerFactor;
    BYTE    byReserved[64];
} AOI_SEQUENCE_PARAMS;


IDSEXP is_AOI(HIDS hCam, UINT nCommand, void *pParam, UINT SizeOfParam); 

/*!
 * \brief Data type for 32-bit unsigned int value ranges.
 *
 * \since uEye SDK 3.80.
 */
typedef struct S_RANGE_OF_VALUES_U32
{
    UINT    u32Minimum;     /*!< \brief Minimum value. */
    UINT    u32Maximum;     /*!< \brief Maximum value, not considered Infinite. */
    UINT    u32Increment;   /*!< \brief Increment value. */
    UINT    u32Default;     /*!< \brief Default value. */
    UINT    u32Infinite;    /*!< \brief Infinite code, where applicable. */

} RANGE_OF_VALUES_U32;


/*!
 * \brief Enumeration of transfer engine's capability flags.
 * 
 * \sa is_Transfer, TRANSFER_CMD_QUERY_CAPABILITIES.
 *
 * \note to come
 */
typedef enum E_TRANSFER_CAPABILITY_FLAGS
{
    /*!
     * \brief Capability flag: 'Image delay' setting supported.
     */
    TRANSFER_CAP_IMAGEDELAY         = 0x01,

    /*!
     * \brief Capability flag: 'Packet clock' setting supported.
     */
    TRANSFER_CAP_PACKETINTERVAL     = 0x20,

} TRANSFER_CAPABILITY_FLAGS;

/*!
 * \brief Enumeration of commands supported by the transfer engine access function.
 *
 * \sa is_Transfer.
 *
 * \note to come
 */
typedef enum E_TRANSFER_CMD
{
    /*!
     * \brief   Query the transfer engine capabilities.
     *          Type of value: unsigned int bitmask, 32-bit.
     * \sa TRANSFER_CAPABILITY_FLAGS.
     */
    TRANSFER_CMD_QUERY_CAPABILITIES                   = 0,

    /*!
     * \brief   Set 'image delay' in us units.
     *          Type of value: unsigned int, 32-bit.
     */
    TRANSFER_CMD_SET_IMAGEDELAY_US                    = 1000,
    /*!
     * \brief   Set 'packet clock' in us units.
     *          Type of value: unsigned int, 32-bit.
     */
    TRANSFER_CMD_SET_PACKETINTERVAL_US                = 1005,

    /*!
     * \brief   Get 'image delay' in us units.
     *          Type of value: unsigned int, 32-bit.
     */
    TRANSFER_CMD_GET_IMAGEDELAY_US                    = 2000,
    /*!
     * \brief   Get 'packet clock' in us units.
     *          Type of value: unsigned int, 32-bit.
     */
    TRANSFER_CMD_GET_PACKETINTERVAL_US                = 2005,

    /*!
     * \brief   Get range of values for 'image delay'.
     *          Type of value: RANGE_OF_VALUES_U32, size of RANGE_OF_VALUES_U32.
     */
    TRANSFER_CMD_GETRANGE_IMAGEDELAY_US               = 3000,
    /*!
     * \brief   Get range of values for 'packet clock'.
     *          Type of value: RANGE_OF_VALUES_U32, size of RANGE_OF_VALUES_U32.
     */
    TRANSFER_CMD_GETRANGE_PACKETINTERVAL_US           = 3005,

    /*!
     * \brief   Sets Image Destination \see TRANSFER_MODE
     *          Type of value: TRANSFER_MODE, size of TRANSFER_MODE.
     */
    TRANSFER_CMD_SET_IMAGE_DESTINATION                = 5000,

    /*!
     * \brief   Retrieves currently set Image Destination \see TRANSFER_MODE
     *          Type of value: TRANSFER_MODE, size of TRANSFER_MODE.
     */
    TRANSFER_CMD_GET_IMAGE_DESTINATION                = 5001,

    /*!
     * \brief   Retrieves possible Image Destinations \see TRANSFER_MODE
     *          Type of value: TRANSFER_MODE, size of TRANSFER_MODE.
     */
    TRANSFER_CMD_GET_IMAGE_DESTINATION_CAPABILITIES   = 5002

} TRANSFER_CMD;


typedef enum E_TRANSFER_TARGET
{
    IS_TRANSFER_DESTINATION_DEVICE_MEMORY   = 1,
    IS_TRANSFER_DESTINATION_USER_MEMORY     = 2

} TRANSFER_TARGET;


/*!
 * \brief Generic interface to the transfer engine setup.
 *
 * \param   hCam            valid device handle.
 * \param   nCommand        code, indicates requested access and accessed value, one from \ref TRANSFER_CMD enumeration.
 * \param   pParam          input or output storage for the accessed param.
 * \param   cbSizeOfParam   size of *pParam.
 * \return  error code
 *
 * \note For 'set' commands: If the resulting transfer engine setup differs from the current state the transfer param
 *          is immediately written to the device.
 * \note For 'set' commands: The function forces the value to the next lower catch according to the related increment.
 */
IDSEXP is_Transfer(HIDS hCam, UINT nCommand, void* pParam, UINT cbSizeOfParam);


/*!
 * \brief definition of the boot boost id
 *
 * \since uEye SDK 3.90.
 */
typedef BYTE IS_BOOTBOOST_ID;

#define IS_BOOTBOOST_ID_MIN     1   /*!< \brief minimum valid id */
#define IS_BOOTBOOST_ID_MAX     254 /*!< \brief maximum valid id */
#define IS_BOOTBOOST_ID_NONE    0   /*!< \brief special value: no id's */
#define IS_BOOTBOOST_ID_ALL     255 /*!< \brief special value: all id's */

/*!
 * \brief definition of the boot boost id list object
 *
 * \since uEye SDK 3.90.
 */
typedef struct S_IS_BOOTBOOST_IDLIST
{
    /*! \brief the number of listed elements. */
    DWORD u32NumberOfEntries;

    /*! \brief the list. */
    IS_BOOTBOOST_ID aList[1];

} IS_BOOTBOOST_IDLIST;

/*! \brief size of id list header */
#define IS_BOOTBOOST_IDLIST_HEADERSIZE  (sizeof(DWORD))
/*! \brief size of id list element */
#define IS_BOOTBOOST_IDLIST_ELEMENTSIZE (sizeof(IS_BOOTBOOST_ID))

/*!
 * \brief Enumeration of commands supported by the boot boost access function \ref is_BootBoost.
 */
typedef enum E_IS_BOOTBOOST_CMD
{
    /*!
     * \brief   Enable BootBoost.
     *          Type of data: no data.
     */
    IS_BOOTBOOST_CMD_ENABLE             = 0x00010001,

    /*!
     * \brief   Disable BootBoost.
     *          Type of data: no data.
     */
    IS_BOOTBOOST_CMD_DISABLE            = 0x00010011,

    /*!
     * \brief   Query enabled state of BootBoost.
     *          Type of data: DWORD.
     */
    IS_BOOTBOOST_CMD_GET_ENABLED        = 0x20010021,

    /*!
     * \brief   Add an id to the id list.
     *          Type of data: \ref IS_BOOTBOOST_ID.
     */
    IS_BOOTBOOST_CMD_ADD_ID             = 0x10100001,

    /*!
     * \brief   Replace id list.
     *          Type of data: \ref IS_BOOTBOOST_IDLIST.
     */
    IS_BOOTBOOST_CMD_SET_IDLIST         = 0x10100005,

    /*!
     * \brief   Remove an id from the id list.
     *          Type of data: \ref IS_BOOTBOOST_ID.
     */
     IS_BOOTBOOST_CMD_REMOVE_ID         = 0x10100011,

    /*!
     * \brief   Clear id list.
     *          Type of data: no data.
     */
    IS_BOOTBOOST_CMD_CLEAR_IDLIST       = 0x00100015,

    /*!
     * \brief   Query id list.
     *          Type of data: \ref IS_BOOTBOOST_IDLIST.
     */
     IS_BOOTBOOST_CMD_GET_IDLIST        = 0x30100021,

    /*!
     * \brief   Query the number of listed id's.
     *          Type of data: \ref DWORD.
     */
    IS_BOOTBOOST_CMD_GET_IDLIST_SIZE    = 0x20100022,

} IS_BOOTBOOST_CMD;

/*!
 * \brief Generic interface to the boot boost functionality.
 *
 * \param   hCam            currently not used / ignored, passing 0 recommended
 * \param   nCommand        code, indicates requested access and accessed parameter, one of \ref BOOTBOOST_CMD enumeration.
 * \param   pParam          input or output storage for the accessed param.
 * \param   cbSizeOfParam   size of *pParam.
 * \return  error code
 *
 * \since uEye SDK 3.90.
 *
 * \note For the reason of your applications compatibility to future extensions
 *       it is strongly recommended to pass the value 0 for hCam.
 */
IDSEXP is_BootBoost(HIDS hCam, UINT nCommand, void* pParam, UINT cbSizeOfParam);


/*!
 * \brief Enumeration of commands supported by the device feature function \ref is_DeviceFeature.
 */
typedef enum E_DEVICE_FEATURE_CMD
{
    IS_DEVICE_FEATURE_CMD_GET_SUPPORTED_FEATURES                                = 1,
    IS_DEVICE_FEATURE_CMD_SET_LINESCAN_MODE                                     = 2,
    IS_DEVICE_FEATURE_CMD_GET_LINESCAN_MODE                                     = 3,
    IS_DEVICE_FEATURE_CMD_SET_LINESCAN_NUMBER                                   = 4,
    IS_DEVICE_FEATURE_CMD_GET_LINESCAN_NUMBER                                   = 5,
    IS_DEVICE_FEATURE_CMD_SET_SHUTTER_MODE                                      = 6,
    IS_DEVICE_FEATURE_CMD_GET_SHUTTER_MODE                                      = 7,
    IS_DEVICE_FEATURE_CMD_SET_PREFER_XS_HS_MODE                                 = 8,
    IS_DEVICE_FEATURE_CMD_GET_PREFER_XS_HS_MODE                                 = 9,
    IS_DEVICE_FEATURE_CMD_GET_DEFAULT_PREFER_XS_HS_MODE                         = 10,   
    IS_DEVICE_FEATURE_CMD_GET_LOG_MODE_DEFAULT                                  = 11,
    IS_DEVICE_FEATURE_CMD_GET_LOG_MODE                                          = 12,
    IS_DEVICE_FEATURE_CMD_SET_LOG_MODE                                          = 13,   
    IS_DEVICE_FEATURE_CMD_GET_LOG_MODE_MANUAL_VALUE_DEFAULT                     = 14,
    IS_DEVICE_FEATURE_CMD_GET_LOG_MODE_MANUAL_VALUE_RANGE                       = 15,
    IS_DEVICE_FEATURE_CMD_GET_LOG_MODE_MANUAL_VALUE                             = 16,
    IS_DEVICE_FEATURE_CMD_SET_LOG_MODE_MANUAL_VALUE                             = 17,
    IS_DEVICE_FEATURE_CMD_GET_LOG_MODE_MANUAL_GAIN_DEFAULT                      = 18,
    IS_DEVICE_FEATURE_CMD_GET_LOG_MODE_MANUAL_GAIN_RANGE                        = 19,
    IS_DEVICE_FEATURE_CMD_GET_LOG_MODE_MANUAL_GAIN                              = 20,
    IS_DEVICE_FEATURE_CMD_SET_LOG_MODE_MANUAL_GAIN                              = 21,
    IS_DEVICE_FEATURE_CMD_GET_VERTICAL_AOI_MERGE_MODE_DEFAULT                   = 22,
    IS_DEVICE_FEATURE_CMD_GET_VERTICAL_AOI_MERGE_MODE                           = 23,
    IS_DEVICE_FEATURE_CMD_SET_VERTICAL_AOI_MERGE_MODE                           = 24,
    IS_DEVICE_FEATURE_CMD_GET_VERTICAL_AOI_MERGE_POSITION_DEFAULT               = 25,
    IS_DEVICE_FEATURE_CMD_GET_VERTICAL_AOI_MERGE_POSITION_RANGE                 = 26,
    IS_DEVICE_FEATURE_CMD_GET_VERTICAL_AOI_MERGE_POSITION                       = 27,
    IS_DEVICE_FEATURE_CMD_SET_VERTICAL_AOI_MERGE_POSITION                       = 28,
    IS_DEVICE_FEATURE_CMD_GET_FPN_CORRECTION_MODE_DEFAULT                       = 29,
    IS_DEVICE_FEATURE_CMD_GET_FPN_CORRECTION_MODE                               = 30,
    IS_DEVICE_FEATURE_CMD_SET_FPN_CORRECTION_MODE                               = 31,
    IS_DEVICE_FEATURE_CMD_GET_SENSOR_SOURCE_GAIN_RANGE                          = 32,
    IS_DEVICE_FEATURE_CMD_GET_SENSOR_SOURCE_GAIN_DEFAULT                        = 33,
    IS_DEVICE_FEATURE_CMD_GET_SENSOR_SOURCE_GAIN                                = 34,
    IS_DEVICE_FEATURE_CMD_SET_SENSOR_SOURCE_GAIN                                = 35,
    IS_DEVICE_FEATURE_CMD_GET_BLACK_REFERENCE_MODE_DEFAULT                      = 36,
    IS_DEVICE_FEATURE_CMD_GET_BLACK_REFERENCE_MODE                              = 37,
    IS_DEVICE_FEATURE_CMD_SET_BLACK_REFERENCE_MODE                              = 38,
    IS_DEVICE_FEATURE_CMD_GET_ALLOW_RAW_WITH_LUT                                = 39,
    IS_DEVICE_FEATURE_CMD_SET_ALLOW_RAW_WITH_LUT                                = 40,
    IS_DEVICE_FEATURE_CMD_GET_SUPPORTED_SENSOR_BIT_DEPTHS                       = 41,
    IS_DEVICE_FEATURE_CMD_GET_SENSOR_BIT_DEPTH_DEFAULT                          = 42,
    IS_DEVICE_FEATURE_CMD_GET_SENSOR_BIT_DEPTH                                  = 43,
    IS_DEVICE_FEATURE_CMD_SET_SENSOR_BIT_DEPTH                                  = 44,
    IS_DEVICE_FEATURE_CMD_GET_TEMPERATURE                                       = 45,
    IS_DEVICE_FEATURE_CMD_GET_JPEG_COMPRESSION                                  = 46,
    IS_DEVICE_FEATURE_CMD_SET_JPEG_COMPRESSION                                  = 47,
    IS_DEVICE_FEATURE_CMD_GET_JPEG_COMPRESSION_DEFAULT                          = 48,
    IS_DEVICE_FEATURE_CMD_GET_JPEG_COMPRESSION_RANGE                            = 49,
    IS_DEVICE_FEATURE_CMD_GET_NOISE_REDUCTION_MODE                              = 50,
    IS_DEVICE_FEATURE_CMD_SET_NOISE_REDUCTION_MODE                              = 51,
    IS_DEVICE_FEATURE_CMD_GET_NOISE_REDUCTION_MODE_DEFAULT                      = 52,
    IS_DEVICE_FEATURE_CMD_GET_TIMESTAMP_CONFIGURATION                           = 53,
    IS_DEVICE_FEATURE_CMD_SET_TIMESTAMP_CONFIGURATION                           = 54,
    IS_DEVICE_FEATURE_CMD_GET_VERTICAL_AOI_MERGE_HEIGHT_DEFAULT                 = 55,
    IS_DEVICE_FEATURE_CMD_GET_VERTICAL_AOI_MERGE_HEIGHT_NUMBER                  = 56,
    IS_DEVICE_FEATURE_CMD_GET_VERTICAL_AOI_MERGE_HEIGHT_LIST                    = 57,
    IS_DEVICE_FEATURE_CMD_GET_VERTICAL_AOI_MERGE_HEIGHT                         = 58,
    IS_DEVICE_FEATURE_CMD_SET_VERTICAL_AOI_MERGE_HEIGHT                         = 59,
    IS_DEVICE_FEATURE_CMD_GET_VERTICAL_AOI_MERGE_ADDITIONAL_POSITION_DEFAULT    = 60,
    IS_DEVICE_FEATURE_CMD_GET_VERTICAL_AOI_MERGE_ADDITIONAL_POSITION_RANGE      = 61,
    IS_DEVICE_FEATURE_CMD_GET_VERTICAL_AOI_MERGE_ADDITIONAL_POSITION            = 62,
    IS_DEVICE_FEATURE_CMD_SET_VERTICAL_AOI_MERGE_ADDITIONAL_POSITION            = 63,
    IS_DEVICE_FEATURE_CMD_GET_SENSOR_TEMPERATURE_NUMERICAL_VALUE                = 64,
    IS_DEVICE_FEATURE_CMD_SET_IMAGE_EFFECT                                      = 65,
    IS_DEVICE_FEATURE_CMD_GET_IMAGE_EFFECT                                      = 66,
    IS_DEVICE_FEATURE_CMD_GET_IMAGE_EFFECT_DEFAULT                              = 67,
    IS_DEVICE_FEATURE_CMD_GET_EXTENDED_PIXELCLOCK_RANGE_ENABLE_DEFAULT			= 68,
	IS_DEVICE_FEATURE_CMD_GET_EXTENDED_PIXELCLOCK_RANGE_ENABLE					= 69,
	IS_DEVICE_FEATURE_CMD_SET_EXTENDED_PIXELCLOCK_RANGE_ENABLE					= 70,
	IS_DEVICE_FEATURE_CMD_MULTI_INTEGRATION_GET_SCOPE            			    = 71,
	IS_DEVICE_FEATURE_CMD_MULTI_INTEGRATION_GET_PARAMS           			    = 72,
	IS_DEVICE_FEATURE_CMD_MULTI_INTEGRATION_SET_PARAMS                 		    = 73,
    IS_DEVICE_FEATURE_CMD_MULTI_INTEGRATION_GET_MODE_DEFAULT                	= 74,
	IS_DEVICE_FEATURE_CMD_MULTI_INTEGRATION_GET_MODE                     	    = 75,
	IS_DEVICE_FEATURE_CMD_MULTI_INTEGRATION_SET_MODE                   	        = 76,
    IS_DEVICE_FEATURE_CMD_SET_I2C_TARGET                                        = 77

} DEVICE_FEATURE_CMD;


/*!
 * \brief Enumeration of device feature capability flags.
 *
 * \sa is_DeviceFeature
 */
typedef enum E_DEVICE_FEATURE_MODE_CAPS
{
    IS_DEVICE_FEATURE_CAP_SHUTTER_MODE_ROLLING                      = 0x00000001,
    IS_DEVICE_FEATURE_CAP_SHUTTER_MODE_GLOBAL                       = 0x00000002,
    IS_DEVICE_FEATURE_CAP_LINESCAN_MODE_FAST                        = 0x00000004,
    IS_DEVICE_FEATURE_CAP_LINESCAN_NUMBER                           = 0x00000008,
    IS_DEVICE_FEATURE_CAP_PREFER_XS_HS_MODE                         = 0x00000010,
    IS_DEVICE_FEATURE_CAP_LOG_MODE                                  = 0x00000020,
    IS_DEVICE_FEATURE_CAP_SHUTTER_MODE_ROLLING_GLOBAL_START         = 0x00000040,
    IS_DEVICE_FEATURE_CAP_SHUTTER_MODE_GLOBAL_ALTERNATIVE_TIMING    = 0x00000080,
    IS_DEVICE_FEATURE_CAP_VERTICAL_AOI_MERGE                        = 0x00000100,
    IS_DEVICE_FEATURE_CAP_FPN_CORRECTION                            = 0x00000200,
    IS_DEVICE_FEATURE_CAP_SENSOR_SOURCE_GAIN                        = 0x00000400,
    IS_DEVICE_FEATURE_CAP_BLACK_REFERENCE                           = 0x00000800,
    IS_DEVICE_FEATURE_CAP_SENSOR_BIT_DEPTH                          = 0x00001000,
    IS_DEVICE_FEATURE_CAP_TEMPERATURE                               = 0x00002000,
    IS_DEVICE_FEATURE_CAP_JPEG_COMPRESSION                          = 0x00004000,
    IS_DEVICE_FEATURE_CAP_NOISE_REDUCTION                           = 0x00008000,
    IS_DEVICE_FEATURE_CAP_TIMESTAMP_CONFIGURATION                   = 0x00010000,
    IS_DEVICE_FEATURE_CAP_IMAGE_EFFECT                              = 0x00020000,
    IS_DEVICE_FEATURE_CAP_EXTENDED_PIXELCLOCK_RANGE					= 0x00040000,
    IS_DEVICE_FEATURE_CAP_MULTI_INTEGRATION         	            = 0x00080000

} DEVICE_FEATURE_MODE_CAPS;

/*!
 * \brief Enumeration of Noise modes.
 *
 * \sa is_DeviceFeature
 */
typedef enum E_NOISE_REDUCTION_MODES
{
    IS_NOISE_REDUCTION_OFF        = 0,
    IS_NOISE_REDUCTION_ADAPTIVE   = 1
} NOISE_REDUCTION_MODES;

/*!
 * \brief Enumeration of Log modes.
 *
 * \sa is_DeviceFeature
 */
typedef enum E_LOG_MODES
{
    IS_LOG_MODE_FACTORY_DEFAULT    = 0,
    IS_LOG_MODE_OFF                = 1,
    IS_LOG_MODE_MANUAL             = 2,
    IS_LOG_MODE_AUTO               = 3

} LOG_MODES;


/*!
 * \brief Enumeration of Vertical AOI merge modes.
 *
 * \sa is_DeviceFeature
 */
typedef enum E_VERTICAL_AOI_MERGE_MODES
{
    IS_VERTICAL_AOI_MERGE_MODE_OFF                      = 0,
    IS_VERTICAL_AOI_MERGE_MODE_FREERUN                  = 1,
    IS_VERTICAL_AOI_MERGE_MODE_TRIGGERED_SOFTWARE       = 2,
    IS_VERTICAL_AOI_MERGE_MODE_TRIGGERED_FALLING_GPIO1  = 3,
    IS_VERTICAL_AOI_MERGE_MODE_TRIGGERED_RISING_GPIO1   = 4,
    IS_VERTICAL_AOI_MERGE_MODE_TRIGGERED_FALLING_GPIO2  = 5,
    IS_VERTICAL_AOI_MERGE_MODE_TRIGGERED_RISING_GPIO2   = 6

} VERTICAL_AOI_MERGE_MODES;


/*!
 * \brief Enumeration of FPN correction modes.
 *
 * \sa is_DeviceFeature
 */
typedef enum E_FPN_CORRECTION_MODES
{
    IS_FPN_CORRECTION_MODE_OFF      = 0,
    IS_FPN_CORRECTION_MODE_HARDWARE = 1

} FPN_CORRECTION_MODES;


/*!
 * \brief Enumeration of black reference pixel modes.
 *
 * \sa is_DeviceFeature
 */
typedef enum E_BLACK_REFERENCE_MODES
{
    IS_BLACK_REFERENCE_MODE_OFF           = 0,
    IS_BLACK_REFERENCE_MODE_COLUMNS_LEFT  = 1

} BLACK_REFERENCE_MODES;


/*!
 * \brief Enumeration of sensor bit depths.
 *
 * \sa is_DeviceFeature
 */
typedef enum E_SENSOR_BIT_DEPTH
{
    IS_SENSOR_BIT_DEPTH_AUTO    = 0x00000000,
    IS_SENSOR_BIT_DEPTH_8_BIT   = 0x00000001,
    IS_SENSOR_BIT_DEPTH_10_BIT  = 0x00000002,
    IS_SENSOR_BIT_DEPTH_12_BIT  = 0x00000004

} SENSOR_BIT_DEPTH;


/*!
 * \brief Enumeration of timestamp configuration modes.
 *
 * \sa is_DeviceFeature
 */
typedef enum E_TIMESTAMP_CONFIGURATION_MODE
{
    IS_RESET_TIMESTAMP_ONCE = 1

} TIMESTAMP_CONFIGURATION_MODE;


/*!
 * \brief Enumeration of timestamp configuration pins.
 *
 * \sa is_DeviceFeature
 */
typedef enum E_TIMESTAMP_CONFIGURATION_PIN
{
    TIMESTAMP_CONFIGURATION_PIN_NONE        = 0,
    TIMESTAMP_CONFIGURATION_PIN_TRIGGER     = 1,
    TIMESTAMP_CONFIGURATION_PIN_GPIO_1      = 2,
    TIMESTAMP_CONFIGURATION_PIN_GPIO_2      = 3

} TIMESTAMP_CONFIGURATION_PIN;


/*!
 * \brief Enumeration of timestamp configuration edges.
 *
 * \sa is_DeviceFeature
 */
typedef enum E_TIMESTAMP_CONFIGURATION_EDGE
{
    TIMESTAMP_CONFIGURATION_EDGE_FALLING    = 0,
    TIMESTAMP_CONFIGURATION_EDGE_RISING     = 1

} TIMESTAMP_CONFIGURATION_EDGE;


typedef struct S_IS_TIMESTAMP_CONFIGURATION
{
    INT s32Mode;
    INT s32Pin;
    INT s32Edge;

} IS_TIMESTAMP_CONFIGURATION;


/*!
 * \brief Enumeration of image effect configuration.
 *
 * \sa is_DeviceFeature
 */
typedef enum E_IMAGE_EFFECT_MODE
{
    IS_IMAGE_EFFECT_DISABLE    = 0,
    IS_IMAGE_EFFECT_SEPIA      = 1,
    IS_IMAGE_EFFECT_MONOCHROME = 2,
    IS_IMAGE_EFFECT_NEGATIVE   = 3,
    IS_IMAGE_EFFECT_CROSSHAIRS = 4

} IMAGE_EFFECT_MODE;


/*!
 * \brief Enumeration of the extended pixelclock ranges.
 *
 * \sa is_DeviceFeature
 */
typedef enum S_IS_EXTENDED_PIXELCLOCK_RANGE
{
    EXTENDED_PIXELCLOCK_RANGE_OFF	= 0,
    EXTENDED_PIXELCLOCK_RANGE_ON	= 1

} IS_EXTENDED_PIXELCLOCK_RANGE;


/*!
 * \brief Enumeration of the multi integration modes.
 *
 * \sa is_DeviceFeature
 */
typedef enum E_IS_MULTI_INTEGRATION_MODE
{
    MULTI_INTEGRATION_MODE_OFF      = 0,
    MULTI_INTEGRATION_MODE_SOFTWARE = 1,
    MULTI_INTEGRATION_MODE_GPIO1    = 2,
    MULTI_INTEGRATION_MODE_GPIO2    = 3
    
} IS_MULTI_INTEGRATION_MODE;


typedef struct S_IS_MULTI_INTEGRATION_CYCLES
{
    double dblIntegration_ms;
    double dblPause_ms;

} IS_MULTI_INTEGRATION_CYCLES;


typedef struct S_IS_MULTI_INTEGRATION_SCOPE
{
    double dblMinIntegration_ms;
    double dblMaxIntegration_ms;
    double dblIntegrationGranularity_ms;
    double dblMinPause_ms;
    double dblMaxPause_ms;
    double dblPauseGranularity_ms;
    double dblMinCycle_ms;
    double dblMaxCycle_ms;
    double dblCycleGranularity_ms;
    double dblMinTriggerCycle_ms;
    double dblMinTriggerDuration_ms;
    UINT nMinNumberOfCycles;
    UINT nMaxNumberOfCycles;
    BYTE m_bReserved[32];

} IS_MULTI_INTEGRATION_SCOPE;


typedef enum E_IS_I2C_TARGET
{
    I2C_TARGET_DEFAULT      = 0,
    I2C_TARGET_SENSOR_1     = 1,
    I2C_TARGET_SENSOR_2     = 2,
    I2C_TARGET_LOGIC_BOARD  = 4

} IS_I2C_TARGET;


/*!
 * \brief Interface to control various device features.
 *
 * \param   hCam            valid device handle.
 * \param   nCommand        Specifies the command and device.
 * \param   pParam          input or output storage for the accessed param.
 * \param   cbSizeOfParam   size of *pParam.
 * \return  error code
 */
IDSEXP is_DeviceFeature(HIDS hCam, UINT nCommand, void* pParam, UINT cbSizeOfParam);


/*!
 * \brief Enumeration of commands supported by the exposure function.
 *
 * \sa is_Exposure.
 */
typedef enum E_EXPOSURE_CMD
{
    IS_EXPOSURE_CMD_GET_CAPS                        = 1,
    IS_EXPOSURE_CMD_GET_EXPOSURE_DEFAULT            = 2,
    IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MIN          = 3,
    IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MAX          = 4,
    IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_INC          = 5,
    IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE              = 6,
    IS_EXPOSURE_CMD_GET_EXPOSURE                    = 7,
    IS_EXPOSURE_CMD_GET_FINE_INCREMENT_RANGE_MIN    = 8,
    IS_EXPOSURE_CMD_GET_FINE_INCREMENT_RANGE_MAX    = 9,
    IS_EXPOSURE_CMD_GET_FINE_INCREMENT_RANGE_INC    = 10,
    IS_EXPOSURE_CMD_GET_FINE_INCREMENT_RANGE        = 11,
    IS_EXPOSURE_CMD_SET_EXPOSURE                    = 12,
    IS_EXPOSURE_CMD_GET_LONG_EXPOSURE_RANGE_MIN     = 13,
    IS_EXPOSURE_CMD_GET_LONG_EXPOSURE_RANGE_MAX     = 14,
    IS_EXPOSURE_CMD_GET_LONG_EXPOSURE_RANGE_INC     = 15,
    IS_EXPOSURE_CMD_GET_LONG_EXPOSURE_RANGE         = 16,
    IS_EXPOSURE_CMD_GET_LONG_EXPOSURE_ENABLE        = 17,
    IS_EXPOSURE_CMD_SET_LONG_EXPOSURE_ENABLE        = 18,
    IS_EXPOSURE_CMD_GET_DUAL_EXPOSURE_RATIO_DEFAULT = 19, 
    IS_EXPOSURE_CMD_GET_DUAL_EXPOSURE_RATIO_RANGE   = 20, 
    IS_EXPOSURE_CMD_GET_DUAL_EXPOSURE_RATIO         = 21,
    IS_EXPOSURE_CMD_SET_DUAL_EXPOSURE_RATIO         = 22

} EXPOSURE_CMD;


typedef enum E_EXPOSURE_CAPS
{
    IS_EXPOSURE_CAP_EXPOSURE                        = 0x00000001,
    IS_EXPOSURE_CAP_FINE_INCREMENT                  = 0x00000002,
    IS_EXPOSURE_CAP_LONG_EXPOSURE                   = 0x00000004,
    IS_EXPOSURE_CAP_DUAL_EXPOSURE                   = 0x00000008

} EXPOSURE_CAPS;


/*!
 * \brief Interface to control the camera exposure
 *
 * \param   hCam            valid device handle.
 * \param   nCommand        Specifies the command
 * \param   pParam          input or output storage for the accessed param.
 * \param   cbSizeOfParam   size of *pParam.
 * \return  error code
 */
IDSEXP is_Exposure(HIDS hCam, UINT nCommand, void* pParam, UINT cbSizeOfParam);



typedef enum E_TRIGGER_CMD
{
    IS_TRIGGER_CMD_GET_BURST_SIZE_SUPPORTED     = 1,
    IS_TRIGGER_CMD_GET_BURST_SIZE_RANGE         = 2,
    IS_TRIGGER_CMD_GET_BURST_SIZE               = 3,
    IS_TRIGGER_CMD_SET_BURST_SIZE               = 4

} TRIGGER_CMD;


/*!
 * \brief Interface to control the trigger functions
 *
 * \param   hCam            valid device handle.
 * \param   nCommand        Specifies the command
 * \param   pParam          input or output storage for the accessed param.
 * \param   cbSizeOfParam   size of *pParam.
 * \return  error code
 */
IDSEXP is_Trigger(HIDS hCam, UINT nCommand, void* pParam, UINT cbSizeOfParam);


#pragma pack(push, 1)

/*!
 * \brief Definition of the uEye device info / heartbeat
 *
 * This data is periodically received from the device.
 *
 * \sa IS_DEVICE_INFO
 * \since uEye SDK 4.00.
 */
typedef struct S_IS_DEVICE_INFO_HEARTBEAT
{
    BYTE    reserved_1[24];             /*!< \brief reserved */

    DWORD   dwRuntimeFirmwareVersion;   /*!< \brief Runtime firmware version */

    BYTE    reserved_2[8];              /*!< \brief reserved */

    WORD    wTemperature;               /*!< \brief camera temperature */
    WORD    wLinkSpeed_Mb;              /*!< \brief current link speed in Mbit/s */

    BYTE    reserved_3[6];              /*!< \brief reserved */

    WORD    wComportOffset;             /*!< \brief comport offset from 100, valid range -99 to +156 */

    BYTE    reserved[200];              /*!< \brief reserved */

} IS_DEVICE_INFO_HEARTBEAT;

/*!
 * \brief Definition of the uEye device info / control
 *
 * This data is provided by the uEye driver.
 *
 * \sa IS_DEVICE_INFO
 * \since uEye SDK 4.00.
 */
typedef struct S_IS_DEVICE_INFO_CONTROL
{
    DWORD   dwDeviceId;     /*!< \brief device id */

    BYTE    reserved[148];  /*!< \brief reserved */

} IS_DEVICE_INFO_CONTROL;

/*!
 * \brief Definition of the uEye device info
 *
 * This data is provided by the uEye driver.
 *
 * \since uEye SDK 4.00.
 */
typedef struct S_IS_DEVICE_INFO
{
    IS_DEVICE_INFO_HEARTBEAT    infoDevHeartbeat;   /*!< \brief device info / heartbeat */

    IS_DEVICE_INFO_CONTROL      infoDevControl;     /*!< \brief device info / control */

    BYTE                        reserved[240];      /*!< \brief reserved */

} IS_DEVICE_INFO;

#pragma pack(pop)

/*!
 * \brief Enumeration of commands supported by the device info access function \ref is_DeviceInfo.
 *
 * \since uEye SDK 4.00.
 */
typedef enum E_IS_DEVICE_INFO_CMD
{
    /*!
     * \brief Get the device info.
     *        Type of data: \ref IS_DEVICE_INFO.
     */
    IS_DEVICE_INFO_CMD_GET_DEVICE_INFO  = 0x02010001,

} IS_DEVICE_INFO_CMD;

/*!
 * \brief Generic interface to the device info functionality.
 *
 * \param   hCam            currently not used / ignored, passing 0 recommended
 * \param   nCommand        code, indicates requested access and accessed parameter,
 *                          one from \ref IS_DEVICE_INFO_CMD enumeration.
 * \param   pParam          input or output storage for the accessed param.
 * \param   cbSizeOfParam   size of *pParam.
 * \return  error code
 *
 * \since uEye SDK 4.00.
 */
IDSEXP is_DeviceInfo(HIDS hCam, UINT nCommand, void* pParam, UINT cbSizeOfParam);


/*! \brief Enumeration of commands supported by the callback functionality access function \ref is_Callback().
 *
 * \sa is_Callback()
 *
 * \since uEye SDK 4.20.
 */
typedef enum E_IS_CALLBACK_CMD
{
    /*!
     * \brief Install a callback.
     *        Type of data: \ref IS_CALLBACK_INSTALLATION_DATA.
     */
    IS_CALLBACK_CMD_INSTALL     = 0x00000001,

    /*!
     * \brief Uninstall a callback.
     *        Type of data: \ref HIDS_CALLBACK.
     */
    IS_CALLBACK_CMD_UNINSTALL   = 0x00000002,

} IS_CALLBACK_CMD;

/*! \brief Enumeration of events supported by the callback functionality access function \ref is_Callback().
 *
 * \sa is_Callback()
 *
 * \since uEye SDK 4.20.
 */
typedef enum E_IS_CALLBACK_EVENT
{
    /*!
     * \brief The callback will be executed before the postprocessing of an acquired image starts.
     *
     * This %Callback event requires objects of types \ref IS_CALLBACK_EVCTX_IMAGE_PROCESSING
     * and \ref IS_CALLBACK_FDBK_IMAGE_PROCESSING to be processed in the callback function.
     *
     * \sa IS_CALLBACK_EVCTX_IMAGE_PROCESSING, IS_CALLBACK_FDBK_IMAGE_PROCESSING
     */
    IS_CALLBACK_EV_IMGPOSTPROC_START = 0x00000001

} IS_CALLBACK_EVENT;

/*! \brief %Callback event context for data processing callbacks.
 *
 * For data processing callbacks a pointer to this data is provided to the %Callback function as the second argument.
 *
 * \sa IS_CALLBACK_FUNC
 *
 * \since uEye SDK 4.20.
 */
typedef struct S_IS_CALLBACK_EVCTX_DATA_PROCESSING
{
    /*! \brief Memory address of the source data. */
    const void* const   pSrcBuf;

    /*! \brief Size of the source data in units number of bytes. */
    const UINT          cbSrcBuf;

    /*! \brief Memory address of the destination buffer. */
    void* const         pDestBuf;

    /*! \brief Size of the destination buffer in units number of bytes. */
    const UINT          cbDestBuf;

} IS_CALLBACK_EVCTX_DATA_PROCESSING;

/*! \brief %Callback event context for image processing callbacks.
 *
 * For image processing callbacks a pointer to this data is provided to the %Callback function as the second argument.
 *
 * This type is used for the following callback events:
 * \li \ref IS_CALLBACK_EV_IMGPOSTPROC_START
 *
 * \sa IS_CALLBACK_FUNC, IS_CALLBACK_EVCTX_DATA_PROCESSING,
 *     IS_CALLBACK_EV_IMGPOSTPROC_START
 *
 * \since uEye SDK 4.20.
 */
typedef struct S_IS_CALLBACK_EVCTX_IMAGE_PROCESSING
{
    /*! brief Indication of the related memory buffers. */
    IS_CALLBACK_EVCTX_DATA_PROCESSING   bufferInfo;

    /*! \todo what else do we need here ?
     * - Memory ID?
     * - Source width?
     * - Source pixelformat?
     * - ...?
     */

} IS_CALLBACK_EVCTX_IMAGE_PROCESSING;

/*! \brief %Callback feedback for image processing callbacks.
 *
 * For image processing callbacks a pointer to this data is provided to the %Callback function as the third argument.
 * The callback function is required to fill provide this information to the API.
 *
 * This type is used for the following callback events:
 * \li IS_CALLBACK_EV_IMGPOSTPROC_START
 *
 * \sa IS_CALLBACK_FUNC,
 *     IS_CALLBACK_EV_IMGPOSTPROC_START
 *
 * \since uEye SDK 4.20.
 */
typedef struct S_IS_CALLBACK_FDBK_IMAGE_PROCESSING
{
    /*! \todo what do we need here ? */
    UINT    nDummy;

} IS_CALLBACK_FDBK_IMAGE_PROCESSING;

/*! \brief %Callback function for the callback funcionality \ref is_Callback().
 *
 * Params: \li User context pointer.
 *         \li Pointer to the %Callback event context.
 *         \li Feedback pointer.
 * Return: Success code: \li 0 for success.
 *                       \li 32-bit positive value for special indications.
 *
 * \attention The range of negative 32-bit values is reserved and must not be used as a return value!
 * \attention It is crucial for the system performance that you return from you callback function
 *            as soon as possible. The callback functions are called in blocking mode!
 *
 * \sa IS_CALLBACK_INSTALLATION_DATA
 *
 * \since uEye SDK 4.20.
 */
typedef INT (__cdecl *IS_CALLBACK_FUNC)(void*, void*, void*);

/*! \brief Handle type for the callback functionality.
 *
 * \sa IS_CALLBACK_INSTALLATION_DATA
 *
 * \since uEye SDK 4.20.
 */
typedef HIDS HIDS_CALLBACK;

/*! \brief %Callback installation data.
 *
 * This data is to be passed to is_Callback for the command \ref IS_CALLBACK_CMD_INSTALL.
 *
 * \since uEye SDK 4.20.
 */
typedef struct S_IS_CALLBACK_INSTALLATION_DATA
{
    /*! \name User input */
    ///@{

    /*! \brief The event to install the callback for. One from \ref IS_CALLBACK_EVENT enumeration. */
    UINT nEvent;

    /*! \brief The callback function to be called on the specified event. */
    IS_CALLBACK_FUNC pfFunc;

    /*! \brief The user context pointer to be passed to the callback function. */
    void* pUserContext;

    ///@}

    /*! \name API output */
    ///@{

    /*! \brief The handle to the installed callback.
     *
     * The handle value is generated by \ref is_Callback().
     *
     * \note Pass this handle value to \ref is_Callback for to uninstall the callback.
     */
    HIDS_CALLBACK hCallback;

    ///@}

} IS_CALLBACK_INSTALLATION_DATA;

/*! \brief Generic interface to the callback functionality.
 *
 * \param   hCam            valid device handle.
 * \param   nCommand        code, indicates requested access, one from \ref IS_CALLBACK_CMD enumeration.
 * \param   pParam          input or output storage for the I/O data.
 * \param   cbSizeOfParam   size of *pParam.
 * \return  error code
 *
 * \since uEye SDK 4.20.
 */
IDSEXP is_Callback(HIDS hCam, UINT nCommand, void* pParam, UINT cbSizeOfParam);



typedef enum E_IS_OPTIMAL_CAMERA_TIMING_CMD
{
    /*!
     * \brief Get optimal camera timing.
     *        Type of data: \ref IS_OPTIMAL_CAMERA_TIMING_CMD_GET_PIXELCLOCK.
     */
    IS_OPTIMAL_CAMERA_TIMING_CMD_GET_PIXELCLOCK = 0x00000001,

    /*!
     * \brief Get optimal camera timing.
     *        Type of data: \ref IS_OPTIMAL_CAMERA_TIMING_CMD_GET_FRAMERATE.
     */
    IS_OPTIMAL_CAMERA_TIMING_CMD_GET_FRAMERATE  = 0x00000002

} IS_OPTIMAL_CAMERA_TIMING_CMD;


/*!
 * \brief I/O structure of the optimal camera parameters
 *
 * \since uEye SDK 4.30.
 */
typedef struct S_IS_OPTIMAL_CAMERA_TIMING
{
    INT     s32Mode;
    INT     s32TimeoutFineTuning;
    INT*    ps32PixelClock;
    double* pdFramerate;

} IS_OPTIMAL_CAMERA_TIMING;


/*! \brief Generic interface to the optimal camera timing functionality.
 *
 * \param   hCam            valid device handle.
 * \param   nCommand        specify the optimal value to be read back (pixelclock and/or framerate).
 * \param   pParam          input or output storage for the I/O data.
 * \param   cbSizeOfParam   size of *pParam.
 * \return  error code
 *
 * \since uEye SDK 4.30.
 */
IDSEXP is_OptimalCameraTiming(HIDS hCam, UINT u32Command, void* pParam, UINT u32SizeOfParam);



// ----------------------------------------------------------------------------
// functions and datatypes only valid for uEye ETH
// ----------------------------------------------------------------------------

#pragma pack(push, 1)

  // IP V4 address
  typedef union _UEYE_ETH_ADDR_IPV4
  {
      struct
      {
          BYTE  by1;
          BYTE  by2;
          BYTE  by3;
          BYTE  by4;
      } by;

      DWORD dwAddr;

  } UEYE_ETH_ADDR_IPV4, *PUEYE_ETH_ADDR_IPV4;

  // Ethernet address
  typedef struct _UEYE_ETH_ADDR_MAC
  {
      BYTE abyOctet[6];

  } UEYE_ETH_ADDR_MAC, *PUEYE_ETH_ADDR_MAC;

  // IP configuration
  typedef struct _UEYE_ETH_IP_CONFIGURATION
  {
      UEYE_ETH_ADDR_IPV4    ipAddress;      /*!< IP address */
      UEYE_ETH_ADDR_IPV4    ipSubnetmask;   /*!< IP subnetmask */

      BYTE                  reserved[4];    /*!< reserved */

  } UEYE_ETH_IP_CONFIGURATION, *PUEYE_ETH_IP_CONFIGURATION;

  // values for UEYE_ETH_DEVICE_INFO_HEARTBEAT::dwDeviceStatusFlags
  typedef enum _UEYE_ETH_DEVICESTATUS
  {
      IS_ETH_DEVSTATUS_READY_TO_OPERATE=            0x00000001, /*!< device is ready to operate */
      IS_ETH_DEVSTATUS_TESTING_IP_CURRENT=          0x00000002, /*!< device is (arp-)probing its current ip */
      IS_ETH_DEVSTATUS_TESTING_IP_PERSISTENT=       0x00000004, /*!< device is (arp-)probing its persistent ip */
      IS_ETH_DEVSTATUS_TESTING_IP_RANGE=            0x00000008, /*!< device is (arp-)probing the autocfg ip range */

      IS_ETH_DEVSTATUS_INAPPLICABLE_IP_CURRENT=     0x00000010, /*!< current ip is inapplicable */
      IS_ETH_DEVSTATUS_INAPPLICABLE_IP_PERSISTENT=  0x00000020, /*!< persistent ip is inapplicable */
      IS_ETH_DEVSTATUS_INAPPLICABLE_IP_RANGE=       0x00000040, /*!< autocfg ip range is inapplicable */

      IS_ETH_DEVSTATUS_UNPAIRED=                    0x00000100, /*!< device is unpaired */
      IS_ETH_DEVSTATUS_PAIRING_IN_PROGRESS=         0x00000200, /*!< device is being paired */
      IS_ETH_DEVSTATUS_PAIRED=                      0x00000400, /*!< device is paired */

      IS_ETH_DEVSTATUS_FORCE_100MBPS=               0x00001000, /*!< device phy is configured to 100 Mbps */
      IS_ETH_DEVSTATUS_NO_COMPORT=                  0x00002000, /*!< device does not support ueye eth comport */

      IS_ETH_DEVSTATUS_RECEIVING_FW_STARTER=        0x00010000, /*!< device is receiving the starter firmware */
      IS_ETH_DEVSTATUS_RECEIVING_FW_RUNTIME=        0x00020000, /*!< device is receiving the runtime firmware */
      IS_ETH_DEVSTATUS_INAPPLICABLE_FW_RUNTIME=     0x00040000, /*!< runtime firmware is inapplicable */
      IS_ETH_DEVSTATUS_INAPPLICABLE_FW_STARTER=     0x00080000, /*!< starter firmware is inapplicable */

      IS_ETH_DEVSTATUS_REBOOTING_FW_RUNTIME=        0x00100000, /*!< device is rebooting to runtime firmware */
      IS_ETH_DEVSTATUS_REBOOTING_FW_STARTER=        0x00200000, /*!< device is rebooting to starter firmware */
      IS_ETH_DEVSTATUS_REBOOTING_FW_FAILSAFE=       0x00400000, /*!< device is rebooting to failsafe firmware */

      IS_ETH_DEVSTATUS_RUNTIME_FW_ERR0=             0x80000000  /*!< checksum error runtime firmware */

  } UEYE_ETH_DEVICESTATUS;


  // heartbeat info transmitted periodically by a device
  // contained in UEYE_ETH_DEVICE_INFO
  typedef struct _UEYE_ETH_DEVICE_INFO_HEARTBEAT
  {
      BYTE                  abySerialNumber[12];        /*!< \brief camera's serial number (string) */

      BYTE                  byDeviceType;               /*!< \brief device type / board type, 0x80 for ETH */
      
      BYTE                  byCameraID;                 /*!< \brief camera id */

      WORD                  wSensorID;                  /*!< \brief camera's sensor's id */

      WORD                  wSizeImgMem_MB;             /*!< \brief size of camera's image memory in MB */

      BYTE                  reserved_1[2];              /*!< \brief reserved */

      DWORD                 dwVerStarterFirmware;       /*!< \brief starter firmware version */

      DWORD                 dwVerRuntimeFirmware;       /*!< \brief runtime firmware version */

      DWORD                 dwStatus;                   /*!< \brief camera status flags */

      BYTE                  reserved_2[4];              /*!< \brief reserved */
      
      WORD                  wTemperature;               /*!< \brief camera temperature */
      
      WORD                  wLinkSpeed_Mb;              /*!< \brief link speed in Mb */

      UEYE_ETH_ADDR_MAC     macDevice;                  /*!< \brief camera's MAC address */
      
      WORD                  wComportOffset;             /*!< \brief comport offset from 100, valid range -99 to +156 */

      UEYE_ETH_IP_CONFIGURATION ipcfgPersistentIpCfg;   /*!< \brief persistent IP configuration */

      UEYE_ETH_IP_CONFIGURATION ipcfgCurrentIpCfg;      /*!< \brief current IP configuration */

      UEYE_ETH_ADDR_MAC     macPairedHost;              /*!< \brief paired host's MAC address */

      BYTE                  reserved_4[2];              /*!< \brief reserved */

      UEYE_ETH_ADDR_IPV4    ipPairedHostIp;             /*!< \brief paired host's IP address */

      UEYE_ETH_ADDR_IPV4    ipAutoCfgIpRangeBegin;      /*!< \brief begin of IP address range */

      UEYE_ETH_ADDR_IPV4    ipAutoCfgIpRangeEnd;        /*!< \brief end of IP address range */

      BYTE                  abyUserSpace[8];            /*!< \brief user space data (first 8 bytes) */

      BYTE                  reserved_5[84];             /*!< \brief reserved */

      BYTE                  reserved_6[64];             /*!< \brief reserved */

  } UEYE_ETH_DEVICE_INFO_HEARTBEAT, *PUEYE_ETH_DEVICE_INFO_HEARTBEAT;

  // values for UEYE_ETH_DEVICE_INFO_CONTROL::dwControlStatus
  typedef enum _UEYE_ETH_CONTROLSTATUS
  {
      IS_ETH_CTRLSTATUS_AVAILABLE =             0x00000001, /*!< device is available TO SELF */
      IS_ETH_CTRLSTATUS_ACCESSIBLE1 =           0x00000002, /*!< device is accessible BY SELF, i.e. directly 'unicastable' */
      IS_ETH_CTRLSTATUS_ACCESSIBLE2 =           0x00000004, /*!< device is accessible BY SELF, i.e. not on persistent ip and adapters ip autocfg range is valid */

      IS_ETH_CTRLSTATUS_PERSISTENT_IP_USED =    0x00000010, /*!< device is running on persistent ip configuration */
      IS_ETH_CTRLSTATUS_COMPATIBLE =            0x00000020, /*!< device is compatible TO SELF */
      IS_ETH_CTRLSTATUS_ADAPTER_ON_DHCP =       0x00000040, /*!< adapter is configured to use dhcp */
      IS_ETH_CTRLSTATUS_ADAPTER_SETUP_OK =      0x00000080, /*!< adapter's setup is ok with respect to uEye needs */

      IS_ETH_CTRLSTATUS_UNPAIRING_IN_PROGRESS = 0x00000100, /*!< device is being unpaired FROM SELF */
      IS_ETH_CTRLSTATUS_PAIRING_IN_PROGRESS =   0x00000200, /*!< device is being paired TO SELF */

      IS_ETH_CTRLSTATUS_PAIRED =                0x00001000, /*!< device is paired TO SELF */
      IS_ETH_CTRLSTATUS_OPENED =                0x00004000, /*!< device is opened BY SELF */

      IS_ETH_CTRLSTATUS_FW_UPLOAD_STARTER =     0x00010000, /*!< device is receiving the starter firmware */
      IS_ETH_CTRLSTATUS_FW_UPLOAD_RUNTIME =     0x00020000, /*!< device is receiving the runtime firmware */

      IS_ETH_CTRLSTATUS_REBOOTING =             0x00100000, /*!< device is rebooting */

      IS_ETH_CTRLSTATUS_BOOTBOOST_ENABLED =     0x01000000, /*!< boot-boosting is enabled for this device */
      IS_ETH_CTRLSTATUS_BOOTBOOST_ACTIVE =      0x02000000, /*!< boot-boosting is active for this device */
      IS_ETH_CTRLSTATUS_INITIALIZED =           0x08000000, /*!< device object is initialized */

      IS_ETH_CTRLSTATUS_TO_BE_DELETED =         0x40000000, /*!< device object is being deleted */
      IS_ETH_CTRLSTATUS_TO_BE_REMOVED =         0x80000000, /*!< device object is being removed */

  } UEYE_ETH_CONTROLSTATUS;

  // control info for a listed device
  // contained in UEYE_ETH_DEVICE_INFO
  typedef struct _UEYE_ETH_DEVICE_INFO_CONTROL
  {
      DWORD     dwDeviceID;         /*!< device's unique id */

      DWORD     dwControlStatus;    /*!< device control status */

      BYTE      reserved_1[80];     /*!< reserved */

      BYTE      reserved_2[64];     /*!< reserved */

  } UEYE_ETH_DEVICE_INFO_CONTROL, *PUEYE_ETH_DEVICE_INFO_CONTROL;

  // Ethernet configuration
  typedef struct _UEYE_ETH_ETHERNET_CONFIGURATION
  {
      UEYE_ETH_IP_CONFIGURATION ipcfg;
      UEYE_ETH_ADDR_MAC         mac;

  } UEYE_ETH_ETHERNET_CONFIGURATION, *PUEYE_ETH_ETHERNET_CONFIGURATION;

  // autocfg ip setup
  typedef struct _UEYE_ETH_AUTOCFG_IP_SETUP
  {
      UEYE_ETH_ADDR_IPV4    ipAutoCfgIpRangeBegin;      /*!< begin of ip address range for devices */
      UEYE_ETH_ADDR_IPV4    ipAutoCfgIpRangeEnd;        /*!< end of ip address range for devices */

      BYTE                  reserved[4];    /*!< reserved */

  } UEYE_ETH_AUTOCFG_IP_SETUP, *PUEYE_ETH_AUTOCFG_IP_SETUP;

  // values for incoming packets filter setup
  typedef enum _UEYE_ETH_PACKETFILTER_SETUP
  {
      // notice: arp and icmp (ping) packets are always passed!

      IS_ETH_PCKTFLT_PASSALL=       0,  /*!< pass all packets to OS */
      IS_ETH_PCKTFLT_BLOCKUEGET=    1,  /*!< block UEGET packets to the OS */
      IS_ETH_PCKTFLT_BLOCKALL=      2   /*!< block all packets to the OS */

  } UEYE_ETH_PACKETFILTER_SETUP;

  // values for link speed setup
  typedef enum _UEYE_ETH_LINKSPEED_SETUP
  {
      IS_ETH_LINKSPEED_100MB=		100,    /*!< 100 MBits */
      IS_ETH_LINKSPEED_1000MB=	    1000    /*!< 1000 MBits */

  } UEYE_ETH_LINKSPEED_SETUP;


  // control info for a device's network adapter
  // contained in UEYE_ETH_DEVICE_INFO
  typedef struct _UEYE_ETH_ADAPTER_INFO
  {
      DWORD                             dwAdapterID;        /*!< adapter's unique id */

      DWORD                             dwDeviceLinkspeed;  /*!< device's linked to this adapter are forced to use this link speed */

      UEYE_ETH_ETHERNET_CONFIGURATION   ethcfg;         /*!< adapter's eth configuration */
      BYTE                              reserved_2[2];  /*!< reserved */
      BOOL                              bIsEnabledDHCP; /*!< adapter's dhcp enabled flag */

      UEYE_ETH_AUTOCFG_IP_SETUP         autoCfgIp;                  /*!< the setup for the ip auto configuration */
      BOOL                              bIsValidAutoCfgIpRange;     /*!<    the given range is valid when: 
                                                                            - begin and end are valid ip addresses
                                                                            - begin and end are in the subnet of the adapter
                                                                            - */

      DWORD                             dwCntDevicesKnown;      /*!< count of listed Known devices */
      DWORD                             dwCntDevicesPaired;     /*!< count of listed Paired devices */

      WORD                              wPacketFilter;          /*!< Setting for the Incoming Packets Filter. see UEYE_ETH_PACKETFILTER_SETUP enum above. */

      BYTE                              reserved_3[38];         /*!< reserved */
      BYTE                              reserved_4[64];         /*!< reserved */

  } UEYE_ETH_ADAPTER_INFO, *PUEYE_ETH_ADAPTER_INFO;

  // driver info
  // contained in UEYE_ETH_DEVICE_INFO
  typedef struct _UEYE_ETH_DRIVER_INFO
  {
      DWORD     dwMinVerStarterFirmware;    /*!< minimum version compatible starter firmware */
      DWORD     dwMaxVerStarterFirmware;    /*!< maximum version compatible starter firmware */

      BYTE      reserved_1[8];              /*!< reserved */
      BYTE      reserved_2[64];             /*!< reserved */

  } UEYE_ETH_DRIVER_INFO, *PUEYE_ETH_DRIVER_INFO;



  // use is_GetEthDeviceInfo() to obtain this data.
  typedef struct _UEYE_ETH_DEVICE_INFO
  {
      UEYE_ETH_DEVICE_INFO_HEARTBEAT    infoDevHeartbeat;

      UEYE_ETH_DEVICE_INFO_CONTROL      infoDevControl;

      UEYE_ETH_ADAPTER_INFO             infoAdapter;

      UEYE_ETH_DRIVER_INFO              infoDriver;

  } UEYE_ETH_DEVICE_INFO, *PUEYE_ETH_DEVICE_INFO;


  typedef struct _UEYE_COMPORT_CONFIGURATION
  {
      WORD wComportNumber;

  } UEYE_COMPORT_CONFIGURATION, *PUEYE_COMPORT_CONFIGURATION;


#pragma pack(pop)

  IDSEXP is_SetStarterFirmware  (HIDS hCam, const CHAR* pcFilepath, UINT uFilepathLen);
  IDSEXP is_SetPacketFilter     (INT iAdapterID, UINT uFilterSetting);
  IDSEXP is_GetComportNumber    (HIDS hCam, UINT *pComportNumber);


/*!
 * \brief Enumeration of IpConfig capability flags.
 *
 * \sa is_IpConfig, IPCONFIG_CMD_QUERY_CAPABILITIES.
 * \since uEye SDK 3.80.
 */
typedef enum E_IPCONFIG_CAPABILITY_FLAGS
{
    /*! \brief Capability flag indicates support of Persistent IP for the device in question. */
    IPCONFIG_CAP_PERSISTENT_IP_SUPPORTED    = 0x01,

    /*! \brief Capability flag indicates support of IP auto configuration for the device in question. */
    IPCONFIG_CAP_AUTOCONFIG_IP_SUPPORTED    = 0x04

} IPCONFIG_CAPABILITY_FLAGS;

/*!
 * \brief Enumeration of commands supported by the IP configuration access function \ref is_IpConfig.
 *
 * \since uEye SDK 3.80.
 */
typedef enum E_IPCONFIG_CMD
{
    /*!
     * \brief   Query the IP configuration capabilities.
     *          Type of value: unsigned int bitmask, 32-bit.
     * \sa IPCONFIG_CAPABILITY_FLAGS.
     */
    IPCONFIG_CMD_QUERY_CAPABILITIES             = 0,

    /*!
     * \brief   Set persistent IP configuration.
     *          Type of value: \ref UEYE_ETH_IP_CONFIGURATION.
     *
     * \note Use camera's device id or camera's MAC address as identifier for \ref is_IpConfig.
     * \note Changing persistent IP configuration enabled status is allowed only if device is not paired.
     */
    IPCONFIG_CMD_SET_PERSISTENT_IP              = 0x01010000,
    /*!
     * \brief   Set IP auto configuration setup.
     *          Type of value: \ref UEYE_ETH_AUTOCFG_IP_SETUP.
     *
     * \note Use NIC's adapter id or NIC's MAC address as identifier for \ref is_IpConfig.
     * \note Changed autoconfig IP setup is applied for each device connected to the addressed NIC
     *          at the next pairing of the device.
     */
    IPCONFIG_CMD_SET_AUTOCONFIG_IP              = 0x01040000,
    /*!
     * \brief   Set IP auto configuration setup by device identification.
     *          Type of value: \ref UEYE_ETH_AUTOCFG_IP_SETUP.
     *
     * \note Use camera's device id or camera's MAC address as identifier for \ref is_IpConfig.
     * \note Changed autoconfig IP setup is applied for each device connected to the addressed NIC
     *          at the next pairing of the device.
     *
     * \since uEye SDK 4.00.
     */
    IPCONFIG_CMD_SET_AUTOCONFIG_IP_BYDEVICE     = 0x01040100,

    /*!
     * \brief   Get persistent IP configuration.
     *          Type of value: \ref UEYE_ETH_IP_CONFIGURATION.
     *
     * \note Use camera's device id or camera's MAC address as identifier for \ref is_IpConfig.
     */
    IPCONFIG_CMD_GET_PERSISTENT_IP              = 0x02010000,
    /*!
     * \brief   Get IP auto configuration setup.
     *          Type of value: \ref UEYE_ETH_AUTOCFG_IP_SETUP.
     *
     * \note Use NIC's adapter id or NIC's MAC address as identifier for \ref is_IpConfig.
     */
    IPCONFIG_CMD_GET_AUTOCONFIG_IP              = 0x02040000,
    /*!
     * \brief   Get IP auto configuration setup by device identification.
     *          Type of value: \ref UEYE_ETH_AUTOCFG_IP_SETUP.
     *
     * \note Use camera's device id or camera's MAC address as identifier for \ref is_IpConfig.
     *
     * \since uEye SDK 4.00.
     */
    IPCONFIG_CMD_GET_AUTOCONFIG_IP_BYDEVICE     = 0x02040100

} IPCONFIG_CMD;

/*!
 * \brief Generic interface to the device's IP configuration.
 * \param   iID             if != -1: id of the addressed camera device / network adapter.
 * \param   mac             if iID == -1: mac of the addressed camera device / network adapter.
 * \param   nCommand        code, indicates requested access and accessed value, \ref IPCONFIG_CMD enumeration.
 * \param   pParam          input or output storage for the accessed param.
 * \param   cbSizeOfParam   size of *pParam.
 * \return  error code
 *
 * \since uEye SDK 3.80.
 */
IDSEXP is_IpConfig(INT iID, UEYE_ETH_ADDR_MAC mac, UINT nCommand, void* pParam, UINT cbSizeOfParam);


/*!
 * \brief Enumeration of defines used by is_Configuration(), \ref is_Configuration.
 */
typedef enum E_CONFIGURATION_SEL
{
    IS_CONFIG_CPU_IDLE_STATES_BIT_AC_VALUE         = 0x01, /*!< Mains power */
    IS_CONFIG_CPU_IDLE_STATES_BIT_DC_VALUE         = 0x02, /*!< Battery power */

    IS_CONFIG_IPO_NOT_ALLOWED                      = 0,
    IS_CONFIG_IPO_ALLOWED                          = 1,

    IS_CONFIG_OPEN_MP_DISABLE                      = 0,
    IS_CONFIG_OPEN_MP_ENABLE                       = 1,

    IS_CONFIG_INITIAL_PARAMETERSET_NONE            = 0,
    IS_CONFIG_INITIAL_PARAMETERSET_1               = 1,
    IS_CONFIG_INITIAL_PARAMETERSET_2               = 2,

    IS_CONFIG_ETH_CONFIGURATION_MODE_OFF           = 0,
    IS_CONFIG_ETH_CONFIGURATION_MODE_ON            = 1

} CONFIGURATION_SEL;

/*!
 * \brief Enumeration of commands of function is_Configuration , \ref is_Configuration.
 */
typedef enum E_CONFIGURATION_CMD
{
    IS_CONFIG_CMD_GET_CAPABILITIES                         = 1, /*!< Get supported configuration capabilities (bitmask of CONFIGURATION_CAPS)*/
    
    IS_CONFIG_CPU_IDLE_STATES_CMD_GET_ENABLE               = 2, /*!< Get the current CPU idle states enable state (bitmask of CONFIGURATION_SEL)*/
    IS_CONFIG_CPU_IDLE_STATES_CMD_SET_DISABLE_ON_OPEN      = 4, /*!< Disable migration to other CPU idle states (other than C0) if the first USB camera is being opened*/
    IS_CONFIG_CPU_IDLE_STATES_CMD_GET_DISABLE_ON_OPEN      = 5, /*!< Get the current setting of the command IS_CPU_IDLE_STATES_CMD_SET_DISABLE_ON_OPEN*/

    IS_CONFIG_OPEN_MP_CMD_GET_ENABLE                       = 6,
    IS_CONFIG_OPEN_MP_CMD_SET_ENABLE                       = 7,
    IS_CONFIG_OPEN_MP_CMD_GET_ENABLE_DEFAULT               = 8,

    IS_CONFIG_INITIAL_PARAMETERSET_CMD_SET                 = 9,
    IS_CONFIG_INITIAL_PARAMETERSET_CMD_GET                 = 10,

    IS_CONFIG_ETH_CONFIGURATION_MODE_CMD_SET_ENABLE        = 11,
    IS_CONFIG_ETH_CONFIGURATION_MODE_CMD_GET_ENABLE        = 12,

    IS_CONFIG_IPO_CMD_GET_ALLOWED                          = 13,
    IS_CONFIG_IPO_CMD_SET_ALLOWED                          = 14

} CONFIGURATION_CMD;

/*!
 * \brief Enumeration of configuration command capability flags, \ref is_Configuration.
 */
typedef enum E_CONFIGURATION_CAPS
{
    IS_CONFIG_CPU_IDLE_STATES_CAP_SUPPORTED                = 0x00000001, /*!< CPU idle state commands are supported by the SDK */
    IS_CONFIG_OPEN_MP_CAP_SUPPORTED                        = 0x00000002, /*!< Open MP commands are supported by the SDK */
    IS_CONFIG_INITIAL_PARAMETERSET_CAP_SUPPORTED           = 0x00000004, /*!< Initial parameter set commands are supported by the SDK */
    IS_CONFIG_IPO_CAP_SUPPORTED                            = 0x00000008  /*!< "Intel Performance Thread" is supported by the SDK */
} CONFIGURATION_CAPS;

/*!
 * \brief Command to set general configuration parameters (e.g. the CPU idle state settings)
 * \param   nCommand        specifies the command, \ref CONFIGURATION_CMD.
 * \param   pParam          input or output storage for the accessed param.
 * \param   cbSizeOfParam   size of *pParam.
 * \return  error code.
 */
IDSEXP is_Configuration(UINT nCommand, void* pParam, UINT cbSizeOfParam);


/*!
 * \brief Structure for flash delay and duration \ref is_IO.
 */
typedef struct S_IO_FLASH_PARAMS
{
    INT s32Delay;
    UINT u32Duration;
} IO_FLASH_PARAMS;


/*!
 * \brief Structure for the PWM params \ref is_IO.
 */
typedef struct S_IO_PWM_PARAMS
{
    double dblFrequency_Hz;
    double dblDutyCycle;
} IO_PWM_PARAMS;


/*!
 * \brief Structure for the configuration params of the GPIOs \ref is_IO.
*/
typedef struct S_IO_GPIO_CONFIGURATION
{
    UINT u32Gpio;
    UINT u32Caps;
    UINT u32Configuration;
    UINT u32State;
    UINT u32Reserved[12];

} IO_GPIO_CONFIGURATION;


/*!
 * \brief Defines used by is_IO(), \ref is_IO.
 */
#define IO_LED_STATE_1                      0
#define IO_LED_STATE_2                      1

#define IO_FLASH_MODE_OFF                   0
#define IO_FLASH_MODE_TRIGGER_LO_ACTIVE     1
#define IO_FLASH_MODE_TRIGGER_HI_ACTIVE     2
#define IO_FLASH_MODE_CONSTANT_HIGH         3
#define IO_FLASH_MODE_CONSTANT_LOW          4
#define IO_FLASH_MODE_FREERUN_LO_ACTIVE     5
#define IO_FLASH_MODE_FREERUN_HI_ACTIVE     6

#define IS_FLASH_MODE_PWM                   0x8000
#define IO_FLASH_MODE_GPIO_1                0x0010
#define IO_FLASH_MODE_GPIO_2                0x0020
#define IO_FLASH_MODE_GPIO_3                0x0040
#define IO_FLASH_MODE_GPIO_4                0x0080
#define IO_FLASH_MODE_GPIO_5                0x0100
#define IO_FLASH_MODE_GPIO_6                0x0200

#define IO_FLASH_GPIO_PORT_MASK             (IO_FLASH_MODE_GPIO_1 | IO_FLASH_MODE_GPIO_2 | IO_FLASH_MODE_GPIO_3 | IO_FLASH_MODE_GPIO_4 | IO_FLASH_MODE_GPIO_5 | IO_FLASH_MODE_GPIO_6)  

#define IO_GPIO_1                           0x0001
#define IO_GPIO_2                           0x0002
#define IO_GPIO_3                           0x0004
#define IO_GPIO_4                           0x0008
#define IO_GPIO_5                           0x0010
#define IO_GPIO_6                           0x0020

#define IS_GPIO_INPUT                       0x0001
#define IS_GPIO_OUTPUT                      0x0002
#define IS_GPIO_FLASH                       0x0004
#define IS_GPIO_PWM                         0x0008
#define IS_GPIO_COMPORT_RX                  0x0010
#define IS_GPIO_COMPORT_TX                  0x0020
#define IS_GPIO_MULTI_INTEGRATION_MODE      0x0040

#define IS_FLASH_AUTO_FREERUN_OFF           0
#define IS_FLASH_AUTO_FREERUN_ON            1

/*!
 * \brief Enumeration of commands of function is_IO , \ref is_IO.
 */
typedef enum E_IO_CMD
{
    IS_IO_CMD_GPIOS_GET_SUPPORTED               = 1,
    IS_IO_CMD_GPIOS_GET_SUPPORTED_INPUTS        = 2,
    IS_IO_CMD_GPIOS_GET_SUPPORTED_OUTPUTS       = 3,
    IS_IO_CMD_GPIOS_GET_DIRECTION               = 4,
    IS_IO_CMD_GPIOS_SET_DIRECTION               = 5,
    IS_IO_CMD_GPIOS_GET_STATE                   = 6,
    IS_IO_CMD_GPIOS_SET_STATE                   = 7,
    IS_IO_CMD_LED_GET_STATE                     = 8,
    IS_IO_CMD_LED_SET_STATE                     = 9,
    IS_IO_CMD_LED_TOGGLE_STATE                  = 10,
    IS_IO_CMD_FLASH_GET_GLOBAL_PARAMS           = 11, 
    IS_IO_CMD_FLASH_APPLY_GLOBAL_PARAMS         = 12,
    IS_IO_CMD_FLASH_GET_SUPPORTED_GPIOS         = 13,
    IS_IO_CMD_FLASH_GET_PARAMS_MIN              = 14,
    IS_IO_CMD_FLASH_GET_PARAMS_MAX              = 15,
    IS_IO_CMD_FLASH_GET_PARAMS_INC              = 16,
    IS_IO_CMD_FLASH_GET_PARAMS                  = 17,
    IS_IO_CMD_FLASH_SET_PARAMS                  = 18,
    IS_IO_CMD_FLASH_GET_MODE                    = 19,
    IS_IO_CMD_FLASH_SET_MODE                    = 20,
    IS_IO_CMD_PWM_GET_SUPPORTED_GPIOS           = 21,    
    IS_IO_CMD_PWM_GET_PARAMS_MIN                = 22,
    IS_IO_CMD_PWM_GET_PARAMS_MAX                = 23,
    IS_IO_CMD_PWM_GET_PARAMS_INC                = 24,
    IS_IO_CMD_PWM_GET_PARAMS                    = 25,
    IS_IO_CMD_PWM_SET_PARAMS                    = 26,
    IS_IO_CMD_PWM_GET_MODE                      = 27,
    IS_IO_CMD_PWM_SET_MODE                      = 28,
    IS_IO_CMD_GPIOS_GET_CONFIGURATION           = 29,
    IS_IO_CMD_GPIOS_SET_CONFIGURATION           = 30,
    IS_IO_CMD_FLASH_GET_GPIO_PARAMS_MIN         = 31,
    IS_IO_CMD_FLASH_SET_GPIO_PARAMS             = 32,
    IS_IO_CMD_FLASH_GET_AUTO_FREERUN_DEFAULT    = 33,
    IS_IO_CMD_FLASH_GET_AUTO_FREERUN            = 34,
    IS_IO_CMD_FLASH_SET_AUTO_FREERUN            = 35
    
} IO_CMD;


/*!
 * \brief Function for all IO features (including flash)
 * \param   nCommand        specifies the command, \ref CONFIGURATION_CMD.
 * \param   pParam          input or output storage for the accessed param.
 * \param   cbSizeOfParam   size of *pParam.
 * \return  error code.
 */
IDSEXP is_IO(HIDS hCam, UINT nCommand, void* pParam, UINT cbSizeOfParam);


/*!
 * \brief Enumeration of commands of function is_AutoParameter, \ref is_AutoParameter.
 */
typedef enum E_AUTOPARAMETER_CMD
{
    IS_AWB_CMD_GET_SUPPORTED_TYPES              = 1,
    IS_AWB_CMD_GET_TYPE                         = 2,
    IS_AWB_CMD_SET_TYPE                         = 3,
    IS_AWB_CMD_GET_ENABLE                       = 4,
    IS_AWB_CMD_SET_ENABLE                       = 5,
    IS_AWB_CMD_GET_SUPPORTED_RGB_COLOR_MODELS   = 6,
    IS_AWB_CMD_GET_RGB_COLOR_MODEL              = 7,
    IS_AWB_CMD_SET_RGB_COLOR_MODEL              = 8
    
} AUTOPARAMETER_CMD;


/*!
 * \brief Defines used by is_AutoParameter, \ref is_AutoParameter.
 */
#define IS_AWB_GREYWORLD                0x0001
#define IS_AWB_COLOR_TEMPERATURE        0x0002

#define IS_AUTOPARAMETER_DISABLE             0
#define IS_AUTOPARAMETER_ENABLE              1
#define IS_AUTOPARAMETER_ENABLE_RUNONCE      2


/*!
 * \brief Function to control all auto parameter settings
 * \param   nCommand        specifies the command, \ref AUTOPARAMETER_CMD.
 * \param   pParam          input or output storage for the accessed param.
 * \param   cbSizeOfParam   size of *pParam.
 * \return  error code.
 */
IDSEXP is_AutoParameter(HIDS hCam, UINT nCommand, void* pParam, UINT cbSizeOfParam);


typedef struct
{
    char* pSourceBuffer;
    char* pDestBuffer;
    INT nDestPixelFormat;
    INT nDestPixelConverter;
    INT nDestGamma;
    INT nDestEdgeEnhancement;
    INT nDestColorCorrectionMode;
    INT nDestSaturationU;
    INT nDestSaturationV;
    BYTE reserved[32];

} BUFFER_CONVERSION_PARAMS;


/*!
 * \brief Enumeration of commands of function is_Convert , \ref is_Convert.
 */
typedef enum E_CONVERT_CMD
{
    IS_CONVERT_CMD_APPLY_PARAMS_AND_CONVERT_BUFFER = 1

} CONVERT_CMD;


/*!
 * \brief Function for different conversions (e.g. raw images to different pixelformats)
 * \param   hCam            valid device handle.
 * \param   nCommand        specifies the command
 * \param   pParam          input or output storage for the accessed param.
 * \param   cbSizeOfParam   size of *pParam.
 * \return  error code.
 */
IDSEXP is_Convert(HIDS hCam, UINT nCommand, void* pParam, UINT cbSizeOfParam);


/*!
 * \brief Enumeration of commands of function is_ParameterSet , \ref is_ParameterSet.
 */
typedef enum E_PARAMETERSET_CMD
{
    IS_PARAMETERSET_CMD_LOAD_EEPROM                         = 1,
    IS_PARAMETERSET_CMD_LOAD_FILE                           = 2,
    IS_PARAMETERSET_CMD_SAVE_EEPROM                         = 3,
    IS_PARAMETERSET_CMD_SAVE_FILE                           = 4,
    IS_PARAMETERSET_CMD_GET_NUMBER_SUPPORTED                = 5,
    IS_PARAMETERSET_CMD_GET_HW_PARAMETERSET_AVAILABLE       = 6,
    IS_PARAMETERSET_CMD_ERASE_HW_PARAMETERSET               = 7
} PARAMETERSET_CMD;

/*!
* \brief Interface to control all parameters settings
* \param   hCam            valid device handle.
* \param   nCommand        Specifies the command
* \param   pParam          input or output storage for the accessed param.
* \param   cbSizeOfParam   size of *pParam.
* \return  error code
*/
IDSEXP is_ParameterSet(HIDS hCam, UINT nCommand, void* pParam, UINT cbSizeOfParam);


/*!
 * \brief Enumeration of commands of function is_EdgeEnhancement , \ref is_EdgeEnhancement.
 */
typedef enum E_EDGE_ENHANCEMENT_CMD
{
    IS_EDGE_ENHANCEMENT_CMD_GET_RANGE   = 1,
    IS_EDGE_ENHANCEMENT_CMD_GET_DEFAULT = 2,
    IS_EDGE_ENHANCEMENT_CMD_GET         = 3,
    IS_EDGE_ENHANCEMENT_CMD_SET         = 4
    
} EDGE_ENHANCEMENT_CMD;


/*!
* \brief Interface to set the edge enhancement
* \param   hCam            valid device handle.
* \param   nCommand        Specifies the command
* \param   pParam          input or output storage for the accessed param.
* \param   cbSizeOfParam   size of *pParam.
* \return  error code
*/
IDSEXP is_EdgeEnhancement(HIDS hCam, UINT nCommand, void* pParam, UINT cbSizeOfParam);


/*!
 * \brief Enumeration of commands of function is_PixelClock , \ref is_PixelClock.
 */
typedef enum E_PIXELCLOCK_CMD
{
    IS_PIXELCLOCK_CMD_GET_NUMBER    = 1,
    IS_PIXELCLOCK_CMD_GET_LIST      = 2,
    IS_PIXELCLOCK_CMD_GET_RANGE     = 3,
    IS_PIXELCLOCK_CMD_GET_DEFAULT   = 4,
    IS_PIXELCLOCK_CMD_GET           = 5, 
    IS_PIXELCLOCK_CMD_SET           = 6

} PIXELCLOCK_CMD;


/*!
* \brief Interface to set the pixel clock
* \param   hCam            valid device handle.
* \param   nCommand        Specifies the command
* \param   pParam          input or output storage for the accessed param.
* \param   cbSizeOfParam   size of *pParam.
* \return  error code
*/
IDSEXP is_PixelClock(HIDS hCam, UINT nCommand, void* pParam, UINT cbSizeOfParam);



typedef struct
{
    wchar_t* pwchFileName;
    UINT     nFileType;
    UINT     nQuality;
    char**   ppcImageMem;
    UINT*    pnImageID;
    BYTE     reserved[32];

} IMAGE_FILE_PARAMS;


/*!
 * \brief Enumeration of commands of function \ref is_ImageFile,
 *
 * \sa is_ImageFile.
 */
typedef enum E_IMAGE_FILE_CMD
{
    IS_IMAGE_FILE_CMD_LOAD    = 1,
    IS_IMAGE_FILE_CMD_SAVE    = 2

} IMAGE_FILE_CMD;


/*!
* \brief Interface for all image file operations
* \param   hCam            valid device handle.
* \param   nCommand        Specifies the command
* \param   pParam          input or output storage for the accessed param.
* \param   cbSizeOfParam   size of *pParam.
* \return  error code
*/
IDSEXP is_ImageFile(HIDS hCam, UINT nCommand, void* pParam, UINT cbSizeOfParam);


/*!
 * \brief Data type for 32-bit signed int value ranges.
 */
typedef struct S_IS_RANGE_S32
{
    INT s32Min;
    INT s32Max;
    INT s32Inc;

} IS_RANGE_S32;

/*!
 * \brief Data type for 64-bit signed double value ranges.
 */
typedef struct S_IS_RANGE_F64
{
    double f64Min;
    double f64Max;
    double f64Inc;

} IS_RANGE_F64;

/*!
 * \brief Enumeration of modes of function \ref is_Blacklevel.
 */
typedef enum E_BLACKLEVEL_MODES
{
    IS_AUTO_BLACKLEVEL_OFF = 0,
    IS_AUTO_BLACKLEVEL_ON  = 1

} BLACKLEVEL_MODES;


/*!
 * \brief Enumeration of caps of function \ref is_Blacklevel.
 */
typedef enum E_BLACKLEVEL_CAPS
{
    IS_BLACKLEVEL_CAP_SET_AUTO_BLACKLEVEL   = 1,
    IS_BLACKLEVEL_CAP_SET_OFFSET            = 2,
        
} BLACKLEVEL_CAPS;

/*!
 * \brief Enumeration of commands of function \ref is_Blacklevel.
 */
typedef enum E_BLACKLEVEL_CMD
{
    IS_BLACKLEVEL_CMD_GET_CAPS           = 1,
    IS_BLACKLEVEL_CMD_GET_MODE_DEFAULT   = 2,
    IS_BLACKLEVEL_CMD_GET_MODE           = 3,
    IS_BLACKLEVEL_CMD_SET_MODE           = 4,
    IS_BLACKLEVEL_CMD_GET_OFFSET_DEFAULT = 5,
    IS_BLACKLEVEL_CMD_GET_OFFSET_RANGE   = 6,
    IS_BLACKLEVEL_CMD_GET_OFFSET         = 7,
    IS_BLACKLEVEL_CMD_SET_OFFSET         = 8

} BLACKLEVEL_CMD;

/*!
* \brief Interface for all black level operations
* \param   hCam            valid device handle.
* \param   nCommand        Specifies the command
* \param   pParam          input or output storage for the accessed param.
* \param   cbSizeOfParam   size of *pParam.
* \return  error code
*/
IDSEXP is_Blacklevel(HIDS hCam, UINT nCommand, void* pParam, UINT cbSizeOfParam);


/*!
 * \brief Enumeration of commands of function \ref is_ImageBuffer.
 */
typedef enum E_IMGBUF_CMD
{
    IS_IMGBUF_DEVMEM_CMD_GET_AVAILABLE_ITERATIONS      = 1,
    IS_IMGBUF_DEVMEM_CMD_GET_ITERATION_INFO            = 2,
    IS_IMGBUF_DEVMEM_CMD_TRANSFER_IMAGE                = 3,
    IS_IMGBUF_DEVMEM_CMD_RELEASE_ITERATIONS            = 4

} IMGBUF_CMD;


/*!
 * \brief Data type for ID ranges of iterations or image IDs 
 */
typedef struct S_ID_RANGE
{
    INT s32First;
    INT s32Last;

} ID_RANGE;


/*!
 * \brief Info structure for capture iterations.
 */
typedef struct S_IMGBUF_ITERATION_INFO
{
    UINT u32IterationID;
    ID_RANGE rangeImageID;
    BYTE bReserved[52];

} IMGBUF_ITERATION_INFO;


/*!
 * \brief Structure to specify concrete image
 */
typedef struct S_IMGBUF_ITEM
{
    UINT u32IterationID;
    INT  s32ImageID;

} IMGBUF_ITEM;


/*!
* \brief Interface for all ImageBuffer operations
* \param   hCam            valid device handle.
* \param   nCommand        Specifies the command
* \param   pParam          input or output storage for the accessed param.
* \param   cbSizeOfParam   size of *pParam.
* \return  error code
*/
IDSEXP is_ImageBuffer(HIDS hCam, UINT nCommand, void* pParam, UINT cbSizeOfParam);


/*!
 * \brief Info structure about the calculated sharpness value from the function \ref is_Measure.
*/
typedef struct S_MEASURE_SHARPNESS_AOI_INFO
{
    UINT    u32NumberAOI;
    UINT    u32SharpnessValue;
    IS_RECT rcAOI;

} MEASURE_SHARPNESS_AOI_INFO;

/*!
 * \brief Enumeration of commands of function \ref is_Measure.
*/
typedef enum E_MEASURE_CMD
{
    IS_MEASURE_CMD_SHARPNESS_AOI_SET		= 1,
    IS_MEASURE_CMD_SHARPNESS_AOI_INQUIRE	= 2,
	IS_MEASURE_CMD_SHARPNESS_AOI_SET_PRESET	= 3

} MEASURE_CMD;

/*!
 * \brief sharpness AOI presets for the function \ref is_Measure.
*/
typedef enum E_MEASURE_SHARPNESS_AOI_PRESETS
{
	IS_MEASURE_SHARPNESS_AOI_PRESET_1 = 1

} MEASURE_SHARPNESS_AOI_PRESETS;

/*!
* \brief Interface for all measure operations
* \param   hCam            valid device handle.
* \param   nCommand        Specifies the command
* \param   pParam          input or output storage for the accessed param.
* \param   cbSizeOfParam   size of *pParam.
* \return  error code
*/
IDSEXP is_Measure(HIDS hCam, UINT nCommand, void* pParam, UINT cbSizeOfParam);


/*!
 * \brief Defines for the is_LUT() function
*/
#define IS_LUT_64                      64
#define IS_LUT_128                     128

#define IS_LUT_PRESET_ID_IDENTITY      0
#define IS_LUT_PRESET_ID_NEGATIVE      1
#define IS_LUT_PRESET_ID_GLOW1         2
#define IS_LUT_PRESET_ID_GLOW2         3
#define IS_LUT_PRESET_ID_ASTRO1        4
#define IS_LUT_PRESET_ID_RAINBOW1      5
#define IS_LUT_PRESET_ID_MAP1          6
#define IS_LUT_PRESET_ID_HOT           7
#define IS_LUT_PRESET_ID_SEPIC         8
#define IS_LUT_PRESET_ID_ONLY_RED      9
#define IS_LUT_PRESET_ID_ONLY_GREEN    10
#define IS_LUT_PRESET_ID_ONLY_BLUE     11

typedef INT IS_LUT_PRESET;

/*!
 * \brief Structure with LUT values \ref is_LUT.
*/
typedef struct
{
    double  dblValues[3][IS_LUT_64];
    BOOL    bAllChannelsAreEqual;

} IS_LUT_CONFIGURATION_64;

/*!
 * \brief Structure for LUT presets \ref is_LUT.
*/
typedef struct
{
    IS_LUT_PRESET               predefinedLutID;
    IS_LUT_CONFIGURATION_64     lutConfiguration;

} IS_LUT_CONFIGURATION_PRESET_64;

/*!
 * \brief Commands for the is_LUT function \ref is_LUT.
*/
#define IS_LUT_CMD_SET_ENABLED                  0x0001
#define IS_LUT_CMD_SET_MODE                     0x0002
#define IS_LUT_CMD_GET_STATE                    0x0005
#define IS_LUT_CMD_GET_SUPPORT_INFO             0x0006
#define IS_LUT_CMD_SET_USER_LUT                 0x0010
#define IS_LUT_CMD_GET_USER_LUT                 0x0011
#define IS_LUT_CMD_GET_COMPLETE_LUT             0x0012
#define IS_LUT_CMD_GET_PRESET_LUT               0x0013
#define IS_LUT_CMD_LOAD_FILE                    0x0100
#define IS_LUT_CMD_SAVE_FILE                    0x0101

/*!
 * \brief Defines for the state of the LUT \ref is_LUT.
*/
#define IS_LUT_STATE_ID_FLAG_HARDWARE           0x0010
#define IS_LUT_STATE_ID_FLAG_SOFTWARE           0x0020
#define IS_LUT_STATE_ID_FLAG_GAMMA              0x0100
#define IS_LUT_STATE_ID_FLAG_LUT                0x0200

#define IS_LUT_STATE_ID_INACTIVE                0x0000
#define IS_LUT_STATE_ID_NOT_SUPPORTED           0x0001
#define IS_LUT_STATE_ID_HARDWARE_LUT            (IS_LUT_STATE_ID_FLAG_HARDWARE | IS_LUT_STATE_ID_FLAG_LUT)
#define IS_LUT_STATE_ID_HARDWARE_GAMMA          (IS_LUT_STATE_ID_FLAG_HARDWARE | IS_LUT_STATE_ID_FLAG_GAMMA)
#define IS_LUT_STATE_ID_HARDWARE_LUTANDGAMMA    (IS_LUT_STATE_ID_FLAG_HARDWARE | IS_LUT_STATE_ID_FLAG_LUT | IS_LUT_STATE_ID_FLAG_GAMMA)
#define IS_LUT_STATE_ID_SOFTWARE_LUT            (IS_LUT_STATE_ID_FLAG_SOFTWARE | IS_LUT_STATE_ID_FLAG_LUT)
#define IS_LUT_STATE_ID_SOFTWARE_GAMMA          (IS_LUT_STATE_ID_FLAG_SOFTWARE | IS_LUT_STATE_ID_FLAG_GAMMA)
#define IS_LUT_STATE_ID_SOFTWARE_LUTANDGAMMA    (IS_LUT_STATE_ID_FLAG_SOFTWARE | IS_LUT_STATE_ID_FLAG_LUT | IS_LUT_STATE_ID_FLAG_GAMMA)

#define IS_LUT_MODE_ID_DEFAULT                  0
#define IS_LUT_MODE_ID_FORCE_HARDWARE           1
#define IS_LUT_MODE_ID_FORCE_SOFTWARE           2

#define IS_LUT_DISABLED                         0
#define IS_LUT_ENABLED                          1

typedef INT IS_LUT_ENABLED_STATE;
typedef INT IS_LUT_MODE;

/*!
 * \brief Structure for the state of the LUT \ref is_LUT.
*/
typedef struct
{
    /* use userdefined lut (or only gamma?) */
    BOOL bLUTEnabled;

    /* lut state id */
    INT nLUTStateID;
    
    /* lut mode */
    INT nLUTModeID;

    /* lut bit depth */
    INT nLUTBits;

} IS_LUT_STATE;

/*!
 * \brief Structure with infos about the LUT settings \ref is_LUT.
*/
typedef struct
{
    /* supported lut features in current color/conversion mode */
    BOOL bSupportLUTHardware;
    BOOL bSupportLUTSoftware;

    INT nBitsHardware;
    INT nBitsSoftware;

    /* supported lut channels */
    INT nChannelsHardware;
    INT nChannelsSoftware;

} IS_LUT_SUPPORT_INFO;

/*!
* \brief Function to control the LUT
* \param   hCam            valid device handle.
* \param   nCommand        Specifies the command
* \param   pParam          input or output storage for the accessed param.
* \param   cbSizeOfParam   size of *pParam.
* \return  error code
*/
IDSEXP is_LUT(HIDS hCam, UINT nCommand, void* pParam, UINT cbSizeOfParams);


/*!
 * \brief Defines for the is_Gamma function \ref is_Gamma.
*/
#define IS_GAMMA_CMD_SET            0x0001
#define IS_GAMMA_CMD_GET_DEFAULT    0x0002
#define IS_GAMMA_CMD_GET            0x0003

#define IS_GAMMA_VALUE_MIN          1
#define IS_GAMMA_VALUE_MAX          1000

/*!
* \brief Function to set/get gamma
* \param   hCam            valid device handle.
* \param   nCommand        Specifies the command
* \param   pParam          input or output storage for the accessed param.
* \param   cbSizeOfParam   size of *pParam.
* \return  error code
*/
IDSEXP is_Gamma(HIDS hCam, UINT nCommand, void* pParam, UINT cbSizeOfParams);


typedef enum E_IS_MEMORY_CMD
{
    IS_MEMORY_GET_SIZE  = 1,
    IS_MEMORY_READ      = 2,
    IS_MEMORY_WRITE     = 3

} IS_MEMORY_CMD;


typedef enum E_IS_MEMORY_DESCRIPTION
{
    IS_MEMORY_USER_1    = 1,
    IS_MEMORY_USER_2    = 2

} IS_MEMORY_DESCRIPTION;


typedef struct 
{
    UINT u32Description;
    UINT u32Offset;
    unsigned char* pu8Data;
    UINT u32SizeOfData;

} IS_MEMORY_ACCESS;


typedef struct
{
    UINT u32Description;
    UINT u32SizeBytes;

} IS_MEMORY_SIZE;


/*! 
 * \brief Generic function to access memory.
 * 
 * \param hf            Valid camera handle.
 * \param u32Command    The read / write command.
 * \param pParam        I/O parameter, depends on the command.
 * \param cbParam       Size of *pParam.
 * \return Status of the execution.
 */
IDSEXP is_Memory(HIDS hf, UINT nCommand, void* pParam, UINT cbSizeOfParam);


#ifdef __cplusplus
};
#endif  /* __cplusplus */

#pragma pack(pop)

#endif  // #ifndef __IDS_HEADER__
