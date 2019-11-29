Attribute VB_Name = "PixeLINKApi"
Option Explicit
'/****************************************************************************************************************
' * COPYRIGHT © 2009 PixeLINK CORPORATION.  ALL RIGHTS RESERVED.                                                 *
' *                                                                                                              *
' * Copyright Notice and Disclaimer of Liability:                                                                *
' *                                                                                                              *
' *                                                                                                              *
' * PixeLINK Corporation is henceforth referred to as PixeLINK or PixeLINK Corporation.                          *
' * Purchaser henceforth refers to the original purchaser(s) of the equipment, and/or any legitimate user(s).    *
' *                                                                                                              *
' * PixeLINK hereby explicitly prohibits any form of reproduction (with the express strict exception for backup  *
' * and archival purposes, which are allowed as stipulated within the License Agreement for PixeLINK Corporation *
' * Software), modification, and/or distribution of this software and/or its associated documentation unless     *
' * explicitly specified in a written agreement signed by both parties.                                          *
' *                                                                                                              *
' * To the extent permitted by law, PixeLINK disclaims all other warranties or conditions of any kind, either    *
' * express or implied, including but not limited to all warranties or conditions of merchantability and         *
' * fitness for a particular purpose and those arising by statute or otherwise in law or from a course of        *
' * dealing or usage of trade. Other written or oral statements by PixeLINK, its representatives, or others do   *
' * not constitute warranties or conditions of PixeLINK.                                                         *
' *                                                                                                              *
' * PixeLINK makes no guarantees or representations, including but not limited to: the use of, or the result(s)  *
' * of the use of: the software and associated documentation in terms of correctness, accuracy, reliability,     *
' * being current, or otherwise. The Purchaser hereby agree to rely on the software, associated hardware and     *
' * documentation and results stemming from the use thereof solely at their own risk.                            *
' *                                                                                                              *
' * By using the products(s) associated with the software, and/or the software, the Purchaser and/or user(s)     *
' * agree(s) to abide by the terms and conditions set forth within this document, as well as, respectively,      *
' * any and all other documents issued by PixeLINK in relation to the product(s).                                *
' *                                                                                                              *
' * PixeLINK is hereby absolved of any and all liability to the Purchaser, and/or a third party, financial or    *
' * otherwise, arising from any subsequent loss, direct and indirect, and damage, direct and indirect,           *
' * resulting from intended and/or unintended usage of its software, product(s) and documentation, as well       *
' * as additional service(s) rendered by PixeLINK, such as technical support, unless otherwise explicitly        *
' * specified in a written agreement signed by both parties. Under no circumstances shall the terms and          *
' * conditions of such an agreement apply retroactively.                                                         *
' *                                                                                                              *
' ****************************************************************************************************************/
' ------------------------------------------------------------------------
'
'    PixeLINKAPI.bas -- PixeLINK(tm) API Declarations for Visual Basic
'
'       Needs to be installed in
'         <PROGRAM FILES>\Microsoft Visual Studio\Common\Tools\Winapi
' ------------------------------------------------------------------------
 
'------------------------------------------------------------------------
'
'    For safety, the Following Constants  and Types should be obtained from the system
'
'    Make the Api Viewer available
'        Add-Ins -> Add-In Manager -> VB6 Api Viewer
'
'    Add in the Constants - Note that they must be in order:
'       Add-Ins -> API Viewer
'         File -> Load Text File
'           <PROGRAM FILES>\Microsoft Visual Studio\Common\Tools\Winapi\Win32Api.txt
'
'           API Type    - Constants
'
'           API Type    - Types
' ------------------------------------------------------------------------
Public Const WS_CHILD = &H40000000
Public Const WS_VISIBLE = &H10000000
Public Const WS_CAPTION = &HC00000                  '  WS_BORDER Or WS_DLGFRAME
Public Const WS_SYSMENU = &H80000
Public Const WS_THICKFRAME = &H40000
Public Const WS_MAXIMIZEBOX = &H10000
Public Const WS_MINIMIZEBOX = &H20000
Public Const WS_OVERLAPPED = &H0
Public Const WS_OVERLAPPEDWINDOW = (WS_OVERLAPPED Or WS_CAPTION Or WS_SYSMENU Or WS_THICKFRAME Or WS_MINIMIZEBOX Or WS_MAXIMIZEBOX)
Public Const CW_USEDEFAULT = &H80000000

 
'**************************
'   CONSTANT Declarations
'**************************
'
Public Const ApiSuccess = &H0                            ' The function completed successfully
Public Const ApiSuccessParametersChanged = &H1           ' Indicates that a set feature is successful but one or more parameter had to be changed (ROI)
Public Const ApiSuccessAlreadyRunning = &H2              ' The stream is already started
Public Const ApiSuccessLowMemory = &H3                   ' There is not as much memory as needed for optimum performance. Performance may be affected.
Public Const ApiSuccessParameterWarning = &H4            ' The operation completed succesfully, but one or more of the parameters are suspect (such as using an invalid IP address)
Public Const ApiSuccessReducedSpeedWarning = &H5         ' The operation completed sucessfully, but the camera is operating at a speed less than it's maximum
Public Const ApiSuccessExposureAdjustmentMade = &H6      ' The operation completed sucessfully, but the camera had to adjust exposure to accomodate
Public Const ApiSuccessWhiteBalanceTooDark = &H7         ' The Auto WhiteBalance algorithm could not achieve good results because the auto-adjust area is too dark
Public Const ApiSuccessWhiteBalanceTooBright = &H8       ' The Auto WhiteBalance algorithm could not achieve good results because the auto-adjust area has too many saturated pxels
Public Const ApiSuccessWithFrameLoss = &H9               ' The operation completed sucessfully, but some frame loss occurred (the system could not keep pace with the camera


Public Const ApiUnknownError = &H80000001                ' Unknown error
Public Const ApiInvalidHandleError = &H80000002          ' The handle parameter invalid
Public Const ApiInvalidParameterError = &H80000003       ' Invalid parameter
Public Const ApiBufferTooSmall = &H80000004              ' A buffer passed as parameter is too small.
Public Const ApiInvalidFunctionCallError = &H80000005    ' The function cannot be called at this time
Public Const ApiNotSupportedError = &H80000006           ' The API cannot complete the request
Public Const ApiCameraInUseError = &H80000007            ' The camera is already being used by another application
Public Const ApiNoCameraError = &H80000008               ' There is no response from the camera
Public Const ApiHardwareError = &H80000009               ' The Camera responded with an error
Public Const ApiCameraUnknownError = &H8000000A          ' The API does not recognize the camera
Public Const ApiOutOfBandwidthError = &H8000000B         ' There is not enough 1394 bandwidth to start the stream
Public Const ApiOutOfMemoryError = &H8000000C            ' The API can not allocate the required memory
Public Const ApiOSVersionError = &H8000000D              ' The API cannot run on the current operating system
Public Const ApiNoSerialNumberError = &H8000000E         ' The serial number coundn't be obtained from the camera
Public Const ApiInvalidSerialNumberError = &H8000000F    ' A camera with that serial number coundn't be found
Public Const ApiDiskFullError = &H80000010               ' Not enough disk space to complete an IO operation
Public Const ApiIOError = &H80000011                     ' An error occurred during an IO operation
Public Const ApiStreamStopped = &H80000012               ' Application requested streaming termination
Public Const ApiNullPointerError = &H80000013            ' The pointer parameter = NULL
Public Const ApiCreatePreviewWndError = &H80000014       ' Error creating the preview window
Public Const ApiOutOfRangeError = &H80000016             ' Indicates that a feature set value is out of range
Public Const ApiNoCameraAvailableError = &H80000017      ' There is no camera available
Public Const ApiInvalidCameraName = &H80000018           ' Indicated that the name specified is not a valid camera name
Public Const ApiGetNextFrameBusy = &H80000019            ' GextNextFrame() can't be called at this time because its being use by an FRAME_OVERLAY callback function
Public Const ApiFrameInUseError = &H8000001A             ' A frame was still in use when the buffers were deallocated.
  
Public Const ApiStreamExistingError = &H90000001
Public Const ApiEnumDoneError = &H90000002
Public Const ApiNotEnoughResourcesError = &H90000003
Public Const ApiBadFrameSizeError = &H90000004
Public Const ApiNoStreamError = &H90000005
Public Const ApiVersionError = &H90000006
Public Const ApiNoDeviceError = &H90000007
Public Const ApiCannotMapFrameError = &H90000008
Public Const ApiOhciDriverError = &H90000009
Public Const ApiInvalidIoctlParameter = &H9000000A
Public Const ApiInvalidOhciDriverError = &H9000000B
Public Const ApiCameraTimeoutError = &H9000000C
Public Const ApiInvalidFrameReceivedError = &H9000000D
Public Const ApiOSServiceError = &H9000000E
Public Const ApiTimeoutError = &H9000000F
Public Const ApiRequiresControlAccess = &H90000010
Public Const ApiGevInitializationError = &H90000011
Public Const ApiIpServicesError = &H90000012
Public Const ApiIpAddressingError = &H90000013
Public Const ApiDriverCommunicationError = &H90000014
Public Const ApiInvalidXmlError = &H90000015
Public Const ApiCameraRejectedValueError = &H90000016
Public Const ApiSuspectedFirewallBlockError = &H90000017
Public Const ApiIncorrectLinkSpeed = &H90000018
Public Const ApiCameraNotReady = &H90000019
Public Const ApiInconsistentConfiguration = &H9000001A
Public Const ApiNotPermittedWhileStreaming = &H9000001B
Public Const ApiOSAccessDeniedError = &H9000001C
Public Const ApiInvalidAutoRoiError = &H9000001D
Public Const ApiGpiHardwareTriggerConflict = &H9000001E
Public Const ApiGpioConfigurationError = &H9000001F
Public Const ApiUnsupportedPixelFormatError = &H90000020
Public Const ApiUnsupportedClipEncoding = &H90000021
Public Const ApiH264EncodingError = &H90000022
Public Const ApiH264FrameTooLargeError = &H90000023
'
' Feature IDs
'
Public Const FEATURE_BRIGHTNESS = 0
Public Const FEATURE_PIXELINK_RESERVED_1 = 1
Public Const FEATURE_SHARPNESS = 2
Public Const FEATURE_WHITE_BAL = 3
Public Const FEATURE_HUE = 4
Public Const FEATURE_SATURATION = 5
Public Const FEATURE_GAMMA = 6
Public Const FEATURE_SHUTTER = 7
Public Const FEATURE_GAIN = 8
Public Const FEATURE_IRIS = 9
Public Const FEATURE_FOCUS = 10
Public Const FEATURE_TEMPERATURE = 11
Public Const FEATURE_TRIGGER = 12
Public Const FEATURE_ZOOM = 13
Public Const FEATURE_PAN = 14
Public Const FEATURE_TILT = 15
Public Const FEATURE_OPT_FILTER = 16
Public Const FEATURE_GPIO = 17
Public Const FEATURE_FRAME_RATE = 18
Public Const FEATURE_ROI = 19
Public Const FEATURE_FLIP = 20
Public Const FEATURE_PIXEL_ADDRESSING = 21
Public Const FEATURE_PIXEL_FORMAT = 22
Public Const FEATURE_EXTENDED_SHUTTER = 23
Public Const FEATURE_AUTO_ROI = 24
Public Const FEATURE_LOOKUP_TABLE = 25
Public Const FEATURE_MEMORY_CHANNEL = 26
Public Const FEATURE_WHITE_SHADING = 27
Public Const FEATURE_ROTATE = 28
Public Const FEATURE_IMAGER_CLK_DIVISOR = 29
Public Const FEATURE_TRIGGER_WITH_CONTROLLED_LIGHT = 30
Public Const FEATURE_MAX_PIXEL_SIZE = 31
Public Const FEATURE_BODY_TEMPERATURE = 32
Public Const FEATURE_MAX_PACKET_SIZE = 33
Public Const FEATURE_BANDWIDTH_LIMIT = 34
Public Const FEATURE_ACTUAL_FRAME_RATE = 35
Public Const FEATURE_SHARPNESS_SCORE = 36
Public Const FEATURE_SPECIAL_CAMERA_MODE = 37
Public Const FEATURES_TOTAL = 38

' For PxLGetCameraFeatures
Public Const FEATURE_ALL = &HFFFFFFFF

' Feature aliases
Public Const FEATURE_EXPOSURE = FEATURE_SHUTTER '// IIDC feature 'exposure' is equivalent to feature 'shutter'
Public Const FEATURE_DECIMATION = FEATURE_PIXEL_ADDRESSING '// Decimation is just one kind of pixel addressing mode

' Video clip Stream Format
Public Const CLIP_STREAM_PDS = 0
Public Const CLIP_STREAM_H264 = 1
' Video Clip File Format
Public Const CLIP_FORMAT_AVI = 0
Public Const CLIP_FORMAT_MP4 = 1

' Feature Flags
Public Const FEATURE_FLAG_PRESENCE = &H1
Public Const FEATURE_FLAG_MANUAL = &H2
Public Const FEATURE_FLAG_AUTO = &H4
Public Const FEATURE_FLAG_ONEPUSH = &H8
Public Const FEATURE_FLAG_OFF = &H10
Public Const FEATURE_FLAG_DESC_SUPPORTED = &H20
Public Const FEATURE_FLAG_READ_ONLY = &H40
Public Const FEATURE_FLAG_SETTABLE_WHILE_STREAMING = &H80
Public Const FEATURE_FLAG_PERSISTABLE = &H100
Public Const FEATURE_FLAG_EMULATION = &H200
Public Const FEATURE_FLAG_VOLATILE = &H400


        
' Image File Format
Public Const IMAGE_FORMAT_BMP = 0
Public Const IMAGE_FORMAT_TIFF = 1
Public Const IMAGE_FORMAT_PSD = 2
Public Const IMAGE_FORMAT_JPEG = 3

' Pixel Format
Public Const PIXEL_FORMAT_MONO8 = 0
Public Const PIXEL_FORMAT_MONO16 = 1
Public Const PIXEL_FORMAT_YUV422 = 2
Public Const PIXEL_FORMAT_BAYER8_GRBG = 3
Public Const PIXEL_FORMAT_BAYER16_GRBG = 4
Public Const PIXEL_FORMAT_RGB24 = 5
Public Const PIXEL_FORMAT_RGB48 = 6
Public Const PIXEL_FORMAT_BAYER8_RGGB = 7
Public Const PIXEL_FORMAT_BAYER8_GBRG = 8
Public Const PIXEL_FORMAT_BAYER8_BGGR = 9
Public Const PIXEL_FORMAT_BAYER16_RGGB = 10
Public Const PIXEL_FORMAT_BAYER16_GBRG = 11
Public Const PIXEL_FORMAT_BAYER16_BGGR = 12
Public Const PIXEL_FORMAT_BAYER8 = PIXEL_FORMAT_BAYER8_GRBG
Public Const PIXEL_FORMAT_BAYER16 = PIXEL_FORMAT_BAYER16_GRBG
Public Const PIXEL_FORMAT_MONO12_PACKED = 13
Public Const PIXEL_FORMAT_BAYER12_GRBG_PACKED = 14
Public Const PIXEL_FORMAT_BAYER12_RGGB_PACKED = 15
Public Const PIXEL_FORMAT_BAYER12_GBRG_PACKED = 16
Public Const PIXEL_FORMAT_BAYER12_BGGR_PACKED = 17
Public Const PIXEL_FORMAT_BAYER12_PACKED = PIXEL_FORMAT_BAYER12_GRBG_PACKED

' Preview State
Public Const START_PREVIEW = 0
Public Const PAUSE_PREVIEW = 1
Public Const STOP_PREVIEW = 2
        
' Stream State
Public Const START_STREAM = 0
Public Const PAUSE_STREAM = 1
Public Const STOP_STREAM = 2
        
' Trigger types
Public Const TRIGGER_TYPE_FREE_RUNNING = 0
Public Const TRIGGER_TYPE_SOFTWARE = 1
Public Const TRIGGER_TYPE_HARDWARE = 2

' Descriptors
Public Const PXL_MAX_STROBES = 16
Public Const PXL_MAX_KNEE_POINTS = 4

' Descriptors (advanced features)
Public Const PXL_UPDATE_CAMERA = 0
Public Const PXL_UPDATE_HOST = 1
 
' Camera Features
Public Type FEATURE_PARAM
    MinValue As Single
    MaxValue As Single
End Type

Public Type CAMERA_FEATURE
    FeatureId As UInteger
    Flags As UInteger
    NumberOfParameters As UInteger
    pParams As UInteger ' FEATURE_PARAM pointer
End Type

Public Type CAMERA_FEATURES
    Size As UInteger
    NumberOfFeatures As UInteger
    pFeatures As UInteger  ' CAMERA_FEATURE pointer
End Type

Public Const MAX_CAMERA_PARAM = 6

Public Type SINGLE_CAMERA_FEATURE
    FeatureId As UInteger
    Flags As UInteger
    NumberOfParameters As UInteger
    Params(1 To MAX_CAMERA_PARAM) As FEATURE_PARAM
End Type

'Camera Info
Public Type CAMERA_INFO
    VendorName As String * 33
    ModelName As String * 33
    Description As String * 256
    SerialNumber As String * 33
    FirmwareVersion As String * 12
    FPGAVersion As String * 12
    CameraName As String * 256
End Type

' Media Access Control (MAC) Address
Public Type PXL_MAC_ADDRESS
    MacAddr(1 To 6) As Byte
End Type

' Internet Procotol (IP) Address
Public Type PXL_IP_ADDRESS
    U8Address(1 To 4) As Byte
End Type

' Camera Id Info
' For use with PxLGetNumberCamerasEx
Public Type CAMERA_ID_INFO
    StructSize As UInteger
    CameraSerialNum As UInteger
    CameraMac As PXL_MAC_ADDRESS
    ' Padding necessary to match padding in the C struct
    padding1(1 To 2) As Byte
    CameraIpAddress As PXL_IP_ADDRESS
    CameraIPMask As PXL_IP_ADDRESS
    CameraIpGateway As PXL_IP_ADDRESS
    NicIpAddress As PXL_IP_ADDRESS
    NicIpMask As PXL_IP_ADDRESS
    NicAccessMode As UInteger
    CameraIpAssignmentType as Byte
    XmlVersionMajor as Byte
    XmlVersionMinor as Byte
    XmlVersionSubminor as Byte
    IpEngineLoadVersionMajor as Byte
    IpEngineLoadVersionMinor as Byte
    IpEngineLoadVersionSubminor as Byte
    CameraProperties as Byte
    ControllingIpAddress as PXL_IP_ADDRESS
End Type

' CameraIpAssignmentTypes
Public Const PXL_IP_UNKNOWN_ASSIGNMENT = &H0
Public Const PXL_IP_DHCP_ASSIGNED = &H1
Public Const PXL_IP_LLA_ASSIGNED = &H2
Public Const PXL_IP_STATIC_PERSISTENT = &H3
Public Const PXL_IP_STATIC_VOLATILE = &H4

' Overlay usage
Public Const OVERLAY_PREVIEW = &H1
Public Const OVERLAY_FORMAT_IMAGE = &H2
Public Const OVERLAY_FORMAT_CLIP = &H4
Public Const OVERLAY_FRAME = &H8

' Frame Descriptor Definition
Public Type BRIGHTNESS_
    Value As Single
End Type

Public Type AUTOEXPOSURE_
    Value As Single
End Type

Public Type SHARPNESS_
    Value As Single
End Type

Public Type WHITEBALANCE_
    Value As Single
End Type

Public Type HUE_
    Value As Single
End Type

Public Type SATURATION_
    Value As Single
End Type

Public Type GAMMA_
    Value As Single
End Type

Public Type SHUTTER_
    Value As Single
End Type

Public Type GAIN_
    Value As Single
End Type

Public Type IRIS_
    Value As Single
End Type

Public Type FOCUS_
    Value As Single
End Type

Public Type TEMPERATURE_
    Value As Single
End Type
  
Public Type TRIGGER_
    Mode As Single
    Type As Single
    Polarity As Single
    Delay As Single
    Parameter As Single
End Type

Public Type ZOOM_
    Value As Single
End Type

Public Type PAN_
    Value As Single
End Type

Public Type TILT_
    Value As Single
End Type

Public Type OPTICALFILTER_
    Value As Single
End Type

Public Type GPIO_
    Mode(1 To PXL_MAX_STROBES) As Single
    Polarity(1 To PXL_MAX_STROBES) As Single
    Parameter1(1 To PXL_MAX_STROBES) As Single
    Parameter2(1 To PXL_MAX_STROBES) As Single
    Parameter3(1 To PXL_MAX_STROBES) As Single
End Type

Public Type FRAMERATE_
    Value As Single
End Type

Public Type ROI_
    Left As Single
    Top As Single
    Width As Single
    Height As Single
End Type

Public Type FLIP_
    Horizontal As Single
    Vertical As Single
End Type

Public Type DECIMATION_
     Value As Single
End Type

Public Type PIXELFORMAT_
     Value As Single
End Type

Public Type EXTENDEDSHUTTER_
     KneePoint(1 To PXL_MAX_KNEE_POINTS) As Single
End Type

Public Type AUTOROI_
    Left As Single
    Top As Single
    Width As Single
    Height As Single
End Type

Public Type DecimationMode_
     Value As Single
End Type

Public Type WhiteShading_
     RedGain As Single
     GreenGain As Single
     BlueGain As Single
End Type

Public Type Rotate_
     Value As Single
End Type

Public Type ImagerClkDivisor_
     Value As Single
End Type

Public Type TriggerWithControlledLight_
     Value As Single
End Type

Public Type MaxPixelSize_
     Value As Single
End Type

Public Type TriggerNumber_
     Value As Single
End Type

Public Type ImageProcessing_
     Mask As UInteger
End Type

Public Type PixelAddressingValue_
     Horizontal As Single
     Vertical As Single
End Type

Public Type BandwidthLimit_
     Value As Single
End Type

Public Type ActualFrameRate_
     Value As Single
End Type

Public Type SharpnessScoreParams_
     Left As Single
     Top As Single
     Width As Single
     Height As Single
     MaxValue As Single
End Type

Public Type SharpnessScore_
     Value As Single
End Type

Public Type FRAME_DESC
    Size As UInteger
    FrameTime As Single
    FrameNumber As UInteger
    Brightness As BRIGHTNESS_
    AutoExposure As AUTOEXPOSURE_
    Sharpness As SHARPNESS_
    WhiteBalance As WHITEBALANCE_
    Hue As HUE_
    Saturation As SATURATION_
    Gamma As GAMMA_
    Shutter As SHUTTER_
    Gain As GAIN_
    Iris As IRIS_
    Focus As FOCUS_
    Temperature As TEMPERATURE_
    Trigger As TRIGGER_
    Zoom As ZOOM_
    Pan As PAN_
    Tilt As TILT_
    OpticalFilter As OPTICALFILTER_
    GPIO As GPIO_
    FrameRate As FRAMERATE_
    Roi As ROI_
    Flip As FLIP_
    Decimation As DECIMATION_
    PixelFormat As PIXELFORMAT_
    ExtendedShutter As EXTENDEDSHUTTER_
    AutoRoi As AUTOROI_
    DecimationMode As DecimationMode_
    WhiteShading As WhiteShading_
    Rotate As Rotate_
    ImagerClkDivisor As ImagerClkDivisor_
    TriggerWithControlledLight As TriggerWithControlledLight_
    MaxPixelSize As MaxPixelSize_
    TriggerNumber As TriggerNumber_
    ImageProcessing as ImageProcessing_
    PixelAddressingValue As PixelAddressingValue_
    FrameTime as Double
    LongFrameNumber As Long
    BandwidthLimit As BandwidthLimit_
    ActualFrameRate As ActualFrameRate_
    SharpnessScoreParams As SharpnessScoreParams_
    SharpnessScore As SharpnessScore_
End Type

Public Type ERROR_REPORT
    lReturnCode As UInteger
    strFunctionName As String * 32
    strReturnCode As String * 32
    strReport As String * 256
End Type

'Camera Info
Public Type CLIP_ENCODING_INFO
    StreamEncoding As UInteger
    DecimationFactor As UInteger
    PlaybackFrameRate As Single
    playbackBitRate As UInteger
End Type

   
'***************************************************
'   FUNCTION Declarations - must follow constants
'
'   Note: These declartions can be used directly - just by adding this
'         file to the VB project because the Default for Declare Function
'         is Public.
'
'***************************************************
Declare Function PxLFormatClip Lib "PxLAPI40.DLL" Alias "_PxLFormatClip@12" (
        ByVal InputFileName As String,
        ByVal OutputFileName As String,
        ByVal OutputFormat As UInteger) As UInteger

Declare Function PxLFormatClipEx Lib "PxLAPI40.DLL" Alias "_PxLFormatClipEx@16" (
        ByVal InputFileName As String,
        ByVal OutputFileName As String,
        ByVal InputFormat As UInteger,
        ByVal OutputFormat As UInteger) As UInteger

Declare Function PxLFormatImage Lib "PxLAPI40.DLL" Alias "_PxLFormatImage@20" (
         ByRef Src As Byte,
         ByRef frameDesc As FRAME_DESC,
         ByVal OutputFormat As UInteger,
         ByRef Dest As Byte,
         ByRef DestBufferSize As UInteger) As UInteger

Declare Function PxLGetCameraFeatures Lib "PxLAPI40.DLL" Alias "_PxLGetCameraFeatures@16" (
         ByVal hCamera As UInteger,
         ByVal FeatureId As UInteger,
         ByRef FeatureInfo As Byte,
         ByRef bufferSize As UInteger) As UInteger

Declare Function PxLGetCameraInfo Lib "PxLAPI40.DLL" Alias "_PxLGetCameraInfo@8" (
         ByVal hCamera As UInteger,
         ByRef Information As CAMERA_INFO) As UInteger

Declare Function PxLGetCameraInfoEx Lib "PxLAPI40.DLL" Alias "_PxLGetCameraInfoEx@12" (
         ByVal hCamera As UInteger,
         ByRef Information As CAMERA_INFO,
         ByVal informationSize) As UInteger

Declare Function PxLGetClip Lib "PxLAPI40.DLL" Alias "_PxLGetClip@16" (
         ByVal hCamera As UInteger,
         ByVal NumberOfFrames As UInteger,
         ByVal Filename As String,
         ByVal TerminationFunction As UInteger) As UInteger

Declare Function PxLGetEncodedClip Lib "PxLAPI40.DLL" Alias "_PxLGetEncodedClip@20" (
         ByVal hCamera As UInteger,
         ByVal NumberOfFrames As UInteger,
         ByVal Filename As String,
         ByVal ClipInfo As CLIP_ENCODING_INFO,
         ByVal TerminationFunction As UInteger) As UInteger

Declare Function PxLGetErrorReport Lib "PxLAPI40.DLL" Alias "_PxLGetErrorReport@8" (
         ByVal hCamera As UInteger,
         ByRef ErrorReport As ERROR_REPORT) As UInteger

Declare Function PxLGetFeature Lib "PxLAPI40.DLL" Alias "_PxLGetFeature@20" (
         ByVal hCamera As UInteger,
         ByVal FeatureId As UInteger,
         ByRef Flags As UInteger,
         ByRef NumberParms As UInteger,
         ByRef Parms As Single) As UInteger

Declare Function PxLGetNextFrame Lib "PxLAPI40.DLL" Alias "_PxLGetNextFrame@16" (
         ByVal hCamera As UInteger,
         ByVal bufferSize As UInteger,
         ByRef Frame As Byte,
         ByRef Descriptor As FRAME_DESC) As UInteger

Declare Function PxLGetNumberCameras Lib "PxLAPI40.DLL" Alias "_PxLGetNumberCameras@8" (
         ByRef SerialNumbers As UInteger,
         ByRef NumberSerial As UInteger) As UInteger

Declare Function PxLGetNumberCamerasEx Lib "PxLAPI40.DLL" Alias "_PxLGetNumberCamerasEx@8" (
         ByRef cameraIdInfo As Any,
         ByRef numberOfCameras As UInteger) As UInteger

Declare Function PxLSetCameraIpAddress Lib "PxLAPI40.DLL" Alias "_PxLSetCameraIpAddress@20" (
        ByRef CameraMac As PXL_MAC_ADDRESS,
        ByRef CameraIpAddress As PXL_IP_ADDRESS,
        ByRef cameraSubnetMask As PXL_IP_ADDRESS,
        ByRef cameraDefaultGateway As PXL_IP_ADDRESS,
        ByRef persistent As UInteger) As UInteger

Declare Function PxLInitialize Lib "PxLAPI40.DLL" Alias "_PxLInitialize@8" (
         ByVal SerialNumber As UInteger,
         ByRef hCamera As UInteger) As UInteger

Declare Function PxLInitializeEx Lib "PxLAPI40.DLL" Alias "_PxLInitializeEx@8" (
         ByVal SerialNumber As UInteger,
         ByRef hCamera As UInteger,
         ByVal Flags As UInteger) As UInteger

Declare Function PxLResetPreviewWindow Lib "PxLAPI40.DLL" Alias "_PxLResetPreviewWindow@4" (
         ByVal hCamera As UInteger) As UInteger

Declare Function PxLSetCallback Lib "PxLAPI40.DLL" Alias "_PxLSetCallback@16" (
         ByVal hCamera As UInteger,
         ByVal OverlayUse As UInteger,
         ByVal context As UInteger,
         ByVal DataProcessFunctionPtr As UInteger) As UInteger

Declare Function PxLSetCameraName Lib "PxLAPI40.DLL" Alias "_PxLSetCameraName@8" (
         ByVal hCamera As UInteger,
         ByVal CameraName As String) As UInteger

Declare Function PxLSetFeature Lib "PxLAPI40.DLL" Alias "_PxLSetFeature@20" (
         ByVal hCamera As UInteger,
         ByVal FeatureId As UInteger,
         ByVal Flags As UInteger,
         ByVal NumberParms As UInteger,
         ByRef Parms As Single) As UInteger

Declare Function PxLGetCurrentTimestamp Lib "PxLAPI40.DLL" Alias "_PxLGetCurrentTimestamp@8" (
         ByVal hCamera As UInteger,
         ByRef CurrentTimestamp As Double) As UInteger

Declare Function PxLSetPreviewSettings Lib "PxLAPI40.DLL" Alias "_PxLSetPreviewSettings@36" (
         ByVal hCamera As UInteger,
         Optional ByVal Title As String = "PixeLINK Preview",
         Optional ByVal Style As UInteger = WS_OVERLAPPEDWINDOW Or WS_VISIBLE,
         Optional ByVal Left As UInteger = CW_USEDEFAULT,
         Optional ByVal Top As UInteger = CW_USEDEFAULT,
         Optional ByVal Width As UInteger = CW_USEDEFAULT,
         Optional ByVal Height As UInteger = CW_USEDEFAULT,
         Optional ByVal Parent As UInteger = 0,
         Optional ByVal ChildId As UInteger = 0) As UInteger

Declare Function PxLSetPreviewState Lib "PxLAPI40.DLL" Alias "_PxLSetPreviewState@12" (
         ByVal hCamera As UInteger,
         ByVal PreviewState As UInteger,
         ByRef HWnd As UInteger) As UInteger

Declare Function PxLSetPreviewStateEx Lib "PxLAPI40.DLL" Alias "_PxLSetPreviewStateEx@16" (
         ByVal hCamera As UInteger,
         ByVal PreviewState As UInteger,
         ByRef HWnd As UInteger,
         ByVal ChangeFunctionPtr As UInteger) As UInteger

Declare Function PxLSetStreamState Lib "PxLAPI40.DLL" Alias "_PxLSetStreamState@8" (
         ByVal hCamera As UInteger,
         ByVal StreamState As UInteger) As UInteger

Declare Function PxLUninitialize Lib "PxLAPI40.DLL" Alias "_PxLUninitialize@4" (
         ByVal hCamera As UInteger) As UInteger

Declare Function PxLCreateDescriptor Lib "PxLAPI40.DLL" Alias "_PxLCreateDescriptor@12" (
         ByVal hCamera As UInteger,
         ByRef DescriptorHandle As UInteger,
         ByVal UpdateMode As UInteger) As UInteger

Declare Function PxLRemoveDescriptor Lib "PxLAPI40.DLL" Alias "_PxLRemoveDescriptor@8" (
         ByVal hCamera As UInteger,
         ByVal Descriptor As UInteger) As UInteger

Declare Function PxLUpdateDescriptor Lib "PxLAPI40.DLL" Alias "_PxLUpdateDescriptor@12" (
         ByVal hCamera As UInteger,
         ByVal Descriptor As UInteger,
         ByVal UpdateMode As UInteger) As UInteger

Declare Function PxLSaveSettings Lib "PxLAPI40.DLL" Alias "_PxLSaveSettings@8" (
         ByVal hCamera As UInteger,
         ByVal Channel As UInteger) As UInteger

Declare Function PxLLoadSettings Lib "PxLAPI40.DLL" Alias "_PxLLoadSettings@8" (
         ByVal hCamera As UInteger,
         ByVal Channel As UInteger) As UInteger

Declare Function PxLCameraRead Lib "PxLAPI40.DLL" Alias "_PxLCameraRead@12" (
         ByVal hCamera As UInteger,
         ByVal uBufferSize As UInteger,
         ByRef pBuffer As Byte) As UInteger

Declare Function PxLCameraWrite Lib "PxLAPI40.DLL" Alias "_PxLCameraWrite@12" (
         ByVal hCamera As UInteger,
         ByVal uBufferSize As UInteger,
         ByRef pBuffer As Byte) As UInteger

Declare Function PxLPrivateCmd Lib "PxLAPI40.DLL" Alias "_PxLPrivateCmd@12" (
         ByVal hCamera As UInteger,
         ByVal bufferSize As UInteger,
         ByRef pBuffer As Byte) As UInteger


' Required by PxLGetSingleCameraFeature
Private Declare Sub CopyMemory Lib "kernel32" Alias "RtlMoveMemory" (ByVal pDest As Any, ByVal pSrc As Any, ByVal ByteLen As UInteger)

' PxLGetSingleCameraFeature
'
' Purpose: Simple PxLGetSingleCameraFeature wrapper
'          to obtain a single feature
'
Function PxLGetSingleCameraFeature(
         ByVal hCamera As UInteger,
         ByVal FeatureId As UInteger,
         ByRef CameraFeature As SINGLE_CAMERA_FEATURE) As UInteger

    Dim rc As UInteger
    Dim bufferSize As UInteger
    Dim bufferFeature() As Byte

    bufferSize = 0
    rc = PxLGetCameraFeatures(hCamera, FeatureId, ByVal 0, bufferSize)
    If rc <> ApiSuccess Then
        PxLGetSingleCameraFeature = rc
        Exit Function
    End If
    
    ReDim bufferFeature(1 To bufferSize)
    rc = PxLGetCameraFeatures(hCamera, FeatureId, bufferFeature(1), bufferSize)
    If rc <> ApiSuccess Then
        PxLGetSingleCameraFeature = rc
        Exit Function
    End If
     
    Dim cameraFeatures As CAMERA_FEATURES
    CopyMemory VarPtr(cameraFeatures), VarPtr(bufferFeature(1)), LenB(cameraFeatures)
        
    Dim feature As CAMERA_FEATURE
    CopyMemory VarPtr(feature.FeatureId), _
               cameraFeatures.pFeatures, cameraFeatures.NumberOfFeatures * LenB(feature)
            
    ' Fill the SINGLE_CAMERA_FEATURE structure
    CameraFeature.FeatureId = feature.FeatureId
    CameraFeature.Flags = feature.Flags
    CameraFeature.NumberOfParameters = feature.NumberOfParameters

    CopyMemory VarPtr(CameraFeature.Params(1)), _
            feature.pParams, _
            feature.NumberOfParameters * _
            LenB(CameraFeature.Params(1)) 'size of FEATURE_PARAM
    
    PxLGetSingleCameraFeature = rc
    
End Function


