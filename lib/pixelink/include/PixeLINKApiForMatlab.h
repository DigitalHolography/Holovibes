/*
  PixeLINKApiForMatlab.h V1.1

  A simplified version of the PixeLINK API headers because Matlab has troubles parsing the 
  offical PixeLINK headers.
  Using this header file and Matlab's library functions, it is possible to use the PixeLINK 4.0 API.

  This header is provided as is. Feedback and suggestions are appreciated.

  Revision History
  V0.1    2009/12/09    - First pass
  V0.2    2010/01/18    - Fixed errors in documentation
  V0.3    2010/01/18    - Added sample session to the documentation
  V1.0    2010/08/13    - Fixed typo: FrameDesc.Flipertical => FlipVertical
  V1.1    2011/04/25    - Fixed comments: Was extracting hCamera too early in example session
  V1.2    2011/12/20    - Minor additions for GEV


  Example session

//
// Load the library, using the simplified header file
//
>> [notfound, warnings] = loadlibrary('C:\Windows\system32\PxLApi40.dll', 'C:\Program Files\PixeLINK\include\PixeLINKApiForMatlab.h', 'alias', 'PixeLINK')
// Double check that it's loaded
>> libisloaded('PixeLINK')

//
// List all the functions MATLAB now knows about from the DLL
//
>> libfunctions 'PixeLINK' -full

//
// Getting a list of serial numbers of available cameras
//
>> pNumberOfSerialNumbers = libpointer('uint32Ptr', 0)
>> calllib('PixeLINK', 'PxLGetNumberCameras', 0, pNumberOfSerialNumbers)
>> pSerialNumbers = libpointer('uint32Ptr', 1:pNumberOfSerialNumbers.Value)
>> calllib('PixeLINK', 'PxLGetNumberCameras', pSerialNumbers, pNumberOfSerialNumbers)


//
// Initializing a camera.
// Declare a camera handle, and pass the serial number to the camera.
// Two options here: Pass the serial number of a camera, or pass in the serial number 0, meaning
// connect to any camera.
//
>> phCamera = libpointer('uint32Ptr', 0)
>> calllib('PixeLINK', 'PxLInitialize', 0, phCamera)
//
// Could also use a specific serial number
// calllib('PixeLINK', 'PxLInitialize', pSerialNumbers.Value(1), phCamera)

//
// Get the handle value
//
>> hCamera = phCamera.Value


//
// Get information about the camera
//
>> cameraInfo.VendorName = zeros(1,33, 'int8')
>> pCameraInfo = libpointer('s_CAMERA_INFOPtr', cameraInfo)
>> calllib('PixeLINK', 'PxLGetCameraInfo', hCamera, pCameraInfo)
>> char(pCameraInfo.Value.VendorName)

// Query the exposure
>> pFlags = libpointer('uint32Ptr', 0)
>> pNumberOfParams = libpointer('uint32Ptr', 1)
>> pExposure = libpointer('singlePtr', 0.0)
>> calllib('PixeLINK', 'PxLGetFeature', hCamera, 7, pFlags, pNumberOfParams, pExposure)
>> pExposure.Value

// Now set the exposure to double its value
// (Just to test calling PxLSetFeature)
>> pExposure.Value = pExposure.Value * 2
>> calllib('PixeLINK', 'PxLSetFeature', hCamera, 7, pFlags.Value, pNumberOfParams.Value, pExposure)


// Query the ROI
>> pParams = libpointer('singlePtr', 1:4)
>> pNumberOfParams.Value = 4;
>> calllib('PixeLINK', 'PxLGetFeature', hCamera, 19, pFlags, pNumberOfParams, pParams)
>> roiWidth  = pParams.Value(3)
>> roiHeight = pParams.Value(4)

// Query the Pixel Addressing mode and value
>> pNumberOfParams.Value = 2
>> calllib('PixeLINK', 'PxLGetFeature', hCamera, 21, pFlags, pNumberOfParams, pParams)
>> pixelAddressingValue = pParams.Value(1)
>> roiHeight = roiHeight / pixelAddressingValue
>> roiWidth = roiWidth / pixelAddressingValue
>> numberOfPixels = roiWidth * roiHeight


// Start streaming
>> calllib('PixeLINK', 'PxLSetStreamState', hCamera, 0)

// Start a preview window
>> phWnd = libpointer('uint32Ptr', 0)
>> calllib('PixeLINK', 'PxLSetPreviewState', hCamera, 0, phWnd)

// Capture an image
// Assuming that the camera is set in MONO8 or BAYER8
// Declare a frame buffer, and a FRAME_DESC so that we can capture a frame
>> pFrameBuffer = libpointer('uint8Ptr', 1:numberOfPixels)
>> frameDesc.uSize = 516
>> pFrameDesc = libpointer('s_FRAME_DESCPtr', frameDesc)
>> calllib('PixeLINK', 'PxLGetNextFrame', hCamera, length(pFrameBuffer.Value), pFrameBuffer, pFrameDesc)

// Display as a monochrome image
>> imagesc(reshape(pFrameBuffer.Value, roiWidth, roiHeight) .', [0, 255]); colormap(gray);


//
// Shutting everything down
//


// Stop previewing
>> calllib('PixeLINK', 'PxLSetPreviewState', hCamera, 2, phWnd)

// Stop streaming
>> calllib('PixeLINK', 'PxLSetStreamState', hCamera, 2)


// Uninitialize the camera
>> calllib('PixeLINK', 'PxLUninitialize', hCamera)


// Have to clear variables using PixeLINK structs before unloading the library
>> clear pFrameDesc

// Unload the PixeLINK library
>> unloadlibrary('PixeLINK')

// Double check that it's UNloaded
>> libisloaded('PixeLINK')

See the example session below for an example of what you should see.

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Example Session In MATLAB
// For this demo, two cameras are connected to the host computer (One FireWire (999742000) and one USB (999955049))
// Ask for a list of connected cameras, connect to the first one, then and query a few values.
//
//
<examplesession>

>> [notfound, warnings] = loadlibrary('C:\Windows\system32\PxLApi40.dll', 'C:\Program Files\PixeLINK\include\PixeLINKApiForMatlab.h', 'alias', 'PixeLINK')

notfound = 

   Empty cell array: 0-by-1


warnings =

     ''


>> libisloaded('PixeLINK')

ans =

     1

>> libfunctions 'PixeLINK' -full

Functions in library PixeLINK:

[uint32, voidPtr, s_FRAME_DESCPtr, voidPtr, uint32Ptr] PxLFormatImage(voidPtr, s_FRAME_DESCPtr, uint32, voidPtr, uint32Ptr)
[uint32, s_CAMERA_INFOPtr] PxLGetCameraInfo(uint32, s_CAMERA_INFOPtr)
[uint32, uint32Ptr, uint32Ptr, singlePtr] PxLGetFeature(uint32, uint32, uint32Ptr, uint32Ptr, singlePtr)
[uint32, voidPtr, s_FRAME_DESCPtr] PxLGetNextFrame(uint32, uint32, voidPtr, s_FRAME_DESCPtr)
[uint32, uint32Ptr, uint32Ptr] PxLGetNumberCameras(uint32Ptr, uint32Ptr)
[uint32, uint32Ptr] PxLInitialize(uint32, uint32Ptr)
uint32 PxLLoadSettings(uint32, uint32)
uint32 PxLResetPreviewWindow(uint32)
uint32 PxLSaveSettings(uint32, uint32)
[uint32, singlePtr] PxLSetFeature(uint32, uint32, uint32, uint32, singlePtr)
[uint32, cstring] PxLSetPreviewSettings(uint32, cstring, uint32, uint32, uint32, uint32, uint32, uint32, uint32)
[uint32, uint32Ptr] PxLSetPreviewState(uint32, uint32, uint32Ptr)
uint32 PxLSetStreamState(uint32, uint32)
uint32 PxLUninitialize(uint32)


>> pNumberOfSerialNumbers = libpointer('uint32Ptr', 0)
 
pNumberOfSerialNumbers =
 
libpointer
>> calllib('PixeLINK', 'PxLGetNumberCameras', 0, pNumberOfSerialNumbers)

ans =

     0

>> get(pNumberOfSerialNumbers)
       Value: 2
    DataType: 'uint32Ptr'

>> pSerialNumbers = libpointer('uint32Ptr', 1:pNumberOfSerialNumbers.Value)
 
pSerialNumbers =
 
libpointer
>> calllib('PixeLINK', 'PxLGetNumberCameras', pSerialNumbers, pNumberOfSerialNumbers)

ans =

     0

>> get(pSerialNumbers)
       Value: [999955049 999742000]
    DataType: 'uint32Ptr'

>> phCamera = libpointer('uint32Ptr', 0)
 
phCamera =
 
libpointer
>> calllib('PixeLINK', 'PxLInitialize', pSerialNumbers.Value(1), phCamera)

ans =

     0

>> hCamera = phCamera.Value

hCamera =

  2147483649

>> cameraInfo.VendorName = zeros(1,33, 'int8')

cameraInfo = 

    VendorName: [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]

>> pCameraInfo = libpointer('s_CAMERA_INFOPtr', cameraInfo)
 
pCameraInfo =
 
libpointer
>> calllib('PixeLINK', 'PxLGetCameraInfo', hCamera, pCameraInfo)

ans =

     0

>> char(pCameraInfo.Value.VendorName)

ans =

PixeLINK                         

>> pFlags = libpointer('uint32Ptr', 0)
 
pFlags =
 
libpointer
>> pNumberOfParams = libpointer('uint32Ptr', 1)
 
pNumberOfParams =
 
libpointer
>> pExposure = libpointer('singlePtr', 0.0)
 
pExposure =
 
libpointer
>> calllib('PixeLINK', 'PxLGetFeature', hCamera, 7, pFlags, pNumberOfParams, pExposure)

ans =

     0

>> pExposure.Value

ans =

    0.0400

>> 

</examplesession>
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
*/


// Some handy defines to simplify our declarations
#define PXL_RETURN_CODE         unsigned int
#define U32                     unsigned int
#define S8                        signed char
#define U8                      unsigned char
#define HANDLE                  unsigned int
#define HWND                    unsigned int
#define PXL_API                 __declspec(dllimport) __stdcall
#define PXL_MAX_STROBES         16
#define PXL_MAX_KNEE_POINTS      4


//
// The FRAME_DESC struct in PixeLINKTypes.h
//
typedef struct _FRAME_DESC
{
    U32 uSize;
    float fFrameTime;
    U32 uFrameNumber;
    float Brightness;
    float AutoExposure;
    float Sharpness;
    float WhiteBalance;
    float Hue;
    float Saturation;
    float Gamma;
    float Shutter;
    float Gain;
    float Iris;
    float Focus;
    float Temperature;
    float TriggerMode;
    float TriggerType;
    float TriggerPolarity;
    float TriggerDelay;
    float TriggerParameter;
    float Zoom;
    float Pan;
    float Tilt;
    float OpticalFilter;
    float GPIOMode[PXL_MAX_STROBES];
    float GPIOPolarity[PXL_MAX_STROBES];
    float GPIOParameter1[PXL_MAX_STROBES];
    float GPIOParameter2[PXL_MAX_STROBES];
    float GPIOParameter3[PXL_MAX_STROBES];
    float FrameRate;
    float RoiLeft;
    float RoiTop;
    float RoiWdth;
    float RoiHeight;
    float FlipHorizontal;
    float FlipVertical;
    float Decimation;
    float PixelFormat;
    float KneePoints[PXL_MAX_KNEE_POINTS];
    float AutoRoiLeft;
    float AutoRoiTop;
    float AutoRoiWidth;
    float AutoRoiHeight;
    float DecimationMode;
    float WhiteShadingRedGain;
    float WhiteShadingGreenGain;
    float WhiteShadingBlueGain;
    float Rotate;
    float ImagerClockDivisor;
    float TriggerWithControlledLight;
    float MaxPixelSize;
	float TriggerNumber;
	float ImageProcessing;
} FRAME_DESC;

//
// The CAMERA_INFO struct in PixeLINKTypes.h
//
typedef struct _CAMERA_INFO
{
    S8 VendorName      [33];
    S8 ModelName       [33];
    S8 Description    [256];
    S8 SerialNumber    [33];
    S8 FirmwareVersion [12];
    S8 FPGAVersion     [12];
    S8 CameraName     [256];
} CAMERA_INFO, *PCAMERA_INFO;

/* MAC Address */
typedef struct _PXL_MAC_ADDRESS 
{
    U8  MacAddr[6];
} PXL_MAC_ADDRESS, *PPXL_MAC_ADDRESS;

/* IP Address */
typedef struct _PXL_IP_ADDRESS
{
    union
    {
        U8   U8Address[4];
        U32  U32Address;
    };
} PXL_IP_ADDRESS, *PPXL_IP_ADDRESS;


/* Camera ID Info */
typedef struct _CAMERA_ID_INFO
{
    U32              StructSize;  /* Set this to sizeof (CAMERA_ID_INFO) to facilitate growth */
    U32              CameraSerialNum;
    PXL_MAC_ADDRESS  CameraMac;
    PXL_IP_ADDRESS   CameraIpAddress;
    PXL_IP_ADDRESS   CameraIpMask;
    PXL_IP_ADDRESS   CameraIpGateway;
    PXL_IP_ADDRESS   NicIpAddress;
    PXL_IP_ADDRESS   NicIpMask;
    U32              NicAccessMode;
    U8               CameraIpAssignmentType;
    U8               XmlVersionMajor;
    U8               XmlVersionMinor;
    U8               XmlVersionSubminor;
    U8               IpEngineLoadVersionMajor;
    U8               IpEngineLoadVersionMinor;
    U8               IpEngineLoadVersionSubminor;
    U8               CameraProperties;
    PXL_IP_ADDRESS   ControllingIpAddress; /* Identifies the host which is currently controlling this camera */
} CAMERA_ID_INFO, *PCAMERA_ID_INFO;

//
// PixeLINK API Functions exported from PxLApi40.dll
//
PXL_RETURN_CODE PXL_API PxLGetNumberCameras  (U32* pSerialNumbers, U32* pNumberSerialNumbers);
PXL_RETURN_CODE PXL_API PxLGetNumberCamerasEx(CAMERA_ID_INFO* pCameraIdInfo, U32* pNumberCameraIdInfos);

PXL_RETURN_CODE PXL_API PxLInitialize  (U32 serialNumber, HANDLE* phCamera);
PXL_RETURN_CODE PXL_API PxLInitializeEx(U32 serialNumber, HANDLE* phCamera, U32 flags);
PXL_RETURN_CODE PXL_API PxLUninitialize(HANDLE hCamera);

PXL_RETURN_CODE PXL_API PxLGetCameraInfo(HANDLE hCamera, CAMERA_INFO* pInformation);

PXL_RETURN_CODE PXL_API PxLSaveSettings(HANDLE hCamera, U32 channelNumber);
PXL_RETURN_CODE PXL_API PxLLoadSettings(HANDLE hCamera, U32 channelNumber);

PXL_RETURN_CODE PXL_API PxLSetStreamState(HANDLE hCamera, U32 streamState);
PXL_RETURN_CODE PXL_API PxLSetPreviewState(HANDLE hCamera,U32 previewState, HWND* pHWnd);
PXL_RETURN_CODE PXL_API PxLSetPreviewSettings(HANDLE hCamera, const char* pTitle, U32 style, U32 left, U32 top, U32 width, U32 height, HWND   hParent, U32 childId);
PXL_RETURN_CODE	PXL_API PxLResetPreviewWindow(HANDLE hCamera);

PXL_RETURN_CODE PXL_API PxLGetFeature(HANDLE hCamera, U32 featureId, U32* pFlags, U32* pNumberOfParams, float* pParams);
PXL_RETURN_CODE PXL_API PxLSetFeature(HANDLE hCamera, U32 featureId, U32   flags, U32   numberOfParams, float* pParams);

PXL_RETURN_CODE PXL_API PxLGetNextFrame(HANDLE hCamera, U32 bufferSizeInBytes,  void* pFrame, FRAME_DESC* pFrameDesc);
PXL_RETURN_CODE PXL_API PxLFormatImage(void const * pSrcFrame, FRAME_DESC const * pSrcFrameDesc, U32 outputFormat, void* pDestBuffer, U32* pDestBufferSize);

