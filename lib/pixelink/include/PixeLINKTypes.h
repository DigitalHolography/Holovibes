/****************************************************************************************************************
 * COPYRIGHT � 2010 PixeLINK CORPORATION.  ALL RIGHTS RESERVED.                                                 *
 * Copyright Notice and Disclaimer of Liability:                                                                *
 *                                                                                                              *
 *                                                                                                              *
 * PixeLINK Corporation is henceforth referred to as PixeLINK or PixeLINK Corporation.                          *
 * Purchaser henceforth refers to the original purchaser(s) of the equipment, and/or any legitimate user(s).    *
 *                                                                                                              *
 * PixeLINK hereby explicitly prohibits any form of reproduction (with the express strict exception for backup  *
 * and archival purposes, which are allowed as stipulated within the License Agreement for PixeLINK Corporation *
 * Software), modification, and/or distribution of this software and/or its associated documentation unless     *
 * explicitly specified in a written agreement signed by both parties.                                          *
 *                                                                                                              *
 * To the extent permitted by law, PixeLINK disclaims all other warranties or conditions of any kind, either    *
 * express or implied, including but not limited to all warranties or conditions of merchantability and         *
 * fitness for a particular purpose and those arising by statute or otherwise in law or from a course of        *
 * dealing or usage of trade. Other written or oral statements by PixeLINK, its representatives, or others do   *
 * not constitute warranties or conditions of PixeLINK.                                                         *
 *                                                                                                              *
 * PixeLINK makes no guarantees or representations, including but not limited to: the use of, or the result(s)  *
 * of the use of: the software and associated documentation in terms of correctness, accuracy, reliability,     *
 * being current, or otherwise. The Purchaser hereby agree to rely on the software, associated hardware and     *
 * documentation and results stemming from the use thereof solely at their own risk.                            *
 *                                                                                                              *
 * By using the products(s) associated with the software, and/or the software, the Purchaser and/or user(s)     *
 * agree(s) to abide by the terms and conditions set forth within this document, as well as, respectively,      *
 * any and all other documents issued by PixeLINK in relation to the product(s).                                *
 *                                                                                                              *
 * PixeLINK is hereby absolved of any and all liability to the Purchaser, and/or a third party, financial or    *
 * otherwise, arising from any subsequent loss, direct and indirect, and damage, direct and indirect,           *
 * resulting from intended and/or unintended usage of its software, product(s) and documentation, as well       *
 * as additional service(s) rendered by PixeLINK, such as technical support, unless otherwise explicitly        *
 * specified in a written agreement signed by both parties. Under no circumstances shall the terms and          *
 * conditions of such an agreement apply retroactively.                                                         *
 *                                                                                                              *
 ****************************************************************************************************************/

#ifndef PIXELINK_COM_PIXELINKTYPES_H
#define PIXELINK_COM_PIXELINKTYPES_H

#ifdef PIXELINK_LINUX

typedef unsigned long long int U64;
typedef unsigned int           U32;
typedef unsigned short         U16;
typedef unsigned char          U8;

typedef signed long long int   S64;
typedef signed int             S32;
typedef signed short           S16;
typedef signed char            S8;

typedef U32    ULONG;
typedef U16    USHORT;
typedef U8     UCHAR;

/* Some types and decorators we borrow from Windows */
typedef void*  HANDLE;
typedef void*  HWND;
typedef U8     BYTE;
typedef float  FLOAT;

typedef void   VOID;
typedef void*  PVOID;
typedef void*  LPVOID;

typedef char*  LPSTR;
typedef const char* LPCSTR;
typedef char* PCHAR;

typedef ULONG* PULONG;
typedef UCHAR* PUCHAR;
typedef FLOAT* PFLOAT;

#ifndef NULL
#define NULL   0
#endif
#define INFINITE 0xFFFFFFFF
#define IN     /**/
#define OUT    /**/

#else

/* These types assume you've included windows.h or some DDK equivalent */

typedef ULONGLONG   U64;
typedef ULONG       U32;
typedef USHORT      U16;
typedef UCHAR       U8;

typedef LONGLONG    S64;
typedef LONG        S32;
typedef SHORT       S16;
typedef CHAR        S8;

#endif

/* Common and composite types */
typedef float F32;
typedef U32   BOOL32;

/* Pointers to unsigned types */
typedef U64*    PU64;
typedef U32*    PU32;
typedef U16*    PU16;
typedef  U8*    PU8;

/* Pointers to signed types */
typedef S64*  PS64;
typedef S32*  PS32;
typedef S16*  PS16;
typedef  S8*  PS8;

/* Pointers to other types */
typedef F32*    PF32;
typedef BOOL32* PBOOL32;


#ifndef FALSE
#define FALSE               0
#endif
#ifndef TRUE
#define TRUE                1
#endif

/* Return codes */
typedef int PXL_RETURN_CODE;

/* Video Clip Encoding Format (AKA compression strategy) */
#define CLIP_ENCODING_PDS     0 /* Uncompressed, PixeLINK format */
#define CLIP_ENCODING_H264    1 /* Compressed using h264 encoding */
/* Video Clip File 'Container' Format */
#define CLIP_FORMAT_AVI     0
#define CLIP_FORMAT_MP4     1

/* Feature IDs */
#define FEATURE_BRIGHTNESS              0
#define FEATURE_PIXELINK_RESERVED_1     1
#define FEATURE_SHARPNESS               2
#define FEATURE_COLOR_TEMP              3
#define FEATURE_HUE                     4
#define FEATURE_SATURATION              5
#define FEATURE_GAMMA                   6
#define FEATURE_SHUTTER                 7
#define FEATURE_GAIN                    8
#define FEATURE_IRIS                    9
#define FEATURE_FOCUS                   10
#define FEATURE_SENSOR_TEMPERATURE      11
#define FEATURE_TRIGGER                 12
#define FEATURE_ZOOM                    13
#define FEATURE_PAN                     14
#define FEATURE_TILT                    15
#define FEATURE_OPT_FILTER              16
#define FEATURE_GPIO                    17
#define FEATURE_FRAME_RATE              18
#define FEATURE_ROI                     19
#define FEATURE_FLIP                    20
#define FEATURE_PIXEL_ADDRESSING        21
#define FEATURE_PIXEL_FORMAT            22
#define FEATURE_EXTENDED_SHUTTER        23
#define FEATURE_AUTO_ROI                24
#define FEATURE_LOOKUP_TABLE            25
#define FEATURE_MEMORY_CHANNEL          26
#define FEATURE_WHITE_SHADING           27          /* Seen in Capture OEM as White Balance */
#define FEATURE_ROTATE                  28
#define FEATURE_IMAGER_CLK_DIVISOR      29          /* DEPRECATED - New applications should not use. */
#define FEATURE_TRIGGER_WITH_CONTROLLED_LIGHT   30  /* Allows trigger to be used more deterministically where */
                                                    /* lighting cannot be controlled.                         */
#define FEATURE_MAX_PIXEL_SIZE          31          /* The number of bits used to represent 16-bit data (10 or 12) */
#define FEATURE_BODY_TEMPERATURE		32	
#define FEATURE_MAX_PACKET_SIZE  		33	
#define FEATURE_BANDWIDTH_LIMIT         34
#define FEATURE_ACTUAL_FRAME_RATE       35
#define FEATURE_SHARPNESS_SCORE         36
#define FEATURE_SPECIAL_CAMERA_MODE     37
#define FEATURE_GAIN_HDR                38
#define FEATURE_POLAR_WEIGHTINGS        39
#define FEATURE_POLAR_HSV_INTERPRETATION 40
#define FEATURES_TOTAL                  41

/* Feature aliases for backward compatibility*/
#define FEATURE_DECIMATION              FEATURE_PIXEL_ADDRESSING   /* Really, decimation is just one type of pixel addressing          */
#define FEATURE_EXPOSURE                FEATURE_SHUTTER            /* IIDC'c EXPOSURE is equivalent to feature SHUTTER                 */
#define FEATURE_WHITE_BAL               FEATURE_COLOR_TEMP         /* IIDC's white balance is usually referred to as color temperature */
#define FEATURE_TEMPERATURE             FEATURE_SENSOR_TEMPERATURE /* Now more specific, as the temperature is from the sensor */

/* For PxLGetCameraFeatures */
#define FEATURE_ALL 0xFFFFFFFF

/* Feature Flags */
#define FEATURE_FLAG_PRESENCE       0x00000001  /* The feature is supported on this camera. */
#define FEATURE_FLAG_MANUAL         0x00000002
#define FEATURE_FLAG_AUTO           0x00000004
#define FEATURE_FLAG_ONEPUSH        0x00000008
#define FEATURE_FLAG_OFF            0x00000010
#define FEATURE_FLAG_DESC_SUPPORTED 0x00000020
#define FEATURE_FLAG_READ_ONLY      0x00000040
#define FEATURE_FLAG_SETTABLE_WHILE_STREAMING 0x00000080
#define FEATURE_FLAG_PERSISTABLE    0x00000100  /* The feature will be saved with PxLSaveSettings */
#define FEATURE_FLAG_EMULATION      0x00000200  /* The feature is implemented in the API, not the camera */
#define FEATURE_FLAG_VOLATILE       0x00000400  /* The features (settable) value or limits, may change as the result of
                                                   changing some other feature.  See help file for details on
                                                   feature interaction */
#define FEATURE_FLAG_CONTROLLER     0x00000800  /* The feature is implemented in a seperate controller, not the camera */
/* FEATURE_FLAG_ASSERT_LOWER_LIMIT and FEATURE_FLAG_ASSERT_UPPER_LIMIT are generally supported by features that are
 * implemented using a motor control.  Setting the feature with this flag set, will move the motor to its 
 * absolute limit -- allowing an application to 'reclaibrate' for any mechanical drift that may have occurred
 * over time. */
#define FEATURE_FLAG_ASSERT_LOWER_LIMIT  0x00001000 /* (re)assert the feature to its absolute lowest value */
#define FEATURE_FLAG_ASSERT_UPPER_LIMIT  0x00002000 /* (re)assert the feature to its absolute highest value */
#define FEATURE_FLAG_USES_AUTO_ROI  0x00004000 /* This feature will use the Auto ROI (if enabled) for any auto adjustments*/

/* Exactly one of these 'mode' bits should be set with each feature set operation */
#define FEATURE_FLAG_MODE_BITS (FEATURE_FLAG_MANUAL | FEATURE_FLAG_AUTO | FEATURE_FLAG_ONEPUSH | FEATURE_FLAG_OFF)                                                  


/* 
 * 
 * Setting and clearing FEATURE_FLAG_OFF to enable and disable a feature can be 
 * a little confusing, so these macros make things a little more readable.
 * 
 * Examples
 * // Is a feature enabled?
 * int isEnabled = IS_FEATURE_ENABLED(flags)
 *
 * // Modify the flags to enable a feature
 * flags = ENABLE_FEATURE(flags, 1)
 *
 * // Modify the flags to disable a feature
 * flags = ENABLE_FEATURE(flags, 0)
 *
 *
 */
#define CLEAR_MODE_BITS(flags)			( (flags) & ~FEATURE_FLAG_MODE_BITS )
#define IS_FEATURE_ENABLED(flags)       ( (((flags) & FEATURE_FLAG_OFF) != 0) ? 0 : 1 )
#define IS_FEATURE_SUPPORTED(flags)     ( (((flags) & FEATURE_FLAG_PRESENCE) == 0) ? 0 : 1 )
#define ENABLE_FEATURE(flags, enable)   ( ((enable)) ?  (CLEAR_MODE_BITS((flags))|FEATURE_FLAG_MANUAL) : (CLEAR_MODE_BITS((flags))|FEATURE_FLAG_OFF) )



/* Image File Format */
/*   Note that DIB formats will appear (in memory), with the last row first, and the first row last */
#define IMAGE_FORMAT_BMP                0x0000
#define IMAGE_FORMAT_TIFF               0x0001
#define IMAGE_FORMAT_PSD                0x0002
#define IMAGE_FORMAT_JPEG               0x0003
#define IMAGE_FORMAT_PNG                0x0004
#define IMAGE_FORMAT_RAW_MONO8          0x1000
#define IMAGE_FORMAT_RAW_RGB24          0x1005
#define IMAGE_FORMAT_RAW_RGB48          0x1006                 /* RGB48 are always non-DIB */
#define IMAGE_FORMAT_RAW_RGB24_DIB      IMAGE_FORMAT_RAW_RGB24 /* RGB24 is DIB format */
#define IMAGE_FORMAT_RAW_RGB24_NON_DIB  0x1012
#define IMAGE_FORMAT_RAW_BGR24          0x1022                 /* BGR24 are always non-DIB */
#define IMAGE_FORMAT_RAW_BGR24_NON_DIB  IMAGE_FORMAT_RAW_BGR24 /* BGR24 are always non-DIB */

/* Pixel Formats */
#define PIXEL_FORMAT_MONO8            0
#define PIXEL_FORMAT_MONO16           1
#define PIXEL_FORMAT_YUV422           2
#define PIXEL_FORMAT_BAYER8_GRBG      3
#define PIXEL_FORMAT_BAYER16_GRBG     4
#define PIXEL_FORMAT_RGB24            5
#define PIXEL_FORMAT_RGB48            6
#define PIXEL_FORMAT_BAYER8_RGGB      7
#define PIXEL_FORMAT_BAYER8_GBRG      8
#define PIXEL_FORMAT_BAYER8_BGGR      9
#define PIXEL_FORMAT_BAYER16_RGGB    10
#define PIXEL_FORMAT_BAYER16_GBRG    11
#define PIXEL_FORMAT_BAYER16_BGGR    12
#define PIXEL_FORMAT_BAYER8          PIXEL_FORMAT_BAYER8_GRBG
#define PIXEL_FORMAT_BAYER16         PIXEL_FORMAT_BAYER16_GRBG
#define PIXEL_FORMAT_MONO12_PACKED   13
#define PIXEL_FORMAT_BAYER12_GRBG_PACKED  14
#define PIXEL_FORMAT_BAYER12_RGGB_PACKED  15
#define PIXEL_FORMAT_BAYER12_GBRG_PACKED  16
#define PIXEL_FORMAT_BAYER12_BGGR_PACKED  17
#define PIXEL_FORMAT_BAYER12_PACKED       PIXEL_FORMAT_BAYER12_GRBG_PACKED
/* Linux prefers non DIB formats, so use these to differentiate */
/*    By 'default' RGB24 are DIB, while RGB48 are non-DIB */
#define PIXEL_FORMAT_RGB24_DIB       PIXEL_FORMAT_RGB24
#define PIXEL_FORMAT_RGB24_NON_DIB   18
#define PIXEL_FORMAT_RGB48_NON_DIB   PIXEL_FORMAT_RGB48
#define PIXEL_FORMAT_RGB48_DIB       19
/* Added the 12-bit packed format used by Vision cameras */
/* Created after GigE Vision 1.0 (see comment below) */
#define PIXEL_FORMAT_MONO12_PACKED_MSFIRST   20
#define PIXEL_FORMAT_BAYER12_GRBG_PACKED_MSFIRST  21
#define PIXEL_FORMAT_BAYER12_RGGB_PACKED_MSFIRST  22
#define PIXEL_FORMAT_BAYER12_GBRG_PACKED_MSFIRST  23
#define PIXEL_FORMAT_BAYER12_BGGR_PACKED_MSFIRST  24
#define PIXEL_FORMAT_BAYER12_PACKED_MSFIRST       PIXEL_FORMAT_BAYER12_GRBG_PACKED_MSFIRST
#define PIXEL_FORMAT_MONO10_PACKED_MSFIRST   25
#define PIXEL_FORMAT_BAYER10_GRBG_PACKED_MSFIRST  26
#define PIXEL_FORMAT_BAYER10_RGGB_PACKED_MSFIRST  27
#define PIXEL_FORMAT_BAYER10_GBRG_PACKED_MSFIRST  28
#define PIXEL_FORMAT_BAYER10_BGGR_PACKED_MSFIRST  29
#define PIXEL_FORMAT_BAYER10_PACKED_MSFIRST       PIXEL_FORMAT_BAYER10_GRBG_PACKED_MSFIRST

/* Polar camera formats */
#define PIXEL_FORMAT_STOKES4_12    30  /* 12 bits for each of S0 (x2), S1, and S2 representing Stokes values, for a 48 bit total*/
#define PIXEL_FORMAT_POLAR4_12     31  /* 12 bits for each of each pixel, represented 4 times, for a 48 bit total*/
#define PIXEL_FORMAT_POLAR_RAW4_12 32  /* 12 bits for each of the 4 polar channels (i0, i45, i90, i135), for a 48 bit total*/
#define PIXEL_FORMAT_HSV4_12       33  /* 12 bits for each of Hue (x2), Saturation, and Value, for a 48 bit total*/

/* BGR camera formats */
#define PIXEL_FORMAT_BGR24       34
#define PIXEL_FORMAT_BGR24_NON_DIB PIXEL_FORMAT_BGR24 /* BGR formats are always non-DIB */

/* The standard body European Machine Vision Association (EMVA) have created is 'convention' for pixel naming
 * that describes pixel formats, with this convention being adopted by Vision standards such as the USB3 Vision
 * standard spficied by the Advanced Vision and Imaging Association (AIA).  This convention document is is called
 * Pixel Format Naming Convention (PFNC).  This API predates all of this standardization effort, but the pixel format
 * ued by camers via this API, do use the same pixel formatting specifed by these standards.
 *
 * For reference, the following defines shows how the above pixel formats map to eqivalent ones specified in PFNC
 */
#define PFNC_Mono8             PIXEL_FORMAT_MONO8
#define PFNC_Mono16            PIXEL_FORMAT_MONO16
#define PFNC_YCbCr422_8_CbYCrY PIXEL_FORMAT_YUV422
#define PFNC_RGB8              PIXEL_FORMAT_RGB24
#define PFNC_RGB16             PIXEL_FORMAT_RGB48
#define PFNC_BayerGR8          PIXEL_FORMAT_BAYER8_GRBG
#define PFNC_BayerRG8          PIXEL_FORMAT_BAYER8_RGGB
#define PFNC_BayerGB8          PIXEL_FORMAT_BAYER8_GBRG
#define PFNC_BayerBG8          PIXEL_FORMAT_BAYER8_BGGR
#define PFNC_BayerGR16         PIXEL_FORMAT_BAYER16_GRBG
#define PFNC_BayerRG16         PIXEL_FORMAT_BAYER16_RGGB
#define PFNC_BayerGB16         PIXEL_FORMAT_BAYER16_GBRG
#define PFNC_BayerBG16         PIXEL_FORMAT_BAYER16_BGGR
#define PFNC_Mono12g           PIXEL_FORMAT_MONO12_PACKED_MSFIRST
#define PFNC_BayerGR12g        PIXEL_FORMAT_BAYER12_GRBG_PACKED_MSFIRST
#define PFNC_BayerRG12g        PIXEL_FORMAT_BAYER12_RGGB_PACKED_MSFIRST
#define PFNC_BayerGB12g        PIXEL_FORMAT_BAYER12_GBRG_PACKED_MSFIRST
#define PFNC_BayerBG12g        PIXEL_FORMAT_BAYER12_BGGR_PACKED_MSFIRST
#define PFNC_Mono10g           PIXEL_FORMAT_MONO10_PACKED_MSFIRST
#define PFNC_BayerGR10g        PIXEL_FORMAT_BAYER10_GRBG_PACKED_MSFIRST
#define PFNC_BayerRG10g        PIXEL_FORMAT_BAYER10_RGGB_PACKED_MSFIRST
#define PFNC_BayerGB10g        PIXEL_FORMAT_BAYER10_GBRG_PACKED_MSFIRST
#define PFNC_BayerBG10g        PIXEL_FORMAT_BAYER10_BGGR_PACKED_MSFIRST
/* Furthermore, the GigE Vision specification used a packing scheme not adopted by PFNC, but is used by some
 * of our 'earler' (pre-PFNC) GigE Vision cameras. */
#define GigEVision_1_0_GVSP_PIX_MONO12_PACKED    PIXEL_FORMAT_MONO12_PACKED
#define GigEVision_1_0_GVSP_PIX_BAYGR12_PACKED   PIXEL_FORMAT_BAYER12_GRBG_PACKED
#define GigEVision_1_0_GVSP_PIX_BAYRG12_PACKED   PIXEL_FORMAT_BAYER12_RGGB_PACKED
#define GigEVision_1_0_GVSP_PIX_BAYGB12_PACKED   PIXEL_FORMAT_BAYER12_GBRG_PACKED
#define GigEVision_1_0_GVSP_PIX_BAYBG12_PACKED   PIXEL_FORMAT_BAYER12_BGGR_PACKED

/* Stream State */
#define START_STREAM    0
#define PAUSE_STREAM    1
#define STOP_STREAM     2

/* Preview State */
#define START_PREVIEW   0
#define PAUSE_PREVIEW   1
#define STOP_PREVIEW    2

/* Preview Window Events */
#define PREVIEW_CLOSED      0
#define PREVIEW_MINIMIZED   1
#define PREVIEW_RESTORED    2
#define PREVIEW_ACTIVATED   3 /* The preview window is the active window */
#define PREVIEW_DEACTIVATED 4 /* The preview window is no longer the active window */
#define PREVIEW_RESIZED     5
#define PREVIEW_MOVED       6

/* Trigger types */
#define TRIGGER_TYPE_FREE_RUNNING         0
#define TRIGGER_TYPE_SOFTWARE             1
#define TRIGGER_TYPE_HARDWARE             2

/* Descriptors */
#define PXL_MAX_STROBES         16
#define PXL_MAX_KNEE_POINTS     4

/* Descriptors (advanced features) */
#define PXL_UPDATE_CAMERA 0
#define PXL_UPDATE_HOST   1

/* Default Memory Channel */
#define FACTORY_DEFAULTS_MEMORY_CHANNEL 0

/* Camera Features */
typedef struct _FEATURE_PARAM
{
    float fMinValue;
    float fMaxValue;
} FEATURE_PARAM, *PFEATURE_PARAM;

typedef struct _CAMERA_FEATURE
{
    U32 uFeatureId;
    U32 uFlags;
    U32 uNumberOfParameters;
    FEATURE_PARAM *pParams;
} CAMERA_FEATURE, *PCAMERA_FEATURE; 

typedef struct _CAMERA_FEATURES
{
    U32 uSize;
    U32 uNumberOfFeatures;
    CAMERA_FEATURE *pFeatures;
}  CAMERA_FEATURES, *PCAMERA_FEATURES;


/* Camera Info */
typedef struct _CAMERA_INFO
{
    S8 VendorName [33];
    S8 ModelName [33];
    S8 Description [256];
    S8 SerialNumber[33];
    S8 FirmwareVersion[12];
    S8 FPGAVersion[12];
    S8 CameraName[256];
    S8 XMLVersion[12];      // New as of Release 9
    S8 BootloadVersion[12]; // New as of Release 9
    S8 LensDescription [64];   // New as of Release 9.1
} CAMERA_INFO, *PCAMERA_INFO;

/* External Controller Info */
typedef struct _CONTROLLER_INFO
{
    U32    ControllerSerialNumber;
    U32    TypeMask;               // bit mask defining all control functions of this controller (See CONTROLLER_FLAG_XXXX)
    U32    CameraSerialNumber;     // Assigned camera, or 0 if unassigned.
    char   COMPort [64];           // String identifying the host COM port to which the controller is connected
    BOOL32 USBVitrualPort;         // TRUE if this is a Virtual COM port over USB
    char   VendorName [64];
    char   ModelName [64];
    char   Description [256];
    char   FirmwareVersion[64];
} CONTROLLER_INFO, *PCONTROLLER_INFO;

#define CONTROLLER_FLAG_FOCUS       0x00000001  
#define CONTROLLER_FLAG_ZOOM        0x00000002
#define CONTROLLER_FLAG_IRIS        0x00000004
#define CONTROLLER_FLAG_SHUTTER     0x00000008 // external, mechanical shuter
#define CONTROLLER_FLAG_LIGHTING    0x00000010

/* Frame Descriptor */
typedef struct _FRAME_DESC
{
    U32 uSize;
    float fFrameTime; 
    U32 uFrameNumber;

    struct _Brightness{
        float fValue;
    } Brightness;

    struct{
        float fValue;
    } AutoExposure;

    struct{
        float fValue;
    } Sharpness;

    struct{
        float fValue;
    } WhiteBalance;

    struct{
        float fValue;
    } Hue;

    struct{
        float fValue;
    } Saturation;

    struct{
        float fValue;
    } Gamma;

    struct{
        float fValue;
    } Shutter;

    struct{
        float fValue;
    } Gain;

    struct{
        float fValue;
    } Iris;

    struct{
        float fValue;
    } Focus;

    struct{
        float fValue;
    } Temperature;

    struct{
        float fMode;
        float fType;
        float fPolarity;
        float fDelay;
        float fParameter;
    } Trigger;

    struct{
        float fValue;
    } Zoom;

    struct{
        float fValue;
    } Pan;

    struct{
        float fValue;
    } Tilt;

    struct{
        float fValue;
    } OpticalFilter;

    struct{
        float fMode[PXL_MAX_STROBES];
        float fPolarity[PXL_MAX_STROBES];
        float fParameter1[PXL_MAX_STROBES];
        float fParameter2[PXL_MAX_STROBES];
        float fParameter3[PXL_MAX_STROBES];
    } GPIO;

    struct{
        float fValue;
    } FrameRate;

    struct{
        float fLeft;
        float fTop;
        float fWidth;
        float fHeight;
    } Roi;

    struct{
        float fHorizontal;
        float fVertical;
    } Flip;

    struct{
        float fValue;
    } Decimation;

    struct{
        float fValue;
    } PixelFormat;

    struct{
        float fKneePoint[PXL_MAX_KNEE_POINTS];
    } ExtendedShutter;

    struct{
        float fLeft;
        float fTop;
        float fWidth;
        float fHeight;
    } AutoROI;

    struct{
        float fValue;
    } DecimationMode;

    struct{
        float fRedGain;
        float fGreenGain;
        float fBlueGain;
    } WhiteShading;

    struct{
        float fValue;
    } Rotate;

    struct{
        float fValue;
    } ImagerClkDivisor; /* Added to slow down imager to support slower frame rates */

    struct{
        float fValue;
    } TriggerWithControlledLight;  
          
    struct{
        float fValue;
    } MaxPixelSize;   /* The number of bits used to represent 16-bit data (10 or 12) */

    struct{
        float fValue; 
    } TriggerNumber; /* Valid only for hardware trigger mode 14.  It identifies the frame number in a particular trigger sequence */
    
    struct{
        U32 uMask;
    } ImageProcessing; /* Bit mask describing processing that was performed on the image */

    struct{
        float fHorizontal;
        float fVertical;
    } PixelAddressingValue; /* Valid only for cameras with independant X & Y Pixel Addressing */
    
    double dFrameTime; /* Same as fRametime, but with better resolution/capacity */
    U64    u64FrameNumber; /* Same as uFrameNumber, but with greater capacity */
    
    struct{
        float fValue;
    } BandwidthLimit; /* Upper bound on the amount of bandwidth the camera can use (in mb/s) */

    struct{
        float fValue;
    } ActualFrameRate; /* The frame rate (in frames/sec) being used by the camera */

    struct{
       float fLeft;
       float fTop;
       float fWidth;
       float fHeight;
       float fMaxValue;
    } SharpnessScoreParams;  /* Controls calculation of SharpnessScore.  Valid ony for those cameras that support FEATURE_SHARPNESS_SCORE */

    struct{
       float fValue;
    } SharpnessScore;  /* SharpnessScore of this image.  Valid ony for those cameras that support FEATURE_SHARPNESS_SCORE */

    struct{
        U32   uMode;       /* The type of HDR applied to the image (if any) */
        float fDarkGain;   /* Gain used for dark pixel components */
        float fBrightGain; /* Gain used for bright pixel components */
    } HDRInfo;  /* The type of HDR applied to the image (if any) */
    
    struct{
       U32   uCFA;       /* Type of color filter array used (if color camera) */
       float f0Weight;   /* Contributions of 0 degree polarized light (horizontal).  In percentage */
       float f45Weight;  /* Contributions of 45 degree polarized light (CW from horizontal).  In percentage */
       float f90Weight;  /* Contributions of 90 degree polarized light (vertical).  In percentage */
       float f135Weight; /* Contributions of 135 degree polarized light (CW from horizontal).  In percentage */
       U32   uHSVInterpretation; /* How PIXEL_FORMAT_HSV4_12 should be interpreted */
    } PolarInfo;  /* Polar sub-channel weighting factors (if supported) */

} FRAME_DESC, *PFRAME_DESC;

typedef struct _ERROR_REPORT
{
    PXL_RETURN_CODE uReturnCode;
    S8 strFunctionName[32];
    S8 strReturnCode[32];
    S8 strReport[256];
} ERROR_REPORT, *PERROR_REPORT;

/* Callback identifiers for PxLSetCallback */
#define CALLBACK_PREVIEW      0x01
#define CALLBACK_FORMAT_IMAGE 0x02
#define CALLBACK_FORMAT_CLIP  0x04
#define CALLBACK_FRAME        0x08
#define CALLBACK_PREVIEW_RAW  (0x10 | CALLBACK_PREVIEW)

/* Aliases for backward compatibility */
#define OVERLAY_PREVIEW         CALLBACK_PREVIEW
#define OVERLAY_FORMAT_IMAGE    CALLBACK_FORMAT_IMAGE
#define OVERLAY_FORMAT_CLIP     CALLBACK_FORMAT_CLIP
#define OVERLAY_FRAME           CALLBACK_FRAME

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

/* Camera Properties Flags (8 bits available) */
#define CAMERA_PROPERTY_MONITOR_ACCESS_ONLY  0x01  /* Someone else has control of the camera -- only monitor access allowed */
#define CAMERA_PROPERTY_NOT_ACCESSIBLE       0x02  /* Someone else has exclusive access of the camera -- no other access are permitted */
#define CAMERA_PROPERTY_IP_UNREACHABLE       0x04  /* Cannot access this camera as the camera and NIC are on different subnets */

/* For PxLLoadSettings and PxLSaveSettings */
#define PXL_SETTINGS_FACTORY    FACTORY_DEFAULTS_MEMORY_CHANNEL
#define PXL_SETTINGS_USER       1

/* FEATURE_EXPOSURE (FEATURE_SHUTTER) parameters */
#define FEATURE_EXPOSURE_PARAM_VALUE        0
#define FEATURE_EXPOSURE_PARAM_AUTO_MIN     1
#define FEATURE_EXPOSURE_PARAM_AUTO_MAX     2

/* Used by FEATURE_TRIGGER and FEATURE_GPIO */
#define POLARITY_ACTIVE_LOW     0
#define POLARITY_ACTIVE_HIGH    1

#define POLARITY_NEGATIVE       POLARITY_ACTIVE_LOW
#define POLARITY_POSITIVE       POLARITY_ACTIVE_HIGH    

/* FEATURE_TRIGGER parameters */
#define FEATURE_TRIGGER_PARAM_MODE          0
#define FEATURE_TRIGGER_PARAM_TYPE          1
#define FEATURE_TRIGGER_PARAM_POLARITY      2
#define FEATURE_TRIGGER_PARAM_DELAY         3
#define FEATURE_TRIGGER_PARAM_PARAMETER     4
#define FEATURE_TRIGGER_PARAM_NUMBER        4   /* Alias */
#define FEATURE_TRIGGER_NUM_PARAMS          5

/*
 * Most frequently supported triggering modes.
 * See Capture OEM for a description of each 
 */
#define TRIGGER_MODE_0              0
#define TRIGGER_MODE_1              1
#define TRIGGER_MODE_2              2
#define TRIGGER_MODE_3              3
#define TRIGGER_MODE_4              4
#define TRIGGER_MODE_5              5
#define TRIGGER_MODE_14             14


/* FEATURE_GPIO parameters */
#define FEATURE_GPIO_PARAM_GPIO_INDEX       0  /* Also known as strobe number */
#define FEATURE_GPIO_PARAM_MODE             1
#define FEATURE_GPIO_PARAM_POLARITY         2
/* See below for more information on these parameters */
#define FEATURE_GPIO_PARAM_PARAM_1          3
#define FEATURE_GPIO_PARAM_PARAM_2          4
#define FEATURE_GPIO_PARAM_PARAM_3          5
#define FEATURE_GPIO_NUM_PARAMS             6

/* GPIO Modes */
#define GPIO_MODE_STROBE    0
#define GPIO_MODE_NORMAL    1
#define GPIO_MODE_PULSE     2
#define GPIO_MODE_BUSY      3
#define GPIO_MODE_FLASH     4
#define GPIO_MODE_INPUT     5

/*
 * FEATURE_GPIO parameters 3 to 5 depend on what 
 * mode is being used
 */

/* GPIO Mode Strobe */
#define FEATURE_GPIO_MODE_STROBE_PARAM_DELAY    3
#define FEATURE_GPIO_MODE_STROBE_PARAM_DURATION 4

/* GPIO Mode Normal  */
/* Params 3-5 unused */

/* GPIO Mode Pulse */
#define FEATURE_GPIO_MODE_PULSE_PARAM_NUMBER   3
#define FEATURE_GPIO_MODE_PULSE_PARAM_DURATION 4
#define FEATURE_GPIO_MODE_PULSE_PARAM_INTERVAL 5

/* GPIO Mode Busy    */
/* Params 3-5 unused */

/* GPIO Mode Flash Window */
/* Params 3-5 unused      */

/* GPIO Mode Input */
#define FEATURE_GPIO_MODE_INPUT_PARAM_STATUS   3 /* The current status of the GPI signal */

/* FEATURE_ROI parameters */
#define FEATURE_ROI_PARAM_LEFT      0
#define FEATURE_ROI_PARAM_TOP       1
#define FEATURE_ROI_PARAM_WIDTH     2
#define FEATURE_ROI_PARAM_HEIGHT    3
#define FEATURE_ROI_NUM_PARAMS      4

/* FEATURE_FLIP parameters */
#define FEATURE_FLIP_PARAM_HORIZONTAL   0
#define FEATURE_FLIP_PARAM_VERTICAL     1
#define FEATURE_FLIP_NUM_PARAMS         2

/* FEATURE_SHARPNESS_SCORE parameters */
#define FEATURE_SHARPNESS_SCORE_PARAM_LEFT      0
#define FEATURE_SHARPNESS_SCORE_PARAM_TOP       1
#define FEATURE_SHARPNESS_SCORE_PARAM_WIDTH     2
#define FEATURE_SHARPNESS_SCORE_PARAM_HEIGHT    3
#define FEATURE_SHARPNESS_SCORE_MAX_VALUE       4
#define FEATURE_SHARPNESS_SCORE_NUM_PARAMS      5

/*
 * Pixel Addressing  
 * (FEATURE_DECIMATION/FEATURE_PIXEL_ADDRESSING)
*/
/* Create an alias if there isn't one already */
#ifndef FEATURE_PIXEL_ADDRESSING
#define FEATURE_PIXEL_ADDRESSING    FEATURE_DECIMATION
#endif
#define FEATURE_PIXEL_ADDRESSING_PARAM_VALUE    0
#define FEATURE_PIXEL_ADDRESSING_PARAM_MODE     1
#define FEATURE_PIXEL_ADDRESSING_PARAM_X_VALUE  2
#define FEATURE_PIXEL_ADDRESSING_PARAM_Y_VALUE  3
#define FEATURE_PIXEL_ADDRESSING_NUM_PARAMS     4

#define PIXEL_ADDRESSING_MODE_DECIMATE  0
#define PIXEL_ADDRESSING_MODE_AVERAGE   1
#define PIXEL_ADDRESSING_MODE_BIN       2
#define PIXEL_ADDRESSING_MODE_RESAMPLE  3

#define PIXEL_ADDRESSING_VALUE_NONE     1
#define PIXEL_ADDRESSING_VALUE_BY_2     2

/* FEATURE_EXTENDED_SHUTTER parameters */
#define FEATURE_EXTENDED_SHUTTER_PARAM_NUM_KNEES    0
#define FEATURE_EXTENDED_SHUTTER_PARAM_KNEE_1       1
#define FEATURE_EXTENDED_SHUTTER_PARAM_KNEE_2       2
#define FEATURE_EXTENDED_SHUTTER_PARAM_KNEE_3       3
#define FEATURE_EXTENDED_SHUTTER_PARAM_KNEE_4       4

/* FEATURE_AUTO_ROI parameters */
#define FEATURE_AUTO_ROI_PARAM_LEFT     0
#define FEATURE_AUTO_ROI_PARAM_TOP      1
#define FEATURE_AUTO_ROI_PARAM_WIDTH    2
#define FEATURE_AUTO_ROI_PARAM_HEIGHT   3

/* FEATURE_WHITE_SHADING (Displayed in Capture OEM as White Balance) */
#define FEATURE_WHITE_SHADING_PARAM_RED     0
#define FEATURE_WHITE_SHADING_PARAM_GREEN   1
#define FEATURE_WHITE_SHADING_PARAM_BLUE    2

#define FEATURE_WHITE_BALANCE_PARAM_RED     0
#define FEATURE_WHITE_BALANCE_PARAM_GREEN   1
#define FEATURE_WHITE_BALANCE_PARAM_BLUE    2
#define FEATURE_WHITE_BALANCE_NUM_PARAMS    3

/* Standard Rotations */
#define FEATURE_ROTATE_0_DEG        0
#define FEATURE_ROTATE_90_DEG       90
#define FEATURE_ROTATE_180_DEG      180
#define FEATURE_ROTATE_270_DEG      270

/* FEATURE_MAX_PACKET_SIZE */
/*      Parameter 0 -- supported packet sizes */
#define FEATURE_MAX_PACKET_SIZE_NORMAL 1500
#define FEATURE_MAX_PACKET_SIZE_JUMBO  9000

/* FEATURE_SPECIAL_CAMERA_MODE */
#define FEATURE_SPECIAL_CAMERA_MODE_NONE              0
#define FEATURE_SPECIAL_CAMERA_MODE_FIXED_FRAME_RATE  1

/* FEATURE_GAIN_HDR */
#define FEATURE_GAIN_HDR_MODE_NONE        0
#define FEATURE_GAIN_HDR_MODE_CAMERA      1
#define FEATURE_GAIN_HDR_MODE_INTERLEAVED 2

/* FEATURE_POLAR_WEIGHTINGS */
#define FEATURE_POLAR_WEIGHTINGS_0_DEG    0
#define FEATURE_POLAR_WEIGHTINGS_45_DEG   1
#define FEATURE_POLAR_WEIGHTINGS_90_DEG   2
#define FEATURE_POLAR_WEIGHTINGS_135_DEG  3

/* FEATURE_POLAR_HSV_INTERPRETATION */
#define FEATURE_POLAR_HSV_AS_COLOR        0
#define FEATURE_POLAR_HSV_AS_ANGLE        1
#define FEATURE_POLAR_HSV_AS_DEGREE       2

/* Color filter array used to 'colorize' the camera */
#define PXL_CFA_NONE    0 /* 'Mono' camera */
#define PXL_CFA_RGGB    1
#define PXL_CFA_GBRG    2
#define PXL_CFA_GRBG    3
#define PXL_CFA_BGGR    4

/* Camera flags used with PxLInitializeEx (32 bits available) */
#define CAMERA_INITIALIZE_EX_FLAG_MONITOR_ACCESS_ONLY  0x00000001  /* We only want monitor (read-only) access to the camera */
#define CAMERA_INITIALIZE_EX_FLAG_ISSUE_STREAM_STOP    0x00000002  /* A stream stop is issued as part of camera initialization */

/* Flat-field Calibration Types */
#define FFC_TYPE_UNKNOWN            0
#define FFC_TYPE_UNCALIBRATED       1
#define FFC_TYPE_FACTORY            2
#define FFC_TYPE_FIELD              3

/* Types of IP Address Assignments */
#define PXL_IP_UNKNOWN_ASSIGNMENT      0
#define PXL_IP_DHCP_ASSIGNED           1
#define PXL_IP_LLA_ASSIGNED            2
#define PXL_IP_STATIC_PERSISTENT       3  /* User-assigned - Survives power cycles */
#define PXL_IP_STATIC_VOLATILE         4  /* User-assigned - Lost on power cycle   */

typedef struct _CLIP_ENCODING_INFO
{
    U32 uStreamEncoding;     /* Encoding (compression) scheme used for the stream */
    U32 uDecimationFactor;   /* Used to reduce the number of frames used while encoding the stream */
    F32 playbackFrameRate;   /* Frame rate to be used for decoding (during playback) */
    U32 playbackBitRate;     /* Provides guidance to the compression algorithms for encoding; higher bitrate == less compression */
}  CLIP_ENCODING_INFO, *PCLIP_ENCODING_INFO;

/*
 * 'Magic number' that appears as the first 4 bytes of
 * a PixeLINK data stream (.pds) file
 * see PxLGetClipPds();
 */
#define PIXELINK_DATA_STREAM_MAGIC_NUMBER 0x04040404

#define CLIP_PLAYBACK_FRAMERATE_DEFAULT (30.0f) /* 30 frames/second -- a playback rate commonly used */
#define CLIP_PLAYBACK_FRAMERATE_CAPTURE (-1.0f) /* Use this value to indicate the playback rate matches the camera's FEATURE_ACTUAL_FRAME_RATE */
#define CLIP_PLAYBACK_BITRATE_DEFAULT (1000000) /* 1,000,000 bits / second -- a playack bitrate commonly used */
#define CLIP_DECIMATION_NONE (1)                /* all images streamed from the camera will be captured into the clip */

#endif /* PIXELINK_COM_PIXELINKTYPES_H */
