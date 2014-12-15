////////////////////////////////////////////////////////////////////////////////
// Header file for frame grabber camera.
// intek, c. kuehnel, 25.2.2005
////////////////////////////////////////////////////////////////////////////////

#ifndef FGCAMERA_H
#define FGCAMERA_H

#include <basetype.h>
#include <errstat.h>

/* Always use standard call if not suppressed */
#ifndef FGLUSECDECL
 #define FGLUSESTDCALL 1
#endif

/* See what link convention to use */
#ifdef FGLUSESTDCALL
 #define LINKCONVENTION __stdcall
#else
 #define LINKCONVENTION __cdecl
#endif

/* Compile the library and export our functions */
#ifdef FGLEXPORT
 #define FSPEC __declspec(dllexport) LINKCONVENTION
 #define CSPEC __declspec(dllexport)
#endif

/* Link library into program code */
#ifdef FGLLINK
 #define FSPEC LINKCONVENTION
 #define CSPEC
#endif

/* Import functions from dynamic link library */
#ifndef FSPEC
 #define FSPEC __declspec(dllimport) LINKCONVENTION
 #define CSPEC __declspec(dllimport)
#endif

/* Remove all modifiers if requested */
#ifdef NOMODIFIER
  #undef FSPEC
  #undef CSPEC
#endif

// Notification WPARAMs
#define WPARAM_NODELISTCHANGED          0       // LPARAM = no meaning
#define WPARAM_ERROR                    1       // LPARAM = Ored error flags from all cards
#define WPARAM_FRAMESREADY              2       // LPARAM = CFGCamera Iso Context
#define WPARAM_ERRORFLAGSCARD0          3       // LPARAM = Detailed error field
#define WPARAM_ERRORFLAGSCARD1          4       // LPARAM = Detailed error field
#define WPARAM_ERRORFLAGSCARD2          5       // LPARAM = Detailed error field
#define WPARAM_ERRORFLAGSCARD3          6       // LPARAM = Detailed error field
#define WPARAM_ERRORFLAGSCARD4          7       // LPARAM = Detailed error field
#define WPARAM_ERRORFLAGSCARD5          8       // LPARAM = Detailed error field
#define WPARAM_ERRORFLAGSCARD6          9       // LPARAM = Detailed error field
#define WPARAM_ERRORFLAGSCARD7          10      // LPARAM = Detailed error field
#define WPARAM_ERRORFLAGSCARD8          11      // LPARAM = Detailed error field
#define WPARAM_ERRORFLAGSCARD9          12      // LPARAM = Detailed error field
#define WPARAM_BUSRESET                 13      // Bus reset event

// Frame flags
#define FGF_INVALID             0x00000001      // Data area might be damaged
#define FGF_LAST                0x00000002      // Last in queue
#define FGF_DMAHALTED           0x00000004      // Dma was halted in between
#define FGF_FORCEPOST           0x10000000      // Force post to driver in LIMPMODE

// Parameter ImageFormat: Rate specifies the mode when RES_SCALABLE selected!
#define MAKEIMAGEFORMAT(Res,Col,RateOrMode) (((UINT32)(UINT8)Res<<16)|\
                                            ((UINT32)(UINT8)Col<<8)|\
                                            ((UINT32)(UINT8)RateOrMode))
#define IMGRES(n)          (((n)>>16)&0xFF)
#define IMGCOL(n)          (((n)>>8)&0xFF)
#define IMGRATE(n)         ((n)&0xFF)
#define IMGMODE(n)         IMGRATE(n)

// Parameter DCAM: Rate specifies the color mode when FORMAT7 selected!
#define MAKEDCAMFORMAT(Format,Mode,RateOrColor) (0x80000000| \
                                                ((UINT32)(UINT8)Format<<16)|\
                                                ((UINT32)(UINT8)Mode<<8)| \
                                                ((UINT32)(UINT8)RateOrColor))
#define DCAMFORMAT(n)      (((n)>>16)&0xFF)
#define DCAMMODE(n)        (((n)>>8)&0xFF)
#define DCAMRATE(n)        ((n)&0xFF)
#define DCAMCOL(n)         DCAMRATE(n)

#define ISDCAMFORMAT(n)    ((n)&0x80000000?TRUE:FALSE)

// Parameter Trigger
#define MAKETRIGGER(On,Pol,Src,Mode,Parm) (((UINT32)(UINT8)On<<25)|\
                                          ((UINT32)(UINT8)Pol<<24)|\
                                          ((UINT32)(UINT8)Src<<21)|\
                                          ((UINT32)(UINT8)Mode<<16)|\
                                          (Parm))
#define TRGON(n)           (((n)>>25)&0x1)
#define TRGPOL(n)          (((n)>>24)&0x1)
#define TRGSRC(n)          (((n)>>21)&0x7)
#define TRGMODE(n)         (((n)>>16)&0xF)
#define TRGPARM(n)         ((n)&0xFFF)

// Special parameter value for DCAM 'feature'
#define PVAL_OFF                ((UINT32)-1)
#define PVAL_AUTO               ((UINT32)-2)
#define PVAL_ONESHOT            ((UINT32)-3)

// Special parameter value for burst count
#define BC_INFINITE             0
#define BC_ONESHOT              1

// Useful macros
#define HIUINT16(n)             ((UINT16)((n)>>16))
#define LOUINT16(n)             ((UINT16)((n)&0xFFFF))
#define MAKEUINT32(Hi,Lo)       (((UINT32)(Hi)<<16)|(Lo))

// DMA flags: ISO security checking flags
#define DMAF_CHECKSYNC          0x01            // Check for double sync field
#define DMAF_CHECKCYCLE         0x02            // Check for wrong cycle
#define DMAF_CHECKLENGTH        0x04            // Check each packet length
#define DMAF_FRAMESTART         0x80            // Create frame start events

#define DMAF_CONTEXTISEVT       0x10            // IsoContext is event handle
#define DMAF_CONTEXTISCB        0x20            // IsoContext is callback

#ifndef INFINITE
 #define INFINITE 0xFFFFFFFF
#endif

// Enumeration for resolutions
enum FG_RESOLUTION
{
  RES_160_120=0,
  RES_320_240,
  RES_640_480,
  RES_800_600,
  RES_1024_768,
  RES_1280_960,
  RES_1600_1200,
  RES_SCALABLE,
  RES_LAST
};

// Enumeration for color modes
enum FG_COLORMODE
{
  CM_Y8=0,
  CM_YUV411,
  CM_YUV422,
  CM_YUV444,
  CM_RGB8,
  CM_Y16,
  CM_RGB16,
  CM_SY16,
  CM_SRGB16,
  CM_RAW8,
  CM_RAW16,
  CM_LAST
};

// Enumeration for frame rates
enum FG_FRAMERATE
{
  FR_1_875=0,
  FR_3_75,
  FR_7_5,
  FR_15,
  FR_30,
  FR_60,
  FR_120,
  FR_240,
  FR_LAST
};

// Enumeration for DMA mode
enum FG_DMA
{
  DMA_CONTINOUS=0,
  DMA_LIMP,
  DMA_REPLACE,
  DMA_LAST
};

// Enumeration for Bayer pattern
enum FG_BAYERPATTERN
{
  BP_RGGB=0,
  BP_GRBG,
  BP_BGGR,
  BP_GBRG,
  BP_LAST
};

// Specific Parameter data types
enum FG_PARSPECIFIC
{
  FGPS_INVALID=0,
  FGPS_FEATUREINFO,
  FGPS_TRIGGERINFO,
  FGPS_COLORFORMAT,
  FGPS_LAST
};

// Enumeration for physical speed
enum FG_PHYSPEED
{
  PS_100MBIT=0,
  PS_200MBIT,
  PS_400MBIT,
  PS_800MBIT,
  PS_AUTO,
  PS_LAST
};

// Parameters
enum FG_PARAMETER
{
  FGP_IMAGEFORMAT=0,                            // Compact image format
  FGP_ENUMIMAGEFORMAT,                          // Enumeration (Reset,Get)
  FGP_BRIGHTNESS,                               // Set image brightness
  FGP_AUTOEXPOSURE,                             // Set auto exposure
  FGP_SHARPNESS,                                // Set image sharpness
  FGP_WHITEBALCB,                               // Blue
  FGP_WHITEBALCR,                               // Red
  FGP_HUE,                                      // Set image hue
  FGP_SATURATION,                               // Set color saturation
  FGP_GAMMA,                                    // Set gamma
  FGP_SHUTTER,                                  // Shutter time
  FGP_GAIN,                                     // Gain
  FGP_IRIS,                                     // Iris
  FGP_FOCUS,                                    // Focus
  FGP_TEMPERATURE,                              // Color temperature
  FGP_TRIGGER,                                  // Trigger
  FGP_TRIGGERDLY,                               // Delay of trigger
  FGP_WHITESHD,                                 // Whiteshade
  FGP_FRAMERATE,                                // Frame rate
  FGP_ZOOM,                                     // Zoom
  FGP_PAN,                                      // Pan
  FGP_TILT,                                     // Tilt
  FGP_OPTICALFILTER,                            // Filter
  FGP_CAPTURESIZE,                              // Size of capture
  FGP_CAPTUREQUALITY,                           // Quality
  FGP_PHYSPEED,                                 // Set speed for asy/iso
  FGP_XSIZE,                                    // Image XSize
  FGP_YSIZE,                                    // Image YSize
  FGP_XPOSITION,                                // Image x position
  FGP_YPOSITION,                                // Image y position
  FGP_PACKETSIZE,                               // Packet size
  FGP_DMAMODE,                                  // DMA mode (continuous or limp)
  FGP_BURSTCOUNT,                               // Number of images to produce
  FGP_FRAMEBUFFERCOUNT,                         // Number of frame buffers
  FGP_USEIRMFORBW,                              // Allocate bandwidth or not (IsoRscMgr)
  FGP_ADJUSTPARAMETERS,                         // Adjust parameters or fail
  FGP_STARTIMMEDIATELY,                         // Start bursting immediately
  FGP_FRAMEMEMORYSIZE,                          // Read only: Frame buffer size
  FGP_COLORFORMAT,                              // Read only: Colorformat
  FGP_IRMFREEBW,                                // Read only: Free iso bytes for 400MBit
  FGP_DO_FASTTRIGGER,                           // Fast trigger (no ACK)
  FGP_DO_BUSTRIGGER,                            // Broadcast trigger
  FGP_RESIZE,                                   // Start/Stop resizing
  FGP_USEIRMFORCHN,                             // Get channel over isochronous resource manager
  FGP_CAMACCEPTDELAY,                           // Delay after writing values
  FGP_ISOCHANNEL,                               // Iso channel
  FGP_CYCLETIME,                                // Read cycle time
  FGP_DORESET,                                  // Reset camera
  FGP_DMAFLAGS,                                 // Flags for ISO DMA
  FGP_R0C,                                      // Ring 0 call gate
  FGP_BUSADDRESS,                               // Exact bus address
  FGP_CMDTIMEOUT,                               // Global bus command timeout
  FGP_CARD,                                     // Card number of this camera (set before connect)
  FGP_LICENSEINFO,                              // Query license information
  FGP_PACKETCOUNT,                              // Read only: Packet count
  FGP_DO_MULTIBUSTRIGGER,                       // Do trigger on several busses
  FGP_CARDRESET,                                // Do reset on card (for hard errors)

  FGP_LAST
};

typedef void (LINKCONVENTION FGCALLBACK)(void* Context,UINT32 wParam,void* lParam);

#define FGIF_ISOCONTEXTISHWND   0x00000001      // FGInit flag: IsoContext is window handle

typedef struct                                  // Parameters for init
{
  void*         hWnd;                           // Window handle for notification
  UINT16        Msg;                            // Message to send to window
  FGCALLBACK   *pCallback;                      // Pointer to callback
  void*         Context;                        // Context for callback
  UINT32        Flags;                          // Flags for this module
}FGINIT;

typedef struct                                  // Struct for a frame
{
  FGHANDLE      System;                         // For system use only
  UINT32        Flags;                          // Flags: Last, Invalid, no post etc...
  UINT16        Id;                             // Continous ID
  UINT8        *pData;                          // Data pointer
  UINT32        Length;                         // Buffers length
  UINT32HL      RxTime;                         // Receive time as 100ns ticks since 1.1.1601
  UINT32        BeginCycleTime;                 // Frame begin as bus time
  UINT32        EndCycleTime;                   // Frame end as bus time
  UINT32        Reserved[2];                    // Reserved for system use
}FGFRAME;

typedef struct                                  // Specific for feature
{
  UINT8         ReadOutCap   : 1;               // Value can be read
  UINT8         OnOffCap     : 1;               // Switchable
  UINT8         OnOffState   : 1;               // On-State
  UINT8         AutoCap      : 1;               // Auto mode capable
  UINT8         AutoState    : 1;               // Auto state
  UINT8         OnePushCap   : 1;               // One push capable
  UINT8         OnePushState : 1;               // One push state
}FGPSFEATURE;

typedef struct                                  // Specific for trigger
{
  UINT32        ReadOutCap    : 1;              // Value can be read
  UINT32        OnOffCap      : 1;              // Switchable
  UINT32        OnOffState    : 1;              // On-State
  UINT32        PolarityCap   : 1;              // Polarity capable
  UINT32        PolarityState : 1;              // Polarity state
  UINT32        ReadInputCap  : 1;              // Input readable
  UINT32        InputState    : 1;              // Input state
  UINT32        TriggerSrcCap : 8;              // Trigger source capabilities
  UINT32        TriggerSrc    : 4;              // Actual trigger source
  UINT32        TriggerModeCap: 16;             // Trigger mode capabilities
  UINT32        TriggerMode   : 4;              // Actual Trigger mode
  UINT32        TriggerParameter : 12;          // Actual parameter
}FGPSTRIGGER;

typedef struct                                  // Ring 0 call gate parameter
{
  void*         pF;
  void*         pN;
}FGPSR0C;

typedef struct
{
  UINT8         Type;                           // Type, 0=invalid, no specific
  union
  {
    FGPSFEATURE FeatureInfo;                    // Type=1
    FGPSTRIGGER TriggerInfo;                    // Type=2
    FGPSR0C     R0CInfo;                        // Type=3
  }Data;
}FGPSPECIFIC;

typedef struct
{
  UINT32        IsValue;                        // Actual value
  UINT32        MinValue;                       // Parameters min. value
  UINT32        MaxValue;                       // Parameters max. value
  UINT32        Unit;                           // Parameters unit
  FGPSPECIFIC   Specific;                       // Parameters specific
}FGPINFO;

typedef struct                                  // Info for a device
{
  UINT32HL      Guid;                           // GUID of this device
  UINT8         CardNumber;                     // Card number
  UINT8         NodeId;                         // Depends on bus topology
  UINT8         Busy;                           // Actually busy
}FGNODEINFO;

typedef union
{
  struct
  {
    UINT32      Port        : 8;
    UINT32      PCIFunction : 8;
    UINT32      PCIDevice   : 8;
    UINT32      PCIBus      : 8;
  };
  UINT32        AsUINT32;
}FGBUSADDRESS;

typedef struct                                  // Argument for CFGIsoDma::Start
{
  UINT8         CardNr;                         // On which card
  UINT8         IsoChn;                         // Isochronous channel
  UINT16        FrameCnt;                       // Frames to allocate
  UINT32        PktCnt;                         // Number of packets
  UINT16        PktSize;                        // Size of single packet (bytes)
  UINT8         DmaMode;                        // DMA mode
  UINT8         DmaFlags;                       // DMA flags
  void*         NotifyContext;                  // Notification context
  UINT32        RealBufCnt;                     // Allocated buffer count
}FGISODMAPARMS;

////////////////////////////////////////////////////////////////////////////////
// String arrays.
////////////////////////////////////////////////////////////////////////////////

#ifdef __cplusplus
extern "C" {
#endif

extern CSPEC char* FGBayerPatternStr[BP_LAST];
extern CSPEC char* FGSpeedStr[PS_LAST];
extern CSPEC char* FGColorFormatStr[CM_LAST];
extern CSPEC char* FGFrameRateStr[FR_LAST];
extern CSPEC char* FGParameterStr[FGP_LAST];
extern CSPEC char* FGResolutionStr[RES_LAST];

#ifdef __cplusplus
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Class for frame grabber like camera.
// We only include class specific if we have C++.
////////////////////////////////////////////////////////////////////////////////

#ifdef __cplusplus

class CCamera;

class CSPEC CFGCamera
{
protected:
  UINT8                 m_Speed;                // Selected speed
  UINT8                 m_Card;                 // Card index
  UINT16                m_Reserved;             // Reserved
  void*                 m_Context;              // Context for user
  CCamera              *m_pDCam;                // Internal work class

public:
                        CFGCamera();
  virtual              ~CFGCamera();

  virtual CCamera*      GetPtrDCam();

  virtual UINT32        WriteRegister(UINT32 Address,UINT32 Value);
  virtual UINT32        ReadRegister(UINT32 Address,UINT32 *pValue);

  virtual UINT32        WriteBlock(UINT32 Address,UINT8 *pData,UINT32 Length);
  virtual UINT32        ReadBlock(UINT32 Address,UINT8 *pData,UINT32 Length);

  virtual UINT32        Connect(UINT32HL *pGuid,void* IsoContext=NULL);
  virtual UINT32        Disconnect();

  virtual UINT32        SetParameter(UINT16 Which,UINT32 Value);
  virtual UINT32        GetParameter(UINT16 Which,UINT32 *pValue);
  virtual UINT32        GetParameterInfo(UINT16 Which,FGPINFO *pInfo);

  virtual UINT32        OpenCapture();
  virtual UINT32        CloseCapture();

  virtual UINT32        AssignUserBuffers(UINT32 Cnt,UINT32 Size,void* *ppMemArray);

  virtual UINT32        StartDevice();
  virtual UINT32        StopDevice();

  virtual UINT32        GetFrame(FGFRAME *pFrame,UINT32 TimeoutInMs=INFINITE);
  virtual UINT32        PutFrame(FGFRAME *pFrame);
  virtual UINT32        DiscardFrames();

  virtual UINT32        GetDeviceName(char *pAll,UINT32 MaxLength,char *pDev=NULL);
  virtual void*         GetContext();

  virtual UINT32        GetLicenseRequest(char *pStr,UINT32 MaxLen);
};

////////////////////////////////////////////////////////////////////////////////
// Class for a broadcast object.
////////////////////////////////////////////////////////////////////////////////

class CSPEC CBroadcast
{
protected:
  CFGCamera            *m_pFGCamera;
  void                 *m_Handle;

public:
                        CBroadcast(CFGCamera *pFGCamera);
  virtual               ~CBroadcast();

  virtual UINT32        WriteRegister(UINT32 Address,UINT32 Value);
  virtual UINT32        WriteBlock(UINT32 Address,UINT8 *pData,UINT32 Length);
};

////////////////////////////////////////////////////////////////////////////////
// Class for a pure DMA object.
////////////////////////////////////////////////////////////////////////////////

class CSPEC CFGIsoDma
{
public:
  class CIsoDma        *m_pIsoDma;              // Worker object

                        CFGIsoDma();
  virtual               ~CFGIsoDma();

  virtual UINT32        OpenCapture(FGISODMAPARMS *pParms);
  virtual UINT32        CloseCapture();

  virtual UINT32        GetFrame(FGFRAME *pFrame,UINT32 TimeoutInMs);
  virtual UINT32        PutFrame(FGFRAME *pFrame);
  virtual UINT32        DiscardFrames();

  virtual UINT32        AssignUserBuffers(UINT32 Cnt,UINT32 Size,void* *ppMemArray);
  virtual UINT32        Resize(UINT32 PktCnt,UINT32 PktSize);
  
  virtual UINT32        AnnounceBuffer(UINT8 *pMem,UINT32 Size,FGHANDLE *phBuffer);
  virtual UINT32        RevokeBuffer(FGHANDLE hBuffer);
};

#endif

////////////////////////////////////////////////////////////////////////////////
// Global management functions.
////////////////////////////////////////////////////////////////////////////////

#ifdef __cplusplus
extern "C" {
#endif

UINT32 FSPEC FGInitModule(FGINIT *pArg /* can be NULL */);
void   FSPEC FGExitModule();

UINT32 FSPEC FGGetNodeList(FGNODEINFO *pInfo,UINT32 MaxCnt,UINT32 *pRealCnt);

UINT32 FSPEC FGGetHostLicenseRequest(char *pStr,UINT32 MaxLen);
UINT32 FSPEC FGGetLicenseInfo(UINT8 *pLicenseType);

#ifdef __cplusplus
}
#endif

#endif

