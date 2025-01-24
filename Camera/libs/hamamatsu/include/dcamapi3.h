/* **************************************************************** *

        dcamapi3.h:	July 18, 2013

 * **************************************************************** */

#ifndef _INCLUDE_DCAMAPI_H

#ifdef __cplusplus

/* C++ */

extern "C"
{

#endif

    /* **************************************************************** *

            platform absorber

     * **************************************************************** */

    /* determine TARGET OS */

#if !defined(DCAM_TARGETOS_IS_WIN32) && !defined(DCAM_TARGETOS_IS_MACOSX) && !defined(DCAM_TARGETOS_IS_LINUX)

#if defined(WIN32) || defined(_INC_WINDOWS)

#ifndef _INC_WINDOWS
#error WINDOWS.H is not included.
#endif

#define DCAM_TARGETOS_IS_WIN32

#elif defined(LINUX)

#define DCAM_TARGETOS_IS_LINUX

#elif defined(MACOSX) || __ppc64__ || __i386__ || __x86_64__

#define DCAM_TARGETOS_IS_MACOSX

#else

    /* now DCAM only supports Windows, Linux and MacOSX */

#error DCAM requires one of definition WIN32, MACOSX and LINUX

#endif

#endif // ! defined( DCAM_TARGETOS_IS_WIN32 ) && ! defined( DCAM_TARGETOS_IS_MACOSX ) && ! defined(
       // DCAM_TARGETOS_IS_LINUX )

    /* **************************************************************** *

            defines

     * **************************************************************** */

#ifdef DCAM_TARGETOS_IS_WIN32
    typedef unsigned long _DWORD;
#else
typedef unsigned int _DWORD;
typedef unsigned char BYTE;
#endif

    /* define for Linux */

#ifdef DCAM_TARGETOS_IS_LINUX

    typedef unsigned int BOOL;

#endif

#ifndef FALSE
#define FALSE 0
#endif

#ifndef TRUE
#define TRUE 1
#endif

    /*** --- initialize option --- ***/

#define DCAMINIT_APIVER_0310 "|apiver=310"
#define DCAMINIT_APIVER_LATEST "|apiver=latest"

#define DCAMINIT_DEFAULT DCAMINIT_APIVER_LATEST

    /* define - HDCAMSIGNAL */

#if defined(DCAM_TARGETOS_IS_WIN32)
    typedef HANDLE HDCAMSIGNAL;
#elif defined(DCAM_TARGETOS_IS_MACOSX)
typedef MPEventID HDCAMSIGNAL;
#else
typedef pthread_cond_t* HDCAMSIGNAL;
#endif

    /* define for MacOSX */

#if !defined(DCAMAPI_VERMIN) || DCAMAPI_VERMIN <= 3200
#ifdef DCAM_TARGETOS_IS_MACOSX

    typedef unsigned long SIZE;

#ifndef _OBJC_OBJC_H_
    typedef signed char BOOL;
#endif
#endif
#endif // ! defined(DCAMAPI_VERMIN) || DCAMAPI_VERMIN <= 3200

    typedef struct
    {
        int32 cx;
        int32 cy;
    } DCAM_SIZE;

    /*** --- datatypes --- ***/

    DCAM_DECLARE_BEGIN( enum, DCAM_DATATYPE )
{
	DCAM_DATATYPE_NONE			=	0,

	DCAM_DATATYPE_UINT8			=	0x00000001,	/* bit 0 */
	DCAM_DATATYPE_UINT16		=	0x00000002,	/* bit 1 */
	DCAM_DATATYPE_UINT32		=	0x00000008,	/* bit 3 */

	DCAM_DATATYPE_MOSAIC8		=	0x00010000,	/* bit 16,  8bit mosaic */
	DCAM_DATATYPE_MOSAIC16		=	0x00020000,	/* bit 17, 16bit mosaic */
	DCAM_DATATYPE_RGB24			=	0x00040000,	/* bit 18,  8bit*3, [ r0, g0, b0], [r1, g1, b1] */
	DCAM_DATATYPE_RGB48			=	0x00100000,	/* bit 20, 16bit*3, [ r0, g0, b0], [r1, g1, b1] */

	/* just like 1394 format, Y is unsigned, U and V are signed. */
	/* about U and V, signal level is from -128 to 128, data value is from 0x00 to 0xFF */
	DCAM_DATATYPE_YUV411		=	0x01000000,	/* 8bit, [ u0, y0, y1, v0, y2, y3 ], [u4, y4, y5, v4, v6, y7], */
	DCAM_DATATYPE_YUV422		=	0x02000000,	/* 8bit, [ u0, y0, v0, y1 ], [u2, y2, v2, v3 ], */
	DCAM_DATATYPE_YUV444		=	0x04000000, /* 8bit, [ u0, y0, v0 ], [ u1, y1, v1 ], */

	/* for backward compatibility */
	DCAM_DATATYPE_INT8			=	0x00000010,	/* bit 4 */
	DCAM_DATATYPE_INT16			=	0x00000020,	/* bit 5 */
	DCAM_DATATYPE_INT32			=	0x00000080,	/* bit 7 */

	DCAM_DATATYPE_BGR24			=	0x00000400,	/* bit 10,  8bit*3, [ b0, g0, r0], [b1, g1, r1] */
	DCAM_DATATYPE_BGR48			=	0x00001000,	/* bit 12, 16bit*3, [ b0, g0, r0], [b1, g1, r1] */

	_end_of_dcam_datatype
}
DCAM_DECLARE_END( DCAM_DATATYPE )

/*** --- bitstypes --- ***/

DCAM_DECLARE_BEGIN( enum, DCAM_BITSTYPE )
{
/**/	DCAM_BITSTYPE_NONE		=	0x00000000,
/**/	DCAM_BITSTYPE_INDEX8	=	0x00000001,
/**/	DCAM_BITSTYPE_RGB16		=	0x00000002,
/**/	DCAM_BITSTYPE_RGB24		=	0x00000004,	/* 8bit, [ b0, g0, r0] */
/**/	DCAM_BITSTYPE_RGB32		=	0x00000008,

	_end_of_dcam_bitstype
}
DCAM_DECLARE_END( DCAM_BITSTYPE )

/*** --- capture mode --- ***/

DCAM_DECLARE_BEGIN( enum, DCAM_CAPTUREMODE )
{
		DCAM_CAPTUREMODE_SNAP		= 0,
		DCAM_CAPTUREMODE_SEQUENCE	= 1,

	_end_of_dcam_capturemode
}
DCAM_DECLARE_END( DCAM_CAPTUREMODE )

/*** --- camera capability --- ***/
enum {
	DCAM_QUERYCAPABILITY_FUNCTIONS			= 0,
	DCAM_QUERYCAPABILITY_DATATYPE			= 1,
	DCAM_QUERYCAPABILITY_BITSTYPE			= 2,
	DCAM_QUERYCAPABILITY_EVENTS				= 3,

	DCAM_QUERYCAPABILITY_AREA				= 4
};

    enum
    {
        DCAM_CAPABILITY_BINNING2 = 0x00000002,
        DCAM_CAPABILITY_BINNING4 = 0x00000004,
        DCAM_CAPABILITY_BINNING8 = 0x00000008,
        DCAM_CAPABILITY_BINNING16 = 0x00000010,
        DCAM_CAPABILITY_BINNING32 = 0x00000020,
        DCAM_CAPABILITY_TRIGGER_EDGE = 0x00000100,
        DCAM_CAPABILITY_TRIGGER_LEVEL = 0x00000200,
        DCAM_CAPABILITY_TRIGGER_MULTISHOT_SENSITIVE = 0x00000400,
        DCAM_CAPABILITY_TRIGGER_CYCLE_DELAY = 0x00000800,
        DCAM_CAPABILITY_TRIGGER_SOFTWARE = 0x00001000,
        DCAM_CAPABILITY_TRIGGER_FASTREPETITION = 0x00002000,
        DCAM_CAPABILITY_TRIGGER_TDI = 0x00004000,
        DCAM_CAPABILITY_TRIGGER_TDIINTERNAL = 0x00008000,
        DCAM_CAPABILITY_TRIGGER_POSI = 0x00010000,
        DCAM_CAPABILITY_TRIGGER_NEGA = 0x00020000,
        DCAM_CAPABILITY_TRIGGER_START = 0x00040000,
        /* reserved = 0x00080000, */
        /* reserved = 0x00400000, */
        DCAM_CAPABILITY_TRIGGER_SYNCREADOUT = 0x00800000,
        DCAM_CAPABILITY_BINNING6 = 0x01000000,
        DCAM_CAPABILITY_BINNING12 = 0x02000000,

        /*** --- from 2.1.2 --- ***/
        DCAM_CAPABILITY_ATTACHBUFFER = 0x00100000,
        DCAM_CAPABILITY_RAWDATA = 0x00200000,

        DCAM_CAPABILITY_ALL = 0x07b7FF3E
    };

    /*** --- update --- ***/
    enum
    {
        DCAM_UPDATE_RESOLUTION = 0x0001,
        DCAM_UPDATE_AREA = 0x0002,
        DCAM_UPDATE_DATATYPE = 0x0004,
        DCAM_UPDATE_BITSTYPE = 0x0008,
        DCAM_UPDATE_EXPOSURE = 0x0010,
        DCAM_UPDATE_TRIGGER = 0x0020,
        DCAM_UPDATE_DATARANGE = 0x0040,
        DCAM_UPDATE_DATAFRAMEBYTES = 0x0080,

        DCAM_UPDATE_PROPERTY = 0x0100, /* reserved */

        DCAM_UPDATE_ALL = 0x01ff
    };

    /*** --- trigger mode --- ***/
    enum
    {
        DCAM_TRIGMODE_INTERNAL = 0x0001,
        DCAM_TRIGMODE_EDGE = 0x0002,
        DCAM_TRIGMODE_LEVEL = 0x0004,
        DCAM_TRIGMODE_MULTISHOT_SENSITIVE = 0x0008,
        DCAM_TRIGMODE_CYCLE_DELAY = 0x0010,
        DCAM_TRIGMODE_SOFTWARE = 0x0020,
        DCAM_TRIGMODE_FASTREPETITION = 0x0040,
        DCAM_TRIGMODE_TDI = 0x0080,
        DCAM_TRIGMODE_TDIINTERNAL = 0x0100,
        DCAM_TRIGMODE_START = 0x0200,
        DCAM_TRIGMODE_SYNCREADOUT = 0x0400
    };

    /*** --- trigger polarity --- ***/
    enum
    {
        DCAM_TRIGPOL_NEGATIVE = 0x0000,
        DCAM_TRIGPOL_POSITIVE = 0x0001
    };

    /* **************************************************************** *

            DCAM-API v3.1 or older functions

     * **************************************************************** */

    /*** --- error function --- ***/

    int32 DCAMAPI dcam_getlasterror(HDCAM h, char* buf DCAM_DEFAULT_ARG, _DWORD bytesize DCAM_DEFAULT_ARG);

    /*** --- initialize and finalize --- ***/

    BOOL DCAMAPI dcam_init(void* reserved1 DCAM_DEFAULT_ARG,
                           int32* pCount DCAM_DEFAULT_ARG,
                           const char* option DCAMINIT_DEFAULT_ARG);
    BOOL DCAMAPI dcam_uninit(void* reserved1 DCAM_DEFAULT_ARG, const char* reserved2 DCAM_DEFAULT_ARG);
    BOOL DCAMAPI dcam_getmodelinfo(int32 index, int32 dwStringID, char* buf, _DWORD bytesize);

    BOOL DCAMAPI dcam_open(HDCAM* ph, int32 index, const char* reserved DCAM_DEFAULT_ARG);
    BOOL DCAMAPI dcam_close(HDCAM h);

    /*** --- camera infomation --- ***/

    BOOL DCAMAPI dcam_getstring(HDCAM h, int32 dwStringID, char* buf, _DWORD bytesize);
    BOOL DCAMAPI dcam_getcapability(HDCAM h, _DWORD* pCapability, _DWORD dwCapTypeID);

    BOOL DCAMAPI dcam_getdatatype(HDCAM h, DCAM_DATATYPE* pType);
    BOOL DCAMAPI dcam_getbitstype(HDCAM h, DCAM_BITSTYPE* pType);
    BOOL DCAMAPI dcam_setdatatype(HDCAM h, DCAM_DATATYPE type);
    BOOL DCAMAPI dcam_setbitstype(HDCAM h, DCAM_BITSTYPE type);

#if defined(DCAM_TARGETOS_IS_WIN32) || defined(DCAM_TARGETOS_IS_MACOSX)
    BOOL DCAMAPI dcam_getdatasize(HDCAM h, SIZE* pSize);
    BOOL DCAMAPI dcam_getbitssize(HDCAM h, SIZE* pSize);
#endif

#if DCAMAPI_VER >= 3010
    BOOL DCAMAPI dcam_getdatasizeex(HDCAM h, DCAM_SIZE* pSize);
    BOOL DCAMAPI dcam_getbitssizeex(HDCAM h, DCAM_SIZE* pSize);
#endif

    /*** --- parameters --- ***/

    BOOL DCAMAPI dcam_queryupdate(HDCAM h, _DWORD* pFlag, _DWORD reserved DCAM_DEFAULT_ARG);

    BOOL DCAMAPI dcam_getbinning(HDCAM h, int32* pBinning);
    BOOL DCAMAPI dcam_getexposuretime(HDCAM h, double* pSec);
    BOOL DCAMAPI dcam_gettriggermode(HDCAM h, int32* pMode);
    BOOL DCAMAPI dcam_gettriggerpolarity(HDCAM h, int32* pPolarity);

    BOOL DCAMAPI dcam_setbinning(HDCAM h, int32 binning);
    BOOL DCAMAPI dcam_setexposuretime(HDCAM h, double sec);
    BOOL DCAMAPI dcam_settriggermode(HDCAM h, int32 mode);
    BOOL DCAMAPI dcam_settriggerpolarity(HDCAM h, int32 polarity);

    /*** --- capturing --- ***/

    BOOL DCAMAPI dcam_precapture(HDCAM h, DCAM_CAPTUREMODE mode);
    BOOL DCAMAPI dcam_getdatarange(HDCAM h, int32* pMax, int32* pMin DCAM_DEFAULT_ARG);
    BOOL DCAMAPI dcam_getdataframebytes(HDCAM h, _DWORD* pSize);

    BOOL DCAMAPI dcam_allocframe(HDCAM h, int32 framecount);
    BOOL DCAMAPI dcam_getframecount(HDCAM h, int32* pFrame);

    BOOL DCAMAPI dcam_capture(HDCAM h);
    BOOL DCAMAPI dcam_idle(HDCAM h);
    BOOL DCAMAPI dcam_wait(HDCAM h,
                           _DWORD* pCode,
                           _DWORD timeout DCAM_DEFAULT_ARG,
                           HDCAMSIGNAL abortsignal DCAM_DEFAULT_ARG);

    BOOL DCAMAPI dcam_getstatus(HDCAM h, _DWORD* pStatus);
    BOOL DCAMAPI dcam_gettransferinfo(HDCAM h, int32* pNewestFrameIndex, int32* pFrameCount);

    BOOL DCAMAPI dcam_freeframe(HDCAM h);

    /*** --- user memory support --- ***/

    BOOL DCAMAPI dcam_attachbuffer(HDCAM h, void** frames, _DWORD size);
    BOOL DCAMAPI dcam_releasebuffer(HDCAM h);

    /*** --- data transfer --- ***/

    BOOL DCAMAPI dcam_lockdata(HDCAM h, void** pTop, int32* pRowbytes, int32 frame);
    BOOL DCAMAPI dcam_lockbits(HDCAM h, BYTE** pTop, int32* pRowbytes, int32 frame);
    BOOL DCAMAPI dcam_unlockdata(HDCAM h);
    BOOL DCAMAPI dcam_unlockbits(HDCAM h);

    /*** --- LUT --- ***/

    BOOL DCAMAPI dcam_setbitsinputlutrange(HDCAM h, int32 inMax, int32 inMin DCAM_DEFAULT_ARG);
    BOOL DCAMAPI dcam_setbitsoutputlutrange(HDCAM h, BYTE outMax, BYTE outMin DCAM_DEFAULT_ARG);

    /*** --- Control Panel --- ***/

    /* BOOL DCAMAPI dcam_showpanel				( HDCAM h, HWND hWnd, _DWORD reserved DCAM_DEFAULT_ARG
     * );
     */

    /*** --- extended --- ***/

    BOOL DCAMAPI dcam_extended(HDCAM h, _ui32 iCmd, void* param, _DWORD size);

    /*** --- software trigger --- ***/
    BOOL DCAMAPI dcam_firetrigger(HDCAM h);

    /* **************************************************************** *

            for extended function

     * **************************************************************** */

    /*** -- iCmd parameter of dcam_extended() -- ***/
    enum
    {
        DCAM_IDMSG_QUERYPARAMCOUNT =
            0x0200 /*		 _DWORD* 		 param,   bytesize = byte size to receive IDs	  */
        ,
        DCAM_IDMSG_SETPARAM = 0x0201 /* const DCAM_HDR_PARAM* param,   bytesize = sizeof( parameters); */
        ,
        DCAM_IDMSG_GETPARAM = 0x0202 /*		 DCAM_HDR_PARAM* param,   bytesize = sizeof( parameters); */
        ,
        DCAM_IDMSG_SETGETPARAM = 0x0203 /*		 DCAM_HDR_PARAM* param,   bytesize = sizeof( parameters); */
        ,
        DCAM_IDMSG_QUERYPARAMID =
            0x0204 /*		 _DWORD			 param[], bytesize = sizeof( param );	  */

    };

    /*** -- parameter header -- ***/
    typedef struct _DCAM_HDR_PARAM
    {
        _DWORD cbSize; /* size of whole structure */
        _DWORD id;     /* specify the kind of this structure */
        _DWORD iFlag;  /* specify the member to be set or requested by application */
        _DWORD oFlag;  /* specify the member to be set or gotten by module */
    } DCAM_HDR_PARAM;

    /*** -- parameter IDs -- ***/
    enum
    {
        DCAM_IDPARAM_RGBRATIO = 0xC00481E2,

        DCAM_IDPARAM_FEATURE = 0xC00001E1,
        DCAM_IDPARAM_FEATURE_INQ = 0x800001A1,
        DCAM_IDPARAM_SUBARRAY = 0xC00001E2,
        DCAM_IDPARAM_SUBARRAY_INQ = 0x800001A2,
        DCAM_IDPARAM_FRAME_READOUT_TIME_INQ = 0x800001A3,

        DCAM_IDPARAM_SCANMODE_INQ = 0x800001A4,
        DCAM_IDPARAM_SCANMODE = 0xC00001E4,
        DCAM_IDPARAM_GATING_INQ = 0x800001A5,
        DCAM_IDPARAM_GATING = 0xC00001E5
    };

#ifdef __cplusplus

    /* end of extern "C" */
};

#endif

    /* **************************************************************** *

            for backward compatibility

     * **************************************************************** */

#ifndef _NO_DCAM_BACKWORD_COMPATIBILITY_

#ifdef __cplusplus

    /* C++ */

#define BEGIN_DECLARE_ENUM(c) enum c
#define END_DECLARE_ENUM(c) ;

#define BEGIN_DECLARE_STRUCT(c) struct c
#define END_DECLARE_STRUCT(c) ;

#define OPTION = 0
#define OPTION_DCAMINIT = DCAMINIT_DEFAULT

#else

    /* C */

#define BEGIN_DECLARE_ENUM(c) typedef enum
#define END_DECLARE_ENUM(c) c;

#define BEGIN_DECLARE_STRUCT(c) typedef struct
#define END_DECLARE_STRUCT(c) c;

#define OPTION
#define OPTION_DCAMINIT

#endif // __cplusplus

#ifdef DCAM_TARGETOS_IS_MACOSX

typedef unsigned short WORD;
// typedef	unsigned int		DWORD;
typedef struct HINSTANCE__* HINSTANCE;

#endif

/*** following values may be removed in future DCAM-API  ***/
#define DCAM_IDMSG_SOFTWARE_TRIGGER 0x0400 /*	no parameter	*/

#define DCAM_CAPABILITY_TRIGGER_SOFTWARE_C7780 0x00080000
#define DCAM_CAPABILITY_TRIGGER_PIV DCAM_CAPABILITY_TRIGGER_FASTREPETITION
#define DCAM_TRIGMODE_PIV DCAM_TRIGMODE_FASTREPETITION
#define DCAM_IDSTR_SPECIFICATIONVERSION DCAM_IDSTR_DCAMAPIVERSION
#define DCAM_EVENT_VVALIDBEGIN DCAM_EVENT_EXPOSUREEND
#define DCAM_CAPABILITY_USERMEMORY DCAM_CAPABILITY_ATTACHBUFFER

#define ccDatatype DCAM_DATATYPE
#define ccDatatype_uint8 DCAM_DATATYPE_UINT8
#define ccDatatype_uint16 DCAM_DATATYPE_UINT16
#define ccDatatype_uint32 DCAM_DATATYPE_UINT32
#define ccDatatype_int8 DCAM_DATATYPE_INT8
#define ccDatatype_int16 DCAM_DATATYPE_INT16
#define ccDatatype_int32 DCAM_DATATYPE_INT32
#define ccDatatype_bgr24 DCAM_DATATYPE_BGR24
#define ccDatatype_bgr48 DCAM_DATATYPE_BGR48
#define ccDatatype_rgb24 DCAM_DATATYPE_RGB24
#define ccDatatype_rgb48 DCAM_DATATYPE_RGB48
#define ccDatatype_none DCAM_DATATYPE_NONE

#define ccBitstype DCAM_BITSTYPE
#define ccBitstype_index8 DCAM_BITSTYPE_INDEX8
#define ccBitstype_rgb16 DCAM_BITSTYPE_RGB16
#define ccBitstype_rgb24 DCAM_BITSTYPE_RGB24
#define ccBitstype_rgb32 DCAM_BITSTYPE_RGB32
#define ccBitstype_none DCAM_BITSTYPE_NONE

#define ccCaptureMode DCAM_CAPTUREMODE
#define ccCapture_Snap DCAM_CAPTUREMODE_SNAP
#define ccCapture_Sequence DCAM_CAPTUREMODE_SEQUENCE

#define ccErr_busy DCAMERR_BUSY
#define ccErr_notready DCAMERR_NOTREADY
#define ccErr_notstable DCAMERR_NOTSTABLE
#define ccErr_unstable DCAMERR_UNSTABLE
#define ccErr_notbusy DCAMERR_NOTBUSY
#define ccErr_coolingtrouble DCAMERR_COOLINGTROUBLE
#define ccErr_abort DCAMERR_ABORT
#define ccErr_timeout DCAMERR_TIMEOUT
#define ccErr_lostframe DCAMERR_LOSTFRAME
#define ccErr_noresource DCAMERR_NORESOURCE
#define ccErr_nomemory DCAMERR_NOMEMORY
#define ccErr_nomodule DCAMERR_NOMODULE
#define ccErr_nodriver DCAMERR_NODRIVER
#define ccErr_nocamera DCAMERR_NOCAMERA
#define ccErr_failopenbus DCAMERR_FAILOPENBUS
#define ccErr_failopencamera DCAMERR_FAILOPENCAMERA
#define ccErr_invalidcamera DCAMERR_INVALIDCAMERA
#define ccErr_invalidhandle DCAMERR_INVALIDHANDLE
#define ccErr_invalidparam DCAMERR_INVALIDPARAM
#define ccErr_notsupport DCAMERR_NOTSUPPORT
#define ccErr_failreadcamera DCAMERR_FAILREADCAMERA
#define ccErr_failwritecamera DCAMERR_FAILWRITECAMERA
#define ccErr_unknownmsgid DCAMERR_UNKNOWNMSGID
#define ccErr_unknownstrid DCAMERR_UNKNOWNSTRID
#define ccErr_unknownparamid DCAMERR_UNKNOWNPARAMID
// #define	ccErr_unknowncamera					DCAMERR_UNKNOWNCAMERA
#define ccErr_unknownbitstype DCAMERR_UNKNOWNBITSTYPE
#define ccErr_unknowndatatype DCAMERR_UNKNOWNDATATYPE
#define ccErr_none DCAMERR_NONE
#define ccErr_unreach DCAMERR_UNREACH
#define ccErr_notimplement DCAMERR_NOTIMPLEMENT

#define ccErr_failopen DCAMERR_FAILOPEN
#define DCAMERR_FAILEOPEN DCAMERR_FAILOPEN // typo

enum
{
    DCAM_IDPARAM_C4742_95 = 0xC00181E1,
    DCAM_IDPARAM_C4742_95ER = 0xC00181E2,
    DCAM_IDPARAM_C7300_10 = 0xC00181E3,
    DCAM_IDPARAM_C4880 = 0xC00181E5,
    DCAM_IDPARAM_C8000 = 0xC00181E6,
    DCAM_IDPARAM_C8484 = 0xC00181E7,
    DCAM_IDPARAM_C4742_98BT = 0xC00181E8,
    DCAM_IDPARAM_C4742_95HR = 0xC00181E9,
    DCAM_IDPARAM_C7190_2X = 0xC00181EA,
    DCAM_IDPARAM_C8000_20 = 0xC00181EB,
    DCAM_IDPARAM_C7780 = 0xC00181EC,
    DCAM_IDPARAM_C4742_98 = 0xC00281ED,
    DCAM_IDPARAM_C4742_98ER = 0xC00181EE,
    DCAM_IDPARAM_C7390 = 0xC00181EF,
    DCAM_IDPARAM_C8190 = 0xC00281E1,
    DCAM_IDPARAM_C7190_10 = 0xC00281E2,
    DCAM_IDPARAM_C8000_10 = 0xC00281E3,
    DCAM_IDPARAM_C4880_80 = 0xC00281E5,
    DCAM_IDPARAM_C7780ADJ = 0xC00281E6,
    DCAM_IDPARAM_C8484_00C = 0xC00281E7,
    DCAM_IDPARAM_C7770 = 0xC00281E8,
    DCAM_IDPARAM_C7770_LUT = 0xC00281E9,
    DCAM_IDPARAM_C4880_SSDUNIV = 0xC00281EA,
    DCAM_IDPARAM_C8133 = 0xC00281EC,
    DCAM_IDPARAM_XRAYLINE = 0xC00281EC,

    DCAM_IDPARAM_PCDIG = 0xC00381E4,
    DCAM_IDPARAM_PCDIG_INQ = 0x800381A4,
    DCAM_IDPARAM_ICPCI = 0xC00381E5,
    DCAM_IDPARAM_ICPCI_INQ = 0x800381A5,
    DCAM_IDPARAM_IQV50 = 0xC00381E6,
    DCAM_IDPARAM_IQV50_LUT = 0xC00381E7,
    DCAM_IDPARAM_IQV50_STATUS = 0x800381A8,
    DCAM_IDPARAM_GRAPHIN = 0xC00381E9,
    DCAM_IDPARAM_DiSCS = 0xC00381EA,
    DCAM_IDPARAM_ORCAERSP1 = 0xC00381EB
};

#define DCAM_IDPARAM_C7300 DCAM_IDPARAM_C7300_10
#define DCAM_IDPARAM_C7300_INQ DCAM_IDPARAM_C7300_10_INQ
#define DCAM_IDPARAM_C4742_95NRK 0xC00281E4 /* not supported */

// only old error codes
//	ccErr_closed
//	ccErr_unexpected
//	ccErr_unexpected2

// new error codes
//		DCAMERR_NOGRABBER
//	/*	DCAMERR_NOCOMBINATION */
//		DCAMERR_INVALIDMODULE
//		DCAMERR_INVALIDCOMMPORT
//		DCAMERR_INVALIDVALUE
//		DCAMERR_OUTOFRANGE
//		DCAMERR_NOTWRITABLE
//		DCAMERR_NOTREADABLE
//		DCAMERR_INVALIDPROPERTYID
//		DCAMERR_NEWAPIREQUIRED
//		DCAMERR_WRONGHANDSHAKE
//		DCAMERR_NOPROPERTY
//		DCAMERR_INVALIDCHANNEL
//		DCAMERR_SUCCESS
//		DCAMERR_UNLOADED
//		DCAMERR_THRUADAPTER

#endif // _NO_DCAM_BACKWORD_COMPATIBILITY_

#ifndef _INCLUDE_DCAM_FEATURES_H_
#define _INCLUDE_DCAM_FEATURES_H_

/* ================================================================ *
        feature
 * ---------------------------------------------------------------- */

/*** === DCAM_PARAM_FEATURE === ***/

enum
{
    dcamparam_feature_featureid = 0x00000001,
    dcamparam_feature_flags = 0x00000002,
    dcamparam_feature_featurevalue = 0x00000004
};

typedef struct _DCAM_PARAM_FEATURE
{
    DCAM_HDR_PARAM hdr; /* id == DCAM_IDPARAM_FEATURE */
    _DWORD featureid;   /* [in]		*/
    _DWORD flags;       /* [in/out]		*/
    float featurevalue; /* [in/out]		*/
} DCAM_PARAM_FEATURE;

/*** === DCAM_PARAM_FEATURE_INQ === ***/

enum
{
    dcamparam_featureinq_featureid = 0x00000001,
    dcamparam_featureinq_capflags = 0x00000002,
    dcamparam_featureinq_min = 0x00000004,
    dcamparam_featureinq_max = 0x00000008,
    dcamparam_featureinq_step = 0x00000010,
    dcamparam_featureinq_defaultvalue = 0x00000020,
    dcamparam_featureinq_units = 0x00000040
};

typedef struct _DCAM_PARAM_FEATURE_INQ
{
    DCAM_HDR_PARAM hdr; /* id == DCAM_IDPARAM_FEATURE_INQ */
    _DWORD featureid;   /* [in]		*/
    _DWORD capflags;    /* [out]		*/
    float min;          /* [out]		*/
    float max;          /* [out]		*/
    float step;         /* [out]		*/
    float defaultvalue; /* [out]		*/
    char units[16];     /* [out]		*/
} DCAM_PARAM_FEATURE_INQ;

/*** === feature index === ***/

enum
{
    DCAM_IDFEATURE_INITIALIZE = 0x00000000,
    DCAM_IDFEATURE_BRIGHTNESS = 0x00000001,
    DCAM_IDFEATURE_GAIN = 0x00000002,
    DCAM_IDFEATURE_CONTRAST = 0x00000002,
    DCAM_IDFEATURE_HUE = 0x00000003,
    DCAM_IDFEATURE_SATURATION = 0x00000004,
    DCAM_IDFEATURE_SHARPNESS = 0x00000005,
    DCAM_IDFEATURE_GAMMA = 0x00000006,
    DCAM_IDFEATURE_WHITEBALANCE = 0x00000007,
    DCAM_IDFEATURE_PAN = 0x00000008,
    DCAM_IDFEATURE_TILT = 0x00000009,
    DCAM_IDFEATURE_ZOOM = 0x0000000a,
    DCAM_IDFEATURE_IRIS = 0x0000000b,
    DCAM_IDFEATURE_FOCUS = 0x0000000c,
    DCAM_IDFEATURE_AUTOEXPOSURE = 0x0000000d,
    DCAM_IDFEATURE_SHUTTER = 0x0000000e,
    DCAM_IDFEATURE_EXPOSURETIME = 0x0000000e,
    DCAM_IDFEATURE_TEMPERATURE = 0x0000000f,
    DCAM_IDFEATURE_OPTICALFILTER = 0x00000010,
    DCAM_IDFEATURE_MECHANICALSHUTTER = 0x00000010,
    DCAM_IDFEATURE_LIGHTMODE = 0x00000011,
    DCAM_IDFEATURE_OFFSET = 0x00000012,
    DCAM_IDFEATURE_CONTRASTOFFSET = 0x00000012,
    DCAM_IDFEATURE_CONTRASTGAIN = 0x00000013,
    DCAM_IDFEATURE_AMPLIFIERGAIN = 0x00000014,
    DCAM_IDFEATURE_TEMPERATURETARGET = 0x00000015,
    DCAM_IDFEATURE_SENSITIVITY = 0x00000016,
    DCAM_IDFEATURE_TRIGGERTIMES = 0x00000017
};

/*** --- capflags only --- ***/
#define DCAM_FEATURE_FLAGS_READ_OUT 0x00010000
/* Allows the feature values to be read out.		*/
#define DCAM_FEATURE_FLAGS_DEFAULT 0x00020000
/* Allows DEFAULT function. If supported, when a feature's DEFAULT is turned ON, then		*/
/* the values and flags are ignored and the default setting is used. DEFAULT must be in the OFF */
/* state before you can adjust any other flags and/or values for the feature.
 */
#define DCAM_FEATURE_FLAGS_ONOFF 0x00020000
/* Allows ON/OFF function. If supported, when a feature is turned OFF, then
 */
/* the values and flags are ignored and the feature control is disabled. The feature must be in the OFF		*/
/* state before you can adjust any other flags and/or values for the feature.
 */

#define DCAM_FEATURE_FLAGS_STEPPING_INCONSISTENT 0x00040000
/* step value of DCAM_PARAM_FEATURE_INQ function is not consistent across the		*/
/* entire range of values. For example, if this flag is set, and:		*/
/*		min = 0		*/
/*		max = 3		*/
/*		step = 1		*/
/* Valid values you can set may be 0,1,3 only. 2 is invalid. Therefore,		*/
/* if you implement a scroll bar, Step is the minimum stepping within		*/
/* the range, but a value within the range may be invalid and produce		*/
/* an error. The application should be aware of this case.		*/

/*** --- capflags, flags get, and flags set --- ***/

#define DCAM_FEATURE_FLAGS_AUTO 0x00000001
/* Auto mode (Controlled automatically by camera).		*/

#define DCAM_FEATURE_FLAGS_MANUAL 0x00000002
/* Manual mode (Controlled by user).		*/

#define DCAM_FEATURE_FLAGS_ONE_PUSH 0x00100000
/* Capability allows One Push operation. Getting means One Push mode is in progress.		*/
/* Setting One Push flag processes feature values once, then		*/
/* turns off the feature and returns to default settings.		*/

/*** --- flags get and flags set --- ***/

#define DCAM_FEATURE_FLAGS_DEFAULT_OFF 0x01000000
/* Enable feature control by turning off DEFAULT. (See DCAM_FEATURE_FLAGS_DEFAULT)		*/
#define DCAM_FEATURE_FLAGS_ON 0x01000000
/* Enable feature control by turning it ON. (See DCAM_FEATURE_FLAGS_ONOFF)				*/

#define DCAM_FEATURE_FLAGS_DEFAULT_ON 0x02000000
/* Disable feature control and use default. (See DCAM_FEATURE_FLAGS_DEFAULT) 		*/
/* ** Note: If DEFAULT is ON or you turn DEFAULT ON, you must turn it OFF before		*/
/*			trying to update a new feature value or mode.		*/
#define DCAM_FEATURE_FLAGS_OFF 0x02000000
/* Disable feature control.					(See DCAM_FEATURE_FLAGS_ONOFF) */
/* ** Note: If a feature is OFF or you turn it OFF, you must turn it ON before			*/
/*			trying to update a new feature value or mode.
 */

/*** --- flags set only --- ***/

#define DCAM_FEATURE_FLAGS_IMMEDIATE 0x04000000
/* When setting a feature, you request for an immediate change.		*/
/* For example, when the camera is streaming and you request immediate		*/
/* action, the camera's stream is haulted to stop the camera's		*/
/* current shutter exposure, then the feature is changed and restarted.		*/

#define DCAM_FEATURE_FLAGS_COOLING_ONOFF 0x00020000 /* capflags with DCAM_IDFEATURE_TEMPERATURE */
#define DCAM_FEATURE_FLAGS_COOLING_ON 0x01000000    /* flags with DCAM_IDFEATURE_TEMPERATURE */
#define DCAM_FEATURE_FLAGS_COOLING_OFF 0x02000000   /* flags with DCAM_IDFEATURE_TEMPERATURE */

#define DCAM_FEATURE_FLAGS_MECHANICALSHUTTER_OPEN 0x02000000  /* flags with DCAM_IDFEATURE_MECHANICALSHUTTER */
#define DCAM_FEATURE_FLAGS_MECHANICALSHUTTER_AUTO 0x01000001  /* flags with DCAM_IDFEATURE_MECHANICALSHUTTER */
#define DCAM_FEATURE_FLAGS_MECHANICALSHUTTER_CLOSE 0x01000002 /* flags with DCAM_IDFEATURE_MECHANICALSHUTTER */

/* ================================================================ *
        Sub Array
 * ---------------------------------------------------------------- */

/*** === DCAM_PARAM_SUBARRAY === ***/

enum
{
    dcamparam_subarray_hpos = 0x00000001,
    dcamparam_subarray_vpos = 0x00000002,
    dcamparam_subarray_hsize = 0x00000004,
    dcamparam_subarray_vsize = 0x00000008
};

typedef struct _DCAM_PARAM_SUBARRAY
{
    DCAM_HDR_PARAM hdr; /* id == DCAM_IDPARAM_SUBARRAY */
    int32 hpos;         /* [in/out]			*/
    int32 vpos;         /* [in/out]			*/
    int32 hsize;        /* [in/out]			*/
    int32 vsize;        /* [in/out]			*/
} DCAM_PARAM_SUBARRAY;

/*** === DCAM_PARAM_SUBARRAY_INQ === ***/

enum
{
    dcamparam_subarrayinq_binning = 0x00000001,
    dcamparam_subarrayinq_hmax = 0x00000002,
    dcamparam_subarrayinq_vmax = 0x00000004,
    dcamparam_subarrayinq_hposunit = 0x00000008,
    dcamparam_subarrayinq_vposunit = 0x00000010,
    dcamparam_subarrayinq_hunit = 0x00000020,
    dcamparam_subarrayinq_vunit = 0x00000040
};

typedef struct _DCAM_PARAM_SUBARRAY_INQ
{
    DCAM_HDR_PARAM hdr; /* id == DCAM_IDPARAM_SUBARRAY_INQ */
    int32 binning;      /* [in]			*/
    int32 hmax;         /* [out]			*/
    int32 vmax;         /* [out]			*/
    int32 hposunit;     /* [out]			*/
    int32 vposunit;     /* [out]			*/
    int32 hunit;        /* [out]			*/
    int32 vunit;        /* [out]			*/
} DCAM_PARAM_SUBARRAY_INQ;

/* ================================================================ *
        readout time
 * ---------------------------------------------------------------- */

enum
{
    dcamparam_framereadouttimeinq_framereadouttime = 0x00000001
};

/*** === DCAM_PARAM_FRAME_READOUT_TIME_INQ === ***/

typedef struct _DCAM_PARAM_FRAME_READOUT_TIME_INQ
{
    DCAM_HDR_PARAM hdr;      /* id == DCAM_IDPARAM_FRAME_READOUT_TIME_INQ */
    double framereadouttime; /* [out]			*/
} DCAM_PARAM_FRAME_READOUT_TIME_INQ;

/* ================================================================ *
        scan mode
 * ---------------------------------------------------------------- */

/*** === DCAM_IDPARAM_SCANMODE === ***/

typedef struct _DCAM_PARAM_SCANMODE
{
    DCAM_HDR_PARAM hdr; /* id == DCAM_IDPARAM_SCANMODE */
    int32 speed;        /* [in/out]			*/
    int32 special;      /* [in/out]			*/
} DCAM_PARAM_SCANMODE;

enum
{
    dcamparam_scanmode_speed = 0x00000001,
    dcamparam_scanmode_special = 0x00000002
};

enum _dcamparam_scanmode_speed
{
    dcamparam_scanmode_speed_slowest = 0x00000001,
    dcamparam_scanmode_speed_fastest = 0x000000FF
    /* user specified this value, module may round down		*/
};

enum _dcamparam_scanmode_spcial
{
    dcamparam_scanmode_special_slitscan = 0x00000001
};

/*** === DCAM_IDPARAM_SCANMODE_INQ === ***/

typedef struct _DCAM_PARAM_SCANMODE_INQ
{
    DCAM_HDR_PARAM hdr; /* id == DCAM_IDPARAM_SCANMODE_INQ */
    int32 speedmax;     /* [out]			*/
    int32 special;      /* [in/out]			*/
} DCAM_PARAM_SCANMODE_INQ;

enum
{
    dcamparam_scanmodeinq_speedmax = 0x00000001,
    dcamparam_scanmodeinq_special = 0x00000002
};

/* ================================================================ *
        gating
 * ---------------------------------------------------------------- */

/*** === DCAM_IDPARAM_GATING_INQ  === ***/

typedef struct _DCAM_PARAM_GATING_INQ
{
    DCAM_HDR_PARAM hdr;  /* id = DCAM_IDPARAM_GATING_INQ */
    _DWORD trigmode;     /* [in/out] trigger mode that supports this parameter.	*/
    int32 capflags;      /* [out] gating capability						*/
    double intervalmin;  /* [out] minimum time of interval by second.	*/
    double intervalmax;  /* [out] maximum time of interval by second.	*/
    double intervalstep; /* [out] step time of interval by second. This was intervalystep		*/
    double delaymin;     /* [out] minimum time of delay by second.		*/
    double delaymax;     /* [out] maximum time of delay by second.		*/
    double delaystep;    /* [out] step time of delay by second.			*/
    double widthmin;     /* [out] minimum time of width by second.		*/
    double widthmax;     /* [out] maximum time of width by second.		*/
    double widthstep;    /* [out] step time of width by second.			*/
} DCAM_PARAM_GATING_INQ;

enum _dcamparam_gatinginq
{
    dcamparam_gatinginq_capflags = 0x00000001,
    dcamparam_gatinginq_trigmode = 0x00000002,
    dcamparam_gatinginq_intervalmin = 0x00000004,
    dcamparam_gatinginq_intervalmax = 0x00000008,
    dcamparam_gatinginq_intervalstep = 0x00000010,
    dcamparam_gatinginq_delaymin = 0x00000020,
    dcamparam_gatinginq_delaymax = 0x00000040,
    dcamparam_gatinginq_delaystep = 0x00000080,
    dcamparam_gatinginq_widthmin = 0x00000100,
    dcamparam_gatinginq_widthmax = 0x00000200,
    dcamparam_gatinginq_widthstep = 0x00000400,

    dcamparam_gatinginq_intervalystep = dcamparam_gatinginq_intervalstep
};

enum _dcamparam_gating_flag
{
    dcamparam_gating_flag_off = 0x00000001,
    dcamparam_gating_flag_continuous = 0x00000002,
    dcamparam_gating_flag_single = 0x00000004
};

/*** === DCAM_IDPARAM_GATING  === ***/

typedef struct _DCAM_PARAM_GATING
{
    DCAM_HDR_PARAM hdr; /* id == DCAM_IDPARAM_GATING */
    int32 flags;        /* [in/out] gating control				*/
    double interval;    /* [in/out] interval time by second		*/
    double delay;       /* [in/out] delay time by second		*/
    double width;       /* [in/out] gating period by second.	*/
} DCAM_PARAM_GATING;

enum _dcamparam_gating
{
    dcamparam_gating_flags = 0x00000001,
    dcamparam_gating_interval = 0x00000002,
    dcamparam_gating_delay = 0x00000004,
    dcamparam_gating_width = 0x00000008
};

/* ================================================================ *
        for backward compatibility
 * ---------------------------------------------------------------- */

#define DCAM_IDFEATURE_OFFSET_MAC 0x00001001

#define dcamparam_feature_inq_featureid dcamparam_featureinq_featureid
#define dcamparam_feature_inq_capflags dcamparam_featureinq_capflags
#define dcamparam_feature_inq_min dcamparam_featureinq_min
#define dcamparam_feature_inq_max dcamparam_featureinq_max
#define dcamparam_feature_inq_step dcamparam_featureinq_step
#define dcamparam_feature_inq_defaultvalue dcamparam_featureinq_defaultvalue

#define dcamparam_subarray_inq_binning dcamparam_subarrayinq_binning
#define dcamparam_subarray_inq_hmax dcamparam_subarrayinq_hmax
#define dcamparam_subarray_inq_vmax dcamparam_subarrayinq_vmax
#define dcamparam_subarray_inq_hposunit dcamparam_subarrayinq_hposunit
#define dcamparam_subarray_inq_vposunit dcamparam_subarrayinq_vposunit
#define dcamparam_subarray_inq_hunit dcamparam_subarrayinq_hunit
#define dcamparam_subarray_inq_vunit dcamparam_subarrayinq_vunit

#define _INCLUDE_DCAM_FEATURES_H_
#endif

#ifndef _INCLUDE_PARAMCOLOR_H_
#define _INCLUDE_PARAMCOLOR_H_

typedef struct
{
    double red;
    double green;
    double blue;
} dcam_rgbratio;

typedef struct _DCAM_PARAM_RGBRATIO
{
    DCAM_HDR_PARAM hdr; // id == DCAM_IDPARAM_RGBRATIO

    dcam_rgbratio exposure;
    dcam_rgbratio gain;
} DCAM_PARAM_RGBRATIO;

enum
{
    dcamparam_rgbratio_exposure = 0x00000001,
    dcamparam_rgbratio_gain = 0x00000002
};

#endif

#define _INCLUDE_DCAMAPI_H
#endif
