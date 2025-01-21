/* **************************************************************** *

        dcamapi.h:	July 18, 2013

 * **************************************************************** */

#ifndef _INCLUDE_DCAMAPI_H_

#ifndef DCAMAPI_VER
#define DCAMAPI_VER 4000
#endif

/* **************************************************************** *

        language absorber

 * **************************************************************** */

#ifdef __cplusplus

/* C++ */

#define DCAM_DECLARE_BEGIN(kind, tag) kind tag
#define DCAM_DECLARE_END(tag) ;

#define DCAM_DEFAULT_ARG = 0
#define DCAMINIT_DEFAULT_ARG = DCAMINIT_DEFAULT

extern "C"
{

#else

/* C */

#define DCAM_DECLARE_BEGIN(kind, tag) typedef kind
#define DCAM_DECLARE_END(tag) tag;

#define DCAM_DEFAULT_ARG
#define DCAMINIT_DEFAULT_ARG

#endif

    /* **************************************************************** *

            defines

     * **************************************************************** */

    /* define - HDCAM */

    typedef struct tag_dcam* HDCAM;

    /* define - DCAMAPI */

#ifndef DCAMAPI
#ifdef PASCAL
#define DCAMAPI PASCAL /* DCAM-API based on PASCAL calling */
#else
#define DCAMAPI
#endif
#endif /* DCAMAPI */

    /* define - int32 & _ui32 */

#if defined(WIN32) || defined(_INC_WINDOWS)
    typedef long int32;
    typedef unsigned long _ui32;
#else
typedef int int32;
typedef unsigned int _ui32;
#endif

    /* **************************************************************** *

            constant declaration

     * **************************************************************** */

    /*** --- values --- ***/

#define DCAMCONST_FRAMESTAMP_MISMATCH 0xFFFFFFFF

    /*** --- errors --- ***/

    DCAM_DECLARE_BEGIN(enum, DCAMERR){
        /* status error */
        DCAMERR_BUSY = 0x80000101,      /*		API cannot process in busy state.		*/
        DCAMERR_NOTREADY = 0x80000103,  /*		API requires ready state.				*/
        DCAMERR_NOTSTABLE = 0x80000104, /*		API requires stable or unstable state.	*/
        DCAMERR_UNSTABLE = 0x80000105,  /*		API does not support in unstable state.	*/
        DCAMERR_NOTBUSY = 0x80000107,   /*		API requires busy state.				*/

        DCAMERR_EXCLUDED = 0x80000110, /*		some resource is exclusive and already used	*/

        DCAMERR_COOLINGTROUBLE = 0x80000302,      /*		something happens near cooler */
        DCAMERR_NOTRIGGER = 0x80000303,           /*		no trigger when necessary. Some camera supports this error. */
        DCAMERR_TEMPERATURE_TROUBLE = 0x80000304, /*		camera warns its temperature */

        /* wait error */
        DCAMERR_ABORT = 0x80000102,                /*		abort process			*/
        DCAMERR_TIMEOUT = 0x80000106,              /*		timeout					*/
        DCAMERR_LOSTFRAME = 0x80000301,            /*		frame data is lost		*/
        DCAMERR_MISSINGFRAME_TROUBLE = 0x80000f06, /*		frame is lost but reason is low lever driver's bug */

        /* initialization error */
        DCAMERR_NORESOURCE = 0x80000201,    /*		not enough resource except memory	*/
        DCAMERR_NOMEMORY = 0x80000203,      /*		not enough memory		*/
        DCAMERR_NOMODULE = 0x80000204,      /*		no sub module			*/
        DCAMERR_NODRIVER = 0x80000205,      /*		no driver				*/
        DCAMERR_NOCAMERA = 0x80000206,      /*		no camera				*/
        DCAMERR_NOGRABBER = 0x80000207,     /*		no grabber				*/
        DCAMERR_NOCOMBINATION = 0x80000208, /*		no combination on registry */

        DCAMERR_FAILOPEN = 0x80001001,
        DCAMERR_INVALIDMODULE = 0x80000211,   /*		dcam_init() found invalid module */
        DCAMERR_INVALIDCOMMPORT = 0x80000212, /*		invalid serial port		*/
        DCAMERR_FAILOPENBUS = 0x81001001,     /*		the bus or driver are not available	*/
        DCAMERR_FAILOPENCAMERA = 0x82001001,  /*		camera report error during opening	*/

        /* calling error */
        DCAMERR_INVALIDCAMERA = 0x80000806,     /*		invalid camera		 */
        DCAMERR_INVALIDHANDLE = 0x80000807,     /*		invalid camera handle	*/
        DCAMERR_INVALIDPARAM = 0x80000808,      /*		invalid parameter		*/
        DCAMERR_INVALIDVALUE = 0x80000821,      /*		invalid property value	*/
        DCAMERR_OUTOFRANGE = 0x80000822,        /*		value is out of range	*/
        DCAMERR_NOTWRITABLE = 0x80000823,       /*		the property is not writable	*/
        DCAMERR_NOTREADABLE = 0x80000824,       /*		the property is not readable	*/
        DCAMERR_INVALIDPROPERTYID = 0x80000825, /*		the property id is invalid	*/
        DCAMERR_NEWAPIREQUIRED =
            0x80000826, /*		old API does not support the value because only new API supports the value */
        DCAMERR_WRONGHANDSHAKE =
            0x80000827, /*		this error happens DCAM get error code from camera unexpectedly */
        DCAMERR_NOPROPERTY =
            0x80000828,                      /*		there is no altenative or influence id, or no more property id */
        DCAMERR_INVALIDCHANNEL = 0x80000829, /*		the property id specifies channel but channel is invalid */
        DCAMERR_INVALIDVIEW = 0x8000082a,    /*		the property id specifies channel but channel is invalid */
        DCAMERR_INVALIDSUBARRAY =
            0x8000082b, /*		the conbination of subarray values are invalid. e.g. DCAM_IDPROP_SUBARRAYHPOS +
                           DCAM_IDPROP_SUBARRAYHSIZE is greater than the number of horizontal pixel of sensor. */
        DCAMERR_ACCESSDENY = 0x8000082c,         /*		the property cannot access during this DCAM STATUS */
        DCAMERR_NOVALUETEXT = 0x8000082d,        /*		the property does not have value text */
        DCAMERR_WRONGPROPERTYVALUE = 0x8000082e, /*		at least one property value is wrong */
        DCAMERR_DISHARMONY = 0x80000830,         /*		the paired camera does not have same parameter */
        DCAMERR_FRAMEBUNDLESHOULDBEOFF =
            0x80000832,                           /*	framebundle mode should be OFF under current property settings */
        DCAMERR_INVALIDFRAMEINDEX = 0x80000833,   /*		the frame index is invalid  */
        DCAMERR_INVALIDSESSIONINDEX = 0x80000834, /*		the session index is invalid */
        DCAMERR_NOCORRECTIONDATA = 0x80000838,    /*		not take the dark and shading correction data yet.*/
        DCAMERR_NOTSUPPORT =
            0x80000f03, /*		camera does not support the function or property with current settings */
        DCAMERR_CHANNELDEPENDENTVALUE =
            0x80000839, /*	each channel has own property value so can't return overall property value. */
        DCAMERR_VIEWDEPENDENTVALUE =
            0x8000083a, /*		each view has own property value so can't return overall property value. */

        /* camera or bus trouble */
        DCAMERR_FAILREADCAMERA = 0x83001002,
        DCAMERR_FAILWRITECAMERA = 0x83001003,
        DCAMERR_CONFLICTCOMMPORT = 0x83001004,
        DCAMERR_OPTICS_UNPLUGGED = 0x83001005, /* 	Optics part is unplugged so please check it. */
        DCAMERR_FAILCALIBRATION = 0x83001006,  /*		fail calibration	*/

        DCAMERR_FAILEDOPENRECFILE = 0x84001001,
        DCAMERR_INVALIDRECHANDLE = 0x84001002,
        DCAMERR_FAILEDWRITEDATA = 0x84001003,
        DCAMERR_FAILEDREADDATA = 0x84001004,
        DCAMERR_NOWRECORDING = 0x84001005,
        DCAMERR_WRITEFULL = 0x84001006,
        DCAMERR_ALREADYOCCUPIED = 0x84001007,
        DCAMERR_TOOLARGEUSERDATASIZE = 0x84001008,
        DCAMERR_INVALIDWAITHANDLE = 0x84002001,

        /* calling error for DCAM-API 2.1.3 */
        DCAMERR_UNKNOWNMSGID = 0x80000801,    /*		unknown message id		*/
        DCAMERR_UNKNOWNSTRID = 0x80000802,    /*		unknown string id		*/
        DCAMERR_UNKNOWNPARAMID = 0x80000803,  /*		unkown parameter id		*/
        DCAMERR_UNKNOWNBITSTYPE = 0x80000804, /*		unknown bitmap bits type			*/
        DCAMERR_UNKNOWNDATATYPE = 0x80000805, /*		unknown frame data type				*/

        /* internal error */
        DCAMERR_NONE = 0,                            /*		no error, nothing to have done		*/
        DCAMERR_INSTALLATIONINPROGRESS = 0x80000f00, /*	installation progress				*/
        DCAMERR_UNREACH = 0x80000f01,                /*		internal error						*/
        DCAMERR_UNLOADED = 0x80000f04,               /*		calling after process terminated	*/
        DCAMERR_THRUADAPTER =
            0x80000f05,                    /*											*/
        DCAMERR_NOCONNECTION = 0x80000f07, /*		HDCAM lost connection to camera		*/

        DCAMERR_NOTIMPLEMENT = 0x80000f02, /*		not yet implementation				*/

        DCAMERR_APIINIT_INITOPTIONBYTES = 0xa4010003,
        DCAMERR_APIINIT_INITOPTION = 0xa4010004,

        DCAMERR_INITOPTION_COLLISION_BASE = 0xa401C000,
        DCAMERR_INITOPTION_COLLISION_MAX = 0xa401FFFF,

        /*	Between DCAMERR_INITOPTION_COLLISION_BASE and DCAMERR_INITOPTION_COLLISION_MAX means there is collision
           with initoption in DCAMAPI_INIT. */
        /*	The value "(error code) - DCAMERR_INITOPTION_COLLISION_BASE" indicates the index which second INITOPTION
           group happens. */

        /* success */
        DCAMERR_SUCCESS = 1 /*		no error, general success code, app should check the value is positive	*/
    } DCAM_DECLARE_END(DCAMERR)

        DCAM_DECLARE_BEGIN(enum, DCAMREC_METADATAOPTION){
            DCAMREC_METADATAOPTION__LOCATION_FRAME = 0x00000000,
            DCAMREC_METADATAOPTION__LOCATION_FILE = 0x01000000,
            DCAMREC_METADATAOPTION__LOCATION_SESSION = 0x02000000,
            DCAMREC_METADATAOPTION__LOCATION__MASK = 0xFF000000,

        } DCAM_DECLARE_END(DCAMREC_METADATAOPTION)

            DCAM_DECLARE_BEGIN(enum, DCAM_PIXELTYPE){

                DCAM_PIXELTYPE_MONO8 = 0x00000001,
                DCAM_PIXELTYPE_MONO16 = 0x00000002,

                DCAM_PIXELTYPE_RGB24 = 0x00000021,
                DCAM_PIXELTYPE_RGB48 = 0x00000022,
                DCAM_PIXELTYPE_BGR24 = 0x00000029,
                DCAM_PIXELTYPE_BGR48 = 0x0000002a,

                DCAM_PIXELTYPE_NONE = 0x00000000} DCAM_DECLARE_END(DCAM_PIXELTYPE)

                DCAM_DECLARE_BEGIN(enum,
                                   DCAMBUF_ATTACHKIND){DCAMBUF_ATTACHKIND_TIMESTAMP = 1,
                                                       DCAMBUF_ATTACHKIND_FRAMESTAMP = 2,

                                                       DCAMBUF_ATTACHKIND_FRAME = 0} DCAM_DECLARE_END(DCAM_ATTACHKIND)

        /*** --- status --- ***/
        DCAM_DECLARE_BEGIN(enum, DCAMCAP_STATUS)
    {
        DCAMCAP_STATUS_ERROR = 0x0000, DCAMCAP_STATUS_BUSY = 0x0001, DCAMCAP_STATUS_READY = 0x0002,
        DCAMCAP_STATUS_STABLE = 0x0003, DCAMCAP_STATUS_UNSTABLE = 0x0004,

#if !defined(DCAMAPI_VERMIN) || DCAMAPI_VERMIN <= 3200
        DCAM_STATUS_ERROR = DCAMCAP_STATUS_ERROR, DCAM_STATUS_BUSY = DCAMCAP_STATUS_BUSY,
        DCAM_STATUS_READY = DCAMCAP_STATUS_READY, DCAM_STATUS_STABLE = DCAMCAP_STATUS_STABLE,
        DCAM_STATUS_UNSTABLE = DCAMCAP_STATUS_UNSTABLE,
#endif

        end_of_dcamcap_status
    }
    DCAM_DECLARE_END(DCAMCAP_STATUS)

    DCAM_DECLARE_BEGIN(enum, DCAMWAIT_EVENT)
    {
        DCAMWAIT_CAPEVENT_TRANSFERRED = 0x0001, DCAMWAIT_CAPEVENT_FRAMEREADY = 0x0002, /* all modules support	*/
            DCAMWAIT_CAPEVENT_CYCLEEND = 0x0004,                                       /* all modules support	*/
            DCAMWAIT_CAPEVENT_EXPOSUREEND = 0x0008, DCAMWAIT_CAPEVENT_STOPPED = 0x0010,

        DCAMWAIT_RECEVENT_STOPPED = 0x0100, DCAMWAIT_RECEVENT_WARNING = 0x0200, DCAMWAIT_RECEVENT_MISSED = 0x0400,
        //	DCAMWAIT_RECEVENT_FULLBUF			= 0x0800,	/* *cancel* */
            DCAMWAIT_RECEVENT_DISKFULL = 0x1000, DCAMWAIT_RECEVENT_WRITEFAULT = 0x2000,

        // backward compatibility
            DCAMCAP_EVENT_TRANSFERRED = DCAMWAIT_CAPEVENT_TRANSFERRED,
        DCAMCAP_EVENT_FRAMEREADY = DCAMWAIT_CAPEVENT_FRAMEREADY, DCAMCAP_EVENT_CYCLEEND = DCAMWAIT_CAPEVENT_CYCLEEND,
        DCAMCAP_EVENT_EXPOSUREEND = DCAMWAIT_CAPEVENT_EXPOSUREEND, DCAMCAP_EVENT_STOPPED = DCAMWAIT_CAPEVENT_STOPPED,

        DCAMREC_EVENT_STOPPED = DCAMWAIT_RECEVENT_STOPPED, DCAMREC_EVENT_WARNING = DCAMWAIT_RECEVENT_WARNING,
        DCAMREC_EVENT_MISSED = DCAMWAIT_RECEVENT_MISSED,
        //	DCAMREC_EVENT_FULLBUF				= DCAMWAIT_RECEVENT_FULLBUF,	/* *cancel* */
            DCAMREC_EVENT_DISKFULL = DCAMWAIT_RECEVENT_DISKFULL,
        DCAMREC_EVENT_WRITEFAULT = DCAMWAIT_RECEVENT_WRITEFAULT,

#if !defined(DCAMAPI_VERMIN) || DCAMAPI_VERMIN <= 3200
        DCAM_EVENT_FRAMESTART = DCAMWAIT_CAPEVENT_TRANSFERRED, DCAM_EVENT_FRAMEEND = DCAMWAIT_CAPEVENT_FRAMEREADY,
        DCAM_EVENT_CYCLEEND = DCAMWAIT_CAPEVENT_CYCLEEND,
        DCAM_EVENT_EXPOSUREEND = DCAMWAIT_CAPEVENT_EXPOSUREEND, /* old name was VVALIDBEGIN */
            DCAM_EVENT_CAPTUREEND = DCAMWAIT_CAPEVENT_STOPPED,
#endif

        end_of_dcamwait_event
    }
    DCAM_DECLARE_END(DCAMWAIT_EVENT)

    /*** --- dcamcap_start --- ***/
    DCAM_DECLARE_BEGIN(enum, DCAMCAP_START){DCAMCAP_START_SEQUENCE = -1,
                                            DCAMCAP_START_SNAP = 0} DCAM_DECLARE_END(DCAMCAP_START)

        /*** --- string id --- ***/
        DCAM_DECLARE_BEGIN(enum, DCAM_IDSTR){

            DCAM_IDSTR_BUS = 0x04000101,
            DCAM_IDSTR_CAMERAID = 0x04000102,
            DCAM_IDSTR_VENDOR = 0x04000103,
            DCAM_IDSTR_MODEL = 0x04000104,
            DCAM_IDSTR_CAMERAVERSION = 0x04000105,
            DCAM_IDSTR_DRIVERVERSION = 0x04000106,
            DCAM_IDSTR_MODULEVERSION = 0x04000107,
            DCAM_IDSTR_DCAMAPIVERSION = 0x04000108,

            DCAM_IDSTR_OPTICALBLOCK_MODEL = 0x04001101,
            DCAM_IDSTR_OPTICALBLOCK_ID = 0x04001102,
            DCAM_IDSTR_OPTICALBLOCK_DESCRIPTION = 0x04001103,
            DCAM_IDSTR_OPTICALBLOCK_CHANNEL_1 = 0x04001104,
            DCAM_IDSTR_OPTICALBLOCK_CHANNEL_2 = 0x04001105} DCAM_DECLARE_END(DCAM_IDSTR)

        /*** --- wait timeout --- ***/
        DCAM_DECLARE_BEGIN(enum, DCAMWAIT_TIMEOUT)
    {
        DCAMWAIT_TIMEOUT_INFINITE = 0x80000000,

#if !defined(DCAMAPI_VERMIN) || DCAMAPI_VERMIN <= 3200
        DCAM_WAIT_INFINITE = DCAMWAIT_TIMEOUT_INFINITE,
#endif

        end_of_dcamwait_timeout
    }
    DCAM_DECLARE_END(DCAMWAIT_TIMEOUT)

#if DCAMAPI_VER >= 4000

    /*** --- initialize parameter --- ***/
    DCAM_DECLARE_BEGIN(enum,
                       DCAMAPI_INITOPTION){DCAMAPI_INITOPTION_APIVER__LATEST = 0x00000001,
                                           DCAMAPI_INITOPTION_APIVER__4_0 = 0x00000400,
                                           DCAMAPI_INITOPTION_MULTIVIEW__DISABLE = 0x00010002,
                                           DCAMAPI_INITOPTION_ENDMARK = 0x00000000} DCAM_DECLARE_END(DCAMAPI_INITOPTION)

        /*** --- meta data kind --- ***/

        DCAM_DECLARE_BEGIN(enum,
                           DCAMBUF_METADATAKIND){DCAMBUF_METADATAKIND_TIMESTAMPS = 0x00010000,
                                                 DCAMBUF_METADATAKIND_FRAMESTAMPS = 0x00020000,

                                                 end_of_dcambuf_metadatakind} DCAM_DECLARE_END(DCAMBUF_METADATAKIND)

            DCAM_DECLARE_BEGIN(enum,
                               DCAMREC_METADATAKIND){DCAMREC_METADATAKIND_USERDATATEXT = 0x00000001,
                                                     DCAMREC_METADATAKIND_USERDATABIN = 0x00000002,
                                                     DCAMREC_METADATAKIND_TIMESTAMPS = 0x00010000,
                                                     DCAMREC_METADATAKIND_FRAMESTAMPS = 0x00020000,

                                                     end_of_dcamrec_metadatakind} DCAM_DECLARE_END(DCAMREC_METADATAKIND)

        /*** --- Code Page --- ***/

        DCAM_DECLARE_BEGIN(enum, DCAM_CODEPAGE){DCAM_CODEPAGE__SHIFT_JIS = 932, // Shift JIS

                                                DCAM_CODEPAGE__UTF16_LE = 1200, // UTF-16 (Little Endian)
                                                DCAM_CODEPAGE__UTF16_BE = 1201, // UTF-16 (Big Endian)

                                                DCAM_CODEPAGE__UTF7 = 65000, // UTF-7 translation
                                                DCAM_CODEPAGE__UTF8 = 65001, // UTF-8 translation

                                                DCAM_CODEPAGE__NONE = 0x00000000} DCAM_DECLARE_END(DCAM_CODEPAGE)

            DCAM_DECLARE_BEGIN(enum, DCAMREC_STATUSFLAG){DCAMREC_STATUSFLAG_NONE = 0x00000000,
                                                         DCAMREC_STATUSFLAG_RECORDING = 0x00000001,

                                                         end_of_dcamrec_statusflag} DCAM_DECLARE_END(DCAMREC_STATUSFLAG)

        /* **************************************************************** *

                structures (ver 4.x)

         * **************************************************************** */

        typedef struct DCAMWAIT* HDCAMWAIT;
    typedef struct DCAMREC* HDCAMREC;

    DCAM_DECLARE_BEGIN(struct, DCAM_GUID)
    {
        _ui32 Data1;
        unsigned short Data2;
        unsigned short Data3;
        unsigned char Data4[8];
    }
    DCAM_DECLARE_END(DCAM_GUID)

    DCAM_DECLARE_BEGIN(struct, DCAMAPI_INIT)
    {
        int32 size;              // [in]
        int32 iDeviceCount;      // [out]
        int32 reserved;          // reserved
        int32 initoptionbytes;   // [in] maximum bytes of initoption array.
        const int32* initoption; // [in ptr] initialize options. Choose from DCAMAPI_INITOPTION
        const DCAM_GUID* guid;   // [in ptr]
    }
    DCAM_DECLARE_END(DCAMAPI_INIT)

    DCAM_DECLARE_BEGIN(struct, DCAMDEV_OPEN)
    {
        int32 size;  // [in]
        int32 index; // [in]
        HDCAM hdcam; // [out]
    }
    DCAM_DECLARE_END(DCAMDEV_OPEN)

    DCAM_DECLARE_BEGIN(struct, DCAMDEV_CAPABILITY)
    {
        int32 size;     // [in]
        int32 reserved; // [in]
        int32 capflag1; // [out]
        int32 capflag2; // [out]
    }
    DCAM_DECLARE_END(DCAMDEV_CAPABILITY)

    DCAM_DECLARE_BEGIN(struct, DCAMDEV_STRING)
    {
        int32 size;      // [in]
        int32 iString;   // [in]
        char* text;      // [in,obuf]
        int32 textbytes; // [in]
    }
    DCAM_DECLARE_END(DCAMDEV_STRING)

    DCAM_DECLARE_BEGIN(struct, DCAMPROP_ATTR)
    {
        /* input parameters */
        int32 cbSize;     // [in] size of this structure
        int32 iProp;      //	DCAMIDPROPERTY
        int32 option;     //	DCAMPROPOPTION
        int32 iReserved1; //	must be 0

        /* output parameters */
        int32 attribute;  //	DCAMPROPATTRIBUTE
        int32 iGroup;     //	0 reserved;
        int32 iUnit;      //	DCAMPROPUNIT
        int32 attribute2; //	DCAMPROPATTRIBUTE2

        double valuemin;     //	minimum value
        double valuemax;     //	maximum value
        double valuestep;    //	minimum stepping between a value and the next
        double valuedefault; //	default value

        int32 nMaxChannel; //	max channel if supports
        int32 iReserved3;  //	reserved to 0
        int32 nMaxView;    //	max view if supports

        int32 iProp_NumberOfElement; //	property id to get number of elements of this property if it is array
        int32 iProp_ArrayBase;       //	base id of array if element
        int32 iPropStep_Element;     //	step for iProp to next element
    }
    DCAM_DECLARE_END(DCAMPROP_ATTR)

    DCAM_DECLARE_BEGIN(struct, DCAMPROP_VALUETEXT)
    {
        int32 cbSize;    // [in] size of this structure
        int32 iProp;     // [in] DCAMIDPROP
        double value;    // [in] value of property
        char* text;      // [in,obuf] text of the value
        int32 textbytes; // [in] text buf size
    }
    DCAM_DECLARE_END(DCAMPROP_VALUETEXT)

    DCAM_DECLARE_BEGIN(struct, DCAMBUF_ATTACH)
    {
        int32 size;        // [in] size of this structure.
        int32 iKind;       // [in] DCAMBUF_ATTACHKIND
        void** buffer;     // [in,ptr]
        int32 buffercount; // [in]
    }
    DCAM_DECLARE_END(DCAMBUF_ATTACH)

    DCAM_DECLARE_BEGIN(struct, DCAM_TIMESTAMP)
    {
        int32 sec;      // [out]
        int32 microsec; // [out]
    }
    DCAM_DECLARE_END(DCAM_TIMESTAMP)

    DCAM_DECLARE_BEGIN(struct, DCAMBUF_FRAME)
    {
        // copyframe() and lockframe() use this structure. Some members have different direction.
        // [i:o] means, the member is input at copyframe() and output at lockframe().
        // [i:i] and [o:o] means always input and output at both function.
        // "input" means application has to set the value before calling.
        // "output" means function filles a value at returning.
        int32 size;               // [i:i] size of this structure.
        int32 iKind;              // reserved. set to 0.
        int32 option;             // reserved. set to 0.
        int32 iFrame;             // [i:i] frame index
        void* buf;                // [i:o] pointer for top-left image
        int32 rowbytes;           // [i:o] byte size for next line.
        DCAM_PIXELTYPE type;      // reserved. set to 0.
        int32 width;              // [i:o] horizontal pixel count
        int32 height;             // [i:o] vertical line count
        int32 left;               // [i:o] horizontal start pixel
        int32 top;                // [i:o] vertical start line
        DCAM_TIMESTAMP timestamp; // [o:o] timestamp
        int32 framestamp;         // [o:o] framestamp
        int32 camerastamp;        // [o:o] camerastamp
    }
    DCAM_DECLARE_END(DCAMBUF_FRAME)

    DCAM_DECLARE_BEGIN(struct,
                       DCAMREC_FRAME) // currently the structure is same as DCAM_FRAME. option flag means are different.
    {
        // copyframe() and lockframe() use this structure. Some members have different direction.
        // [i:o] means, the member is input at copyframe() and output at lockframe().
        // [i:i] and [o:o] means always input and output at both function.
        // "input" means application has to set the value before calling.
        // "output" means function filles a value at returning.
        int32 size;               // [i:i] size of this structure.
        int32 iKind;              // reserved. set to 0.
        int32 option;             // reserved. set to 0.
        int32 iFrame;             // [i:i] frame index
        void* buf;                // [i:o] pointer for top-left image
        int32 rowbytes;           // [i:o] byte size for next line.
        DCAM_PIXELTYPE type;      // reserved. set to 0.
        int32 width;              // [i:o] horizontal pixel count
        int32 height;             // [i:o] vertical line count
        int32 left;               // [i:o] horizontal start pixel
        int32 top;                // [i:o] vertical start line
        DCAM_TIMESTAMP timestamp; // [o:o] timestamp
        int32 framestamp;         // [o:o] framestamp
        int32 camerastamp;        // [o:o] camerastamp
    }
    DCAM_DECLARE_END(DCAMREC_FRAME)

    DCAM_DECLARE_BEGIN(struct, DCAMWAIT_OPEN)
    {
        int32 size;         // [in] size of this structure.
        int32 supportevent; // [out];
        HDCAMWAIT hwait;    // [out];
        HDCAM hdcam;        // [in];
    }
    DCAM_DECLARE_END(DCAMWAIT_OPEN)

    DCAM_DECLARE_BEGIN(struct, DCAMWAIT_START)
    {
        int32 size;          // [in] size of this structure.
        int32 eventhappened; // [out]
        int32 eventmask;     // [in]
        int32 timeout;       // [in]
    }
    DCAM_DECLARE_END(DCAMWAIT_START)

    DCAM_DECLARE_BEGIN(struct, DCAMCAP_TRANSFERINFO)
    {
        int32 size;              // [in] size of this structure.
        int32 reserved;          // [in]
        int32 nNewestFrameIndex; // [out]
        int32 nFrameCount;       // [out]
    }
    DCAM_DECLARE_END(DCAMCAP_TRANSFERINFO)

#ifdef _WIN32

    DCAM_DECLARE_BEGIN(struct, DCAMREC_OPENA)
    {
        int32 size;                 // [in] size of this structure.
        int32 reserved;             // [in]
        HDCAMREC hrec;              // [out]
        const char* path;           // [in]
        const char* ext;            // [in]
        int32 maxframepersession;   // [in]
        int32 userdatasize;         // [in]
        int32 userdatasize_session; // [in]
        int32 userdatasize_file;    // [in]
        int32 usertextsize;         // [in]
        int32 usertextsize_session; // [in]
        int32 usertextsize_file;    // [in]
    }
    DCAM_DECLARE_END(DCAMREC_OPENA)

    DCAM_DECLARE_BEGIN(struct, DCAMREC_OPENW)
    {
        int32 size;                 // [in] size of this structure.
        int32 reserved;             // [in]
        HDCAMREC hrec;              // [out]
        const wchar_t* path;        // [in]
        const wchar_t* ext;         // [in]
        int32 maxframepersession;   // [in]
        int32 userdatasize;         // [in]
        int32 userdatasize_session; // [in]
        int32 userdatasize_file;    // [in]
        int32 usertextsize;         // [in]
        int32 usertextsize_session; // [in]
        int32 usertextsize_file;    // [in]
    }
    DCAM_DECLARE_END(DCAMREC_OPENW)

#else

    DCAM_DECLARE_BEGIN(struct, DCAMREC_OPEN)
    {
        int32 size;                 // [in] size of this structure.
        int32 reserved;             // [in]
        HDCAMREC hrec;              // [out]
        const char* path;           // [in]
        const char* ext;            // [in]
        int32 maxframepersession;   // [in]
        int32 userdatasize;         // [in]
        int32 userdatasize_session; // [in]
        int32 userdatasize_file;    // [in]
        int32 usertextsize;         // [in]
        int32 usertextsize_session; // [in]
        int32 usertextsize_file;    // [in]
    }
    DCAM_DECLARE_END(DCAMREC_OPEN)

#endif

    DCAM_DECLARE_BEGIN(struct, DCAM_METADATAHDR)
    {
        int32 size;   // [in] size of whole structure, not only this.
        int32 iKind;  // [in] DCAM_METADATAKIND
        int32 option; // [in] value meaning depends on DCAM_METADATAKIND
        int32 iFrame; // [in] frame index
    }
    DCAM_DECLARE_END(DCAM_METADATAHDR)

    DCAM_DECLARE_BEGIN(struct, DCAM_METADATABLOCKHDR)
    {
        int32 size;     // [in] size of whole structure, not only this.
        int32 iKind;    // [in] DCAM_METADATAKIND
        int32 option;   // [in] value meaning depends on DCAMBUF_METADATAOPTION or DCAMREC_METADATAOPTION
        int32 iFrame;   // [in] start frame index
        int32 in_count; // [in] max count of meta data
        int32 outcount; // [out] count of got meta data.
    }
    DCAM_DECLARE_END(DCAM_METADATABLOCKHDR)

    DCAM_DECLARE_BEGIN(struct, DCAM_USERDATATEXT)
    {
        DCAM_METADATAHDR hdr; // [in] size member should be size of this structure
                              // [in] iKind should be DCAM_METADATAKIND_USERDATATEXT.
                              // [in] option should be one of DCAMREC_METADATAOPTION

        char* text;     // [in] UTF-8 encoding
        int32 text_len; // [in] byte size of meta data.
        int32 codepage; // [in] DCAM_CODEPAGE.
    }
    DCAM_DECLARE_END(DCAM_USERDATATEXT)

    DCAM_DECLARE_BEGIN(struct, DCAM_USERDATABIN)
    {
        DCAM_METADATAHDR hdr; // [in] size member should be size of this structure
                              // [in] iKind should be DCAM_METADATAKIND_USERDATABIN.
                              // [in] option should be one of DCAMREC_METADATAOPTION

        void* bin;      // [in] binary meta data
        int32 bin_len;  // [in] byte size of binary meta data.
        int32 reserved; // [in] 0 reserved.
    }
    DCAM_DECLARE_END(DCAM_USERDATABIN)

    DCAM_DECLARE_BEGIN(struct, DCAM_TIMESTAMPBLOCK)
    {
        DCAM_METADATABLOCKHDR hdr; // [in] size member should be size of this structure
                                   // [in] iKind should be DCAM_METADATAKIND_TIMESTAMPS.
                                   // [in] option should be one of DCAMBUF_METADATAOPTION or DCAMREC_METADATAOPTION

        DCAM_TIMESTAMP* timestamps; // [in] pointer for TIMESTAMP block
        int32 timestampsize;        // [in] sizeof(DCAM_TIMESTRAMP)
        int32 timestampvaildsize;   // [o] return the written data size of DCAM_TIMESTRAMP.
        int32 timestampkind;        // [o] return timestamp kind(Hardware, Driver, DCAM etc..)
        int32 reserved;
    }
    DCAM_DECLARE_END(DCAM_TIMESTAMPBLOCK)

    DCAM_DECLARE_BEGIN(struct, DCAM_FRAMESTAMPBLOCK)
    {
        DCAM_METADATABLOCKHDR hdr; // [in] size member should be size of this structure
                                   // [in] iKind should be DCAM_METADATAKIND_FRAMESTAMPS.
                                   // [in] option should be one of DCAMBUF_METADATAOPTION or DCAMREC_METADATAOPTION

        int32* framestamps; // [in] pointer for framestamp block
        int32 reserved;
    }
    DCAM_DECLARE_END(DCAM_FRAMESTAMPBLOCK)

    DCAM_DECLARE_BEGIN(struct, DCAM_METADATATEXTBLOCK)
    {
        DCAM_METADATABLOCKHDR hdr;

        void* text;          // [i/o] see below.
        int32* textsizes;    // [i/o] see below.
        int32 bytesperunit;  // [i/o] see below.
        int32 reserved;      // [in] reserved to 0
        int32* textcodepage; // [i/o] see below.

        // Note
        // dcamrec_copymetadatablock()
        //	buf										// [in] pointer for filling
        //userdatatext
        // block 	unitsizes								// [in] pointer for filling
        // each
        // text size of METADATA 	bytesperunit							// [in] max bytes
        // per unit for filling each METADATA 	textcodepage							// [in]
        // pointer for filling each text codepage of METADATA

        // dcamrec_lockmetadatablock()
        //	buf										// [out] return DCAM internal pointer
        //of userdatatext block 	unitsizes								// [out] return
        //DCAM internal array pointer of each size
        //	bytesperunit							// [out] max bytes per unit which is set
        //at DCAMREC_OPEN
        //	textcodepage							// [out] return DCAM internal array pointer
        //of each codepage
    }
    DCAM_DECLARE_END(DCAM_METADATATEXTBLOCK)

    DCAM_DECLARE_BEGIN(struct, DCAM_METADATABINBLOCK)
    {
        DCAM_METADATABLOCKHDR hdr;

        void* bin;          // [i/o] see below.
        int32* binsizes;    // [i/o] see below.
        int32 bytesperunit; // [i/o] see below.
        int32 reserved;     // [in] reserved to 0

        // Note
        // dcamrec_copymetadatablock()
        //	bin										// [in] pointer for filling
        //userdatabin
        // block 	binsizes								// [in] pointer for filling
        // each bin size of METADATA
        //	bytesperunit							// [in] max bytes per unit for filling
        //each METADATA

        // dcamrec_lockmetadatablock()
        //	bin										// [out] return DCAM internal pointer
        //of userdata bin block 	binsizes								// [out] return
        //DCAM internal array pointer of each bin size 	bytesperunit
        // // [out] max bytes per unit which is set at DCAMREC_OPEN
    }
    DCAM_DECLARE_END(DCAM_METADATABINBLOCK)

    DCAM_DECLARE_BEGIN(struct, DCAMREC_STATUS)
    {
        int32 size;
        int32 currentsession_index;
        int32 maxframecount_per_session;
        int32 currentframe_index;
        int32 missingframe_count;
        int32 flags;
        int32 totalframecount;
        int32 reserved;
    }
    DCAM_DECLARE_END(DCAMREC_STATUS)

    /* **************************************************************** *

            functions (ver 4.x)

     * **************************************************************** */

    // Initialize, uninitialize and misc.
    DCAMERR DCAMAPI dcamapi_init(DCAMAPI_INIT* param DCAM_DEFAULT_ARG);
    DCAMERR DCAMAPI dcamapi_uninit();
    DCAMERR DCAMAPI dcamdev_open(DCAMDEV_OPEN* param);
    DCAMERR DCAMAPI dcamdev_close(HDCAM h);
    DCAMERR DCAMAPI dcamdev_showpanel(HDCAM h, int32 iKind);
    DCAMERR DCAMAPI dcamdev_getcapability(HDCAM h, DCAMDEV_CAPABILITY* param);
    DCAMERR DCAMAPI dcamdev_getstring(HDCAM h, DCAMDEV_STRING* param);

    // Property control
    DCAMERR DCAMAPI dcamprop_getattr(HDCAM h, DCAMPROP_ATTR* param);
    DCAMERR DCAMAPI dcamprop_getvalue(HDCAM h, int32 iProp, double* pValue);
    DCAMERR DCAMAPI dcamprop_setvalue(HDCAM h, int32 iProp, double fValue);
    DCAMERR DCAMAPI dcamprop_setgetvalue(HDCAM h, int32 iProp, double* pValue, int32 option DCAM_DEFAULT_ARG);
    DCAMERR DCAMAPI dcamprop_queryvalue(HDCAM h, int32 iProp, double* pValue, int32 option DCAM_DEFAULT_ARG);
    DCAMERR DCAMAPI dcamprop_getnextid(HDCAM h, int32* pProp, int32 option DCAM_DEFAULT_ARG);
    DCAMERR DCAMAPI dcamprop_getname(HDCAM h, int32 iProp, char* text, int32 textbytes);
    DCAMERR DCAMAPI dcamprop_getvaluetext(HDCAM h, DCAMPROP_VALUETEXT* param);

    // Buffer control
    DCAMERR DCAMAPI dcambuf_alloc(HDCAM h, int32 framecount); // call dcambuf_release() to free.
    DCAMERR DCAMAPI dcambuf_attach(HDCAM h, const DCAMBUF_ATTACH* param);
    DCAMERR DCAMAPI dcambuf_release(HDCAM h, int32 iKind DCAM_DEFAULT_ARG);
    DCAMERR DCAMAPI dcambuf_lockframe(HDCAM h, DCAMBUF_FRAME* pFrame);
    DCAMERR DCAMAPI dcambuf_copyframe(HDCAM h, DCAMBUF_FRAME* pFrame);
    DCAMERR DCAMAPI dcambuf_copymetadata(HDCAM h, DCAM_METADATAHDR* hdr);

    // Capturing
    DCAMERR DCAMAPI dcamcap_start(HDCAM h, int32 mode);
    DCAMERR DCAMAPI dcamcap_stop(HDCAM h);
    DCAMERR DCAMAPI dcamcap_status(HDCAM h, int32* pStatus);
    DCAMERR DCAMAPI dcamcap_transferinfo(HDCAM h, DCAMCAP_TRANSFERINFO* param);
    DCAMERR DCAMAPI dcamcap_firetrigger(HDCAM h, int32 iKind DCAM_DEFAULT_ARG);
    DCAMERR DCAMAPI dcamcap_record(HDCAM h, HDCAMREC hrec);

    // Wait abort handle control
    DCAMERR DCAMAPI dcamwait_open(DCAMWAIT_OPEN* param);
    DCAMERR DCAMAPI dcamwait_close(HDCAMWAIT hWait);
    DCAMERR DCAMAPI dcamwait_start(HDCAMWAIT hWait, DCAMWAIT_START* param);
    DCAMERR DCAMAPI dcamwait_abort(HDCAMWAIT hWait);

// Recording
#ifdef _WIN32
    DCAMERR DCAMAPI dcamrec_openA(DCAMREC_OPENA* param);
    DCAMERR DCAMAPI dcamrec_openW(DCAMREC_OPENW* param);

#ifdef _UNICODE
#define DCAMREC_OPEN DCAMREC_OPENW
#define dcamrec_open dcamrec_openW
#else
#define DCAMREC_OPEN DCAMREC_OPENA
#define dcamrec_open dcamrec_openA
#endif

#else
    DCAMERR DCAMAPI dcamrec_open(DCAMREC_OPEN* param);
#endif

    DCAMERR DCAMAPI dcamrec_close(HDCAMREC hrec);
    DCAMERR DCAMAPI dcamrec_lockframe(HDCAMREC hrec, DCAMREC_FRAME* pFrame);
    DCAMERR DCAMAPI dcamrec_copyframe(HDCAMREC hrec, DCAMREC_FRAME* pFrame);
    DCAMERR DCAMAPI dcamrec_writemetadata(HDCAMREC hrec, const DCAM_METADATAHDR* hdr);
    DCAMERR DCAMAPI dcamrec_lockmetadata(HDCAMREC hrec, DCAM_METADATAHDR* hdr);
    DCAMERR DCAMAPI dcamrec_copymetadata(HDCAMREC hrec, DCAM_METADATAHDR* hdr);
    DCAMERR DCAMAPI dcamrec_lockmetadatablock(HDCAMREC hrec, DCAM_METADATABLOCKHDR* hdr);
    DCAMERR DCAMAPI dcamrec_copymetadatablock(HDCAMREC hrec, DCAM_METADATABLOCKHDR* hdr);

    DCAMERR DCAMAPI dcamrec_pause(HDCAMREC hrec);
    DCAMERR DCAMAPI dcamrec_resume(HDCAMREC hrec);
    DCAMERR DCAMAPI dcamrec_status(HDCAMREC hrec, DCAMREC_STATUS* pStatus);

    // backward compatibility

    typedef DCAMBUF_FRAME DCAM_FRAME;

    // ---- obsolete ----

    DCAM_DECLARE_BEGIN(enum, DCAM_METADATAKIND) // obsolete
    {DCAM_METADATAKIND_USERDATATEXT = 0x00000001,
     DCAM_METADATAKIND_USERDATABIN = 0x00000002,
     DCAM_METADATAKIND_TIMESTAMPS = 0x00010000,
     DCAM_METADATAKIND_FRAMESTAMPS = 0x00020000,

     end_of_dcam_metadatakind} DCAM_DECLARE_END(DCAM_METADATAKIND)

        DCAM_DECLARE_BEGIN(enum, DCAM_USERDATAKIND){
            DCAM_USERDATAKIND_FRAME = DCAMREC_METADATAOPTION__LOCATION_FRAME,
            DCAM_USERDATAKIND_FILE = DCAMREC_METADATAOPTION__LOCATION_FILE,
            DCAM_USERDATAKIND_SESSION = DCAMREC_METADATAOPTION__LOCATION_SESSION,

            DCAM_USERDATAKIND_LOCATION_MASK =
                DCAMREC_METADATAOPTION__LOCATION__MASK} DCAM_DECLARE_END(DCAM_USERDATAKIND)

            DCAM_DECLARE_BEGIN(struct, DCAM_METADATABLOCK)
    {
        DCAM_METADATABLOCKHDR hdr;

        void* buf;           // [i/o] see below.
        int32* unitsizes;    // [i/o] see below.
        int32 bytesperunit;  // [i/o] see below.
        int32 userdata_kind; // [in] choose userdata kind(File, Session, Frame)

        // Note
        // dcamrec_copymetadatablock()
        //	buf										// [in] pointer for filling
        //userdata
        // block 	unitsizes								// [in] pointer for filling
        // each
        // unit size of METADATA 	bytesperunit							// [in] max bytes
        // per unit for filling each METADATA

        // dcamrec_lockmetadatablock()
        //	buf										// [out] return DCAM internal pointer
        //of userdata block
        //	unitsizes								// [out] return DCAM internal array pointer
        //of each size
        //	bytesperunit							// [out] max bytes per unit which is set
        //at DCAMREC_OPEN
    }
    DCAM_DECLARE_END(DCAM_METADATABLOCK)

#endif // DCAMAPI_VER >= 4000

    /* **************************************************************** */

#ifdef __cplusplus

    /* end of extern "C" */
};

/*** C++ utility ***/

inline int failed(DCAMERR err) { return int(err) < 0; }

#endif

#if !defined(DCAMAPI_VERMIN) || DCAMAPI_VERMIN <= 3200

#include "dcamapi3.h"

#endif // ! defined(DCAMAPI_VERMIN) || DCAMAPI_VERMIN <= 3200

#if (defined(_MSC_VER) && defined(_LINK_DCAMAPI_LIB))
#pragma comment(lib, "dcamapi.lib")
#endif

#define _INCLUDE_DCAMAPI_H_
#endif
