/* **************************************************************** *
        features.h

2.1.1.02 [Mar. 9,2001]	1. added definitions for LONGLONG, ULONGLONG, LARGE_INTEGER, and ULARGE_INTEGER
                                                as defined by Microsoft, in case the compiler does not have them
defined.
                                                2. made DCAM_IDFEATURE_EXPOSURETIME which is equivalent to
DCAM_IDFEATURE_SHUTTER.
                                                3. gave DCAM_IDFEATURE_LIGHTMODE a unique feature ID from
DCAM_IDFEATURE_GAMMA.
                                                4. defined FEATURE_FLAGS_DEFAULT flag for better clarity.
                                                5. Structures use DCAM_HDR_PARAM because DCAM_HDR_PARAM_64 was not
needed.

2.1.1.03 [Mar.12,2001]	1. Removed LONGLONG, ULONGLONG and so on because DCAM-API is 32bit API.
                                                2. Changed member oder of DCAMPARAM_FEATURE and INQ for better
performance.
                                                3. Changed the constant value to enum.

2.1.1.04 [Mar.16,2001]	1. Renamed DCAM_FEATURE_FLAGS_xxxx.
                                                2. Defined DCAM_SUBARRAY_xxxx members as int32 type.

2.1.1.05 [Mar.26,2001]	Re-defined capflags, flags get, and flags set.
2.1.1.06 [Mar.30,2001]	re-defined
DCAM_PARAM_FEATURE,DCAM_PARAM_FEATURE_INQ,DCAM_PARAM_SUBARRAY,DCAM_PARAM_SUBARRAY_INQ.
                                                **Note: All struct's in DCAM-API have to be defined as:

                                                                        typedef struct structtagname {
                                                                                .
                                                                                .
                                                                        } structname;

                                                                for compatibility with C programming.

2.1.1.07 [April 17,2001] Defined DCAM_PARAM_FRAME_READOUT_TIME_INQ

2.1.1.08 [April 23,2001] 1. Added binning [out] parameter for DCAM_PARAM_SUBARRAY_INQ used to query sub-array region
parameters of different binning modes without adjusting the binning of the camera. For example, allows for asynchronous
                                                inquiry while an image is being captured.
                                                2. Defined which parameters can be [in],[out], or both depending on
DCAM_IDMSG_SETPARAM or DCAM_IDMSG_GETPARAM.

2.1.1.09 [May 9, 2001]	Fixed [in] and [out] definitions for:

                                                        [in]	=	parameter is inputted to the function
                                                        [out]	=	parameter value is returned from the function
                                                        [in/out]=	parameter can be eiter sent in, returned, or
both depending on the context of its use.

2.1.1.10 [May 25, 2001]	Added units to DCAM_PARAM_FEATURE_INQ. If units is empty when returned, then it is relative
control. If units returned is a valid string, then it is absolute control.

2.1.1.11 [May 29, 2001]	Added DCAM_IDFEATURE_INITIALIZE. This can be used to re-initialize the camera settings.
2.1.2.0	[Nov.26, 2001]	Added DCAM_PARAM_SCANMODE and DCAM_PARAM_SCANMODEINQ
2.1.3.0	[Feb. 8, 2002]	Added enum _dcamparam_scanmode_speed
2.1.3.1	[Feb.28, 2002]	Re-defined dcamparam_scanmode_speedslowest -> dcamparam_scanmode_speed_slowest
                                                Re-defined dcamparam_scanmode_speedfastest ->
dcamparam_scanmode_speed_fastest Added enum for DCAM_PARAM_FEATURE parameters Added enum for DCAM_PARAM_FEATURE_INQ
parameters Added enum for DCAM_PARAM_SCANMODE parameters Added enum for DCAM_PARAM_SCANMODEINQ parameters 2.1.3.2 [March
21,2003]	Added helper defines for: DCAM_FEATURE_FLAGS_COOLING_ONOFF DCAM_FEATURE_FLAGS_COOLING_ON
                                                DCAM_FEATURE_FLAGS_COOLING_OFF
                                                DCAM_IDFEATURE_MECHANICALSHUTTER
                                                DCAM_FEATURE_FLAGS_MECHANICALSHUTTER_OPEN
                                                DCAM_FEATURE_FLAGS_MECHANICALSHUTTER_AUTO
                                                DCAM_FEATURE_FLAGS_MECHANICALSHUTTER_CLOSE

 * **************************************************************** */

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
