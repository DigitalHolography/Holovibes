// paramPCDIG.h
// [Apr.26.2001]	add software trigger

#ifndef _INCLUDE_PARAMPCDIG_H_
#define _INCLUDE_PARAMPCDIG_H_

enum
{
    DCAMPCDIG_IDMSG_SOFT_TRIG = 0x400,
};

struct DCAM_PARAM_PCDIG_TRIGGERDELAY
{
    DCAM_HDR_PARAM hdr; // id == DCAM_IDPARAM_PCDIG_TRIGGERDELAY

    int32 bInputPolarityRisingEdge;
    int32 bPolarityActiveHigh;
    int32 start_location;
    int32 bPriorityFirstEdge;
    double delaytime;
    double dPulseWidth;
    int32 bEnable;
    int32 exsync_mode;
};

enum
{

    dcamparam_pcdig_triggerdelay_bInputPolarityRisingEdge = 0x00000001,
    dcamparam_pcdig_triggerdelay_bPolarityActiveHigh = 0x00000002,
    dcamparam_pcdig_triggerdelay_start_location = 0x00000004,
    dcamparam_pcdig_triggerdelay_bPriorityFirstEdge = 0x00000008,
    dcamparam_pcdig_triggerdelay_delaytime = 0x00000010,
    dcamparam_pcdig_triggerdelay_dPulseWidth = 0x00000020,
    dcamparam_pcdig_triggerdelay_exsync_mode = 0x00000040,
    dcamparam_pcdig_triggerdelay_bEnable = 0x00000080,
    dcamparam_pcdig_triggerdelay_update = 0x00000100,
};

// const
enum
{
    kDCAM_PARAM_TRIGGERDELAY_exsync_mode_FREE_RUNNING = 0,
    kDCAM_PARAM_TRIGGERDELAY_exsync_mode_TRIG1,
    kDCAM_PARAM_TRIGGERDELAY_exsync_mode_TRIG2,
    kDCAM_PARAM_TRIGGERDELAY_exsync_mode_SOFT_TRIG,
    kDCAM_PARAM_TRIGGERDELAY_exsync_mode_VB_TRIG,

    kDCAM_PARAM_TRIGGERDELAY_start_location_START_CYCLE_END = 0,
    kDCAM_PARAM_TRIGGERDELAY_start_location_START_CYCLE_BEGIN,
    kDCAM_PARAM_TRIGGERDELAY_start_location_START_MIDPOINT,
};

#endif // _INCLUDE_PARAMPCDIG_H_
