// paramC4742_95HR.h
// [Oct.3,2001]

#ifndef _INCLUDE_PARAMC4742_95HR_H_
#define _INCLUDE_PARAMC4742_95HR_H_

struct DCAM_PARAM_C4742_95HR
{
    DCAM_HDR_PARAM hdr; // id == DCAM_IDPARAM_C4742_95HR
    double AET;         // Exposure Time
    int32 SHO;          // Sub-Array Offset 0 - 3992
    int32 SHW;          // Sub-Array Width  8 - 4000
    int32 SVO;          // Sub-Array Offset 0 - 2616
    int32 SVW;          // Sub-Array Width  8 - 2624
    double RAT;         // Real Acquire Time (read only)
};

enum
{
    dcamparam_c4742_95HR_AET = 0x00000001,
    dcamparam_c4742_95HR_SHO = 0x00000010,
    dcamparam_c4742_95HR_SHW = 0x00000020,
    dcamparam_c4742_95HR_SVO = 0x00000040,
    dcamparam_c4742_95HR_SVW = 0x00000080,
    dcamparam_c4742_95HR_RAT = 0x00000100
};

enum
{
    c4742_95HR_kEST_min = 1,
    c4742_95HR_kEST_max = 45100,

    c4742_95HR_kGAN_min = 0,
    c4742_95HR_kGAN_max = 4095,

    c4742_95HR_kOFS_min = 0,
    c4742_95HR_kOFS_max = 4095,

    c4742_95HR_kOPB_min = 0,
    c4742_95HR_kOPB_max = 4095,

    c4742_95HR_kSHO_unit = 8,
    c4742_95HR_kSHO_min = 0,
    c4742_95HR_kSHO_max = 3992,

    c4742_95HR_kSHW_unit = 8,
    c4742_95HR_kSHW_min = 8,
    c4742_95HR_kSHW_max = 4000,

    c4742_95HR_kSVO_unit = 8,
    c4742_95HR_kSVO_min = 0,
    c4742_95HR_kSVO_max = 2616,

    c4742_95HR_kSVW_unit = 8,
    c4742_95HR_kSVW_min = 8,
    c4742_95HR_kSVW_max = 2624
};

#endif // _INCLUDE_PARAMC4742_95HR_H_
