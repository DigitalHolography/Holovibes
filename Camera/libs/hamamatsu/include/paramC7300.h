// paramC7300.h
// [Sep.26,2002]

#ifndef _INCLUDE_PARAMC7300_H_
#define _INCLUDE_PARAMC7300_H_

struct DCAM_PARAM_C7300
{
    DCAM_HDR_PARAM hdr; // id == DCAM_IDPARAM_C7300

    int32 AMD; // Acquire mode              N/E
    int32 NMD; // Normal mode               N/S/F
    int32 EMD; // External mode             E/L
    int32 SMD; // Scam mode                 N/S/A
    int32 ADS; // A/D Select                8/10/12
    int32 SHT; // Shutter
    int32 FBL; // Frame Blanking
    int32 ESC; // NRK not support External trigger Sourece Connector B/D/I
    int32 SPX; // Super Pixel               2/4/8
    int32 SVO; // Scan V-Offset              0 - 1008
    int32 SVW; // Scan V-Width              16 - 1024
    int32 EST; // External Shutter
    int32 SHA; // read Scan Horizontal Area F/K
    int32 SFD; // Optical Black             O/F
    int32 ATP; // Active Trigger Polarity   N/P
    int32 CEO; // Contrast Enhance Offset   0-255
    int32 CEG; // Contrast Enhance Gain     0-255
};

enum
{
    dcamparam_c7300_AMD = 0x00000001,
    dcamparam_c7300_NMD = 0x00000002,
    dcamparam_c7300_EMD = 0x00000004,
    dcamparam_c7300_SMD = 0x00000008,
    dcamparam_c7300_ADS = 0x00000010,
    dcamparam_c7300_SHT = 0x00000020,
    dcamparam_c7300_FBL = 0x00000040,
    dcamparam_c7300_ESC = 0x00000080,
    dcamparam_c7300_SVO = 0x00000100,
    dcamparam_c7300_SVW = 0x00000200,
    dcamparam_c7300_SPX = 0x00000400,
    dcamparam_c7300_EST = 0x00000800,
    dcamparam_c7300_SHA = 0x00001000,
    dcamparam_c7300_SFD = 0x00002000,
    dcamparam_c7300_ATP = 0x00004000,
    dcamparam_c7300_CEO = 0x00008000,
    dcamparam_c7300_CEG = 0x00010000
};

enum
{
    // C7300-10-12NR
    c7300_kSHT_min_N = 1,
    c7300_kSHT_max_N = 1039,
    c7300_kSHT_min_B2 = 1,
    c7300_kSHT_max_B2 = 519,
    c7300_kSHT_min_B4 = 3,
    c7300_kSHT_max_B4 = 260,
    c7300_kSHT_min_B8 = 3,
    c7300_kSHT_max_B8 = 133,
    c7300_kFBL_min_N = 1,
    c7300_kFBL_max_N = 90,
    c7300_kFBL_min_B2 = 1,
    c7300_kFBL_max_B2 = 180,
    c7300_kFBL_min_B4 = 1,
    c7300_kFBL_max_B4 = 325,
    c7300_kFBL_min_B8 = 1,
    c7300_kFBL_max_B8 = 534,
    c7300_kEST_min = 1,
    c7300_kEST_max = 93600,

    c7300_kESC_BNC = 0,
    c7300_kESC_IF,
    c7300_kESC_DSUB,

    c7300_kSVO_unit = 16,
    c7300_kSVO_min = 0,
    c7300_kSVO_max = 1008,

    c7300_kSVW_unit = 16,
    c7300_kSVW_min = 16,
    c7300_kSVW_max = 1024
};

#endif // _INCLUDE_PARAMC7300_H_
