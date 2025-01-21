// paramC8000_10.h
// [Nov.7,2001]

#ifndef _INCLUDE_PARAMC8000_10_H_
#define _INCLUDE_PARAMC8000_10_H_

struct DCAM_PARAM_C8000_10
{
    DCAM_HDR_PARAM hdr; // id == DCAM_PARAMID_C8000_10

    int32 AMD; // Aquire Mode					Normal/External
    int32 EMD; // External Mode				Edge/Level
    int32 SMD; // Scan Mode					Normal/Super-Pixel/Sub-Array

    int32 AET;   // Acquire Exposure Time		number of frame
    int32 SSP;   // Scan Speed					High/Middle/Slow
    int32 SPX;   // Super Pixel					1/2/4/8
    int32 SHO;   // Scan H-Offset				0to632
    int32 SHW;   // Scan H-Width					8to640
    char SHA[3]; // Scan H-Area					F/HC/HL/HR/QC/QL/QR/EC ( 640, 320, 160, 80 )
                 // only available in subarray mode.
    int32 SVO;   // Scan V-Offset				0to472
    int32 SVW;   // Scan V-Width					8to480
    int32 ATP;   // Active Trigger Polarity		Negative/Positive
    int32 SFD;   // Scan Front Dummy				On/Off

    int32 CEG; // Contrast Enhance Gain		0-255
    int32 CEO; // Contrast Enhance Offset		0-255

    int32 CAI_I; // Number of A/D converter bits
};

enum
{
    dcamparam_c8000_10_AMD = 0x00000001,
    dcamparam_c8000_10_EMD = 0x00000002,
    dcamparam_c8000_10_SMD = 0x00000004,

    dcamparam_c8000_10_AET = 0x00000008,
    dcamparam_c8000_10_SSP = 0x00000010,
    dcamparam_c8000_10_SPX = 0x00000020,
    dcamparam_c8000_10_SHO = 0x00000040,
    dcamparam_c8000_10_SHW = 0x00000080,
    dcamparam_c8000_10_SHA = 0x00000100,
    dcamparam_c8000_10_SVO = 0x00000200,
    dcamparam_c8000_10_SVW = 0x00000400,
    dcamparam_c8000_10_ATP = 0x00000800,
    dcamparam_c8000_10_SFD = 0x00001000,

    dcamparam_c8000_10_CEO = 0x00002000,
    dcamparam_c8000_10_CEG = 0x00004000,

    dcamparam_c8000_10_CAI_I = 0x00008000
};

enum
{
    c8000_10_kAET_min = 1,
    c8000_10_kAET_max = 30,

    c8000_10_kSHO_unit = 8,
    c8000_10_kSHO_min = 0,
    c8000_10_kSHO_max = 632,

    c8000_10_kSHW_unit = 8,
    c8000_10_kSHW_min = 8,
    c8000_10_kSHW_max = 640,

    c8000_10_kSVO_unit = 8,
    c8000_10_kSVO_min = 0,
    c8000_10_kSVO_max = 472,

    c8000_10_kSVW_unit = 8,
    c8000_10_kSVW_min = 8,
    c8000_10_kSVW_max = 480,

    c8000_10_kCEG_min = 0,
    c8000_10_kCEG_max = 255,

    c8000_10_kCEO_min = 0,
    c8000_10_kCEO_max = 255
};

#endif // _INCLUDE_PARAMC8000_10_H_
