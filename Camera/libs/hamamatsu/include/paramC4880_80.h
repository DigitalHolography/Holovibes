// paramC4880_80.h
// [Jan.08,2002]

#ifndef _INCLUDE_PARAMC4880_80_H_
#define _INCLUDE_PARAMC4880_80_H_

struct DCAM_PARAM_C4880_80
{
    DCAM_HDR_PARAM hdr; // id == DCAM_PARAMID_C4880_80

    int32 AMD; // Acquire Mode					Internal/External
    int32 AET; // Acquire Exposure Time		0-255 (frame Number)
    int32 SHT; // Shutter Time					0-509 (line Number)    exptime = (aet - 1) + sht
    int32 ACN; // Acquire Cycle Number			1-9999
    int32 FCN; // Frame Cycle Number(External) 0-255

    int32 SSP;    // Scan Speed					High/Slow
    int32 SAG;    // Scan Amp Gain				Low/High/Super High
    int32 SMD;    // Scan Mode					Normal/Extended(Subarray)
    int32 SAR_XO; // Sub-Array X Offset
    int32 SAR_YO; // Sub-Array Y Offset
    int32 SAR_XW; // Sub-Array X Width			656
    int32 SAR_YW; // Sub-Array Y Width			494
    int32 SAR_B;  // Binning						1, 2, 4, 8, 16, 32

    int32 CEC; // Contrast Enhance Control		Volume/External/Off
    int32 CEG; // Contrast Enhance Gain		0-255
    int32 CEO; // Contrast Enhance Offset		0-255
    int32 CVG; // Read Contrast Enhance Volume Gain
    int32 CVO; // Read Contrast Enhance Volume Offset

    double FRT;  // Frame Read Time
    int32 SCA;   // Status of Camera
    int32 CAI_H; // Horizontal
    int32 CAI_V; // Vertical
    int32 CAI_I; // High Speed Mode A/D Bit
    int32 CAI_S; // Slow Speed Mode A/D Bit
};

enum
{
    dcamparam_c4880_80_AMD = 0x00000001,
    dcamparam_c4880_80_AET = 0x00000002,
    dcamparam_c4880_80_SHT = 0x00000004,
    dcamparam_c4880_80_ACN = 0x00000008,
    dcamparam_c4880_80_FCN = 0x00000010,

    dcamparam_c4880_80_SSP = 0x00000020,
    dcamparam_c4880_80_SAG = 0x00000040,
    dcamparam_c4880_80_SMD = 0x00000080,
    dcamparam_c4880_80_SAR = 0x00000100,

    dcamparam_c4880_80_CEC = 0x00001000,
    dcamparam_c4880_80_CEG = 0x00002000,
    dcamparam_c4880_80_CEO = 0x00004000,
    dcamparam_c4880_80_CVG = 0x00008000,
    dcamparam_c4880_80_CVO = 0x00010000,

    dcamparam_c4880_80_FRT = 0x00020000,
    dcamparam_c4880_80_SCA = 0x00040000,
    dcamparam_c4880_80_CAI_H = 0x00080000,
    dcamparam_c4880_80_CAI_V = 0x00100000,
    dcamparam_c4880_80_CAI_I = 0x00200000,
    dcamparam_c4880_80_CAI_S = 0x00400000
};

enum
{
    c4880_80_kAET_min = 0,
    c4880_80_kAET_max = 255,

    c4880_80_kSHT_min = 0,
    c4880_80_kSHT_max = 509,

    c4880_80_kACN_min = 1,
    c4880_80_kACN_max = 9999,

    c4880_80_kFCN_min = 0,
    c4880_80_kFCN_max = 255,

    c4880_80_kCEG_min = 0,
    c4880_80_kCEG_max = 255,

    c4880_80_kCEO_min = 0,
    c4880_80_kCEO_max = 255
};

#endif // _INCLUDE_PARAMC4880_80_H_
