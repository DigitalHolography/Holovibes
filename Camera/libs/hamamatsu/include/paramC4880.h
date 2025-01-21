// paramC4880.h
// [Nov.09,2001]

#ifndef _INCLUDE_PARAMC4880_H_
#define _INCLUDE_PARAMC4880_H_

struct DCAM_PARAM_C4880
{
    DCAM_HDR_PARAM hdr; // id == DCAM_PARAMID_C4880

    int32 AMD;  // Acquire Mode					Internal/External Trigger/External Time/External
                // Stop/External Level
    int32 ASH;  // Acquire Shutter				Auto/Close/Open
    double AET; // Acquire Exposure Time		mmmm:ss.xxx
    int32 ATN;  // Acquire Trigger Number		1-9999
    int32 ACN;  // Acquire Cycle Number			1-9999
    int32 ATP;  // Acquire Trigger Polarity		Posi/ Nega
    double PET; // Post-Acuire Exposure Time	ss.xxx

    int32 SSP;   // Scan Speed					High/Slow
    int32 SOP;   // Scan OPB						Valid/Invalid
    int32 SAG;   // Scan Amp Gain				Low/High/Super High
    int32 SMD;   // Scan Mode					Normal/Sub-Array/Binning/Super-Pixel
    int32 SVO;   // Scan V-Offset				0-511
    int32 SVW;   // Scan V-Width					1-512
    int32 SVB;   // Scan V-Binning Size			1-512
    char SHA[3]; // Scan H-Area					F/HC/HL/HR/QC/QL/QR/EC
    int32 SHB;   // Scan H-Binning Size			1/2/4/8
    int32 SPX;   // Super Pixel					2/4/8
    int32 SFD;   // Set Front Dummy				On/Off

    int32 CEC; // Contrast Enhance Control		Volume/External/Off
    int32 CEG; // Contrast Enhance Gain		0-255
    int32 CEO; // Contrast Enhance Offset		0-255

    int32 PSW; // Panel Switch					Enable/Disable
    int32 CSW; // Cooler Switch				On/Off
    int32 TST; // Temperature Set				-80....0  5Multiple

    int32 CVG;  // Read Contrast Enhance Volume Gain
    int32 CVO;  // Read Contrast Enhance Volume Offset
    double TMP; // Read Temperature

    int32 CAI_I; // High Speed Mode A/D Bit
    int32 CAI_S; // Slow Speed Mode A/D Bit
};

enum
{
    dcamparam_c4880_AMD = 0x00000001,
    dcamparam_c4880_ASH = 0x00000002,
    dcamparam_c4880_AET = 0x00000004,
    dcamparam_c4880_ATN = 0x00000008,
    dcamparam_c4880_ACN = 0x00000010,
    dcamparam_c4880_ATP = 0x00000020,
    dcamparam_c4880_PET = 0x00000040,

    dcamparam_c4880_SSP = 0x00000080,
    dcamparam_c4880_SOP = 0x00000100,
    dcamparam_c4880_SAG = 0x00000200,
    dcamparam_c4880_SMD = 0x00000400,
    dcamparam_c4880_SVO = 0x00000800,
    dcamparam_c4880_SVW = 0x00001000,
    dcamparam_c4880_SVB = 0x00002000,
    dcamparam_c4880_SHA = 0x00004000,
    dcamparam_c4880_SHB = 0x00008000,
    dcamparam_c4880_SPX = 0x00010000,
    dcamparam_c4880_SFD = 0x00020000,

    dcamparam_c4880_CEC = 0x00040000,
    dcamparam_c4880_CEG = 0x00080000,
    dcamparam_c4880_CEO = 0x00100000,

    dcamparam_c4880_PSW = 0x00200000,
    dcamparam_c4880_CSW = 0x00400000,
    dcamparam_c4880_TST = 0x00800000,

    dcamparam_c4880_CVG = 0x01000000,
    dcamparam_c4880_CVO = 0x02000000,
    dcamparam_c4880_TMP = 0x04000000,

    dcamparam_c4880_CAI_I = 0x08000000,
    dcamparam_c4880_CAI_S = 0x10000000
};

enum
{
    c4880_kATN_min = 1,
    c4880_kATN_max = 9999,

    c4880_kACN_min = 1,
    c4880_kACN_max = 9999,

    c4880_kCEG_min = 0,
    c4880_kCEG_max = 255,

    c4880_kCEO_min = 0,
    c4880_kCEO_max = 255
};

#define c4880_kAET_min (20e-3)
#define c4880_kAET_max (9999 * 60 + 59 + 999e-3)

#define c4880_kPET_min 0
#define c4880_kPET_max 30

#endif // _INCLUDE_PARAMC4880_H_
