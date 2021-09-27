// paramC4742_95.h
// [Oct.3,2001]

#ifndef _INCLUDE_PARAMC4742_95_H_
#define _INCLUDE_PARAMC4742_95_H_

struct DCAM_PARAM_C4742_95
{
    DCAM_HDR_PARAM hdr; // id == DCAM_PARAMID_C4742_95

    int32 AMD; // char;	Acquire mode              N/E
    int32 NMD; // char;	Normal mode               N/S/F
    int32 EMD; // char;	External mode             E/L
    int32 ATP; // char;	Active Trigger Polarity   N/P

    int32 SHT; // int32;	Shutter
    int32 FBL; // int32;	Frame Blanking
    int32 EST; // int32;	External Shutter

    int32 SFD; // char;	Optical Black             O/F
    int32 SMD; // char;	Scam mode                 N/S
    int32 SPX; // int32;	Super Pixel               2/4/8

    int32 ADS; // int32;	A/D Select                8/10/12

    int32 CEO; // int32;	Contrast Enhance Offset   0-255
    int32 CEG; // int32;	Contrast Enhance Gain     0-255

    int32 ESC; // char;	External trigger Sourece Connector
               //			B/D/I	BNC or D-SUB or I/F

    int32 SHA; // char;	read Scan Horizontal Area
               //			F/K  Full(1280) Killo(1024)
};

enum
{
    dcamparam_c4742_95_AMD = 0x00000001,
    dcamparam_c4742_95_NMD = 0x00000002,
    dcamparam_c4742_95_EMD = 0x00000004,

    dcamparam_c4742_95_ATP = 0x00000008,

    dcamparam_c4742_95_SHT = 0x00000100,
    dcamparam_c4742_95_FBL = 0x00000200,
    dcamparam_c4742_95_EST = 0x00000400,

    dcamparam_c4742_95_SFD = 0x00001000,
    dcamparam_c4742_95_SMD = 0x00002000,
    dcamparam_c4742_95_SPX = 0x00004000,
    dcamparam_c4742_95_ADS = 0x00008000,

    dcamparam_c4742_95_CEO = 0x00010000,
    dcamparam_c4742_95_CEG = 0x00020000,

    dcamparam_c4742_95_ESC = 0x00040000,

    dcamparam_c4742_95_SHA = 0x00080000
};

/*
enum {

        c4742_95_kATP_Negative		= 0,
        c4742_95_kATP_Positive,

        c4742_95_kUNIT_Shutter		= 0,
        c4742_95_kUNIT_USec,
        c4742_95_kUNIT_MSec,
        c4742_95_kUNIT_Sec,
        c4742_95_kUNIT_Min,

        c4742_95_kSPX_2				= 0,
        c4742_95_kSPX_4,
        c4742_95_kSPX_8,

        c4742_95_kSFD_On			= 0,
        c4742_95_kSFD_Off,

        c4742_95_kESC_BNC			= 0,
        c4742_95_kESC_DSUB,
        c4742_95_kESC_IF,
};
*/

#endif // _INCLUDE_PARAMC4742_95_H_
