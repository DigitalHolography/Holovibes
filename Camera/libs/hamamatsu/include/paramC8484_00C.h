// paramC8484_00C.h

#ifndef _INCLUDE_PARAMC8484_00C_H_
#define _INCLUDE_PARAMC8484_00C_H_

struct DCAM_PARAM_C8484_00C
{
    DCAM_HDR_PARAM hdr; // id == DCAM_PARAMID_C8484_00C

    int32 AMD; // char;	Acquire mode              N/E
    int32 NMD; // char;	Normal mode               N/S/F
               // not support

    int32 TMD; // char;	TDI mode				  I/E

    int32 FBL; // int32;	Frame Blanking
    int32 SVB; // int32;	Scan Vertical Binning	  1-29999
    int32 HLN; // int32;	Horizontal line
               // C8484-00C:	1500-30000
               // C9162:		 696-65535

    int32 CEG; // int32;	Contrast Enhance Gain     0-255
    int32 TPS; // int32;	Test Patturn Select       0,1,2,3

#if 20030130
    // only C9162
    int32 ATP; // char;	Active Trigger Polarity   N/P
    int32 ESC; // char;	External trigger Sourece Connector
               //			B/I	BNC or I/F

    char UNQ1[8]; //		choose exposure stage for each channel	A/B/C/D
#endif
};

enum
{
    dcamparam_c8484_00C_AMD = 0x00000001,
    dcamparam_c8484_00C_NMD = 0x00000002,
    dcamparam_c8484_00C_TMD = 0x00000004,

    dcamparam_c8484_00C_SVB = 0x00000008,
    dcamparam_c8484_00C_HLN = 0x00000010,

    dcamparam_c8484_00C_FBL = 0x00000020,

    dcamparam_c8484_00C_CEG = 0x00000040,
    dcamparam_c8484_00C_TPS = 0x00000080,

    dcamparam_c8484_00C_TimeStampClear = 0x00000200,

#if 20030130
    dcamparam_c8484_00C_ATP = 0x00000400,
    dcamparam_c8484_00C_ESC = 0x00000800,
    dcamparam_c8484_00C_UNQ1 = 0x00001000
#endif
};

#endif // _INCLUDE_PARAMC8484_00C_H_
