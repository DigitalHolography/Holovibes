// paramC7190_2x.h
// [Oct.16,2001]

#ifndef _INCLUDE_PARAMC7190_2X_H_
#define _INCLUDE_PARAMC7190_2X_H_

struct DCAM_PARAM_C7190_2X
{
    DCAM_HDR_PARAM hdr; // id == DCAM_IDPARAM_C7190

    int32 CEG;   // Contrast Enhance Gain
    int32 CEO;   // Contrast Enhance Offset
    int32 HVC;   // HV control
    int32 HVN;   // HV sensitivity
    int32 REC;   // recur control
    int32 RCN;   // recur N
    int32 BGC;   // back grand control
    int32 BGO;   // back grand offset
    int32 CAI_H; // horizontal
    int32 CAI_V; // vertical
    int32 CAI_A; // bits per pixel
};

enum
{
    dcamparam_c7190_2x_CEG = 0x00000001,
    dcamparam_c7190_2x_CEO = 0x00000002,
    dcamparam_c7190_2x_HVC = 0x00000004,
    dcamparam_c7190_2x_HVN = 0x00000008,
    dcamparam_c7190_2x_REC = 0x00000010,
    dcamparam_c7190_2x_RCN = 0x00000020,
    dcamparam_c7190_2x_BGC = 0x00000040,
    dcamparam_c7190_2x_BGO = 0x00000080,
    dcamparam_c7190_2x_BGS = 0x00000100, // no member assinged
                                         // set: back ground store
                                         // get: available this bit to set
    dcamparam_c7190_2x_CAI_H = 0x00000200,
    dcamparam_c7190_2x_CAI_V = 0x00000400,
    dcamparam_c7190_2x_CAI_A = 0x00000800
};

#endif // _INCLUDE_PARAMC7300_2X_H_
