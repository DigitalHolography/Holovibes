// paramC7780.h
// [Oct.23,2001]

#ifndef _INCLUDE_PARAMC7780_H_
#define _INCLUDE_PARAMC7780_H_

struct DCAM_PARAM_C7780
{
    DCAM_HDR_PARAM hdr; // id == DCAM_IDPARAM_C7780

    int32 ADS;     // A/D Select
    int32 ADS_ADJ; // A/D Select Adjust

    int32 SHT_R; // Sutter n R
    int32 SHT_G; // Sutter n G
    int32 SHT_B; // Sutter n B

    double AET_R; // Exposure Time R
    double AET_G; // Exposure Time G
    double AET_B; // Exposure Time B

    double RAT_R; // Return Exposure Time R
    double RAT_G; // Return Exposure Time G
    double RAT_B; // Return Exposure Time B

    int32 LMD; // Light Mode

    int32 SHO; // Sub-Array H Offset
    int32 SHW; // Sub-Array H Width
    int32 SVO; // Sub-Array V Offset
    int32 SVW; // Sub-Array V Width

    int32 OMD; // Output Mode
    int32 MTX; // Matrix

    int32 WBL; // White Balance Memory Number
};

enum
{
    dcamparam_c7780_ADS = 0x00000001,
    dcamparam_c7780_SHT_R = 0x00000002,
    dcamparam_c7780_SHT_G = 0x00000004,
    dcamparam_c7780_SHT_B = 0x00000008,
    dcamparam_c7780_SHT_A = 0x00000010,
    dcamparam_c7780_MST = 0x00000020,
    dcamparam_c7780_AET_R = 0x00000040,
    dcamparam_c7780_AET_G = 0x00000080,
    dcamparam_c7780_AET_B = 0x00000100,
    dcamparam_c7780_AET_A = 0x00000200,
    dcamparam_c7780_RAT_R = 0x00000400,
    dcamparam_c7780_RAT_G = 0x00000800,
    dcamparam_c7780_RAT_B = 0x00001000,
    dcamparam_c7780_LMD = 0x00002000,
    dcamparam_c7780_SHO = 0x00004000,
    dcamparam_c7780_SHW = 0x00008000,
    dcamparam_c7780_SVO = 0x00010000,
    dcamparam_c7780_SVW = 0x00020000,
    dcamparam_c7780_RAT = 0x00040000,
    dcamparam_c7780_OMD = 0x00080000,
    dcamparam_c7780_MTX = 0x00100000,

    // Command
    dcamparam_c7780_WBL_B = 0x00200000, // Black Balance
    dcamparam_c7780_WBL_E = 0x00400000, // Exposure Time (Int. 80%)
    dcamparam_c7780_WBL_P = 0x00800000, // Gain Peak
    dcamparam_c7780_WBL_T = 0x01000000, // Time Peak
    dcamparam_c7780_WBL_M = 0x02000000, // Matrix
    dcamparam_c7780_WBL_S = 0x04000000, // Store Memory
    dcamparam_c7780_WBL_L = 0x08000000  // Load Memory
};

enum
{
    c7780_kSHT_min = 1,
    c7780_kSHT_max = 95694,

    c7780_kGAN_min = 0,
    c7780_kGAN_max = 2,

    c7780_kSHO_unit = 8,
    c7780_kSHO_min = 0,
    c7780_kSHO_max = 1336,

    c7780_kSHW_unit = 8,
    c7780_kSHW_min = 8,
    c7780_kSHW_max = 1344,

    c7780_kSVO_unit = 8,
    c7780_kSVO_min = 0,
    c7780_kSVO_max = 1016,

    c7780_kSVW_unit = 8,
    c7780_kSVW_min = 8,
    c7780_kSVW_max = 1024
};

#endif // _INCLUDE_PARAMC7780_H_
