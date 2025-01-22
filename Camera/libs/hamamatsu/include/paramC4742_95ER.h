// paramC4742_95ER.h

struct DCAM_PARAM_C4742_95ER
{
    DCAM_HDR_PARAM hdr; // id == DCAM_IDPARAM_C4742_95ER

    int32 SHO; // Sub-Array Offset
    int32 SHW; // Sub-Array Width
    int32 SVO; // Sub-Array Offset  0 - 1008
    int32 SVW; // Sub-Array Width  16 - 1024
    int32 LMD; // Light Mode
};

enum
{
    dcamparam_c4742_95ER_SHO = 0x00000010,
    dcamparam_c4742_95ER_SHW = 0x00000020,
    dcamparam_c4742_95ER_SVO = 0x00008000,
    dcamparam_c4742_95ER_SVW = 0x00010000,
    dcamparam_c4742_95ER_LMD = 0x00020000
};

enum
{
    c4742_95ER_kSHT_N_min = 1,
    c4742_95ER_kSHT_N_max = 1055,
    c4742_95ER_kSHT_B2_min = 1,
    c4742_95ER_kSHT_B2_max = 535,
    c4742_95ER_kSHT_B4_min = 1,
    c4742_95ER_kSHT_B4_max = 266,
    c4742_95ER_kSHT_B8_min = 1,
    c4742_95ER_kSHT_B8_max = 137,

    c4742_95ER_kFBL_N_min = 1,
    c4742_95ER_kFBL_N_max = 90,
    c4742_95ER_kFBL_B2_min = 1,
    c4742_95ER_kFBL_B2_max = 180,
    c4742_95ER_kFBL_B4_min = 1,
    c4742_95ER_kFBL_B4_max = 325,
    c4742_95ER_kFBL_B8_min = 1,
    c4742_95ER_kFBL_B8_max = 534,

    c4742_95ER_kEST_min = 1,
    c4742_95ER_kEST_max = 95040, // 93600,// ER, 95040

    c4742_95ER_kSFD_On = 0,
    c4742_95ER_kSFD_Off,

    c4742_95ER_kATP_Negative = 0,
    c4742_95ER_kATP_Positive,

    c4742_95ER_kGAN_min = 0,
    c4742_95ER_kGAN_max = 4095,

    c4742_95ER_kOFS_min = 0,
    c4742_95ER_kOFS_max = 4095,

    c4742_95ER_kSHO_unit = 8,
    c4742_95ER_kSHO_min = 0,
    c4742_95ER_kSHO_max = 1336,

    c4742_95ER_kSHW_unit = 8,
    c4742_95ER_kSHW_min = 8,
    c4742_95ER_kSHW_max = 1344,

    c4742_95ER_kSVO_unit = 8,
    c4742_95ER_kSVO_min = 0,
    c4742_95ER_kSVO_max = 1016,

    c4742_95ER_kSVW_unit = 8,
    c4742_95ER_kSVW_min = 8,
    c4742_95ER_kSVW_max = 1024
};
