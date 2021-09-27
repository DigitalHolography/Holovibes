// paramC8484.h

struct DCAM_PARAM_C8484
{
    DCAM_HDR_PARAM hdr; // id == DCAM_IDPARAM_C8484

    int32 SHO; // Sub-Array Offset
    int32 SHW; // Sub-Array Width
    int32 SVO; // Sub-Array Offset
    int32 SVW; // Sub-Array Width

    int32 LMD; // Light Mode
};

enum
{
    dcamparam_c8484_SHO = 0x00000001,
    dcamparam_c8484_SHW = 0x00000002,
    dcamparam_c8484_SVO = 0x00000004,
    dcamparam_c8484_SVW = 0x00000008,

    dcamparam_c8484_LMD = 0x00000010
};

enum
{
    c8484_kSHT_N_min = 1,
    c8484_kSHT_N_max = 1055,
    c8484_kSHT_B2_min = 1,
    c8484_kSHT_B2_max = 535,
    c8484_kSHT_B4_min = 1,
    c8484_kSHT_B4_max = 266,
    c8484_kSHT_B8_min = 1,
    c8484_kSHT_B8_max = 137,

    c8484_kFBL_N_min = 1,
    c8484_kFBL_N_max = 9,
    c8484_kFBL_B2_min = 1,
    c8484_kFBL_B2_max = 18,
    c8484_kFBL_B4_min = 1,
    c8484_kFBL_B4_max = 33,
    c8484_kFBL_B8_min = 1,
    c8484_kFBL_B8_max = 54,

    c8484_kEST_min = 1,
    c8484_kEST_max = 9504,

    c8484_kSHO_unit = 8,
    c8484_kSHO_min = 0,
    c8484_kSHO_max = 1336,

    c8484_kSHW_unit = 8,
    c8484_kSHW_min = 8,
    c8484_kSHW_max = 1344,

    c8484_kSVO_unit = 8,
    c8484_kSVO_min = 0,
    c8484_kSVO_max = 1016,

    c8484_kSVW_unit = 8,
    c8484_kSVW_min = 8,
    c8484_kSVW_max = 1024
};
