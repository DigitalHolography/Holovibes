//	paramC4742_98ER.h

struct DCAM_PARAM_C4742_98ER
{
    DCAM_HDR_PARAM hdr; // id == DCAM_IDPARAM_C4742_98ER

    int32 LMD;  // light mode	(L/H)
    int32 ESC;  // external trigger source connector	(B/D/I)
    double RAT; // read real acquire time

    // no CSW, TST command [2002/07/08]
};

enum
{
    dcamparam_c4742_98ER_LMD = 0x00000001,
    dcamparam_c4742_98ER_ESC = 0x00000002,
    dcamparam_c4742_98ER_RAT = 0x00000010
};

/*
enum {
        c4742_98ER_kSHO_unit	= 8,
        c4742_98ER_kSHO_min		= 0,
        c4742_98ER_kSHO_max		= 1336,

        c4742_98ER_kSHW_unit	= 8,
        c4742_98ER_kSHW_min		= 8,
        c4742_98ER_kSHW_max		= 1344,

        c4742_98ER_kSVO_unit	= 8,
        c4742_98ER_kSVO_min		= 0,
        c4742_98ER_kSVO_max		= 1016,

        c4742_98ER_kSVW_unit	= 8,
        c4742_98ER_kSVW_min		= 8,
        c4742_98ER_kSVW_max		= 1024,
};
*/
