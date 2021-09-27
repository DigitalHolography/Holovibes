//	paramC4742_98.h

struct DCAM_PARAM_C4742_98
{
    DCAM_HDR_PARAM hdr; // id == DCAM_IDPARAM_C4742_98

    int32 AMD;  // acquire mode							(N/E)
    int32 EMD;  // external exposure time setting mode	(E/L)
    double AET; // acquire exposure time				(sec)
    int32 ATP;  // acquire trigger polarity				(P/N)

    int32 SSP;   // scan speed		(H/S)
    int32 SFD;   // scan front dummy	(O/F)
    char SHA[3]; // scan h-area		(F/K/M/HC/HL/HR/QC/QL/QR/EC)
    int32 SMD;   // scan mode		(N/S/A)
    int32 SPX;   // super pixel		(1/2/4/8)
    int32 SHO;   // scan h-offset	(0 - 1272 step8)
    int32 SHW;   // scan h-width		(8 - 1280 step8)
    int32 SVO;   // scan v-offset	(0 - 1016 step8)
    int32 SVW;   // scan v-width		(8 - 1024 step8)

    int32 CEG; // contrast enhance gain	(SSP:S 0 - 2)	(SSP:H 0 - 255)
    int32 CEO; // contrast enhance offset	(SSP:S not use)	(SSP:H 0 - 255)

    int32 CAI_I; // high speed scan mode a/d bits
    int32 CAI_S; // slow speed scan mode a/d bits
    double TMP;  // temperature
                 // no CSW, TST command [2002/07/08]

    int32 OMD; // output mode		(P/F)
};

enum
{
    dcamparam_c4742_98_AMD = 0x00000001,
    dcamparam_c4742_98_EMD = 0x00000002,
    dcamparam_c4742_98_AET = 0x00000004,
    dcamparam_c4742_98_ATP = 0x00000008,

    dcamparam_c4742_98_SSP = 0x00000010,
    dcamparam_c4742_98_SFD = 0x00000020,
    dcamparam_c4742_98_SHA = 0x00000040,
    dcamparam_c4742_98_SMD = 0x00000080,
    dcamparam_c4742_98_SPX = 0x00000100,
    dcamparam_c4742_98_SHO = 0x00000200,
    dcamparam_c4742_98_SHW = 0x00000400,
    dcamparam_c4742_98_SVO = 0x00000800,
    dcamparam_c4742_98_SVW = 0x00001000,

    dcamparam_c4742_98_CEG = 0x00010000,
    dcamparam_c4742_98_CEO = 0x00020000,

    dcamparam_c4742_98_CAI_I = 0x00100000,
    dcamparam_c4742_98_CAI_S = 0x00200000,
    dcamparam_c4742_98_TMP = 0x00400000,

    dcamparam_c4742_98_OMD = 0x00800000 // ORCA2-HR
};

/*
enum {
        c4742_98_kSHO_unit		= 8,
        c4742_98_kSHO_min		= 0,
        c4742_98_kSHO_max		= 1272,

        c4742_98_kSHW_unit		= 8,
        c4742_98_kSHW_min		= 8,
        c4742_98_kSHW_max		= 1280,

        c4742_98_kSVO_unit		= 8,
        c4742_98_kSVO_min		= 0,
        c4742_98_kSVO_max		= 1016,

        c4742_98_kSVW_unit		= 8,
        c4742_98_kSVW_min		= 8,
        c4742_98_kSVW_max		= 1024,
};
*/
