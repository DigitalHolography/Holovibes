// paramC7390.h
#ifndef	_INCLUDE_PARAMC7390_H_
#define	_INCLUDE_PARAMC7390_H_

//
//	constants
//
enum
{
	C7390_kPixelCount_128		= 1,
	C7390_kPixelCount_256,
	C7390_kPixelCount_384,
	C7390_kPixelCount_512,
	C7390_kPixelCount_640,
	C7390_kPixelCount_768,
	C7390_kPixelCount_896,
	C7390_kPixelCount_1024,
	C7390_kPixelCount_1152,
	C7390_kPixelCount_1280,
	C7390_kPixelCount_1408,
	C7390_kPixelCount_1536,
	C7390_kPixelCount_1664,
	C7390_kPixelCount_1792,
	C7390_kPixelCount_1920,
	C7390_kPixelCount_2048,
	C7390_kPixelCount_2176,
	C7390_kPixelCount_2304,
	C7390_kPixelCount_2432,
	C7390_kPixelCount_2560,

	C7390_kPixelSize_04			= 1,
	C7390_kPixelSize_08,
	C7390_kPixelSize_16,

	C7390_kOutClock_500			= 1,
	C7390_kOutClock_250,
	C7390_kOutClock_125,
	C7390_kOutClock_063,

	C7390_kLSMode_Internal		= 0,
	C7390_kLSMode_External,

	C7390_kLSMode_Min			= C7390_kLSMode_Internal,
	C7390_kLSMode_Max			= C7390_kLSMode_External,
	C7390_kLSMode_Def			= C7390_kLSMode_Internal,

	C7390_kBSDark_Min			= 32,
	C7390_kBSDark_Max			= 2047,
	C7390_kBSDark_Def			= 255,

	C7390_kBSBright_Min			= 512,
	C7390_kBSBright_Max			= 4095,
	C7390_kBSBright_Def			= 2048,

	C7390_kJSDatCount_Min		= 1,
	C7390_kJSDatCount_Max		= 63,
	C7390_kJSDatCount_Def		= 10,

	C7390_kJSRate_Min			= 0,
	C7390_kJSRate_Max			= 200,
	C7390_kJSRate_Def			= 50,

	C7390_kJSAveCount_Min		= 0,
	C7390_kJSAveCount_Max		= 8,
	C7390_kJSAveCount_Def		= 5,

	C7390_kESCount_Min			= 1,
	C7390_kESCount_Max			= 4,
	C7390_kESCount_Def			= C7390_kESCount_Min,

	C7390_kESLeft_Min			= 0,
	C7390_kESLeft_Max			= 2047,
	C7390_kESLeft_Def			= C7390_kESLeft_Min,

	C7390_kESRight_Min			= 0,
	C7390_kESRight_Max			= 2047,
	C7390_kESRight_Def			= C7390_kESRight_Max,

	C7390_kCMMode_None			= 0,
	C7390_kCMMode_MaxData,
	C7390_kCMMode_AveData,
	C7390_kCMMode_SetData,

	C7390_kCMMode_Min			= C7390_kCMMode_None,
	C7390_kCMMode_Max			= C7390_kCMMode_SetData,
	C7390_kCMMode_Def			= C7390_kCMMode_MaxData,

	C7390_kCMData_Min			= 1024,
	C7390_kCMData_Max			= 4095,
	C7390_kCMData_Def			= 4095,

	C7390_kBMMode_AveData		= 0,
	C7390_kBMMode_MaxData,
	C7390_kBMMode_MinData,

	C7390_kBMMode_Min			= C7390_kBMMode_AveData,
	C7390_kBMMode_Max			= C7390_kBMMode_MinData,
	C7390_kBMMode_Def			= C7390_kBMMode_AveData,

	// action
	C7390_kAct_Initialize		= 0,
	C7390_kAct_Start,
	C7390_kAct_Stop,
	C7390_kAct_Dark,
	C7390_kAct_Bright,
	C7390_kAct_Offset,
	C7390_kAct_Cancel,

	// status
	C7390_kSR_ACQ_ACK			= 0,
	C7390_kSR_ACQ_NAK,
	C7390_kSR_IDLE,
	C7390_kSR_LS_ERR,
	C7390_kSR_DARK_WAIT,
	C7390_kSR_DARK_ACQ,
	C7390_kSR_BRIGHT_WAIT,
	C7390_kSR_BRIGHT_ACQ,
	C7390_kSR_ADOFF_CORR,
	C7390_kSR_SATURATE,
	C7390_kSR_MACHINE_ERR,
	C7390_kSR_UNKNOWN_MASK		= 0xffff0000,

	C7390_kVO_Positive			= 0,
	C7390_kVO_Negative,

	C7390_kTO_Off				= 0,
	C7390_kTO_On,

	C7390_kADNo_Min				= 0,
	C7390_kADNo_Max				= 19,
	C7390_kADData_Min			= 0,
	C7390_kADData_Max			= 255

};

#define	C7390_kLSData_Min		((double)   1.0)
#define	C7390_kLSData_Max		((double) 160.0)
#define	C7390_kLSData_Def		((double)  10.0)
#define	C7390_kLSData_Step		((double)   0.1)

enum {
	dcamparam_c7390_pixelCount		= 0x00000010,
	dcamparam_c7390_pixelSize		= 0x00000020,
	dcamparam_c7390_outClock		= 0x00000040,

	dcamparam_c7390_ls				= 0x00000200,
	dcamparam_c7390_bs				= 0x00000400,
	dcamparam_c7390_js				= 0x00000800,
	dcamparam_c7390_es				= 0x00001000,
	dcamparam_c7390_cm				= 0x00002000,
	dcamparam_c7390_bm				= 0x00004000,

	dcamparam_c7390_vo				= 0x00010000,
	dcamparam_c7390_to				= 0x00020000,

	dcamparam_c7390_st				= 0x01000000,
	dcamparam_c7390_sp				= 0x02000000,
	dcamparam_c7390_da				= 0x04000000,
	dcamparam_c7390_ba				= 0x08000000,

	dcamparam_c7390_sr				= 0x10000000,

	dcamparam_c7390_scrollLine		= 0x80000000
};

struct DCAM_PARAM_C7390 {
	DCAM_HDR_PARAM	hdr;		// id == DCAM_IDPARAM_C7390

	int		pixelCount,
			pixelSize,
			outClock;

	int		lsMode;
	double	lsData;
	int		bsDark,
			bsBright,
			jsDatCount,
			jsRate,
			jsAveCount,
			esCount,
			esLeft [C7390_kESCount_Max],
			esRight [C7390_kESCount_Max],
			cmMode,
			cmData,
		 	bmMode,
			voMode,
			toMode;

	int		sr;
	int		scrollLine;
	int		reserve;
};

#endif // _INCLUDE_PARAMC7390_H_

