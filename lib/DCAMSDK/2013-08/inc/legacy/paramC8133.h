// paramC8133.h

#ifndef	_INCLUDE_PARAMC8133_H_
#define	_INCLUDE_PARAMC8133_H_
//
//	constants
//
enum
{
	C8133_kPixelCount_128		= 1,
	C8133_kPixelCount_256,
	C8133_kPixelCount_384,
	C8133_kPixelCount_512,
	C8133_kPixelCount_640,
	C8133_kPixelCount_768,
	C8133_kPixelCount_896,
	C8133_kPixelCount_1024,
	C8133_kPixelCount_1152,
	C8133_kPixelCount_1280,
	C8133_kPixelCount_1408,
	C8133_kPixelCount_1536,
	C8133_kPixelCount_1664,
	C8133_kPixelCount_1792,
	C8133_kPixelCount_1920,
	C8133_kPixelCount_2048,
	C8133_kPixelCount_2176,
	C8133_kPixelCount_2304,
	C8133_kPixelCount_2432,
	C8133_kPixelCount_2560,

	C8133_kPixelSize_04			= 1,
	C8133_kPixelSize_08,
	C8133_kPixelSize_16,

	C8133_kOutClock_500			= 1,
	C8133_kOutClock_250,
	C8133_kOutClock_125,
	C8133_kOutClock_063,

	C8133_kLSMode_Internal		= 0,
	C8133_kLSMode_External,

	C8133_kLSMode_Min			= C8133_kLSMode_Internal,
	C8133_kLSMode_Max			= C8133_kLSMode_External,
	C8133_kLSMode_Def			= C8133_kLSMode_Internal,

	//{ Select
	C8133_kBSDark_Min			= 16,		// 8
	C8133_kBSDark_Max			= 1023,
	C8133_kBSDark_Def			= 63,

	C8133_kBSBright_Min			= 64,
	C8133_kBSBright_Max			= 1023,
	C8133_kBSBright_Def			= 255,		// 64
		//
	C8133_82_kBSDark_Min			= 8,
	C8133_82_kBSDark_Max			= 1023,
	C8133_82_kBSDark_Def			= 63,

	C8133_82_kBSBright_Min			= 64,
	C8133_82_kBSBright_Max			= 1023,
	C8133_82_kBSBright_Def			= 64,
	//}

	//{
	C8133_kJSDatCount_Min		= 0,
	C8133_kJSDatCount_Max		= 64,		//100
	C8133_kJSDatCount_Def		= 10,

	C8133_kJSRate_Min			= 0,
	C8133_kJSRate_Max			= 200,
	C8133_kJSRate_Def			= 50,		// 100

	C8133_kJSAveCount_Min		= 0,
	C8133_kJSAveCount_Max		= 8,		//10
	C8133_kJSAveCount_Def		= 5,
		//
	C8133_82_kJSDatCount_Min		= 0,
	C8133_82_kJSDatCount_Max		= 100,
	C8133_82_kJSDatCount_Def		= 10,

	C8133_82_kJSRate_Min			= 0,
	C8133_82_kJSRate_Max			= 200,
	C8133_82_kJSRate_Def			= 100,

	C8133_82_kJSAveCount_Min		= 0,
	C8133_82_kJSAveCount_Max		= 10,
	C8133_82_kJSAveCount_Def		= 5,
	//}

	C8133_kESCount_Min			= 1,
	C8133_kESCount_Max			= 4,
	C8133_kESCount_Def			= C8133_kESCount_Min,

	C8133_kESLeft_Min			= 0,
	C8133_kESLeft_Max			= 4095,
	C8133_kESLeft_Def			= C8133_kESLeft_Min,

	C8133_kESRight_Min			= 0,
	C8133_kESRight_Max			= 4095,
	C8133_kESRight_Def			= C8133_kESRight_Max,

	C8133_kCMMode_None			= 0,
	C8133_kCMMode_MaxData,
	C8133_kCMMode_AveData,
	C8133_kCMMode_SetData,

	C8133_kCMMode_Min			= C8133_kCMMode_None,
	C8133_kCMMode_Max			= C8133_kCMMode_SetData,
	C8133_kCMMode_Def			= C8133_kCMMode_MaxData,

	//{
	C8133_kCMData_Min			= 64,		//256
	C8133_kCMData_Max			= 1023,
	C8133_kCMData_Def			= 64,
		//
	C8133_82_kCMData_Min			= 256,
	C8133_82_kCMData_Max			= 1023,
	C8133_82_kCMData_Def			= 64,
	//}

	C8133_kBMMode_AveData		= 0,
	C8133_kBMMode_MaxData,
	C8133_kBMMode_MinData,

	C8133_kBMMode_Min			= C8133_kBMMode_AveData,
	C8133_kBMMode_Max			= C8133_kBMMode_MinData,
	C8133_kBMMode_Def			= C8133_kBMMode_AveData,

	// action
	C8133_kAct_Initialize		= 0,
	C8133_kAct_Start,
	C8133_kAct_Stop,
	C8133_kAct_Dark,
	C8133_kAct_Bright,
	C8133_kAct_Offset,
	C8133_kAct_Cancel,

	// status
	C8133_kSR_ACQ_ACK			= 0,
	C8133_kSR_ACQ_NAK,
	C8133_kSR_IDLE,
	C8133_kSR_LS_ERR,
	C8133_kSR_DARK_WAIT,
	C8133_kSR_DARK_ACQ,
	C8133_kSR_BRIGHT_WAIT,
	C8133_kSR_BRIGHT_ACQ,
	C8133_kSR_ADOFF_CORR,
	C8133_kSR_SATURATE,
	C8133_kSR_MACHINE_ERR,
	C8133_kSR_UNKNOWN_MASK		= 0xffff0000,

	C8133_kVO_Positive			= 0,
	C8133_kVO_Negative,

	C8133_kTO_Off				= 0,
	C8133_kTO_On
};

#define	C8133_kLSData_Min		((double)   1.0)
#define	C8133_kLSData_Max		((double) 160.0)
#define	C8133_kLSData_Def		((double)  10.0)
#define	C8133_kLSData_Step		((double)   0.1)

enum {
	dcamparam_c8133_pixelCount		= 0x00000010,
	dcamparam_c8133_pixelSize		= 0x00000020,
	dcamparam_c8133_outClock		= 0x00000040,

	dcamparam_c8133_ls				= 0x00000200,
	dcamparam_c8133_bs				= 0x00000400,
	dcamparam_c8133_js				= 0x00000800,
	dcamparam_c8133_es				= 0x00001000,
	dcamparam_c8133_cm				= 0x00002000,
	dcamparam_c8133_bm				= 0x00004000,

	dcamparam_c8133_vo				= 0x00010000,
	dcamparam_c8133_to				= 0x00020000,

	dcamparam_c8133_st				= 0x01000000,
	dcamparam_c8133_sp				= 0x02000000,
	dcamparam_c8133_da				= 0x04000000,
	dcamparam_c8133_ba				= 0x08000000,
	dcamparam_c8133_db				= 0x10000000,
	dcamparam_c8133_ai				= 0x20000000,

	dcamparam_c8133_sr				= 0x40000000,

	dcamparam_c8133_scrollLine		= 0x80000000
};

struct DCAM_PARAM_C8133 {
	DCAM_HDR_PARAM	hdr;		// id == DCAM_IDPARAM_C8133

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
			esLeft [C8133_kESCount_Max],
			esRight [C8133_kESCount_Max],
			cmMode,
			cmData,
		 	bmMode,
			voMode,
			toMode,
			aiTableNo;
	int		sr;
	int		scrollLine;
	int		reserve;
};

#endif // _INCLUDE_PARAMC8133_H_

