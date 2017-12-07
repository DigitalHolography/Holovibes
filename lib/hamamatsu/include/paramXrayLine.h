// paramXrayLine.h

#ifndef	_INCLUDE_PARAMXRAYLINE_H_
#define	_INCLUDE_PARAMXRAYLINE_H_

enum {
	dcamparam_xrayline_pixelCount		= 0x00000010,
	dcamparam_xrayline_pixelSize		= 0x00000020,
	dcamparam_xrayline_outClock			= 0x00000040,

	dcamparam_xrayline_ls				= 0x00000200,
	dcamparam_xrayline_bs				= 0x00000400,
	dcamparam_xrayline_js				= 0x00000800,
	dcamparam_xrayline_es				= 0x00001000,
	dcamparam_xrayline_cm				= 0x00002000,
	dcamparam_xrayline_bm				= 0x00004000,

	dcamparam_xrayline_vo				= 0x00010000,
	dcamparam_xrayline_to				= 0x00020000,

	dcamparam_xrayline_st				= 0x01000000,
	dcamparam_xrayline_sp				= 0x02000000,
	dcamparam_xrayline_da				= 0x04000000,
	dcamparam_xrayline_ba				= 0x08000000,
	dcamparam_xrayline_db				= 0x10000000,
	dcamparam_xrayline_ai				= 0x20000000,

	dcamparam_xrayline_sr				= 0x40000000,

	dcamparam_xrayline_scrollLine		= 0x80000000
};

#define kESCount_Max 4

struct DCAM_PARAM_XRAYLINE {
	DCAM_HDR_PARAM	hdr;		// id == DCAM_IDPARAM_XRAYLINE

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
			esLeft [kESCount_Max],
			esRight [kESCount_Max],
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

//
//	constants
//
enum
{
	kXrayLine_CameraBit_08	= 0x0008,
	kXrayLine_CameraBit_10	= 0x000A,
	kXrayLine_CameraBit_12	= 0x000C,
	kXrayLine_CameraBit_14	= 0x000E,
	kXrayLine_CameraBit_16	= 0x0010
};

enum
{
	kXrayLine_PixelCount_128		= 1,
	kXrayLine_PixelCount_256,
	kXrayLine_PixelCount_384,
	kXrayLine_PixelCount_512,
	kXrayLine_PixelCount_640,
	kXrayLine_PixelCount_768,
	kXrayLine_PixelCount_896,
	kXrayLine_PixelCount_1024,
	kXrayLine_PixelCount_1152,
	kXrayLine_PixelCount_1280,
	kXrayLine_PixelCount_1408,
	kXrayLine_PixelCount_1536,
	kXrayLine_PixelCount_1664,
	kXrayLine_PixelCount_1792,
	kXrayLine_PixelCount_1920,
	kXrayLine_PixelCount_2048,
	kXrayLine_PixelCount_2176,
	kXrayLine_PixelCount_2304,
	kXrayLine_PixelCount_2432,
	kXrayLine_PixelCount_2560,
	kXrayLine_PixelCount_2688,
	kXrayLine_PixelCount_2816,
	kXrayLine_PixelCount_2944,
	kXrayLine_PixelCount_3072,
	kXrayLine_PixelCount_3200,
	kXrayLine_PixelCount_3328,
	kXrayLine_PixelCount_3456,
	kXrayLine_PixelCount_3584,
	kXrayLine_PixelCount_3712,
	kXrayLine_PixelCount_3840,
	kXrayLine_PixelCount_3968,
	kXrayLine_PixelCount_4096,

	kXrayLine_PixelSize_04			= 1,
	kXrayLine_PixelSize_08,
	kXrayLine_PixelSize_16,
	kXrayLine_PixelSize_02,
	kXrayLine_PixelSize_0048,

	kXrayLine_OutClock_500			= 1,
	kXrayLine_OutClock_533			= 1,
	kXrayLine_OutClock_250			= 2,
	kXrayLine_OutClock_267			= 2,
	kXrayLine_OutClock_125			= 3,
	kXrayLine_OutClock_133			= 3,
	kXrayLine_OutClock_063			= 4,
	kXrayLine_OutClock_067			= 4,
	kXrayLine_OutClock_667,
	kXrayLine_OutClock_333,
	kXrayLine_OutClock_167,
	kXrayLine_OutClock_083,
	kXrayLine_OutClock_800,
	kXrayLine_OutClock_400,
	kXrayLine_OutClock_200,
	kXrayLine_OutClock_100,
	kXrayLine_OutClock_1000,
	kXrayLine_OutClock_2000,

	kXrayLine_LSMode_Internal		= 0,
	kXrayLine_LSMode_External,

	kXrayLine_LSMode_Min			= kXrayLine_LSMode_Internal,
	kXrayLine_LSMode_Max			= kXrayLine_LSMode_External,
	kXrayLine_LSMode_Def			= kXrayLine_LSMode_Internal,

	kXrayLine_TriggerMode_Internal		= 0,
	kXrayLine_TriggerMode_External_Edge,
	kXrayLine_TriggerMode_External_Pulse,

	kXrayLine_TriggerMode_Min			= kXrayLine_TriggerMode_Internal,
	kXrayLine_TriggerMode_Max			= kXrayLine_TriggerMode_External_Pulse,
	kXrayLine_TriggerMode_Def			= kXrayLine_TriggerMode_Internal,

	kXrayLine_BSDark_Min			= 2,	// 8bit:2 10bit:8 12bit:32
	kXrayLine_BSDark_Max			= 4095,	// 8bit:255 10bit:1023 12bit:4095
	kXrayLine_BSDark_Def			= 63,	//31 63  511

	//{
	kXrayLine_Bit8_BSDark_Min			= 2,
	kXrayLine_Bit8_BSDark_Max			= 255,
	kXrayLine_Bit8_BSDark_Def			= 31,
	
	kXrayLine_Bit10_BSDark_Min			= 8,
	kXrayLine_Bit10_BSDark_Max			= 1023,
	kXrayLine_Bit10_BSDark_Def			= 63,

	kXrayLine_Bit12_BSDark_Min			= 32,
	kXrayLine_Bit12_BSDark_Max			= 4095,
	kXrayLine_Bit12_BSDark_Def			= 511,
	//	}

	kXrayLine_BSBright_Min			= 16,	//16 64 256
	kXrayLine_BSBright_Max			= 4095,	//255 1023 4095
	kXrayLine_BSBright_Def			= 255,	//32 64 512

	//{
	kXrayLine_Bit8_BSBright_Min		= 16,
	kXrayLine_Bit8_BSBright_Max		= 255,
	kXrayLine_Bit8_BSBright_Def		= 32,

	kXrayLine_Bit10_BSBright_Min		= 64,
	kXrayLine_Bit10_BSBright_Max		= 1023,
	kXrayLine_Bit10_BSBright_Def		= 64,

	kXrayLine_Bit12_BSBright_Min		= 256,
	kXrayLine_Bit12_BSBright_Max		= 4095,
	kXrayLine_Bit12_BSBright_Def		= 512,
	//}

	kXrayLine_JSDatCount_Min		= 0,
	kXrayLine_JSDatCount_Max		= 100,
	kXrayLine_JSDatCount_Def		= 10,

	kXrayLine_JSRate_Min			= 0,
	kXrayLine_JSRate_Max			= 200,
	kXrayLine_JSRate_Def			= 100,

	kXrayLine_JSAveCount_Min		= 0,
	kXrayLine_JSAveCount_Max		= 10,
	kXrayLine_JSAveCount_Def		= 5,

	kXrayLine_ESCount_Min			= 1,
	kXrayLine_ESCount_Max			= 4,
	kXrayLine_ESCount_Def			= kXrayLine_ESCount_Min,

	kXrayLine_ESLeft_Min			= 0,
	kXrayLine_ESLeft_Max			= ((kXrayLine_PixelCount_4096*128)-1),
	kXrayLine_ESLeft_Def			= kXrayLine_ESLeft_Min,

	kXrayLine_ESRight_Min			= 0,
	kXrayLine_ESRight_Max			= ((kXrayLine_PixelCount_4096*128)-1),
	kXrayLine_ESRight_Def			= kXrayLine_ESRight_Max,

	kXrayLine_CMMode_None			= 0,
	kXrayLine_CMMode_MaxData,
	kXrayLine_CMMode_AveData,
	kXrayLine_CMMode_SetData,

	kXrayLine_CMMode_Min			= kXrayLine_CMMode_None,
	kXrayLine_CMMode_Max			= kXrayLine_CMMode_SetData,
	kXrayLine_CMMode_Def			= kXrayLine_CMMode_MaxData,

	kXrayLine_CMData_Min			= 64,	//64 256 1024
	kXrayLine_CMData_Max			= 1023,	//255 1023 4095
	kXrayLine_CMData_Def			= kXrayLine_CMData_Min,

	//{
	kXrayLine_Bit8_CMData_Min			= 64,
	kXrayLine_Bit8_CMData_Max			= 255,
	kXrayLine_Bit8_CMData_Def			= kXrayLine_Bit8_CMData_Min,

	kXrayLine_Bit10_CMData_Min			= 256,
	kXrayLine_Bit10_CMData_Max			= 1023,
	kXrayLine_Bit10_CMData_Def			= kXrayLine_Bit10_CMData_Min,

	kXrayLine_Bit12_CMData_Min			= 1024,
	kXrayLine_Bit12_CMData_Max			= 4095,
	kXrayLine_Bit12_CMData_Def			= kXrayLine_Bit12_CMData_Min,

	kXrayLine_Bit14_CMData_Min			= 4096,
	kXrayLine_Bit14_CMData_Max			= 16383,
	kXrayLine_Bit14_CMData_Def			= kXrayLine_Bit14_CMData_Min,

	kXrayLine_Bit16_CMData_Min			= 16384,
	kXrayLine_Bit16_CMData_Max			= 65535,
	kXrayLine_Bit16_CMData_Def			= kXrayLine_Bit16_CMData_Min,
	//}

	kXrayLine_BMMode_AveData		= 0,
	kXrayLine_BMMode_MaxData,
	kXrayLine_BMMode_MinData,

	kXrayLine_BMMode_Min			= kXrayLine_BMMode_AveData,
	kXrayLine_BMMode_Max			= kXrayLine_BMMode_MinData,
	kXrayLine_BMMode_Def			= kXrayLine_BMMode_AveData,

	kXrayLine_TPPattern_Horizontal	= 0,
	kXrayLine_TPPattern_Vertical	= 1,
	kXrayLine_TPPattern_Line		= 2,
	kXrayLine_TPPattern_Brightness	= 3,
	kXrayLine_TPPattern_Diagonal	= 8,

	// action
	kXrayLine_Act_Initialize		= 0,
	kXrayLine_Act_Start,
	kXrayLine_Act_Stop,
	kXrayLine_Act_Dark,
	kXrayLine_Act_Bright,
	kXrayLine_Act_Offset,
	kXrayLine_Act_Cancel,

	// status
	kXrayLine_SR_ACQ_ACK			= 0,
	kXrayLine_SR_ACQ_NAK,
	kXrayLine_SR_IDLE,
	kXrayLine_SR_LS_ERR,
	kXrayLine_SR_DARK_WAIT,
	kXrayLine_SR_DARK_ACQ,
	kXrayLine_SR_BRIGHT_WAIT,
	kXrayLine_SR_BRIGHT_ACQ,
	kXrayLine_SR_ADOFF_CORR,
	kXrayLine_SR_SATURATE,
	kXrayLine_SR_MACHINE_ERR,
	kXrayLine_SR_UNKNOWN_MASK		= 0xffff0000,

	kXrayLine_VO_Positive			= 0,
	kXrayLine_VO_Negative,

	kXrayLine_TO_Off				= 0,
	kXrayLine_TO_On
};

#define	kXrayLine_LSData_Min		((double)   1.0)
#define	kXrayLine_LSData_Max		((double) 160.0)
#define	kXrayLine_LSData_Def		((double)  10.0)
#define	kXrayLine_LSData_Step		((double)   0.1)

#endif // _INCLUDE_PARAMXRAYLINE_H_
