/**
 * @file
 * @brief File
 * @author SHT
 * @version 2.0
 * @image html firewire_buffering_small.jpg Functionality Diagram
 */

/*! \page page1 Reference page
  Contains referenced materials.
  \section add_m Additional material
  This page contains compact size \ref diagram\n
  For more info see page \ref fulldiagram.
  \subsection diagram Functionality Diagram
  Compact size functionality diagram. It displays the workflow chart and the relationship between the functions.\n
  It is the top view of camera's API.\n
  @image html firewire_buffering_small.jpg Functionality Diagram
  \sa \ref fulldiagram 
  
  \page fulldiagram Full Size Functionality Diagram
  \note Picture is too large for browser window\n 
  @image html firewire_buffering.jpg Functionality Diagram
  
  \sa \ref page1
*/
#ifndef __MM40API_H
#define __MM40API_H

#ifdef MM40API_EXPORTS
#define MM40_API __declspec(dllexport)
#else
#define MM40_API __declspec(dllimport)
#endif

// winbase.h has IN and OUT but not both
#ifndef INOUT
#define INOUT
#endif

#include "m3ErrorCodes.h"

#define	szMM40_Name		"SoftHard Technology MV4.0 Camera"
#define REG_PATH	"SoftHard\\mm40api" //Full path to chunk size variable in the registry "HKEY_LOCAL_MACHINE\SOFTWARE\REG_PATH".
#define CHUNK_SIZE_REG_VAR	"Chunk"				//Name of the chunk size variable in the registry, value should be from 0 to 31.
#define PPTB_REG_VAR	"PPTB"					//Number of packets in transport buffer, value should be from 5 to 200.
#define DBPC_REG_VAR	"DoBPC"					//Flag to do automaticaly bad pixel correction
#define DCCCD_REG_VAR	"DoCCCD"				//Flag to do automatically clear CCD for MR cameras in trigger mode.
#define DDN_REG_VAR		"DoDeNoise"				//Apply De-Noise filter.
#define EN_RES_REL_REG_VAR "EnResRel"			//Enable/disable resources release if driver crashed or buzzed.
#define USB20_PKT_SIZE		"usb20pktsize"		//Switches packet size of USB 2.0 bulk transfer 0-512, 1-1024 

#ifdef __cplusplus
extern "C" {
#endif

/**
 *\struct SETMODE
 * \brief structure containing information about the mode that can be set by a user
 */
typedef	struct {
#define SETMODE_STRUCT_VER	3
	WORD	wVersion; //!< Version of the structure
#define SETMODE_STRUCT_LEN sizeof(SETMODE)
	WORD	wLenght;	//!< Length (size) of the structure
	DWORD	dwExpUsec;	//!< Value of exposure set in microseconds
	DWORD	dwModeNum; //!< Mode's number
	DWORD	x0;		//!<X origin of Region Of Intereset (ROI)
	DWORD	y0;		//!< Y origin of ROI
	DWORD	cx;		//!< X extension of ROI
	DWORD	cy;		//!< Y extension of ROI
	DWORD	dwKEEP; //!< Keep coming modes in order (if set queues modes in order of coming and applies them in order, rewrites each time otherwise)
	DWORD	dwLTmode;	//!< mode
	DWORD	dwLTcnt;	//!< mode count
	WORD	wUPD;	//!< Removes modes sequence (and resets TG ???)

	DWORD	dwPackets_per_buffer; //!<Number of packets in transport buffer
/*
	DW	1312,	1456,	1600,	1744	; 0..3
	DW	1888,	2032,	2176,	2320	; 4..7
	DW	2464,	2608,	2752,	2896	; 8..11
	DW	3040,	3184,	3328,	3472	;12..15	
	DW	3616,	3760,	3904,	4048	;16..19
	DW	4192,	4336,	4480,	4624	;20..23
	DW	4768,	4912,	5056,	5200	;24..27
	DW	5344,	5488,	5632,	5776	;28..31
*/
	//From version 1	
	WORD	wChunkSize; //!<Packet size. Value from 0 to 31. 

	//From version 2
	WORD	wNoClean;	//!<Do not clean CCD.
	WORD	wNoData;	//!<No data. 
	//From version 3
	WORD	wSSSequence;	//!< TRUE-start to create sequence of modes, FALSE-stop(First mmSetModeEx call of sequence shuold be TRUE, last FALSE)
}SETMODE, *LPSETMODE;

#define SETMODE_STRUCT_LEN_V1 52
#define SETMODE_STRUCT_LEN_V2 56
#define SETMODE_STRUCT_LEN_V3 56

/**
 *\struct FIELDINFO
 * \brief structure containing information about a specific field
 */
typedef	struct {
#define FIELDINFO_STRUCT_VER	4
	WORD	wVersion; //!< Version of the structure
#define FIELDINFO_STRUCT_LEN sizeof(FIELDINFO)
	WORD	wLenght;	//!< Length (size) of the structure
	DWORD	dwFieldNum; 	//!< Field number (refer to diagram \ref diagram) inside the circular buffer
	//Version 1
	DWORD	dwModeNum; //!< Mode number 
	//Version 2
	WORD	modeX0;
	WORD	modeY0;
	WORD	modeCX;
	WORD	modeCY;
	//Version 3
	DWORD	dwExposure; //!< Exposure time usecs.
	float	fGain;		//!< Gain in dB.
	//Version 4
	BOOL	directly_updated; //!< Some camera parameters were updated directly without sycnronization and couldn't be true.
}FIELDINFO, *LPFIELDINFO;

#define FIELDINFO_STRUCT_LEN_V1 12
#define FIELDINFO_STRUCT_LEN_V2 20
#define FIELDINFO_STRUCT_LEN_V3 28
#define FIELDINFO_STRUCT_LEN_V4 32

#define	MM_GAIN_RANGE	1024	//!< Gain range
#define	MMSHADING_MUL1	0x4000	//!< Shading multiplication value
#define	MMSHADING_SUB0	0x0000	//!< Shading subtraction value

/** @name MM_TRIG
         Group of trigger related definitions
   */
   /** @{ */
#define	MM_TRIG_OFF				0x0		//!< No trigger
#define	MM_TRIG_IN_EDGE			0x1		//!< Input positive/negative edge starts exposure
#define	MM_TRIG_IN_LEVEL		0x2		//!< Input positive/negative level defines exposure
#define	MM_TRIG_OUT				0x3		//!< Output is high when exposure is active
#define	MM_TRIG_SOFTW			0x4		//!< Software ping triggers exposure
#define	MM_TRIG_MAX				0x5		//!< Maximum mode
#define	MM_TRIG_IN_EDGE_INV		0x10	//!< Input positive/negative edge starts exposure, inverted output (not supported, see mmSetGPIO)
/** @} */

/**
 *\struct SETTRIGGER
 * \brief structure containing information about black level
 */
typedef struct {
#define SETSETTRIGGER_STRUCT_VER	0
	WORD	wVersion;	//!< Version of the structure
#define SETTRIGGER_STRUCT_LEN sizeof(SETTRIGGER)
	WORD	wLenght;	//!< Length (size) of the structure
	DWORD	dwMode;		//!< Trigger Mode
	WORD	wNegative;	//!< edge type (leading/trailing a.k.a positive/negative)
	WORD	wDoSetMode; //!< 0-don't call mmSetMode function.
} SETTRIGGER, * LPSETTRIGGER;

#define	MM_BPL_MAXCNT	8		//!< Maximum number of pixels used for correction

/**
 *\struct BADPIXEL
 * \brief structure containing information about a bad pixel.
 */
typedef struct {
	DWORD	dwFlags;			//!< Bit mask of bad pixel
#define MM_BPL_END			0x01	//!< End of the list - no further correction needed
#define MM_BPL_REQ			0x02	//!< This pixel requires correction
#define MM_BPL_COLUMN		0x04	//!< Column defect
#define MM_BPL_ROW			0x08	//!< Row defect
#define MM_BPL_CLUSTER		0x10	//!< Cluster defect
#define MM_BPL_DARKPOINT	0x20	//!< Dark point defect
#define MM_BPL_HIGHDPIX		0x40	//!< HighDPIX point defect
#define MM_BPL_BRIGHTPOINT	0x80	//!< Bright point defect

	DWORD	dwX; 			//!< x location of a bad pixel
	DWORD	dwY;			//!< y location of a bad pixel
	DWORD	dwCnt;				//!< Number of pixels used for correction
	int		ox[MM_BPL_MAXCNT];	//!< X Offset to correction pixel
	int		oy[MM_BPL_MAXCNT];	//!< Y Offset to correction pixel
} BADPIXEL, *LPBADPIXEL;


/**
 *\struct BADPIXELSTATISTICS
 * \brief structure containing information about bad pixels on camera.
 */
typedef struct {
	DWORD	dwPoints;		//!< Bad pixels count
	DWORD	dwColumns;		//!< Bad columns count
	DWORD	dwRows;			//!< Bad rows count
	DWORD	dwClusters;		//!< Clusters count
	DWORD	dwPixClusters;	//!< Bad pixels in clusters count
	DWORD	dwTotal;		//!< Total bad pixels count
	char	cCCD[256];		//!< Sensor type
	char	cCCDSN[256];	//!< Sensor/device serial number
	float	fVAB;			//!< Anti Blooming Voltage
	BOOL	bColor;			//!< Color sensor
	LPBADPIXEL lpBP;		//!< Pointer to bad pixel structures array
} BADPIXELSTATISTICS, *LPBADPIXELSTATISTICS;

// HWN2 structure:
// |cccc cccc cccc mmmm 0000 000b rrrr aaaa|
//  chip---------- mos-         b rel- adc-|
//  chip (MM40_HWN2_CHIP)
//                 mos- (MM40_HWN2_MOSAIC)
//                              b (MM40_HWN2_BRDLEV)
//                                rel- (MM40_HWN2_HW_REL)
//                                     adc- (MM40_HWN2_ADC)

#define MM40_HWN2_SHIFT_CHIP   20   // 12 bits: position of chip id in HWN2 number
#define MM40_HWN2_SHIFT_MOSAIC 16   //  4 bits: position of mosaic id in HWN2 number
//#define MM40_HWN2_SHIFT_FREE  9-15//  7 bits: (free)
#define MM40_HWN2_SHIFT_BRDLEV  8   //  1 bit:  1=this camera is board level, 0=this camera is product with case
#define MM40_HWN2_SHIFT_RELEASE 4   //  4 bits: position of release id in HWN2 number
#define MM40_HWN2_SHIFT_ADC     0   //  4 bits: position of adc id in HWN2 number

#define MM40_HWN2_CHIP(dw)		(((dw)>>MM40_HWN2_SHIFT_CHIP) & 0xFFF)
#define	MM40_HWN2_ICX282		0x282
#define	MM40_HWN2_ICX252		0x252
#define	MM40_HWN2_ICX285		0x285
#define	MM40_HWN2_ICX456		0x456
#define	MM40_HWN2_ICX274		0x274
#define	MM40_HWN2_ICX413		0x413
#define	MM40_HWN2_ICX655        0x655
#define	MM40_HWN2_ICX674        0x674
#define	MM40_HWN2_ICX694        0x694
#define	MM40_HWN2_ICX814        0x814
#define	MM40_HWN2_ICX834        0x834
#define	MM40_HWN2_TC285			0xC85
#define	MM40_HWN2_IBIS4_6M		0xC66
#define	MM40_HWN2_KAI4021M		0xA41
#define	MM40_HWN2_KAI11002M		0xA12
#define	MM40_HWN2_KAI16000M		0xA16
#define	MM40_HWN2_MT9P031		0xB50       // 5MPix Aptina
#define MM40_HWN2_MT9V034		0xB05       // WVGA Aptina
#define MM40_HWN2_EV76C560		0xE13       // E2V 1.3MPix
#define MM40_HWN2_EV76C570		0xE20       // E2V 2.0MPix
#define	MM40_HWN2_MU9P031		0x931       // USB 5MPix Aptina
#define MM40_HWN2_VITA1300      0x103       // Onsemi VITA 1.3Mpix
#define MM40_HWN2_CMV300        0xC03       // CMOSIS 0.3Mpix
#define MM40_HWN2_CMV2000       0xC20       // CMOSIS 2.0Mpix
#define MM40_HWN2_CMV2000_LX16  0x620       // CMOSIS 2.0Mpix with XILINX xc6slx16 fpga
#define MM40_HWN2_CMV4000       0xC40       // CMOSIS 4.0Mpix
#define MM40_HWN2_CMV4000C_TG   0x240       // CMOSIS 4.0Mpix rev3 NoGlass
#define MM40_HWN2_CMV4000_LX16  0x640       // CMOSIS 4.0Mpix with XILINX xc6slx16 fpga
#define MM40_HWN2_XLU3			0x123		// xiLinkU3	

#define MM40_HWN2_MOSAIC(dw)	(((dw)>>MM40_HWN2_SHIFT_MOSAIC) & 0xF)
#define	MM40_HWN2_MOSAIC_BW		0x0
#define	MM40_HWN2_MOSAIC_RGB	0x1
#define	MM40_HWN2_MOSAIC_CMYG	0x2
#define MM40_HWN2_MOSAIC_LINES_MULTISPECTRAL  0x3  // sensor lines has different spectral properties (e.g. 256 different properties)
#define MM40_HWN2_MOSAIC_TILED_MULTISPECTRAL  0x4  // sensor parts has different spectral properties (e.g. 32 different properties)
#define MM40_HWN2_MOSAIC_MULTISPECTRAL        0x5  // special mosaic with different spectral properties (e.g. 8 different properties)
#define MM40_HWN2_MOSAIC_UNKNOWN              0x6  // IMEC boardlevel camera with onknown mosaic to be used on sensor

#define MM40_HWN2_BRDLEV(dw)    (((dw)>>MM40_HWN2_SHIFT_BRDLEV) & 0x1)

#define	MM40_HWN2_HW_REL(dw)	(((dw)>>MM40_HWN2_SHIFT_RELEASE) & 0xF)

#define MM40_HWN2_ADC(dw)		(((dw)>>MM40_HWN2_SHIFT_ADC) & 0xF)
#define MM40_HWN2_ADC_9224		0x1
#define MM40_HWN2_ADC_9844		0x2
#define MM40_HWN2_ADC_9845B		0x3
#define MM40_HWN2_ADC_9824		0x4
#define MM40_HWN2_ADC_9949A		0x5
#define MM40_HWN2_ADC_2247		0x6
#define MM40_HWN2_ADC_9942		0x7
#define MM40_HWN2_ADC_9970		0x8
#define MM40_HWN2_ADC_CMOS		0xF



#define	MM40_MODEXT_BWHBIN	0x00	//!< B/W mode of multicolor shift
#define	MM40_MODEXT_RGGB	0x01	//!< Color mode RG/GB mosaic filter (ICX252AK,etc)
#define	MM40_MODEXT_CMYG	0x02	//!< Color mode CM/YG mosaic filter (ICX252AQ,etc)
#define	MM40_MODEXT_RGR		0x03	//!< Color mode RG/GB mosaic filter mixed readout (ICX282AQ,etc)
#define	MM40_MODEXT_MAX		4
#define	MM40_MODEXT_MASK	0x03

#define	MM40_MODEXT_SHOOT	0x04	//!< Mode supports shooting



#define	MM40_TRIG_OFF		0		//!< No trigger
#define	MM40_TRIG_IN_EDGE	1		//!< Input positive/negative edge starts exposure
#define	MM40_TRIG_IN_LEVEL	2		//!< Input positive/negative level defines exposure
#define	MM40_TRIG_OUT		3		//!< Output is high when exposure is active
#define	MM40_TRIG_MAX		4		//!< Maximum mode


/**************************************************************************/
/* Support for MM cameras *************************************************/
/**************************************************************************/
/**
 *\struct DARKSIDE
 * \brief structure containing information about black level
 */
typedef struct {
	double	blackDummy;				//!< Black level from dummy pixels
	double	blackDark;				//!< Black level from dark pixels
	double	blackLeftUp;			//!< Black level from black pixels on top left
	double	blackLeftMid;			//!< Black level from black pixels on mid left
	double	blackLeftDown;			//!< Black level from black pixels on down left
	double	blackRight;				//!< Black level from black pixels on the right

	double	noiseShift;				//!< Noise from dummy (shift reg and output)
	double	noiseCCD;				//!< Noise from all black pixels

	double	black[3][4];			//!< 2D array of black levels
} DARKSIDE, * LPDARKSIDE;


/**
 *\enum MMMOSAIC
 * \brief structure containing information about color matrix
 */
typedef enum {
	MMM_NONE = 0x000,		//!< B/W sensors
	MMM_RGGB = 0x001,		//!< Regular RGGB readout
	MMM_CMYG = 0x002,		//!< AK Sony sensors
	MMM_RGR  = 0x003,		//!< 2R+G readout of RGGB sensor
	MMM_BGGR = 0x004,		//!< BGGR readout
	MMM_GRBG = 0x005,		//!< GRBG readout
	MMM_GBRG = 0x006		//!< GBRG readout
	// etc
} MMMOSAIC;

typedef enum{
	IMAGE_DATA_FORMAT_BGRA8				=	0,
	IMAGE_DATA_FORMAT_RGBA8				=	1,
	IMAGE_DATA_FORMAT_BGR8				=	2,
	IMAGE_DATA_FORMAT_RGB8				=	3,
	IMAGE_DATA_FORMAT_BGR8_PLANAR		=	4,
	IMAGE_DATA_FORMAT_RGB8_PLANAR		=	5,
	IMAGE_DATA_FORMAT_MONO8				=	6, //packed
	IMAGE_DATA_FORMAT_MONO8U			=	7, //unpacked
	IMAGE_DATA_FORMAT_MONO10			=	8, //unpacked
	IMAGE_DATA_FORMAT_MONO12			=	9, //unpacked
	IMAGE_DATA_FORMAT_MONO14			=	10, //unpacked  
	IMAGE_DATA_FORMAT_MONO16			=	11, //packed //NS(not supported)
	IMAGE_DATA_FORMAT_BGRA10			=	12,//NS(not supported)
	IMAGE_DATA_FORMAT_RGBA10			=	13,//NS(not supported)
	IMAGE_DATA_FORMAT_BGR10				=	14,//NS(not supported)
	IMAGE_DATA_FORMAT_RGB10				=	15,//NS(not supported)
	IMAGE_DATA_FORMAT_BGR10_PLANAR		=	16,//NS(not supported)
	IMAGE_DATA_FORMAT_RGB10_PLANAR		=	17,//NS(not supported)
	IMAGE_DATA_FORMAT_BGRA12			=	18,//NS(not supported)
	IMAGE_DATA_FORMAT_RGBA12			=	19,//NS(not supported)
	IMAGE_DATA_FORMAT_BGR12				=	20,//NS(not supported)
	IMAGE_DATA_FORMAT_RGB12				=	21,//NS(not supported)
	IMAGE_DATA_FORMAT_BGR12_PLANAR		=	22,//NS(not supported)
	IMAGE_DATA_FORMAT_RGB12_PLANAR		=	23,//NS(not supported)
	IMAGE_DATA_FORMAT_BGRA14			=	24,//NS(not supported)
	IMAGE_DATA_FORMAT_RGBA14			=	25,//NS(not supported)
	IMAGE_DATA_FORMAT_BGR14				=	26,//NS(not supported)
	IMAGE_DATA_FORMAT_RGB14				=	27,//NS(not supported)
	IMAGE_DATA_FORMAT_BGR14_PLANAR		=	28,//NS(not supported)
	IMAGE_DATA_FORMAT_RGB14_PLANAR		=	29,//NS(not supported)
	IMAGE_DATA_FORMAT_BGRA16			=	30,//NS(not supported)
	IMAGE_DATA_FORMAT_RGBA16			=	31,//NS(not supported)
	IMAGE_DATA_FORMAT_BGR16				=	32,//NS(not supported)
	IMAGE_DATA_FORMAT_RGB16				=	33,//NS(not supported)
	IMAGE_DATA_FORMAT_BGR16_PLANAR		=	34,//NS(not supported)
	IMAGE_DATA_FORMAT_RGB16_PLANAR		=	35,//NS(not supported)
	IMAGE_DATA_FORMAT_BAYER_RG8			=	36, //packed
	IMAGE_DATA_FORMAT_BAYER_RG10		=	37,
	IMAGE_DATA_FORMAT_BAYER_RG12		=	38,
	IMAGE_DATA_FORMAT_BAYER_RG14		=	39,
	IMAGE_DATA_FORMAT_BAYER_RG16		=	40,//packed
	IMAGE_DATA_FORMAT_BAYER_BG8			=	41,//packed
	IMAGE_DATA_FORMAT_BAYER_BG10		=	42,
	IMAGE_DATA_FORMAT_BAYER_BG12		=	43,
	IMAGE_DATA_FORMAT_BAYER_BG14		=	44,
	IMAGE_DATA_FORMAT_BAYER_BG16		=	45,//packed
	IMAGE_DATA_FORMAT_BAYER_GR8			=	46,//packed
	IMAGE_DATA_FORMAT_BAYER_GR10		=	47,
	IMAGE_DATA_FORMAT_BAYER_GR12		=	48,
	IMAGE_DATA_FORMAT_BAYER_GR14		=	49,
	IMAGE_DATA_FORMAT_BAYER_GR16		=	50,
	IMAGE_DATA_FORMAT_BAYER_GB8			=	51,//packed
	IMAGE_DATA_FORMAT_BAYER_GB10		=	52,
	IMAGE_DATA_FORMAT_BAYER_GB12		=	53,
	IMAGE_DATA_FORMAT_BAYER_GB14		=	54,
	IMAGE_DATA_FORMAT_BAYER_GB16		=	55,//packed
}IMAGE_DATA_FORMAT_API;

typedef struct{
	unsigned int width; 
	unsigned int height;
	unsigned int padding_x;			//Number of extra bytes provided at the end of each line to facilitate image alignment in buffers.
	void * pointer;					//Pointer to output image buffer
	IMAGE_DATA_FORMAT_API format; //output image pixel format
}OUTPUT_IMAGE_API, *LPOUTPUT_IMAGE_API;

typedef struct{
	DWORD skipL;				//Horizontal left skip pixels	
	DWORD blL;				//Horizontal left black pixels	
	DWORD garbL;				//Horizontal left corrupted pixels
	DWORD gapL;				//Horizontal left gap pixels
	DWORD Hactv;				//Horizontal active width pixels
	DWORD gapR;				//Horizontal right gap pixels
	DWORD garbR;				//Horizontal right corrupted pixels
	DWORD blR;				//Horizontal right black pixels	
	DWORD skipR;				//Horizontal right skip pixels	
	DWORD skipT;				//Vertical top skip pixels	
	DWORD blT;				//Vertical top black pixels	
	DWORD garbT;				//Vertical top corrupted pixels
	DWORD gapT;				//Vertical top gap pixels
	DWORD Vactv;				//Vertical active width pixels
	DWORD gapB;				//Vertical buttom gap pixels
	DWORD garbB;				//Vertical buttom corrupted pixels
	DWORD blB;				//Vertical buttom black pixels	
	DWORD skipB;				//Vertical buttom skip pixels
	DWORD totalWidth;		//L+H+R
	DWORD totalHeight;		//T+V+B	
	DWORD AbsoluteOffsetX;	//Horizontal offset origin of sensor 0,0	
	DWORD AbsoluteOffsetY;	//Vertical offset origin of sensor 0,0	
	IMAGE_DATA_FORMAT_API format; //Input image format
}INPUT_IMAGE_API, *LPINPUT_IMAGE_API;

/**
 *\struct RAW_PROCESS_DATA
 * \brief structure containing information about a raw data(processed data)
 */
typedef	struct {
#define RAW_PROCESS_DATA_STRUCT_VER	1
	WORD				wVersion; //!< Version of the structure
#define RAW_PROCESS_DATA_STRUCT_LEN sizeof(RAW_PROCESS_DATA)
	WORD				wLenght;	//!< Length (size) of the structure
	LPVOID				raw_bp; //raw data(MQ MD)
	LPVOID				raw_bp_processed; //Processed raw data(depack....)
#define JUST_RAW					0x00000001//If true do just raw processing and return it.
#define CONVERT_DATA_TO_2D			0x00000002//If data processed of real raw data.
#define RAW_BPC_DONE				0x00000004//Bad pixels correction done on raw data
#define COLOR_INTERPOLATION_DONE	0x00000010//If color interpolation done on incoming data
	DWORD				flags;
	INPUT_IMAGE_API		input;	
	OUTPUT_IMAGE_API	output;
	DWORD				frame_number;
}RAW_PROCESS_DATA, *LPRAW_PROCESS_DATA;

/**
 *\enum BITMAP_FROMAT
 * \brief structure containing information about bitmap output format
 */
typedef enum {
	BITMAP_BGR32 = 0x000,	 //!< BGRA 32/64 bits per pixel (by default)
	BITMAP_BGR24 = 0x001, 	 //!< BGR  24/48 bits per pixel
	BITMAP_BGRPLANAR = 0x002 //!< BGR  24/48 bits per pixel, planar format
	// etc
} BITMAP_FORMAT;

/**
 *\enum PLANE
 * \brief structure containing information about color plane
 */
typedef enum {
	PLANE_RED				= 0x000,		//!< red
	PLANE_GREEN_IN_RED		= 0x001,		//!< green in red
	PLANE_GREEN_IN_BLUE		= 0x002,		//!< green in blue
	PLANE_BLUE				= 0x003			//!< blue
	// etc
} PLANE;

/**
 *\enum FAMILY
 * \brief structure containing information about device family (MM, MR, MU, MS....)
 */
typedef enum {
	FAMILY_MM				= 0x000,		//!< 1394 cameras 
	FAMILY_MR				= 0x001,		//!< 1394 cameras
	FAMILY_MU				= 0x002,		//!< USB 2.0 cameras
	FAMILY_MS				= 0x003,		//!< PCIe family cameras
	FAMILY_MV				= 0x004,		//!< Virtual family cameras
	FAMILY_MQ				= 0x005,		//!< CMOS USB 3.0 cameras
	FAMILY_MD				= 0x006			//!< CCD USB 3.0 cameras
	// etc
} FAMILY;

/**
 *\enum RAW_TYPE
 * \brief structure containing information about raw data type returned by functions mmRetrieveFrame and mmRetrieveFrameEx)
 */
typedef enum {
	RAW_TYPE_HALF_ROW = 0					//!< One color to half row
	// etc
} RAW_TYPE;

/**
 *\enum EVENT
 * \brief structure containing information about external event type
 */
typedef enum {
	EXT_EVENT_P_CAMERA				= 0x000,		//!< Camera captured external event pos edge
	EXT_EVENT_N_CAMERA				= 0x001 		//!< Camera captured external event neg edge
	// etc
} EVENT;

/**
 *\enum GPIO
 * \brief structure containing information about GPIO state (only for MU cameras)
 */
typedef enum {
	SET_ZERO				= 0x000,		//!< set gpo to zero
	SET_ONE					= 0x001,		//!< set gpo to one
	SET_EXT_EVENT			= 0x002,		//!< set gpi to input external event
	SET_STROBE_OUT			= 0x003,		//!< set gpo to output "strobe out" signal (device busy)
	SET_TRIGGER				= 0x004,		//!< set gpi to trigger input
	SET_STROBE_OUT_INV		= 0x005,		//!< set gpo to output inverted "strobe out" (device busy)
	SET_STROBE_OUT_INT		= 0x006,		//!< set gpo to output "strobe out" signal (integration)
	SET_STROBE_OUT_INT_INV	= 0x007,		//!< set gpo to output inverted "strobe out" (integration)
	SET_INPUT_OFF			= 0x008,		//!< set gpi off
	SET_FRAME_TRIGGER_WAIT  = 0x009,		//!< set gpo to output ready for next trigger signal
	SET_FRAME_TRIGGER_WAIT_INV  = 0x00A,	//!< set gpo to output inverted ready for next trigger signal
	SET_EXPOSURE_PULSE  	= 0x00B,		//!< set gpo to output "strobe out" signal (integration 250us pulse) (for MU 2x Trow 4us - 220us)
	SET_EXPOSURE_PULSE_INV  = 0x00C,		//!< set gpo to output inverted "strobe out" (integration 250us pulse) (for MU 2x Trow 4us - 220us)
	SET_BUSY  	= 0x00D,		//!< set gpo to output camera is busy (trigger mode  - starts with trigger reception and ends with end of frame transfer from sensor; freerun - active when acq active)
	SET_BUSY_INV  = 0x00E,		//!< set gpo to output inverted camera is busy (trigger mode  - starts with trigger reception and ends with end of frame transfer from sensor; freerun - active when acq active)
	
	// etc
} GPIO;


/**
 *\enum LED
 * \brief structure containing information about LED state (only for MS cameras)
 */
typedef enum {
	LED_LINK_HB				= 0x000,		//!< set led to blink if link is ok	(led 1), heartbeat (led 2)
	LED_TRIGGER				= 0x001,		//!< set led to blink if trigger detected
	LED_EXT_EVENT			= 0x002,		//!< set led to blink if external signal detected
	LED_NIC					= 0x003,		//!< set led to blink if link is ok
	LED_STREAMING			= 0x004,		//!< set led to blink if data streaming
	LED_INTEGRATION			= 0x005,		//!< set led to blink if sensor integration time
	LED_ISACQ				= 0x006,		//!< set led to blink if device busy/not busy
	LED_ZERO				= 0x007,		//!< set led to zero
	LED_ONE					= 0x008,		//!< set led to one
	LED_BLINK				= 0x009,		//!< set led to blink ~1Hz
	// etc
} LED;

/**
 *\enum VERSIONS
 * \brief enumerator of versions
 */
typedef enum {
	VERSION_API		= 0x0000,	//m3api.dll
	VERSION_DRIVER	= 0x0001,	//Device driver 
	VERSION_MCU1	= 0x0002,	//MCU1/CPU1  
	VERSION_MCU2	= 0x0003,	//MCU2/CPU2  
	VERSION_FPGA1	= 0x0004,	//FPGA1  
} VERSIONS;

/**
 *\struct MMMODE
 * \brief structure containing information about a mode
 */
typedef	struct {
	DWORD		dwCx;				//!< X component of total spacial resolution
	DWORD		dwCy;				//!< Y component of total spacial resolution
	DWORD		dwFields;			//!< Number of fields to get one full frame
	DWORD		dwBinX;				//!< Binning along x
	DWORD		dwBinY;				//!< Binning along y
	MMMOSAIC	mmMosaic;			//!< Mosaic of raw data
	DWORD		dwRDO;				//!< Readout schema
	#define	RP_PS_COLOR	0x0		//!< Progressive scan color
	#define	RP_PS_MONO	0x1		//!< Progressive scan mono
	#define	RP_IL_1LINE	0x2		//!< Interlace 1 line - each field with individual colors: lines F0:0,2,4,6...   F1:1,3,5,7...
	#define	RP_IL_2LINE	0x3		//!< Interlace 2 line - each field with equivalent colors: lines F0:0,1,4,5...   F1:2,3,6,7...
	#define	RP_PS_MONO8 0x8		//!< Progressive scan mono (raw data 8 bits per pixel).
	#define RP_ONLY_RAW	0x10	//!< Can't be used interpolation (only raw data available).
	#define RP_RAW_SUPP	0x20	//!< Indicates RAW data support.
} MMMODE, *LPMMMODE;

/**
 *\struct MMMODEX
 * \brief structure containing information about a mode
 */
typedef	struct {
	DWORD		dwCx;				//!< X component of total spacial resolution
	DWORD		dwCy;				//!< Y component of total spacial resolution
	DWORD		dwFields;			//!< Number of fields to get one full frame
	DWORD		dwBinX;				//!< Binning along x
	DWORD		dwBinY;				//!< Binning along y
	MMMOSAIC	mmMosaic;			//!< Mosaic of raw data
	DWORD		dwRDO;				//!< Readout schema
//-----------------------------------
	int			minCy;				// ...
	int			swBinX,	swBinY;		//
//-----------------------------------
	char	description[64];//	= Binning 2x2 progressive scan
//-----------------------------------
	int		Hshift;			// >=0, CCDx(Horigin)
	int		SynPos;			//
	int		HblackL;		// >=0
	int		HgapL;			// >=pCam->ccMinGap
	int		Horigin;		// = HblackL + HGapL;
	int		HgapR;			// >=pCam->ccMinGap
	int		HblackR;		// >=0
	int		Htotal;			//
//----------------------------
	int		Vtotal;			//  = max(MD.Vblack+MD.VgapT, MD.Vorigin) + MD.dwCy + MD.VgapB + 1
	int		Vskip;			//
	int		VblackT;		//
	int		VgapT;			//
	int		Vorigin;		//
	int		VgapB;			//
	int		VblackB;		//
	int		VdummyB;		//
} MMMODEX, *LPMMMODEX;

/**
 *\struct MMGETBITMAP
 * \brief structure containing information about a bitmap.
 */
typedef	struct {
	LPVOID	bp;				//!< Data
	LONG	z;				//!< Zoom 
	DWORD	w;				//!< Full image width
	DWORD	h;				//!< Full image height
	DWORD	cx;			//!< Retrieve area x delta
	DWORD	cy;				//!< Retrieve area y delta
	LONG	x0;			//!< Retrieve area x position
	LONG	y0;				//!< Retrieve area y position
	DWORD	fn;				//!< Last field number
	DWORD	fpf;			//!< Fields per frame
	DWORD	binX;			//!< Required software binning X
	DWORD	binY;			//!< Required software binning Y
	DWORD	dwBPP;					//!< Output precision of color components (8 or 16)
	DARKSIDE	darkSide;	//!< Black level and noise parameters
} MMGETBITMAP, * LPMMGETBITMAP;

/**
 *\struct MMGETBITMAPOPT
 * \brief structure containing some additional information about a bitmap
 */
typedef	struct {
	BOOL	bDoAll;			//!< Processes all pixels, including blind, etc.
	BOOL	bTranspose;		//!< Flip image along diagonal
	BOOL	bBW;			//!< Convert into BW bitmap
	DWORD	dwBin;			//!< Software binning (=1, =2)
	DWORD	x0;				//!< x coordinate of the first point defining ROI
	DWORD	y0;				//!< y coordinate of the first point defining ROI
	DWORD	xc;				//!< x coordinate of the second point defining ROI
	DWORD	yc;				//!< y coordinate of the second point defining ROI

} MMGETBITMAPOPT, * LPMMGETBITMAPOPT;

/**
 *\struct MMAVER
 * \brief structure containing averaging algorithm settings
 */
typedef	struct {
	DWORD mode;
		#define MM_AVER_RECURSIVE		0	//!< Recursive averaging
		#define MM_AVER_ARITHMETIC		1	//!< Arithmetic averaging
	BOOL  enable;							//!< Enable/disable averaging process	
	BOOL  reset;							//!< Reset frame counter(only for TRUE mode)
	DWORD number_of_frames;					//!< Number of frames to use in TRUE mode.
	DWORD alpha;							//!< f(n) = f(n-1)*(1-alpha/100) + f(n)*(alpha/100), alpha = 0...100
} MMAVER, *LPMMAVER;

/**
 *\struct MMFILTER
 * \brief structure containing information about a filter.
 */
typedef	struct {
	DWORD	dwFilter;				//!< Filter routine used
		#define	MMF_NONE		0	//!< No interpolation
		#define	MMF_RGB9331		1	//!< Standard 9331 color interpolation
		#define	MMF_SHTNEW		2	//!< SHT new
		#define	MMF_BWNONE		3	//!< Intensity

	double	rgbWhiteBal[3];			//!< White balance correction				(0.0..8.0)
	double	rgbCorrection[4][4];	//!< Full RGB->RGB correction matrix		(0.0..16.0)
	double	sharpStrength;			//!< Sharpening filter strength			(0.0..4.0)
	double	gammaY;					//!< Intensity gamma						(0.01..1.0)
	double	gammaC;					//!< Stauration (color) gamma				(0.0..1.0)
	DWORD	dwBPP;					//!< Output precision of color components (8 or 16)
} MMFILTER, * LPMMFILTER;

/**
 *\struct MMGETFRAME
 * \brief structure containing information about a frame.
 */
typedef	struct {
	LPVOID	bp;				//!< Pointer to internal MM data buffer
	DWORD	modeX0;			//!< User defined X origin (active area)
	DWORD	modeCX;			//!< User defined X extension (active area)
	DWORD	modeY0;			//!< User defined Y origin (active area)
	DWORD	modeCY;			//!< User defined Y extension (active area)
	DWORD	modeBinX;		//!< Current mode X binning
	DWORD	modeBinY;		//!< Current mode Y binning
	DWORD	modeFPF;		//!< Current mode fields per frame
	DWORD	modeBPP;		//!< Bits per pixel
	DWORD	modeMTX;		//!< Color matrix code
	DWORD	modeRDO;		//!< Read out schema (RP_xxxx)

	DWORD	curX0;			//!< Partial readout Y origin used - covers some guard areas for interpolation
	DWORD	curCX;			//!< Partial readout Y extension
	DWORD	curY0;			//!< Partial readout X origin used - covers full width
	DWORD	curCY;			//!< Partial readout X extension
	DWORD	curBinX;		//!< Current data X  binning
	DWORD	curBinY;		//!< Current data Y binning
	DWORD	curBPP;			//!< Current data bits per pixel
	DWORD	curMTX;			//!< Color matrix code
	DWORD	curFN;			//!< Current field (frame) number
} MMGETFRAME, * LPMMGETFRAME;

/**
 *\struct MMGETFRAMEEX
 * \brief structure containing information about a frame
 */
typedef	struct {
	LPVOID	bp;				//!< Pointer to internal MM data buffer
	DWORD	modeX0;			//!< User defined X origin
	DWORD	modeCX;			//!< User defined X extension
	DWORD	modeY0;			//!< User defined Y origin
	DWORD	modeCY;			//!< User defined Y extension
	DWORD	modeBinX;		//!< Current mode X binning
	DWORD	modeBinY;		//!< Current mode Y binning
	DWORD	modeFPF;		//!< Current mode fields per frame
	DWORD	modeBPP;		//!< Bits per pixel
	DWORD	modeMTX;		//!< Color matrix code
	DWORD	modeRDO;		//!< Read out schema (RP_xxxx)

	DWORD	curX0;			//!< Partial readout Y origin used - covers some guard areas for interpolation
	DWORD	curCX;			//!< Partial readout Y extention + aligment = buffer image width
	DWORD	curY0;			//!< Partial readout X origin used - covers full width
	DWORD	curCY;			//!< Partial readout X extention + aligment = buffer image height
	DWORD	curBinX;		//!< Current data X binning
	DWORD	curBinY;		//!< Current data Y binning
	DWORD	curBPP;			//!< Current data bits per pixel
	DWORD	curMTX;			//!< Color matrix code
	DWORD	curFN;			//!< Current field (frame) number

	DWORD	origX0;			//!< X coordinate of the point referring to the origin of data needed to be processed
	DWORD	origY0;			//!< Y coordinate of the point referring to the origin of data needed to be processed
} MMGETFRAMEEX, * LPMMGETFRAMEEX;

/**
 *\struct NOISE_PARAMS
 * \brief structure containing information about noise filter parameters
 */
typedef struct 
{
	int iISO;						//!< film speed (0-3200), -1 if unknown,
	int iSensitivity;				//!< sensitivity to noise (0-5, 2 is recommended)
	int iStrength;					//!< noise removal strength (0-100, 60 is recommended)
	int iDetails;					//!< preservation of details (0-10, 4 is recommended)
	BOOL fSharpen;					//!< sharpen image (true/false)
	char *szIniFileName;			//!< pointer to string with INI file name (full path),
									//!< if NULL or "", no INI file is loaded
}NOISE_PARAMS, * LPNOISE_PARAMS;

/**
 *\struct MMSHADING
 * \brief structure containing information about shading
 */
typedef	struct {
	DWORD	shCX;			//!< x dimension
	DWORD	shCY;			//!< y dimension
	LPVOID	lpSub;			//!< Pointer to Sub image - orphan, must be released by application
	LPVOID	lpMul;			//!< Pointer to Mul image - orphan, must be released by application
} MMSHADING, * LPMMSHADING;

/**
 *\enum CMS_INTENT
 * \brief structure containing information about shading
 */
typedef enum {
	CMS_INTENT_PERCEPTUAL = 0,
	CMS_INTENT_RELATIVE_COLORIMETRIC,
	CMS_INTENT_SATURATION,
	CMS_INTENT_ABSOLUTE_COLORIMETRIC
}CMS_INTENT;

/**
 *\struct LUT
 * \brief structure containing information about LUT (converts 12, 10 or 8 bits data to 8 bits).
 */
typedef	struct {
#define LUT_STRUCT_VER	1
	WORD	wVersion; //!< Version of the structure
#define LUT_STRUCT_LEN sizeof(LUT)
	WORD	wLenght;	//!< Length (size) of the structure	
#define LUT_SIZE 4096
	BYTE	cLut[LUT_SIZE];	//!< LUT (12, 10 or 8 bits data to 8 bits data)
	WORD	wStart;			//!< Start of LUT set, must be less than wStop
	WORD	wStop;			//!< Stop of LUT set, must be less than LUT_SIZE 
	BOOL	bUseDefault;	//!< Set default LUT (linear ramp)
} LUT, * LPLUT;

//*****************************************************************
typedef enum{
	cpu_type_intel = 0,
	cpu_type_amd = 1
}CPU_TYPE;

typedef struct _INTER_PROCESS_DATA
{
	HANDLE hInterProcSync;
	HANDLE hInterEnumeration;
	HANDLE hXiSyncMutex;
	CPU_TYPE cpu_type;
} INTER_PROCESS_DATA, *PINTER_PROCESS_DATA;

#define INTER_PROCESS_DATA_NAME "m3api_InterProcessDataMap"
#define INTER_PROCESS_MUTEX_NAME "m3api_InterProcessMutex"
#define INTER_ENUM_MUTEX_NAME "m3api_InterEnumMutex"
#define XI_SYNC_MUTEX_NAME "xi_SyncMutex"
//*****************************************************************

/** @name LDTG_MODE
         ???
   */
   /** @{ */
#define	LTM_NORM		0
#define	LTM_CNT			1	//!< CNT=0 : infinity loop
#define	LTM_CNT_OR_TRG_SYNC	2	//!< CNT=0 : TRIG only
#define	LTM_CNT_OR_TRG_ASYN	3	//!< CNT=0 : TRIG only
/** @} */


/** \example testapp.cpp
	 * This is an example of how to use the Test class.
	 * More details about this example.
	 */
	
/** @name Group_1
         Group one
   */
   /** @{ */
/**
  	\brief Returns the number of all the cameras attached to the computer.
	
	Returns the pointer to the number of all cameras attached to the computer, specified by name.

	\code
	LPCSTR  pFriendlyName = NULL;
	DWORD  	pNumberDevices;
	int result =  mm40GetNumberDevices(pFriendlyName, &pNumberDevices);
	\endcode
	
   @param[in] pFriendlyName		device name
   @param[out] pNumberDevices	number of devices
   @return MM40_OK on success, error value otherwise.
	
 */
MM40_API MM40_RETURN __cdecl mmGetNumberDevices ( 
			IN LPCSTR pFriendlyName,			// Device name
			OUT LPDWORD pNumberDevices);		// Ptr to number

/**
  	\brief Returns the number of all the cameras after enumeration.
	
	Returns the pointer to the number of all cameras, specified by name. mmGetNumberDevices should be called before

	\code
	LPCSTR  pFriendlyName = NULL;
	DWORD  	pNumberDevices;
	int result =  mmGetNumberDevicesWithoutEnumeration(pFriendlyName, &pNumberDevices);
	\endcode
	
   @param[in] pFriendlyName		device name
   @param[out] pNumberDevices	number of devices
   @return MM40_OK on success, error value otherwise.
	
 */
MM40_API MM40_RETURN mmGetNumberDevicesWithoutEnumeration ( 
			IN LPCSTR pFriendlyName, 
			OUT PDWORD pNumberDevices);	

/**
   \brief Returns the camera's unique identifier (serial number) specified by index.
	
	Index represents the camera's number (it should not be greater than the value returned by mm40GetNumberDevices function).
	This function returns the serial number of a specific camera.

	\code
	DWORD pChipId;
	DWORD index = atoi(deviceNum); // where deviceNum (for example) is a char*
	int result =  mm40GetDevice(index, &pChipId);
	\endcode
	
   @param[in] nIndex		index of device
   @param[out] pChipId		device identifier (serial number,chip ID)
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetDevice ( 
			IN  DWORD nIndex,					// Index
			OUT LPDWORD pChipId);				// Ptr to chip ID

/**
   \brief Get Firmware Version

   @param[in] drvHandle			???
   @param[out] pSoftwareVersion			???
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetFWVersion( 
						IN HANDLE hDevice, 			// driver handle
						OUT LPDWORD  pSoftwareVersion);	// Returned firmware version

/**
   \brief Get Software/Hardware Version

   @param[in] drvHandle			???
	@param[in] ver			Enumerator of which version to get
   @param[out] pVersion			???
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetVersion( 
						IN HANDLE hDevice, 			// driver handle
						IN VERSIONS ver,			// Enumerator of which version to get
						OUT LPDWORD  pVersion);	// Returned version

/**
   \brief Gets model's name

   @param[in] hDevice			???
   @param[out] lpdwModel			???
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetModelName			(IN HANDLE hDevice, OUT LPDWORD lpdwModel );

/**
   \brief Gets model's name

   @param[in] hDevice			???
   @param[out] string			string with model name
   @param[in] str_lenght		lenght of string buffer
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetDeviceName			(IN HANDLE hDevice, OUT char * string, IN DWORD str_lenght );
MM40_API MM40_RETURN __cdecl mmGetDeviceNameBySerialNumber (IN DWORD dwSerial, OUT char * string, IN DWORD str_lenght );

/**
   \brief Gets model's family (MM, MR, MU, MS.....)

   @param[in] hDevice			???
   @param[out] lpdwModel			???
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetModelFamily		(IN HANDLE hDevice, OUT LPDWORD lpdwFamily );
MM40_API MM40_RETURN __cdecl mmGetModelFamilyById (IN DWORD nIndex, OUT LPDWORD lpdwFamily );

/**
   \brief Gets device instance path

   @param[in] hDevice			???
   @param[out] string			string with instance path
   @param[in] str_lenght		lenght of string buffer
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetDeviceInstancePath	(IN HANDLE hDevice, OUT char * path, IN DWORD str_lenght );
MM40_API MM40_RETURN __cdecl mmGetDeviceInstancePathById ( IN DWORD nIndex,	OUT char * path, IN DWORD str_lenght );

// HWN2:  CCCM.FRA
/**
  \brief Gets HWN2

   @param[in] hDevice			???
   @param[out] pExten			???
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetHWN2				(IN HANDLE hDevice, OUT PDWORD pExten );

/**
   \brief Gets serial number

   @param[in] hDevice			???
   @param[out] lpdwSern			???
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetSerialNumber		(IN HANDLE hDevice, OUT LPDWORD lpdwSern );

/**
   \brief Gets sensor serial number

   @param[in] hDevice			???
   @param[out] string			string with sensor serial number
   @param[in] str_lenght		lenght of string buffer
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetSensorSerialNumber (IN HANDLE hDevice, OUT char * lpdwSern, IN DWORD str_length );

/**
	\brief Get custom user ID from camera.

   @param[in] hDevice			handle of the specified device
   @param[out] string			string with custom used ID
   @param[in] str_lenght		returns lenght of string buffer
   @return MM40_OK on success, error value otherwise.
*/

MM40_API MM40_RETURN __cdecl mmGetDeviceUserID ( IN HANDLE hDevice, OUT char * user_id, IN DWORD* str_lenght );

/**
	\brief Get custom user ID from camera by camera index.

   @param[in] nIndex			index of camera
   @param[out] string			string with custom used ID
   @param[in] str_lenght		lenght of string buffer
   @return MM40_OK on success, error value otherwise.
*/

MM40_API MM40_RETURN __cdecl mmGetDeviceUserIDById (IN DWORD nIndex, OUT char * user_id, IN DWORD str_lenght );

/**
	\brief Assign custom user ID to from camera, maximum length 64 bytes.

   @param[in] hDevice			handle of the specified device
   @param[in] string			string with custom user ID
   @return MM40_OK on success, error value otherwise.
*/

MM40_API MM40_RETURN __cdecl mmSetDeviceUserID ( IN HANDLE hDevice, IN char * user_id);

/**
   \brief Allows the user to initialize camera's software

	 This function prepares the camera's software for work.
	 It populates structures, runs initializing procedures, allocates resources - prepares the camera specified by name and chip ID for work.

	\note Function creates and returns handle of the specified device. To de-initialize the camera and destroy the handler mm40Uninitialize should be called.	
	
	\sa mm40Uninitialize

	\code
	DWORD lpChipId = atoi(deviceChipId); // where deviceChipId (for example) is a char*
	LPCSTR pFriendlyName = NULL;
	int result =  mmInitialize(pFriendlyName, lpChipId, &hDevice);
	\endcode

   @param[in] pFriendlyName		device friendly name
   @param[in] lChipId			device identifier (serial number, chip ID)	
   @param[out] drvHandle		pointer to the specified device's handle
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmInitialize(
			IN LPCSTR pFriendlyName,			// Device friendly name 	
			IN DWORD lChipId,					// chip ID
			OUT PHANDLE drvHandle);				// Returned driver handle


/**
   \brief Uninitializes camera.

	Closes camera handle and releases allocated resources.

	\code
	//device (HANDLE hDevice) should be already initialized here
	int result =  mmUninitialize(hDevice);
	\endcode

   @param[in] hDevice			handle of the specified device
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmUninitialize			(IN HANDLE hDevice);

/**
   \brief Allows the user to initialize camera's hardware.	
   
   Initializes the camera's hardware, starts timing generator

	\note Call to this function should follow call to mm40Initialize function.

	\code
	//device (HANDLE hDevice) should be already initialized here
	int result =  mmInitializeHardware(hDevice);
	\endcode
	
   @param[in] hDevice			handle of the specified device
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmInitializeHardware (IN HANDLE hDevice );

/**
   \brief Gets sensor pixel clock

	Allows the user to get current(just set) pixel clock (in Hz).
	
	\code
	//device (HANDLE hDevice) should be already initialized here
	//note - numbers below are arbitrary
	int result =  mmGetPixelClock (hDevice);	
	\endcode
	
   @param[in] hDevice			handle of the specified device
   @return MM40_OK on success, error value otherwise.
 */
MM40_API DWORD __cdecl mmGetPixelClock			(IN HANDLE hDevice);

/**
   \brief Resets hardware of the specified camera.
	
	Resets the hardware (timing generator) of the specified camera. This function can be used in some special cases to reset camera and therefore return to a normal working mode after some changes were made, or in order to change from trigger mode back to normal and vice versa.

	\code
	//device (HANDLE hDevice) should be already initialized here
	int result =  mmResetHardware(hDevice);
	\endcode
	
   @param[in] hDevice			handle of the specified device
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmResetHardware	(IN HANDLE hDevice );

/**
   \brief Sets camera's mode

	Allows the user to set mode number, exposition (in microseconds) and Region Of Interest (ROI), defined by two pairs of coordinates of the specified camera.
	
	\sa MMMODE
	
	\code
	//device (HANDLE hDevice) should be already initialized here
	//note - numbers below are arbitrary
	DWORD myexp = 100000;
	DWORD mymodenum = 1;
	DWORD myx0 = 10;
	DWORD myy0 = 10;
	DWORD mycx = 20;
	DWORD mycy = 20;
	int result =  mmSetMode (hDevice,myexp,mymodenum, myx0, myy0, mycx, mycy);	
	\endcode
	
   @param[in] hDevice			handle of the specified device
   @param[in] dwExpUsec			exposure value (in microseconds)
   @param[in] dwModeNum			mode
   @param[in] x0				x (horizontal) coordinate of the origin, point of reference (used for image clipping)
   @param[in] y0				y (vertical) coordinate of the origin, point of reference    (used for image clipping)
   @param[in] cx				width of the image area  (used for image clipping)
   @param[in] cy				height of the image area  (used for image clipping)
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmSetMode			(IN HANDLE hDevice, DWORD dwExpUsec, DWORD dwModeNum, DWORD x0, DWORD y0, DWORD cx, DWORD cy);

/**
   \brief Sets camera's horizontal offset

	Allows the user to set horizontal offset in pixels
	
	\code
	//device (HANDLE hDevice) should be already initialized here
	//note - numbers below are arbitrary
	DWORD offset = 100;
	int result =  mmSetOffsetX (hDevice,offset);	
	\endcode
	
   @param[in] hDevice			handle of the specified device
   @param[in] dwOffsetX			offset
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmSetOffsetX			(IN HANDLE hDevice, DWORD dwOffsetX);

/**
   \brief Sets camera's vertical offset

	Allows the user to set vertical offset in pixels
	
	\code
	//device (HANDLE hDevice) should be already initialized here
	//note - numbers below are arbitrary
	DWORD offset = 100;
	int result =  mmSetOffsetY (hDevice,offset);	
	\endcode
	
   @param[in] hDevice			handle of the specified device
   @param[in] dwOffsetX			offset
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmSetOffsetY			(IN HANDLE hDevice, DWORD dwOffsetY);

/**
   \brief Sets camera's image width

	Allows the user to set image width in pixels
	
	\code
	//device (HANDLE hDevice) should be already initialized here
	//note - numbers below are arbitrary
	DWORD width = 640;
	int result =  mmSetWidth (hDevice,width);	
	\endcode
	
   @param[in] hDevice			handle of the specified device
   @param[in] dwWidth			image width 
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmSetWidth			(IN HANDLE hDevice, DWORD dwWidth);

/**
   \brief Sets camera's image height

	Allows the user to set image height in pixels
	
	\code
	//device (HANDLE hDevice) should be already initialized here
	//note - numbers below are arbitrary
	DWORD height = 480;
	int result =  mmSetHeight (hDevice,height);	
	\endcode
	
   @param[in] hDevice			handle of the specified device
   @param[in] dwHeight			image height
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmSetHeight			(IN HANDLE hDevice, DWORD dwHeight);

/**
   \brief Sets camera's exposure time

	Allows the user to set exposition (in microseconds).
	
	\sa MMMODE
	
	\code
	//device (HANDLE hDevice) should be already initialized here
	//note - numbers below are arbitrary
	DWORD myexp = 100000;
	int result =  mmSetExposure (hDevice,myexp);	
	\endcode
	
   @param[in] hDevice			handle of the specified device
   @param[in] dwExpUsec			exposure value (in microseconds)
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmSetExposure			(IN HANDLE hDevice, DWORD dwExpUsec);

/**
   \brief Gets camera's current exposure time

	Allows the user to get current(just set) exposition (in microseconds).
	
	\sa MMMODE
	
	\code
	//device (HANDLE hDevice) should be already initialized here
	//note - numbers below are arbitrary
	int result =  mmGetExposure (hDevice);	
	\endcode
	
   @param[in] hDevice			handle of the specified device
   @return MM40_OK on success, error value otherwise.
 */
MM40_API DWORD __cdecl mmGetExposure			(IN HANDLE hDevice);

/**
   \brief Gets camera's minimum exposure time

	Allows the user to get minimum exposition (in microseconds).
	
	\sa MMMODE
	
	\code
	//device (HANDLE hDevice) should be already initialized here
	//note - numbers below are arbitrary
	int result =  mmGetMinExposure (hDevice);	
	\endcode
	
   @param[in] hDevice			handle of the specified device
   @return MM40_OK on success, error value otherwise.
 */
MM40_API DWORD __cdecl mmGetMinExposure			(IN HANDLE hDevice);

/**
   \brief Gets camera's maximum exposure time

	Allows the user to get maximum exposition (in microseconds).
	
	\sa MMMODE
	
	\code
	//device (HANDLE hDevice) should be already initialized here
	//note - numbers below are arbitrary
	int result =  mmGetMaxExposure (hDevice);	
	\endcode
	
   @param[in] hDevice			handle of the specified device
   @return MM40_OK on success, error value otherwise.
 */
MM40_API DWORD __cdecl mmGetMaxExposure			(IN HANDLE hDevice);

/**
   \brief Sets camera's mode

	
	Allows the user to set mode, by using the pointer to SETMODE structure, so that the function assigns the user defined SETMODE structure to that of the specified camera.
	
	\sa mmSetMode

	\code
	//device (HANDLE hDevice) should be already initialized here
	SETMODE mode;
	//note - numbers below are arbitrary
	WORD 	myver = 1;
	WORD 	mylen = sizeof(SETMODE);
	DWORD myexp = 100000;
	DWORD mymodenum = 1;
	DWORD myx0 = 10;
	DWORD myy0 = 10;
	DWORD mycx = 20;
	DWORD mycy = 20;
	DWORD 	mykeep = 1;
	DWORD  myltmode = 1;
	DWORD 	myltcnt = 1;
	WORD 	myupd = 1;

	mode.wVersion = myver;
	mode.wLenght = mylen;
	mode.dwExpUsec = myexp;
	mode.dwModeNum = mymodenum;
	mode.x0 = myx0;
	mode.y0 = myy0;
	mode.cx = mycx;
	mode.cy = mycy;
	mode.dwKEEP = mykeep;
	mode.dwLTmode = myltmode;
	mode.dwLTcnt = myltcnt;
	mode.wUPD = myupd;

	int result =  mmSetModeEx(hDevice, &mode);
	\endcode
	
   @param[in] hDevice			handle of the specified device
   @param[in] smode				pointer to SETMODE structure
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmSetModeEx		(IN HANDLE hDevice, SETMODE * smode);

/**
   \brief Gets camera's mode

	
	Allows the user to get mode, by using the pointer to SETMODE structure.
	
	\sa mmGetMode

	\code
	//device (HANDLE hDevice) should be already initialized here
	SETMODE mode;
	//note - numbers below are arbitrary

	mode.wVersion = SETMODE_STRUCT_VER;
	mode.wLenght = SETMODE_STRUCT_LEN;

	int result =  mmGetModeEx(hDevice, &mode);
	\endcode
	
   @param[in] hDevice			handle of the specified device
   @param[in] smode				pointer to SETMODE structure
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetModeEx		(IN HANDLE hDevice, SETMODE * smode);

/**
   \brief Checks if last set mode is complete. 
   
   This function checks if last set mode is complete. 
   
   \code
	//device (HANDLE hDevice) should be already initialized here
	int result = mmIsSetModeComplete(hDevice);
	\endcode

   @param[in] hDevice			handle of the specified device
   @return TRUE on success, FALSE otherwise.
 */
MM40_API BOOL        __cdecl mmIsSetModeComplete	(IN HANDLE hDevice);

/**
   \brief Checks if device ready for acquisition. 
   
   This function checks if device ready for acquisition. 
   
   \code
	//device (HANDLE hDevice) should be already initialized here
	int result = mmIsReady(hDevice);
	\endcode

   @param[in] hDevice			handle of the specified device
   @return TRUE on success, FALSE otherwise.
 */
MM40_API BOOL        __cdecl mmIsReady	(IN HANDLE hDevice);

/**
   \brief Starts data acquisition from the camera.

	Begins the work cycle and starts data acquisition from the camera. This function should be called once both software and hardware are initialized.

	\sa mm40Initialize, mmInitializeHardware

	\code
	//device (HANDLE hDevice) should be already initialized here
	int result =  mmStartAcquisition(hDevice);
	\endcode

   @param[in] hDevice			handle of the specified device
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmStartAcquisition	(IN HANDLE hDevice);

/**
   \brief Returns TRUE if acquisition already been started.

	Returns TRUE if acquisition already been started.

	\sa mmStartAcquisition

	\code
	//device (HANDLE hDevice) should be already initialized here
	int result =  mmIsAcquisition(hDevice);
	\endcode

   @param[in] hDevice			handle of the specified device
   @return TRUE on success.
 */
MM40_API BOOL __cdecl mmIsAcquisition	(IN HANDLE hDevice);

/**
   \brief Stops data acquisition from the camera.
   
   Ends the work cycle of the camera and stops data acquisition.

   \code
	//device (HANDLE hDevice) should be already initialized here
	int result =  mmStopAcquisition(hDevice);
	\endcode

   @param[in] hDevice			handle of the specified device
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmStopAcquisition	(IN HANDLE hDevice);

/**
   \brief Gets available bus speed. Only for USB camera.
   
   \code
	//device (HANDLE hDevice) should be already initialized here
	int result =  mmGetBusSpeed(hDevice, &speed, &errors);
	\endcode

   @param[in] hDevice			handle of the specified device
   @param[out] speed			pointer to DWORD parameter. Speed of bus.
   @param[out] errors			pointer to DWORD parameter. Errors.
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetAvailableBusSpeed	(IN HANDLE hDevice, OUT LPDWORD speed, OUT LPDWORD errors);

/**
   \brief Gets current bus speed. Only for USB camera.
   
   \code
	//device (HANDLE hDevice) should be already initialized here
	int result =  mmGetBusSpeed(hDevice, &speed, &errors);
	\endcode

   @param[in] hDevice			handle of the specified device
   @param[out] speed			pointer to DWORD parameter. Speed of bus.
   @param[out] errors			pointer to DWORD parameter. Errors.
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetBusSpeed	(IN HANDLE hDevice, OUT LPDWORD speed);

/**
   \brief Sets current bus speed. Only for USB camera.
   
   \code
	//device (HANDLE hDevice) should be already initialized here
	int result =  mmGetBusSpeed(hDevice, &speed);
	\endcode

   @param[in] hDevice			handle of the specified device
   @param[out] speed			DWORD parameter. Speed of bus.
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmSetBusSpeed	(IN HANDLE hDevice, IN DWORD speed);

/**
   \brief Tunes bus speed. Only for USB camera.
   
   \code
	//device (HANDLE hDevice) should be already initialized here
	int result =  mmTuneBusSpeed(hDevice, &speed, &error, timeout);
	\endcode

   @param[in] hDevice			handle of the specified device
   @param[out] speed			pointer to DWORD parameter. Speed of bus.
   @param[out] errors			pointer to DWORD parameter. Errors.
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmTuneBusSpeed	(IN HANDLE hDevice, OUT LPDWORD speed, OUT LPDWORD errors, IN DWORD timeout);

/**
   \brief Checks whether or not specified field exists

	Checks whether or not a specified field exists and is ready for acquisition (e.g. a field needed for frame composition).

	\code
	//device (HANDLE hDevice) should be already initialized here
	int result =  mmIsFieldReady(hDevice);
	\endcode

   @param[in] hDevice			handle of the specified device
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmIsFieldReady		(IN HANDLE hDevice);

/**
   \brief Waits for field

	Waits for the next frame field to become ready to be acquired for some user-defined time interval (e.g. field needed for frame composition).

	\code
	//device (HANDLE hDevice) should be already initialized here
	DWORD dwTimeOut = atoi(time); // where time (for example) is a char*
	int result = mmWaitForField(hDevice, dwTimeOut);
	\endcode

   @param[in] hDevice			handle of the specified device
   @param[in] dwTimeOut			time interval required to wait for the field to become ready
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmWaitForField		(IN HANDLE hDevice, IN  DWORD dwTimeOut );

/**
   \brief Waits for field

	Waits for the next frame field to become ready to be acquired for some user-defined time interval (e.g. a field needed for frame composition). The difference between this and the function above is that this function returns the pointer to the structure that contains information about the field.

	\code
	//device (HANDLE hDevice) should be already initialized here
	DWORD dwTimeOut = atoi(time); // where time (for example) is a char*
	FIELDINFO lpFLDinfo;
	int result = mmWaitForFieldEx (hDevice, dwTimeOut, &lpFLDinfo);
	\endcode

   @param[in] hDevice			handle of the specified device
   @param[in] dwTimeOut			time interval required to wait for the field to become ready
   @param[in] lpFLDinfo			structure containing information about field
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmWaitForFieldEx	(IN HANDLE hDevice, IN DWORD dwTimeOut, OUT LPFIELDINFO lpFLDinfo); 

/**
   \brief Returns information of current received filed/frame

	Returns information of current received filed/frame

	\code
	//device (HANDLE hDevice) should be already initialized here
	FIELDINFO lpFLDinfo;
	int result = mmGetFieldInfo (hDevice, &lpFLDinfo);
	\endcode

   @param[in] hDevice			handle of the specified device
   @param[in] lpFLDinfo			structure containing information about field
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN mmGetFieldInfo	(IN HANDLE hDevice, OUT LPFIELDINFO lpFLDinfo );

/**
   \brief Waits for event

	Waits for event.

	\code
	//device (HANDLE hDevice) should be already initialized here
	DWORD dwTimeOut = atoi(time); // where time (for example) is a char*
	int result = mmWaitForEvent(hDevice, EXT_EVENT_CAMERA, dwTimeOut);
	\endcode

   @param[in] hDevice			handle of the specified device
   @param[in] dwEvent			event to wait
   @param[in] dwTimeOut			time interval required to wait for the field to become ready
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmWaitForEvent		(IN HANDLE hDevice, IN EVENT dwEvent, IN  DWORD dwTimeOut );

/**
   \brief Waits for new mode first data

	Waits for the next mode.

	\code
	//device (HANDLE hDevice) should be already initialized here
	DWORD dwTimeOut = atoi(time); // where time (for example) is a char*
	FIELDINFO lpFLDinfo;
	int result = mmWaitForNewMode (hDevice, dwExpUsec, dwModeNum, dwTimeOut);
	\endcode

   @param[in] hDevice			handle of the specified device
   @param[in] dwTimeOut			time interval required to wait for the field to become ready
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmWaitForNewMode	(IN HANDLE hDevice, IN DWORD dwTimeOut); 


/**
   \brief Gets primary raw data from the camera driver.

	Gets primary raw data from the camera driver once field becomes ready for acquisition and returns the pointer to that field.

	\code
	HANDLE hDevice;
	//device (HANDLE hDevice) should be already initialized here
	int result = mmAcquireField(hDevice, &hImage);
	\endcode
	
   @param[in] hDevice			handle of the specified device
   @param[out] hImage			pointer to raw data acquired from the field
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmAcquireField		(IN HANDLE hDevice, OUT HANDLE * hImage );

/**
   \brief Resets field buffer chain

	Zero ready fields/frames after function call.

	\code
	//device (HANDLE hDevice) should be already initialized here
	int result =  mmResetFieldBuffersQueue(hDevice);
	\endcode

   @param[in] hDevice			handle of the specified device
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmResetFieldBuffersQueue		(IN HANDLE hDevice);


/**
   \brief Set number of field buffers.

	Set number of field buffers in the field/frame queue.

	\code
	HANDLE hDevice;
	//device (HANDLE hDevice) should be already initialized here
	int result = mmSetNumberFieldBuffers(hDevice, fields);
	\endcode
	
   @param[in] hDevice			handle of the specified device
   @param[out] fields			number of fields
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmSetNumberFieldBuffers		(IN HANDLE hDevice, IN DWORD fields );

/**
   \brief Return number of field buffers.

	Return number of field buffers in the field/frame queue.

	\code
	HANDLE hDevice;
	//device (HANDLE hDevice) should be already initialized here
	int result = mmSetNumberFieldBuffers(hDevice, fields);
	\endcode
	
   @param[in] hDevice			handle of the specified device
   @param[out] fields			number of fields
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN mmGetNumberFieldBuffers (IN HANDLE hDevice, IN LPDWORD fields );

/**
   \brief Set acquisition buffer size.

	Set acquisition buffer size in bytes.

	\code
	HANDLE hDevice;
	//device (HANDLE hDevice) should be already initialized here
	int result = mmSetAcquisitionBufferSize(hDevice, size);
	\endcode
	
   @param[in] hDevice			handle of the specified device
   @param[out] size				buffer size in bytes
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmSetAcquisitionBufferSize		(IN HANDLE hDevice, IN DWORD size );

/**
   \brief Return packet size

	Set Return packet size

	\code
	HANDLE hDevice;
	//device (HANDLE hDevice) should be already initialized here
	int result = mmGetPacketSize(hDevice, &size);
	\endcode
	
   @param[in] hDevice			handle of the specified device
   @param[out] size				packet size in bytes
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN mmGetPacketSize (
		IN HANDLE hDevice, 
		IN LPDWORD size );

/**
   \brief Enable/disable incoming data packing.
   
   Enable incoming data packing(grouping) to avoid overhead in case of unpacked data.

   \code
	HANDLE hDevice;
	//device (HANDLE hDevice) should be already initialized here
	int result = mmEnableDataPacking(hDevice, TRUE);
	\endcode

   @param[in] hDevice			handle of the specified device
   @param[in] packing			TRUE enable, FALSE disable
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmEnableDataPacking		(IN HANDLE hDevice, IN BOOL packing);

/**
   \brief Builds the new frame.
   
   Builds the new frame. If the camera is interlaced frame is build out of several fields. This function should be called sequentially once per field.

   \code
	HANDLE hDevice;
	//device (HANDLE hDevice) should be already initialized here
	RAW_PROCESS_DATA data
	int result = mmPrepareFrameForColorInterpolation(hDevice, hImage, &data);
	\endcode

   @param[in] hDevice			handle of the specified device
   @param[in] hImage			pointer to field
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmProcess		(IN HANDLE hDevice, IN HANDLE hFieldBuffer, LPRAW_PROCESS_DATA data );

/**
   \brief Builds the new frame.
   
   Builds the new frame. If the camera is interlaced frame is build out of several fields. This function should be called sequentially once per field.

   \code
	HANDLE hDevice;
	//device (HANDLE hDevice) should be already initialized here
	RAW_PROCESS_DATA data
	int result = mmPrepareFrameForColorInterpolation(hDevice, hImage, &data);
	\endcode

   @param[in] hDevice			handle of the specified device
   @param[in] hImage			pointer to field
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetInputImageFeatures		(IN HANDLE hDevice, LPINPUT_IMAGE_API data );

/**
   \brief Builds the new frame.
   
   Builds the new frame. If the camera is interlaced frame is build out of several fields. This function should be called sequentially once per field.

   \code
	HANDLE hDevice;
	//device (HANDLE hDevice) should be already initialized here
	int result = mmUpdateFrame(hDevice, hImage);
	\endcode

   @param[in] hDevice			handle of the specified device
   @param[in] hImage			pointer to field
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmUpdateFrame		(IN HANDLE hDevice, IN HANDLE hImage );

/**
   \brief Builds the new frame.
   
   Builds the new frame. If the camera is interlaced frame is build out of several fields. This function should be called sequentially once per field.

   \code
	HANDLE hDevice;
	//device (HANDLE hDevice) should be already initialized here
	int result = mmUpdateFrame(hDevice, hImage);
	\endcode

   @param[in] hDevice			handle of the specified device
   @param[in] hImage			pointer to field
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmUpdateBayerFrame		(IN HANDLE hDevice, IN HANDLE hImage );

/**
   \brief Sets averaging settings
   
   \code
	HANDLE hDevice;
	//device (HANDLE hDevice) should be already initialized here
	int result = mmAverageFrame(hDevice, lpAver);
	\endcode

   @param[in] hDevice			handle of the specified device
   @param[in] lpAver			pointer to MMAVER structure
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmAverageFrame (IN HANDLE hDevice,	IN LPMMAVER lpAver);

/**
   \brief Checks whether or not frame is complete. 
   
   Checks whether or not the frame is complete. \n (i.e. checks if frame composition is finished and it is ready for further processing).

   \code
	//device (HANDLE hDevice) should be already initialized here
	int result = mmIsFrameComplete(hDevice);
	\endcode

   @param[in] hDevice			handle of the specified device
   @return TRUE on success, FALSE otherwise.
 */
MM40_API BOOL        __cdecl mmIsFrameComplete	(IN HANDLE hDevice);

/**
   \brief Applies internal filters to frame data.

	Applies internal filters to frame data, performs demozaicing, sharpening etc.
	
	\sa \ref diagram

	\code
	//device (HANDLE hDevice) should be already initialized here
	int result = mmProcessFrame(hDevice);
	\endcode

   @param[in] hDevice			handle of the specified device
   @return MM40_OK on success, error value otherwise.
 */
MM40_API BOOL		 __cdecl mmIsBufferOK( IN HANDLE hDevice );

/**
   \brief Applies internal filters to frame data.

	Applies internal filters to frame data, performs demozaicing, sharpening etc.
	
	\sa \ref diagram

	\code
	//device (HANDLE hDevice) should be already initialized here
	int result = mmProcessFrame(hDevice);
	\endcode

   @param[in] hDevice			handle of the specified device
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmProcessFrame		(IN HANDLE hDevice);

/**
   \brief Retrieves specified frame

	Allows the user to retrieve the frame into MMGETFRAME structure.

	\code
	//device (HANDLE hDevice) should be already initialized here
	MMGETFRAME lpGF;
	int result = mmRetrieveFrame (hDevice, &lpGF);
	\endcode

   @param[in] hDevice			handle of the specified device
   @param[in,out] lpGF			LPMMGETFRAME, pointer to MMGETFRAME structure
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmRetrieveFrame	(IN HANDLE hDevice, INOUT LPMMGETFRAME lpGF );

/**
   \brief Retrieves specified frame

	Allows the user to retrieve the frame into MMGETFRAMEEX structure.

	\code
	//device (HANDLE hDevice) should be already initialized here
	MMGETFRAMEEX lpGFex;
	int result = mmRetrieveFrameEx(hDevice, &lpGFex);
	\endcode

   @param[in] hDevice			handle of the specified device
   @param[in,out] lpGF			LPMMGETFRAMEEX, pointer to MMGETFRAMEEX structure
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmRetrieveFrameEx	(IN HANDLE hDevice, INOUT LPMMGETFRAMEEX lpGF );

/**
   \brief Retrieves specified frame raw data type

	Allows the user to retrieve the frame raw data type into PDWORD parameter.

	\code
	//device (HANDLE hDevice) should be already initialized here
	DWORD type
	int result = mmRetrieveFrameType(hDevice, &type);
	\endcode

   @param[in] hDevice			handle of the specified device
   @param[in,out] type			pointer to DWORD
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmRetrieveFrameType	(IN HANDLE hDevice, INOUT LPDWORD type );

/**
   \brief Retrieves specified frame

	Allows the user to retrieve the frame into MMGETFRAMEEX structure.

	\code
	//device (HANDLE hDevice) should be already initialized here
	MMGETFRAMEEX lpGFex;
	int result = mmRetrieveFrameBayerEx(hDevice, &lpGFex);
	\endcode

   @param[in] hDevice			handle of the specified device
   @param[in,out] lpGF			LPMMGETFRAMEEX, pointer to MMGETFRAMEEX structure
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmRetrieveFrameBayerEx	(IN HANDLE hDevice, INOUT LPMMGETFRAMEEX lpGF );

/**
   \brief Retrieves specified frame plane

	Allows the user to retrieve the frame into MMGETFRAMEEX structure.

	\code
	//device (HANDLE hDevice) should be already initialized here
	MMGETFRAMEEX lpGFex;
	int result = mmRetrieveFramePlaneEx(hDevice, &lpGFex);
	\endcode

   @param[in] hDevice			handle of the specified device
   @param[in,out] lpGF			LPMMGETFRAMEEX, pointer to MMGETFRAMEEX structure
   @param[in]	plane			PLANE enum (red, green in red, green in blue, blue)
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmRetrieveFramePlaneEx	(IN HANDLE hDevice, INOUT LPMMGETFRAMEEX lpGF, IN DWORD plane);

/**
   \brief Sets up filter

	Allows the user to set internal filter parameters up using MMFILTER structure, that can later be applied to the frame by calling  \ref mmProcessFrame function.
	
	\sa MMFILTER

	\code
	//device (HANDLE hDevice) should be already initialized here
	MMFILTER lpFlt;
	DWORD 	mydwFilter = MMF_NONE;
	//note - numbers below are arbitrary
	double 	mysharpStrength = 0;
 	//Sharpening filter strength (0.0..4.0).
	double 	mygammaY = 1.0;
 	//Intensity gamma (0.01..1.0).
	double 	mygammaC = 0;
 	//Saturation (color) gamma (0.0..1.0).
	DWORD 	mydwBPP = 8;
	
	memset(&lpFlt, 0, sizeof(lpFlt));
	lpFlt.dwFilter = mydwFilter;
	double 	myrgbCorrection [4][4] ={		{ 1.00, 0.00, 0.00, 0.00},	// R
							{ 0.00, 1.00, 0.00, 0.00},	// G
							{ 0.00, 0.00, 1.00, 0.00},	// B
							{ 0.00, 0.00, 1.00, 0.00}};	// S
	lpFlt.rgbWhiteBal [0] = 1.0;
	lpFlt.rgbWhiteBal [1] = 1.0;
	lpFlt.rgbWhiteBal [2] = 1.0;
	memcpy(lpFlt.rgbCorrection, myrgbCorrection, sizeof(lpFlt.rgbCorrection));
	lpFlt.sharpStrength = mysharpStrength;
	lpFlt.gammaY = mygammaY;
	lpFlt.gammaC = mygammaC;
	lpFlt.dwBPP = mydwBPP;

	int result = mmSetupFilter(hDevice, &lpFlt);
	\endcode

   @param[in] hDevice			handle of the specified device
   @param[in] lpFlt			LPMMFILTER, pointer to MMFILTER structure 
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmSetupFilter		(IN HANDLE hDevice, IN LPMMFILTER lpFlt );

/**
   \brief Applyes default color correction matrix(ccm)

	Allows the user to apply default ccm

	\code
	//device (HANDLE hDevice) should be already initialized here
	int result = mmApplyDefaultColorCorrectionMatrix(hDevice, TRUE);
	\endcode

   @param[in] hDevice			handle of the specified device
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmApplyDefaultColorCorrectionMatrix		(IN HANDLE hDevice);
/**
   \brief Applyes default white balance

	Allows the user to apply default white balance

	\code
	//device (HANDLE hDevice) should be already initialized here
	int result = mmApplyDefaultColorCorrectionMatrix(hDevice, TRUE);
	\endcode

   @param[in] hDevice			handle of the specified device
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmApplyDefaultWhiteBalance		(IN HANDLE hDevice);

/**
   \brief Gets filter

	Allows the user to get internal filter parameters using MMFILTER structure, that can later be applied to the frame by calling  \ref mmProcessFrame function.
	
	\sa MMFILTER

	\code
	//device (HANDLE hDevice) should be already initialized here
	MMFILTER lpFlt;
	DWORD 	mydwFilter = MMF_NONE;
	//note - numbers below are arbitrary
	double 	mysharpStrength = 0;
 	//Sharpening filter strength (0.0..4.0).
	double 	mygammaY = 1.0;
 	//Intensity gamma (0.01..1.0).
	double 	mygammaC = 0;
 	//Saturation (color) gamma (0.0..1.0).
	DWORD 	mydwBPP = 8;
	
	memset(&lpFlt, 0, sizeof(lpFlt));
	lpFlt.dwFilter = mydwFilter;
	double 	myrgbCorrection [4][4] ={		{ 1.00, 0.00, 0.00, 0.00},	// R
							{ 0.00, 1.00, 0.00, 0.00},	// G
							{ 0.00, 0.00, 1.00, 0.00},	// B
							{ 0.00, 0.00, 1.00, 0.00}};	// S
	lpFlt.rgbWhiteBal [0] = 1.0;
	lpFlt.rgbWhiteBal [1] = 1.0;
	lpFlt.rgbWhiteBal [2] = 1.0;
	memcpy(lpFlt.rgbCorrection, myrgbCorrection, sizeof(lpFlt.rgbCorrection));
	lpFlt.sharpStrength = mysharpStrength;
	lpFlt.gammaY = mygammaY;
	lpFlt.gammaC = mygammaC;
	lpFlt.dwBPP = mydwBPP;

	int result = mmGetFilter(hDevice, &lpFlt);
	\endcode

   @param[in] hDevice			handle of the specified device
   @param[in] lpFlt			LPMMFILTER, pointer to MMFILTER structure 
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetFilter		(IN HANDLE hDevice, IN LPMMFILTER lpFlt );

/**
   \brief Defines the width of dynamic range.

	Defines the width of dynamic range used for contrast enhancement.

	\code
	//device (HANDLE hDevice) should be already initialized here
	//note - numbers below are arbitrary
	DWORD dwLow = 0;
	DWORD dwHigh = 4095;
	int result = mmSetupCE(hDevice, dwLow, dwHigh);
	\endcode

   @param[in] hDevice			handle of the specified device
   @param[in] dwLow				dynamic range lower bound	
   @param[in] dwHigh			dynamic range upper bound
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmSetupCE			(IN HANDLE hDevice, IN DWORD dwLow, IN DWORD dwHigh );

/**
   \brief Sets bitmap output format (32/64 or 24/48 bits per pixel).
   
   Sets bitmap output format (32/64 or 24/48 bits per pixel).

    \code
	//device (HANDLE hDevice) should be already initialized here
	//note - values below are arbitrary
	int result = mmSetBitmapFormat(hDevice, BITMAP_BGR32);
	\endcode

   @param[in] hDevice			handle of the specified device
   @param[in] bmpf				BITMAP_FORMAT enum (bitmap output format)
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmSetBitmapFormat(IN HANDLE hDevice, IN BITMAP_FORMAT bmpf );


/**
   \brief Returns pointer to bitmap structure. (advanced)
   
   Allows the user to extract bitmap from image buffer. This function returns the pointers to bitmap specified by MMGETBITMAP and MMGETBITMAPOPT structures

    \code
	//device (HANDLE hDevice) should be already initialized here
	//note - values below are arbitrary
	MMGETBITMAPOPT gbo;
	gbo.bDoAll = FALSE;
	gbo.bTranspose = FALSE;
	gbo.dwBin = 1;
	gbo.bBW = TRUE;
	int result = mmRetrieveBitmapAdv(hDevice, &lpGB, &gbo);
	\endcode

   @param[in] hDevice			handle of the specified device
   @param[in,out] lpGB			LPMMGETBITMAP, pointer to MMGETBITMAP structure 
   @param[in,out] lpGBopt		LPMMGETBITMAPOPT, pointer to MMGETBITMAPOPT structure
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmRetrieveBitmapAdv(IN HANDLE hDevice, INOUT LPMMGETBITMAP lpGB, INOUT LPMMGETBITMAPOPT lpGBopt );

/**
   \brief Returns pointer to bitmap structure. (advanced)
   
   This function returns a pointer to MMGETBITMAP and MMGETBITMAPOPT structures containing frame data.

    \code
	//device (HANDLE hDevice) should be already initialized here
	//note - values below are arbitrary
	MMGETBITMAPOPT gbo;
	gbo.bDoAll = FALSE;
	gbo.bTranspose = FALSE;
	gbo.dwBin = 1;
	gbo.bBW = TRUE;
	int result = mmRetrieveBitmapDn(hDevice, &lpGB, &gbo, &noise);
	\endcode

   @param[in]		hDevice			handle of the specified device
   @param[in,out]	lpGB			address of a MMGETBITMAP structure where frame data will be stored on return
   @param[in,out]	lpGBopt			address of a MMGETBITMAPOPT structure where frame data will be stored on return
   @param[in]		noise			LPNOISE_PARAMS, pointer to NOISE_PARAMS structure
   @return MM40_OK on success, error value otherwise.
   @return MM40_MEMORY_ALLOCATION
   @return MM40_INVALID_ARG
 */
MM40_API MM40_RETURN __cdecl mmRetrieveBitmapDn(IN HANDLE hDevice, INOUT LPMMGETBITMAP lpGB, INOUT LPMMGETBITMAPOPT lpGBopt, IN LPNOISE_PARAMS noise );


/**
   \brief Initialize color managment system
   
   Allows the user to apply camera ICC profile.

   @param[in]	hDevice			handle of the specified device
   @param[in]	inProfile		input profile
   @param[in]	outProfile		output profile
   @param[in]	intent			intent
   @param[in]	fast			If TRUE uses fast method (only 8 bit data)
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN mmInitCMS ( IN HANDLE hDevice, LPVOID inProfile, LPVOID outProfile, CMS_INTENT intent, BOOL fast );


/**
   \brief Uninitialize color managment system
   
   Allows the user to apply camera ICC profile.

   @param[in]	hDevice			handle of the specified device
   @param[in]	inProfile		input profile
   @param[in]	outProfile		output profile
   @param[in]	intent			intent
   @param[in]	fast			If TRUE uses fast method (only 8 bit data)
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN mmUninitCMS ( IN HANDLE hDevice );

/**
   \brief Applies camera CMS profiles
   
   Allows the user to apply camera CMS profiles.

   @param[in] hDevice			handle of the specified device
   @param[in,out]	lpGB		address of a MMGETBITMAP structure 
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN mmApplyPCS ( IN HANDLE hDevice, INOUT LPMMGETBITMAP lpGB );

/**
   \brief Frees bitmap

	Frees bitmap specified by MMGETBITMAP structure, releases allocated for the structure space.

	\code
	//device (HANDLE hDevice) should be already initialized here
	//note - values below are arbitrary
	MMGETBITMAP lpGB = { 0, 0, NULL,
				1,1,1, 0, 0, 0,
				1, 
				0,0, 0,0, 0,0};	;
	int result = mmFreeBitmap(hDevice, &lpGB);
	\endcode

   @param[in] hDevice			handle of the specified device
   @param[in,out] lpGB			LPMMGETBITMAP, pointer to MMGETBITMAP structure 
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmFreeBitmap		(IN HANDLE hDevice, INOUT LPMMGETBITMAP lpGB );

/**
   \brief Sets gain

	Sets gain value of the specified camera.

	\code
	//device (HANDLE hDevice) should be already initialized here
	WORD value = atoi(val); // where val (for example) is a char*
	int result = mmSetGain(hDevice, value);
	\endcode

   @param[in] hDevice			handle of the specified device
   @param[in] wGain			value of gain
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmSetGain			(IN HANDLE hDevice, IN WORD wGain );

/**
   \brief Sets trigger

	Allows the user to set the camera to work in trigger mode. Working in this mode the camera will wait for some event (trigger), either a hardware trigger (positive/negative edge) or a software trigger (\ref mmPingTrigger) to acquire the frame.
	
	\sa \ref mmPingTrigger

	\code
	//device (HANDLE hDevice) should be already initialized here
	DWORD tmode = atoi(mmode); // where mmode (for example) is a char*
	int result = mmSetTrigger(hDevice, tmode, neg)
	\endcode

   @param[in] hDevice			handle of the specified device
   @param[in] dwMode			trigger mode
   @param[in] bNegative			edge type (leading/trailing a.k.a positive/negative)
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmSetTrigger		(IN HANDLE hDevice, IN DWORD dwMode, IN BOOL bNegative );

/**
   \brief Sets trigger

	This function puts the camera into the trigger mode. In trigger mode field acquisition will begin only uppon a trigger event.

	\note Available trigger modes are described in the definitions sections under MM_TRIG.
	\note In order to leave the trigger mode...

	\sa \ref mmPingTrigger

	\code
	//device (HANDLE hDevice) should be already initialized here
	SETTRIGGER trigger;
	trigger.wVersion = SETSETTRIGGER_STRUCT_VER;
	trigger.wLenght = SETTRIGGER_STRUCT_LEN;
	trigger.dwMode = MM_TRIG_OFF;
	trigger.wNegative = 0;
	trigger.wDoSetMode = 0;
	int result = mmSetTriggerEx(hDevice, &trigger)
	\endcode

   @param[in] hDevice			handle of the specified device
   @param[in] trigger			trigger mode
   @return MM40_OK on success, error value otherwise.
   @return MM40_INVALID_ARG
 */
MM40_API MM40_RETURN __cdecl mmSetTriggerEx		(IN HANDLE hDevice, IN LPSETTRIGGER trigger );

/**
   \brief Returns trigger mode.

	Allows the user to return trigger mode that the camera is currently working in and edge type.

	\code
	//device (HANDLE hDevice) should be already initialized here
	DWORD lpdwMode;
	BOOL lpbNegative;
	int result = mmGetTrigger( hDevice, &lpdwMode, &lpbNegative);
	\endcode

   @param[in] hDevice			handle of the specified device
   @param[out] lpdwMode			trigger mode
   @param[out] lpbNegative			edge type (leading/trailing a.k.a positive/negative)
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetTrigger		(IN HANDLE hDevice, OUT DWORD *lpdwMode, OUT BOOL * lpbNegative );

/**
   \brief Sets software trigger value

	This function sets sensor intput level 1 or 0

	\note Available trigger modes are described in the definitions sections under MM_TRIG.
	\note In order to leave the trigger mode...

	\sa \ref mmPingTrigger

	\code
	\endcode

   @param[in] hDevice			handle of the specified device
   @param[in] value				value
   @return MM40_OK on success, error value otherwise.
   @return MM40_INVALID_ARG
 */
MM40_API MM40_RETURN __cdecl mmSetSWTriggerValue		(IN HANDLE hDevice, IN BOOL value );

/**
   \brief Resets camera.

	Resets the bus. This function is an alias of \ref mm40BusReset

	\code
	//device (HANDLE hDevice) should be already initialized here
	int result = mmResetCamera(hDevice);
	\endcode

   @param[in] hDevice			handle of the specified device
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmResetCamera		(IN HANDLE hDevice );
/** @} */

/** @name Shading
         Shading functions
*/
/** @{ */
/**
   \brief Initializes shading.

	Initializes shading of the specified region.

   @param[in] hDevice			handle of the specified device
   @param[in,out] lpMMS			LPMMSHADING, pointer to MMSHADING structure
   @param[in] dwCX			linear x dimension of the specified region
   @param[in] dwCY			linear x dimension of the specified region
   @param[in] wOff			saturated white averaged (add, offset) picture
   @param[in] wMul			black averaged (multiply) picture
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmInitializeShading(IN HANDLE hDevice, INOUT LPMMSHADING lpMMS, DWORD dwCX, DWORD dwCY, WORD wOff, WORD wMul );

/**
   \brief Updates frame shading.
   
   Updates specified frame, performs shading correction.
   
   \sa MMSHADING structure 

   @param[in] hDevice			handle of the specified device
   @param[in] hFieldBuffer		field buffer
   @param[in] lpSahding			LPMMSHADING, pointer to MMSHADING structure
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmUpdateFrameShading (IN HANDLE hDevice, IN HANDLE hFieldBuffer, IN LPMMSHADING lpSahding);

/**
   \brief Calculates shading.

	Calculates shading of the specified region.

   @param[in] hDevice			handle of the specified device
   @param[in,out] lpMMS			LPMMSHADING, pointer to MMSHADING structure
   @param[in] dwCX				linear x dimension of the specified region
   @param[in] dwCY				linear y dimension of the specified region
   @param[in] pBlack			pointer to the averaged black picture
   @param[in] pWhite			pointer to the saturated picture
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmCalculateShading (IN HANDLE hDevice, INOUT LPMMSHADING lpMMS, DWORD dwCX, DWORD dwCY, LPWORD pBlack, LPWORD pWhite );

/** @} */


/** @name Group_2
         Group two
*/
/** @{ */
/**
   \brief Checks whether or not certain device exists

   Allows the user to check whether or not device with specified handle exists.

   @param[in] hDevice			handle of the specified device
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmIsDeviceExist			(IN HANDLE hDevice);

/**
   \brief Gets mode descriptor

   Allows the user to access specific mode by its descriptor (pointer to \ref MMMODE structure).

   @param[in] hDevice			handle of the specified device
   @param[in] dwModeNum			mode's number
   @param[out] mode			mode's descriptor
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetModeDescriptor (IN HANDLE hDevice, IN DWORD dwModeNum, OUT MMMODE * mode);

/**
   \brief Gets mode descriptor string

   Allows the user to access specific mode by its descriptor (pointer to \ref MMMODE structure).

   @param[in] hDevice			handle of the specified device
   @param[in] dwModeNum			mode's number
   @param[out] string			string with mode description
   @param[in] str_lenght		lenght of string buffer
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetModeDescriptorString (IN HANDLE hDevice, IN DWORD dwModeNum, OUT char * string, IN DWORD str_lenght );

/**
   \brief Gets more detailed mode descriptor

   Allows the user to access specific mode by its descriptor (pointer to \ref MMMODE structure).

   @param[in] hDevice			handle of the specified device
   @param[in] dwModeNum			mode's number
   @param[out] mode			    mode's descriptor
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetModeDescriptorEx (IN HANDLE hDevice, IN DWORD dwModeNum, OUT MMMODEX * mode);

/**
   \brief Gets mode descriptor

   Allows the user to access specific mode by its descriptor (pointer to \ref MMMODEX structure).

   @param[in] hDevice			handle of the specified device
   @param[in] dwModeNum			mode's number
   @param[out] mode				mode's descriptor
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetCurrentModeDescriptorEx (IN HANDLE hDevice, OUT MMMODEX * mode);

/**
   \brief Returns true for cameras with cooling support.

   Check whether camera supports cooling.

   @param[in] hDevice			handle of the specified device
   @return TRUE if cooling supported, FALSE otherwise.
 */
MM40_API BOOL __cdecl mmIsCooledCamera (IN HANDLE hDevice);

/**
   \brief Returns true for color cameras.

   Check whether camera can return color image.

   @param[in] hDevice			handle of the specified device
   @return TRUE if color camera, FALSE otherwise.
 */
MM40_API BOOL __cdecl mmIsColorCamera (IN HANDLE hDevice);

/**
   \brief Gets mode skipping description

   Allows the user to retrieve mode skipping values.

   @param[in] hDevice			handle of the specified device
   @param[in] dwModeNum			mode's number
   @param[out] dwSkipX			mode skipping horizontal
   @param[out] dwSkipY			mode skipping vertical
   @param[out] isSkpMode		is mode with skipping applied
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetModeSkipping (IN HANDLE hDevice, IN DWORD dwModeNum, OUT LPDWORD lpdwSkipX, OUT LPDWORD lpdwSkipY, OUT PBOOL isSkpMode);

/**
   \brief Get means

   Calculates and return R (red), GR (green in red), GB (green in blue) and B (blue) mean values within specific user-defined region.

   @param[in] hDevice			handle of the specified device
   @param[in] dwX0			user defined X coordinate of the origin
   @param[in] dwY0			user defined Y coordinate of the origin
   @param[in] dwCX			X coordinate extension, width
   @param[in] dwCY			Y coordinate extension, height
   @param[in,out] fMeans			pointer to means
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetMeans			(IN HANDLE hDevice, IN DWORD dwX0, IN DWORD dwY0, IN DWORD dwCX, IN DWORD dwCY, INOUT float * fMeans );

/**
   \brief Get black level

   @param[in] hDevice			handle of the specified device
   @param[in,out] fBL			pointer to black level
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetBlackLevel			(IN HANDLE hDevice, INOUT DWORD * fBL );
//***********************Raw Data**********************************

/**
   \brief Retrieves raw data

   Allows the user to access raw unprocessed data. This function returns pointer to raw data.

   @param[in] hDevice			handle of the specified device
   @param[in] timeout			timeout of getting new frame	
   @param[out] output			pointer to raw data
   @param[in] data_size			data size
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmRetrieveRaw(IN HANDLE hDevice, DWORD timeout, LPVOID * output, DWORD * data_size);

/**
   \brief Set Raw Data

	Sets raw data.

   @param[in] input			pointer to raw data
   @return MM40_OK on success, error value otherwise.
 */
/**
   \brief Set Raw Data

	???

   @param[in] input			pointer to raw data
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmSetRawData(IN HANDLE hDevice, LPVOID input);

/**
   \brief Releases raw data
   
   Release raw data, frees space allocated to structure containing inforation about speific raw data

   @param[in] output			raw data
   @param[in] data_size			data size
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmReleaseRaw(LPVOID output, DWORD data_size);

//*****************************************************************

/**
   \brief Get mode timing ???
   
   ???

   @param[in] hDevice			???
   @param[in,out] fTPFI0			???
   @param[in,out] fTPFIe			???
   @param[in,out] fTPFR0			???
   @param[in,out] fTPFRe			???
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetModeTiming	(IN HANDLE hDevice, INOUT float *fTPFI0, INOUT float *fTPFIe, INOUT float *fTPFR0, INOUT float *fTPFRe);

/**
   \brief Get mode timing maximum ???
   
   ???

   @param[in] hDevice			handle of the specified device
   @param[in,out] fTPFI0			???
   @param[in,out] fTPFIe			???
   @param[in,out] fTPFR0			???
   @param[in,out] fTPFRe			???
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetModeTimingMaximum	(IN HANDLE hDevice, INOUT float *fTPFI0, INOUT float *fTPFIe, INOUT float *fTPFR0, INOUT float *fTPFRe);

/**
   \brief Set frames per second
   
   ???

   @param[in] hDevice			handle of the specified device
   @param[in] fFPS				disirable frames per second
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmSetFPS	(IN HANDLE hDevice, IN float fFPS);

/**
   \brief Get frames per second
   
   ???

   @param[in] hDevice			handle of the specified device
   @param[in,out] fFPS			frames per second
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetFPS	(IN HANDLE hDevice, IN float * fFPS);

/**
   \brief Returns information about gain.
   
   Allows the user to retrieve information about gain (gain definition, lower ???, stops ???)

   @param[in] hDevice			handle of the specified device
   @param[in,out] fLow_dB			???
   @param[in,out] fHigh_dB			???
   @param[in,out] lpdwStops			???
   @param[in,out] lpdwDefGain			???
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetGainInfo		(IN HANDLE hDevice, INOUT float * fLow_dB, INOUT float * fHigh_dB, INOUT LPDWORD lpdwStops, INOUT LPDWORD lpdwDefGain);

/**
   \brief Returns gain
   
   Returns gain.

   @param[in] hDevice			handle of the specified device
   @param[in] pwGain			???
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetGain			(IN HANDLE hDevice, IN LPWORD pwGain );

/**
   \brief Sets gain value
   
   Allows the user to change current gain value to some user-defined one.

   @param[in] hDevice			handle of the specified device
   @param[in] fGain			new gain value
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmSetGainVal		(IN HANDLE hDevice, IN float fGain );

/**
   \brief Sets gain value
   
   Allows the user to change current gain value to some user-defined one.

   @param[in] hDevice			handle of the specified device
   @param[in] fGain			new gain value
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmSetGainAsync		(IN HANDLE hDevice, IN float fGain );

/**
   \brief Gets gain value
   
   Returns current gain value.

   @param[in] hDevice			handle of the specified device
   @param[in] fGain			pointer to floating point number representing current gain
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetGainVal		(IN HANDLE hDevice, IN float *fGain );

/**
   \brief Sets LED

	If camera is equipped with LED (Light Emitting Device), user can control it by turning it on and off and set led functionality with help of this function.

   @param[in] hDevice			handle of the specified device
   @param[in] bLED			??? which one is on ???
   @param[in] state			??? functionality ???
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmSetLED			(IN HANDLE hDevice, IN BYTE bLED, IN LED state );


/**
   \brief Pings trigger
   
   Allows the user to use software trigger, event that triggers camera's action, instead of using harware variant (positive/negative edge)

   @param[in] hDevice			handle of the specified device
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmPingTrigger		(IN HANDLE hDevice );

/**
   \brief Returns raw pixel
   
   Allows the user to return pointer to pixel defined by specific coordinates out of raw data.

   @param[in] hDevice			handle of the specified device
   @param[in] dwX			user-defined, x coordinate
   @param[in] dwY			user-defined, y coordinate
   @param[out] pPix			pixel from raw data
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetRawPixel		(IN HANDLE hDevice, IN DWORD dwX, IN DWORD dwY,  INOUT LPWORD pPix );

/**
   \brief Returns raw pixel ex ???
   
   ???

   @param[in] hDevice			handle of the specified device
   @param[in] dwX			user-defined, x coordinate
   @param[in] dwY			user-defined, y coordinate
   @param[out] pPix			pixel from raw data
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetRawPixelEx	(IN HANDLE hDevice, IN DWORD dwX, IN DWORD dwY,  INOUT LPWORD pPix );

/**
   \brief Returns temperature
   
   Allows the user to get current value of camera's temperature.

   @param[in] hDevice			handle of the specified device
   @param[in,out] pTemp			temperature
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetTemperature	(IN HANDLE hDevice, INOUT LPDWORD pTemp );

/**
   \brief Returns temperature of the chip and the housing separately
   
   Allows the user to get current value of camera's chip and camera's housing temperature.

   @param[in] hDevice			handle of the specified device
   @param[in,out] pThousing			temperature of the housing
   @param[in,out] pTchip			temperature of the chip
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetTemperatureEx	(IN HANDLE hDevice, INOUT float * pThousing, INOUT float * pTchip );

/**
   \brief Calibrates temperature 
   
   Allows the user to calibrate (maintain) temeratures of the chip and the housing.

	@param[in] hDevice			handle of the specified device
   @param[in] pThousing			target temperature of the housing
   @param[in] pTchip			target temperature of the chip
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmCalibrateTemperature	(IN HANDLE hDevice, IN float fThousing, IN float fTchip );

/**
   \brief Calibrate temperature 2P ???
   
   ???

   @param[in] hDevice			handle of the specified device
   @param[in] pThousing			target temperature of the housing
   @param[in] pTchip			target temperature of the chip
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmCalibrateTemperature2P (IN HANDLE hDevice, IN float fThousing, IN float fTchip );

/**
   \brief Sets cooling
   
   Allows the user to cool device to specific tagt temperature.

   @param[in] hDevice			handle of the specified device
   @param[in] bOn			on/off switch
   @param[in] fTargetCT			target temperature
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmSetCooling		(IN HANDLE hDevice, IN BOOL bOn, float fTargetCT);

/**
   \brief Sets flash mode

	Allows the user to set flash mode (more about flash mode here)

   @param[in] hDevice			handle of the specified device
   @param[in] bFlash			boolean value, true if flash mode should be turned on, false otherwise
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmSetFlashMode		(IN HANDLE hDevice, IN BOOL bFlash );

/**
   \brief Gets flash mode
   
   Allows the user to check if flash mode is on or not.

   @param[in] hDevice			handle of the specified device
   @param[in,out] bFlash			boolean value, true if flash mode is turned on, false otherwise
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetFlashMode		(IN HANDLE hDevice, INOUT BOOL* pbFlash );

/**
   \brief Set GPIO state (only for MU cameras)
   
   Allows the user to check if flash mode is on or not.

   @param[in] hDevice			handle of the specified device
   @param[in] dwGPIO			GPIO number
   @param[in] dwGPIOstate		GPIO state
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmSetGPIO		(IN HANDLE hDevice, IN DWORD dwGPIO, IN GPIO dwGPIOstate);

/**
   \brief Get GPIO state (only for MU cameras)
   
   Allows the user to check if flash mode is on or not.

   @param[in] hDevice			handle of the specified device
   @param[in] dwGPIO			GPIO number
   @param[out] dwGPIOstate		GPIO state (16 HSB GPI state, 16 LSB GPO state)
   @param[out] dwGPIlevel		GPI level
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetGPIO		(IN HANDLE hDevice, IN DWORD dwGPIO, IN DWORD * dwGPIOstate, IN DWORD * dwGPIlevel);

/**
   \brief Returns factory info
   
   Allows the user to retrieve information about the camera's manufacturer.

   @param[in] hDevice			handle of the specified device
   @param[in,out???] pCygnal			???
   @param[in,out???] pFPGA			FPGA type
   @param[in,out???] pCPU			???
   @param[in,out???] pUser			???
   @param[in,out???] pPdate			date
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetFactoryInfo	(IN HANDLE hDevice, LPSTR pCygnal, LPSTR pFPGA, LPSTR pCPU, LPSTR pUser, LPDWORD pPdate);

/**
   \brief Set Lookup table (LUT)
   
   Allows to set hardware conversion table from sensor pixel values to received bytes.
   E.g. Mapping from 12 bit data from sensor to 8 bit using Gamma curve.

   @param[in] hDevice			handle of the specified device
   @param[in] pLUT				pointer to LUT structure
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmSetLUT	(IN HANDLE hDevice, LPLUT lut);

/**
   \brief Corrects bad pixels

   @param[in] hDevice			handle of the specified device
   @param[in???] lpBPS			LPBADPIXELSTATISTICS, pointer to BADPIXELSTATISTICS structure
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmCorrectBadPixels		(IN HANDLE hDevice, LPBADPIXELSTATISTICS lpBPS);

/**
   \brief Maxium BPL size (in pixels)

   @param[in] hDevice			handle of the specified device
   @param[out???] size			number of pixels can be stored to camera 
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN mmGetMaxBPLSize(IN HANDLE hDevice, OUT DWORD * size);

/**
   \brief Parses bad pixel list

	Allows the user to parse string input representing bad pixels list.

   @param[in] hDevice			handle of the specified device
   @param[in] szText			string input
   @param[in] lpBPS			LPBADPIXELSTATISTICS, pointer to BADPIXELSTATISTICS structure
   @param[in] bColor			boolean value, true if color, false if black and white
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmParseBadPixelsList	(IN HANDLE hDevice, LPSTR szText, LPBADPIXELSTATISTICS lpBPS, BOOL bColor);

/**
   \brief Evaluate position of correction pixels in badpixel list.

	Check if correct pixels are used in bad pixel list for correction according to used binning/skiping.

   @param[in] hDevice			handle of the specified device
   @param[in] lpBPS				LPBADPIXELSTATISTICS, pointer to BADPIXELSTATISTICS structure
   @param[in] bColor			boolean value, true if color, false if black and white
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN mmEvaluateCorrectionPixels(IN HANDLE hDevice, LPBADPIXELSTATISTICS lpBPS, BOOL bColor);

/**
   \brief Releases bad pixel list
   
   Releases bad pixel list and frees memory allocated to BADPIXELSTATISTICS structure   

   @param[in] hDevice			handle of the specified device
   @param[in] lpBPS			LPBADPIXELSTATISTICS, pointer to BADPIXELSTATISTICS structure
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmReleaseBadPixelsList	(IN HANDLE hDevice, LPBADPIXELSTATISTICS lpBPS);

/**
   \brief Parses profile
   
   Parses string taken from .tm file into values used by mm40api.dll

   @param[in] hDevice			handle of the specified device
   @param[in] szText			string input
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmParseProfile			(IN HANDLE hDevice, LPSTR szText);


/**
   \brief Applies camera profile
   
   Allows the user to apply camera profile stored in FFS (Flash Filesystem). This function replaces some values used by dll with values stored in FFS.

   @param[in] hDevice			handle of the specified device
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmApplyCameraProfile	(IN HANDLE hDevice );

/**
   \brief Saves camera profile

	Allows the user to save modified camera profile to FFS.

   @param[in] hDevice			handle of the specified device
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmSaveCameraProfile	(IN HANDLE hDevice );

/**
   \brief Save CCD sensor bad pixels list

   @param[in] hDevice			???
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmSaveBadPixelsList (IN HANDLE hDevice );

/**
   \brief Set bad pixel list

   @param[in] hDevice			???
   @param[in] lpBPS			???
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmSetBadPixelsList (IN HANDLE hDevice, LPBADPIXELSTATISTICS lpBPS);

/**
   \brief Get bad pixel list

   @param[in] hDevice			???
   @param[in] lpBPS			???
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetBadPixelsList (IN HANDLE hDevice, LPBADPIXELSTATISTICS * lpBPS);

/**
   \brief Gets time stamp

   @param[in] hDevice			???
   @param[in] lpBPS			???
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetTimestamp (IN HANDLE hDevice, LPDWORD timestampsec, LPDWORD timestampusec);


/**
   \brief Get mm40api.dll Version
   
   @param[out] pSoftwareVersion			???
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetDLLVersion( 
						OUT LPDWORD pSoftwareVersion);	// Returned software version
/**
   \brief Get mm40dcam.sys version
   
   @param[in] drvHandle			???
   @param[out] pSoftwareVersion			???
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmGetSYSVersion( 
						IN HANDLE drvHandle, 			// driver handle
						OUT LPDWORD pSoftwareVersion);	// Returned software version

/**
   \brief Checks if user could execute command or update device.
   
   @param[in] hDevice			???
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmCheckPrivileges( IN HANDLE hDevice );

/**
   \brief Read camera register

   @param[in] hDevice			???
   @return MM40_OK on success, error value otherwise.
 */
MM40_API MM40_RETURN __cdecl mmReadReg (IN HANDLE hDevice, IN DWORD Offset, OUT LPDWORD pRegValue);

/** @name Deprecated
         List of deprecated functions.
   */
   /** @{ */
MM40_API MM40_RETURN __cdecl mm40GetNumberDevices ( 
			IN LPCSTR pFriendlyName,			// Device name
			OUT LPDWORD pNumberDevices);		// Ptr to number

MM40_API MM40_RETURN __cdecl mm40GetDevice ( 
			IN  DWORD nIndex,					// Index
			OUT LPDWORD pChipId);				// Ptr to chip ID

MM40_API MM40_RETURN __cdecl mm40Initialize(
			IN LPCSTR pFriendlyName,			// Device friendly name 	
			IN DWORD lChipId,					// chip ID
			OUT PHANDLE drvHandle);				// Returned driver handle

MM40_API MM40_RETURN __cdecl mm40Uninitialize			(IN HANDLE hDevice);

MM40_API MM40_RETURN __cdecl mm40GetDLLVersion( 
						OUT LPDWORD pSoftwareVersion);	// Returned software version

MM40_API MM40_RETURN __cdecl mm40GetFWVersion( 
						IN HANDLE drvHandle, 			// driver handle
						OUT LPDWORD  pSoftwareVersion);	// Returned firmware version

MM40_API MM40_RETURN __cdecl mm40GetSYSVersion( 
						IN HANDLE drvHandle, 			// driver handle
						OUT LPDWORD pSoftwareVersion);	// Returned software version

MM40_API MM40_RETURN __cdecl mm40GetModelName			(IN HANDLE hDevice, OUT LPDWORD lpdwModel );

MM40_API MM40_RETURN __cdecl mm40GetHWN2				(IN HANDLE hDevice, OUT PDWORD pExten );

MM40_API MM40_RETURN __cdecl mm40GetSerialNumber		(IN HANDLE hDevice, OUT LPDWORD lpdwSern );

MM40_API MM40_RETURN __cdecl mm40IsDeviceExist			(IN HANDLE hDevice);

VOID                 __cdecl mm40Abandon				(VOID);
MM40_API MM40_RETURN __cdecl mm40BusReset				(IN HANDLE hDevice );
MM40_API MM40_RETURN __cdecl mm40StartImageAcquisition	(IN HANDLE hDevice);
MM40_API MM40_RETURN __cdecl mm40StartImageShooting		(IN HANDLE hDevice);
MM40_API MM40_RETURN __cdecl mm40AcquireImage			(IN HANDLE hDevice, OUT PVOID *ppData );
MM40_API MM40_RETURN __cdecl mm40AcquireFirstImage		(IN HANDLE hDevice, OUT PVOID *ppData );
MM40_API MM40_RETURN __cdecl mm40StopImageAcquisition	(IN HANDLE hDevice);
MM40_API MM40_RETURN __cdecl mm40IsImageReady			(IN HANDLE hDevice);
MM40_API MM40_RETURN __cdecl mm40WaitForImage			(IN HANDLE hDevice, IN  DWORD dwTimeOut );
MM40_API MM40_RETURN __cdecl mm40GetLastTimeStamp		(IN HANDLE hDevice, PDWORD pSec, PDWORD pCycle );
MM40_API MM40_RETURN __cdecl mm40StartImageStreaming	(IN HANDLE hDevice, DWORD nPackets, DWORD dwRsrc, DWORD dwDescr, DWORD dwBufCnt);
MM40_API MM40_RETURN __cdecl mm40StopImageStreaming		(IN HANDLE hDevice);
MM40_API MM40_RETURN __cdecl mm40GetStreamingChunkSize	(IN HANDLE hDevice, INOUT PDWORD pdwChunkSize );
MM40_API MM40_RETURN __cdecl mm40StartImageCapture		(IN HANDLE hDevice, OUT PVOID *ppData );
MM40_API MM40_RETURN __cdecl mm40CaptureImage			(IN HANDLE hDevice, OUT PVOID *ppData );
MM40_API MM40_RETURN __cdecl mm40StopImageCapture		(IN HANDLE hDevice);
MM40_API MM40_RETURN __cdecl mm40GetModeCount			(IN HANDLE hDevice, OUT LPDWORD lpdwModeCount );
MM40_API MM40_RETURN __cdecl mm40SetMode				(IN HANDLE hDevice, IN  DWORD dwMode);
MM40_API MM40_RETURN __cdecl mm40SetModeEx				(IN HANDLE hDevice, IN  DWORD dwMode, IN DWORD dwX0, IN DWORD dwY0, IN DWORD dwCX, IN DWORD dwCY);
MM40_API MM40_RETURN __cdecl mm40GetMode				(IN HANDLE hDevice, OUT LPDWORD lpdwMode );
MM40_API MM40_RETURN __cdecl mm40SetExposure			(IN HANDLE hDevice, IN  DWORD dwExposure);
MM40_API MM40_RETURN __cdecl mm40GetExposure			(IN HANDLE hDevice, OUT LPDWORD lpdwExposure );
MM40_API MM40_RETURN __cdecl mm40SetTemperature			(IN HANDLE hDevice, IN  DWORD dwTemp );
MM40_API MM40_RETURN __cdecl mm40GetTemperature			(IN HANDLE hDevice, OUT LPDWORD lpdwTemp );
MM40_API MM40_RETURN __cdecl mm40GetCurrTemp			(IN HANDLE hDevice, OUT int * lpTChip, OUT int * lpTHousing );
MM40_API MM40_RETURN __cdecl mm40SetOffset				(IN HANDLE hDevice, IN  DWORD dwOffset );
MM40_API MM40_RETURN __cdecl mm40GetOffset				(IN HANDLE hDevice, OUT LPDWORD lpdwOffset );
MM40_API MM40_RETURN __cdecl mm40SetGain				(IN HANDLE hDevice, IN  DWORD dwGain );
MM40_API MM40_RETURN __cdecl mm40GetGain				(IN HANDLE hDevice, OUT LPDWORD lpdwGain );
MM40_API MM40_RETURN __cdecl mm40SetSerialNumber		(IN HANDLE hDevice, IN  DWORD dwSern );
MM40_API MM40_RETURN __cdecl mm40GetModeExtension		(IN HANDLE hDevice, OUT PDWORD pExten );
MM40_API MM40_RETURN __cdecl mm40GetModeShooting		(IN HANDLE hDevice );
MM40_API MM40_RETURN __cdecl mm40EnableTrigger			(IN HANDLE hDevice, IN BOOL bEnable, IN BOOL bNegative, IN BOOL bOutput );
MM40_API MM40_RETURN __cdecl mm40SetTrigger				(IN HANDLE hDevice, IN DWORD dwMode, IN BOOL bNegative );
MM40_API MM40_RETURN __cdecl mm40GetTrigger				(IN HANDLE hDevice, OUT DWORD *lpdwMode, OUT DWORD * lpbNegative );
MM40_API MM40_RETURN __cdecl mm40PingTrigger			(IN HANDLE hDevice );
MM40_API BOOL		 __cdecl mm40IsBufferOK					(IN HANDLE hDevice, IN LPVOID pBuf, IN LPDWORD pFrameNum );
MM40_API MM40_RETURN __cdecl mm40GetResolution( 
						IN HANDLE	hDevice,			// driver handle
						OUT PSIZE	pResolution,		// mode resolution
						OUT PDWORD	uBufwidth,			// line pitch in bytes
						OUT PDWORD	uF1offs,			// offset to the second field
						OUT PDWORD	uPixsize);			// bits per pixel

MM40_API MM40_RETURN __cdecl mm40ReadQlet( 
						IN HANDLE	hDevice,
						IN DWORD	Offset,
						OUT LPDWORD	pData);

MM40_API MM40_RETURN __cdecl mm40WriteQlet(
						IN HANDLE	hDevice,
						IN DWORD	Offset,
						IN DWORD	uData);

MM40_API MM40_RETURN __cdecl mm40ReadBlock( 
						IN HANDLE	hDevice,
						IN DWORD	Offset,
						IN DWORD    Count,
						OUT PVOID	pData);

MM40_API MM40_RETURN __cdecl mm40WriteBlock( 
						IN HANDLE	hDevice,
						IN DWORD	Offset,
						IN DWORD    Count,
						IN PVOID	pData);

MM40_API MM40_RETURN __cdecl mm40ReadBlockRaw( 
						IN HANDLE	hDevice,
						IN WORD		OffsetHi,
						IN DWORD	OffsetLo,
						IN DWORD    Count,
						OUT PVOID	pData);

MM40_API MM40_RETURN __cdecl mm40WriteBlockRaw( 
						IN HANDLE	hDevice,
						IN WORD		OffsetHi,
						IN DWORD	OffsetLo,
						IN DWORD    Count,
						IN PVOID	pData);


   /** @} */

//***********************Devices emun mask***********************
#define DEV_MASK_1394	0x1
#define DEV_MASK_USB20	0x2
#define DEV_MASK_PCIE	0x4
#define DEV_MASK_VIRT	0x8
#define DEV_MASK_USB30	0x10
#define DEV_MASK_ALL	0xFF

#ifdef __cplusplus
}
#endif

#endif /* __MM40API_H */
