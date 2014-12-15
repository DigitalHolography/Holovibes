#ifndef __MM40EXT_H
#define __MM40EXT_H

#ifdef __cplusplus
extern "C" {
#endif

//MM40_API MM40_RETURN __cdecl StartVideoStream			(IN HANDLE hDevice );
//MM40_API MM40_RETURN __cdecl StopVideoStream			(IN HANDLE hDevice );
//MM40_API MM40_RETURN __cdecl mmReadEEPROM1394				(IN HANDLE hDevice, IN DWORD wAddr, IN DWORD len, IN LPBYTE lpData );
//MM40_API MM40_RETURN __cdecl mmWriteEEPROM1394				(IN HANDLE hDevice, IN DWORD wAddr, IN DWORD len, INOUT LPBYTE lpData );
//MM40_API MM40_RETURN __cdecl mmEraseEEPROM1394				(IN HANDLE hDevice, IN DWORD wAddr, IN DWORD len );
//MM40_API MM40_RETURN __cdecl mmDetectEEPROMsize1394			(IN HANDLE hDevice );

#define	MM40EXT_VSUB	0x1000		// VSUB value
#define	MM40EXT_GAIN	0x1001		// CRB value
#define	MM40EXT_CRB		0x1002		// CRB value
#define	MM40EXT_DC		0x1003		// DC value
#define	MM40EXT_SAVE	0x1004		// SAVE mode
#define	MM40EXT_PTEN	0x1005		// PTEN
#define	MM40EXT_PELT	0x1006		// PELT
#define	MM40EXT_GA		0x1007		// GA
#define	MM40EXT_TADJ	0x1008		// TADJ
#define	MM40EXT_MAGIC	0x1009		// Camera magic number - Windows uses it for enumarating
#define	MM40EXT_GADJ2	0x1010		// GADJ2
#define	MM40EXT_MAILBOX	0x1011		// Mailbox register

//
// Definitions of Isoch Allocate Resources flags
//
#define RESOURCE_USED_IN_LISTENING              0x00000001
#define RESOURCE_USED_IN_TALKING                0x00000002
#define RESOURCE_BUFFERS_CIRCULAR               0x00000004
#define RESOURCE_STRIP_ADDITIONAL_QUADLETS      0x00000008
#define RESOURCE_TIME_STAMP_ON_COMPLETION       0x00000010
#define RESOURCE_SYNCH_ON_TIME                  0x00000020
#define RESOURCE_USE_PACKET_BASED               0x00000040
#define RESOURCE_VARIABLE_ISOCH_PAYLOAD         0x00000080 
#define RESOURCE_USE_MULTICHANNEL               0x00000100

//
// Definitions of Isoch Descriptor flags
//
#define DESCRIPTOR_SYNCH_ON_SY                  0x00000001
#define DESCRIPTOR_SYNCH_ON_TAG                 0x00000002
#define DESCRIPTOR_SYNCH_ON_TIME                0x00000004
#define DESCRIPTOR_USE_SY_TAG_IN_FIRST          0x00000008
#define DESCRIPTOR_TIME_STAMP_ON_COMPLETION     0x00000010
#define DESCRIPTOR_PRIORITY_TIME_DELIVERY       0x00000020
#define DESCRIPTOR_HEADER_SCATTER_GATHER        0x00000040
#define DESCRIPTOR_SYNCH_ON_ALL_TAGS            0x00000080

//
// Definitions of Isoch synchronization flags
//
#define SYNCH_ON_SY                             DESCRIPTOR_SYNCH_ON_SY
#define SYNCH_ON_TAG                            DESCRIPTOR_SYNCH_ON_TAG
#define SYNCH_ON_TIME                           DESCRIPTOR_SYNCH_ON_TIME

//
// Definitions of Speed flags used throughout 1394 Bus APIs
//
#define SPEED_FLAGS_100                         0x01
#define SPEED_FLAGS_200                         0x02
#define SPEED_FLAGS_400                         0x04
#define SPEED_FLAGS_800                         0x08
#define SPEED_FLAGS_1600                        0x10
#define SPEED_FLAGS_3200                        0x20


MM40_API MM40_RETURN __cdecl mmGetExtention			(IN HANDLE hDevice, IN DWORD dwExtension, OUT LPDWORD lpdwValue );
MM40_API MM40_RETURN __cdecl mmSetExtention			(IN HANDLE hDevice, IN DWORD dwExtension, IN DWORD dwValue );
MM40_API MM40_RETURN __cdecl mmGetExtentionDouble	(IN HANDLE hDevice, IN DWORD dwExtension, OUT double * lpValue );
MM40_API MM40_RETURN __cdecl mmSetExtentionDouble	(IN HANDLE hDevice, IN DWORD dwExtension, IN double Value );

enum MM_EXTENSION {
	/*************MR*************/
	EXT_VSUB			= 1000,
	EXT_DEFGAIN			= 1001,
	EXT_GAINOFFSET		= 1002,
	EXT_FPGAREG			= 1003,
	EXT_CPURAM			= 1004,
	EXT_ADCBPP			= 1005,
	EXT_PRESET			= 1006,
	EXT_FPGAVER			= 1007,
	/*************MS, MQ*************/
	EXT_OVERLAP			= 1008,	
	EXT_LUT				= 1009,
	EXT_LINLOG			= 1012,
	EXT_VERTICAL_FLIP	= 1013,
	EXT_HORIZONTAL_FLIP = 1014,
	/*************MX*************/
	EXT_XC_TTP			= 1015,
	EXT_XC_TKF			= 1016,
	EXT_XC_TKP			= 1017,
	EXT_XC_TKI			= 1018,
	EXT_XC_TKD			= 1019,
	EXT_XC_TMAX0		= 1020,
	EXT_XC_TMAX1		= 1021,
	EXT_XC_FTP			= 1022,
	EXT_XC_FKP			= 1023,
	EXT_XC_FKF			= 1024,
	EXT_XC_FMAX			= 1025,
	EXT_XH_TTP			= 1026,
	EXT_XH_TKF			= 1027,
	EXT_XH_TKP			= 1028,
	EXT_XH_TKI			= 1029,
	EXT_XH_TKD			= 1030,
	EXT_XH_TMAX0		= 1031,
	EXT_XH_TMAX1		= 1032,
	EXT_XH_FTP			= 1033,
	EXT_XH_FKP			= 1034,
	EXT_XH_FKF			= 1035,
	EXT_XH_FMAX			= 1036,

	EXT_X_TCCD1			= 1037,
	EXT_X_TCCD2			= 1038,
	EXT_X_TCAM1			= 1039,
	EXT_X_TCAM2			= 1040,
	EXT_X_TFAN			= 1041,
	EXT_X_TVPP			= 1042,
	EXT_X_TFANP			= 1043,
	EXT_X_TPCB			= 1044,
	EXT_XDO_TADJ		= 1045,	
	/*************MS*************/
	EXT_RAW_EN			= 1046,
	EXT_SET_DBNC_GPI	= 1047,
	EXT_EN_DBNC_GPI		= 1048,
	EXT_T0_DBNC			= 1049,
	EXT_T1_DBNC			= 1050,
	EXT_POL_DBNC		= 1051,
	/*************MS and MQ*************/
	EXT_RECENT_FRAME	= 1052,
	/*************EV76C560*************/
	EXT_ROI1_0C_1		= 1053,	
	EXT_ROI1_W_1		= 1054,
	EXT_ROI1_0L_1		= 1055,
	EXT_ROI1_H_1		= 1056,
	EXT_ROI1_0C_2		= 1057,	
	EXT_ROI1_W_2		= 1058,
	EXT_ROI1_0L_2		= 1059,
	EXT_ROI1_H_2		= 1060,
	EXT_ROI_EN			= 1061,
	/*************HDR*************/
	EXT_HDR				= 1070,
	EXT_HDR_T1			= 1071,
	EXT_HDR_T2			= 1072,
	EXT_HDR_KP1			= 1074,
	EXT_HDR_KP2			= 1075,
	EXT_HDR_KP_COUNT	= 1076,
	/************************************/
	EXT_FREQUENCY		= 1079,	
	/*************CMV Calib*************/
	EXT_CMV_FREQ_MIN	= 1080,	
	EXT_CMV_FREQ_MAX	= 1081,
	EXT_CMV_FREQ_MAX16	= 1082,
	EXT_CMV_FREQ_STEP	= 1083,
	EXT_CMV_FREQ		= 1085,
	EXT_CMV_FREQ_BL_OFFSET	= 1086,
	EXT_CMV_ADC_GAIN		= 1087,
	EXT_CMV_BL25_OFFSET		= 1090,
	EXT_CMV_BL25_SLOPE		= 1091,
	EXT_CMV_BL48_OFFSET		= 1092,
	EXT_CMV_BL48_SLOPE		= 1093,
	/*************E2V Calib*************/
	EXT_E2V_BL_1P			= 1088,
	EXT_E2V_BL_2P			= 1089,
	EXT_E2V_BL_2P_KP_EXP	= 1094,
	EXT_E2V_BL_2P_KP_OFFSET	= 1098,
	EXT_E2V_BL_1P_KP		= 1150,
	EXT_E2V_BL_2P_KP		= 1099,
	/*************ACTIVE AREA**********/
	EXT_ACTIVE_AREA       = 1095,
	EXT_ACTIVE_OFFSET_X   = 1096,
	EXT_ACTIVE_OFFSET_Y   = 1097,
	/*************MQ*************/
	EXT_UPDATE_PARAM_DIRECT = 1100,
	EXT_IS_U3V				= 1101,
	EXT_DEV_HARDWARE_REVISION  = 1102,     // hardware revision of camera
	EXT_IS_USB20				= 1103,
	/************CMV300 Calib***********/
	EXT_CMV300_FREQ_BL_TOP_OFFSET = 1203,
	EXT_CMV300_FREQ_BL_BOT_OFFSET = 1204,
	EXT_CMV300_ADC_GAIN			  = 1205,
	EXT_CMV300_FREQ_STEP		  = 1206,
	EXT_CMV300_FREQ				  = 1207,

	/*************MD*************/
	EXT_SELECT_TAP				= 1299,	
	EXT_H1LOC_POS				= 1300,	
	EXT_H1LOC_NEG				= 1301,
	EXT_H1LOC_POL				= 1302,
	EXT_H2LOC_POS				= 1303,	
	EXT_H2LOC_NEG				= 1304,
	EXT_H2LOC_POL				= 1305,
	EXT_HLLOC_POS				= 1306,	
	EXT_HLLOC_NEG				= 1307,
	EXT_HLLOC_POL				= 1308,
	EXT_RGLOC_POS				= 1309,	
	EXT_RGLOC_NEG				= 1310,
	EXT_RGLOC_POL				= 1311,
	EXT_SHLOC_SHD				= 1312,	
	EXT_SHLOC_SHP				= 1313,
	EXT_SHLOC_SHPW				= 1314,
	EXT_TAP_MULT				= 1315,
	EXT_TAP_OFFSET				= 1319,
	EXT_TAP_MULT_VGA			= 1316,
	EXT_CAM_CAPABILITY_E2V_CLK25	= 1323,
	EXT_CAM_CAPABILITY_E2V_STROBE_MASK	= 1324,
	EXT_CAM_CAPABILITY_CMV_H_ROI	= 1325,
	EXT_CAM_CAPABILITY_PACK_10G160	= 1326,
	EXT_DOUTP_POS				= 1327,
	EXT_DOUTP_NEG				= 1328,
	EXT_HDS_H1					= 1329,
	EXT_HDS_H2					= 1330,
	EXT_HDS_H3					= 1331,
	EXT_HDS_H4					= 1332,
	EXT_HDS_HL					= 1333,
	EXT_HDS_RG					= 1334,
	EXT_HSYNC0					= 1340,
	EXT_CAM_CAPABILITY_CMV_LESS_CH	= 1341,
	EXT_VSKIP					= 1342,
	EXT_DO_BPC					= 1400,	
};


#ifdef __cplusplus
}
#endif

#endif /* __MM40EXT_H */
