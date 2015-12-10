' -------------------------------------------------------------------------
'  Distributed by VXIplug&play Systems Alliance
'  Do not modify the contents of this file.
' -------------------------------------------------------------------------
'  Title   : VISA32.BAS
'  Date    : 01-14-2003
'  Purpose : Include file for the VISA Library 3.0 spec
' -------------------------------------------------------------------------

Global Const VI_SPEC_VERSION  = &H00300000&

' - Resource Template Functions and Operations ----------------------------

Declare Function viOpenDefaultRM Lib "VISA32.DLL" Alias "#141" (sesn As Long) As Long
Declare Function viGetDefaultRM Lib "VISA32.DLL" Alias "#128" (sesn As Long) As Long
Declare Function viFindRsrc Lib "VISA32.DLL" Alias "#129" (ByVal sesn As Long, ByVal expr As String, vi As Long, retCount As Long, ByVal desc As String) As Long
Declare Function viFindNext Lib "VISA32.DLL" Alias "#130" (ByVal vi As Long, ByVal desc As String) As Long
Declare Function viParseRsrc Lib "VISA32.DLL" Alias "#146" (ByVal sesn As Long, ByVal desc As String, intfType As Integer, intfNum As Integer) As Long
Declare Function viParseRsrcEx Lib "VISA32.DLL" Alias "#147" (ByVal sesn As Long, ByVal desc As String, intfType As Integer, intfNum As Integer, ByVal rsrcClass As String, ByVal expandedUnaliasedName As String, ByVal aliasIfExists As String) As Long
Declare Function viOpen Lib "VISA32.DLL" Alias "#131" (ByVal sesn As Long, ByVal viDesc As String, ByVal mode As Long, ByVal timeout As Long, vi As Long) As Long
Declare Function viClose Lib "VISA32.DLL" Alias "#132" (ByVal vi As Long) As Long
Declare Function viGetAttribute Lib "VISA32.DLL" Alias "#133" (ByVal vi As Long, ByVal attrName As Long, attrValue As Any) As Long
Declare Function viSetAttribute Lib "VISA32.DLL" Alias "#134" (ByVal vi As Long, ByVal attrName As Long, ByVal attrValue As Long) As Long
Declare Function viStatusDesc Lib "VISA32.DLL" Alias "#142" (ByVal vi As Long, ByVal status As Long, ByVal desc As String) As Long
Declare Function viLock Lib "VISA32.DLL" Alias "#144" (ByVal vi As Long, ByVal lockType As Long, ByVal timeout As Long, ByVal requestedKey As String, ByVal accessKey As String) As Long
Declare Function viUnlock Lib "VISA32.DLL" Alias "#145" (ByVal vi As Long) As Long
Declare Function viEnableEvent Lib "VISA32.DLL" Alias "#135" (ByVal vi As Long, ByVal eventType As Long, ByVal mechanism As Integer, ByVal context As Long) As Long
Declare Function viDisableEvent Lib "VISA32.DLL" Alias "#136" (ByVal vi As Long, ByVal eventType As Long, ByVal mechanism As Integer) As Long
Declare Function viDiscardEvents Lib "VISA32.DLL" Alias "#137" (ByVal vi As Long, ByVal eventType As Long, ByVal mechanism As Integer) As Long
Declare Function viWaitOnEvent Lib "VISA32.DLL" Alias "#138" (ByVal vi As Long, ByVal inEventType As Long, ByVal timeout As Long, outEventType As Long, outEventContext As Long) As Long

' - Basic I/O Operations --------------------------------------------------

Declare Function viRead Lib "VISA32.DLL" Alias "#256" (ByVal vi As Long, ByVal Buffer As String, ByVal count As Long, retCount As Long) As Long
Declare Function viReadToFile Lib "VISA32.DLL" Alias "#219" (ByVal vi As Long, ByVal filename As String, ByVal count As Long, retCount As Long) As Long
Declare Function viWrite Lib "VISA32.DLL" Alias "#257" (ByVal vi As Long, ByVal Buffer As String, ByVal count As Long, retCount As Long) As Long
Declare Function viWriteFromFile Lib "VISA32.DLL" Alias "#218" (ByVal vi As Long, ByVal filename As String, ByVal count As Long, retCount As Long) As Long
Declare Function viAssertTrigger Lib "VISA32.DLL" Alias "#258"(ByVal vi As Long, ByVal protocol As Integer) As Long
Declare Function viReadSTB Lib "VISA32.DLL" Alias "#259" (ByVal vi As Long, status As Integer) As Long
Declare Function viClear Lib "VISA32.DLL" Alias "#260" (ByVal vi As Long) As Long

' - Formatted and Buffered I/O Operations ---------------------------------

Declare Function viSetBuf Lib "VISA32.DLL" Alias "#267" (ByVal vi As Long, ByVal mask As Integer, ByVal bufSize As Long) As Long
Declare Function viFlush Lib "VISA32.DLL" Alias "#268" (ByVal vi As Long, ByVal mask As Integer) As Long
Declare Function viBufWrite Lib "VISA32.DLL" Alias "#202" (ByVal vi As Long, ByVal Buffer As String, ByVal count As Long, retCount As Long) As Long
Declare Function viBufRead Lib "VISA32.DLL" Alias "#203" (ByVal vi As Long, ByVal Buffer As String, ByVal count As Long, retCount As Long) As Long
Declare Function viVPrintf Lib "VISA32.DLL" Alias "#270" (ByVal vi As Long, ByVal writeFmt As String, params As Any) As Long
Declare Function viVSPrintf Lib "VISA32.DLL" Alias "#205" (ByVal vi As Long, ByVal Buffer As String, ByVal writeFmt As String, params As Any) As Long
Declare Function viVScanf Lib "VISA32.DLL" Alias "#272" (ByVal vi As Long, ByVal readFmt As String, params As Any) As Long
Declare Function viVSScanf Lib "VISA32.DLL" Alias "#207" (ByVal vi As Long, ByVal Buffer As String, ByVal readFmt As String, params As Any) As Long
Declare Function viVQueryf Lib "VISA32.DLL" Alias "#280" (ByVal vi As Long, ByVal writeFmt As String, ByVal readFmt As String, params As Any) As Long

' - Memory I/O Operations -------------------------------------------------

Declare Function viIn8 Lib "VISA32.DLL" Alias "#273" (ByVal vi As Long, ByVal accSpace As Integer, ByVal offset As Long, val8 As Byte) As Long
Declare Function viOut8 Lib "VISA32.DLL" Alias "#274" (ByVal vi As Long, ByVal accSpace As Integer, ByVal offset As Long, ByVal val8 As Byte) As Long
Declare Function viIn16 Lib "VISA32.DLL" Alias "#261" (ByVal vi As Long, ByVal accSpace As Integer, ByVal offset As Long, val16 As Integer) As Long
Declare Function viOut16 Lib "VISA32.DLL" Alias "#262" (ByVal vi As Long, ByVal accSpace As Integer, ByVal offset As Long, ByVal val16 As Integer) As Long
Declare Function viIn32 Lib "VISA32.DLL" Alias "#281" (ByVal vi As Long, ByVal accSpace As Integer, ByVal offset As Long, val32 As Long) As Long
Declare Function viOut32 Lib "VISA32.DLL" Alias "#282" (ByVal vi As Long, ByVal accSpace As Integer, ByVal offset As Long, ByVal val32 As Long) As Long
Declare Function viMoveIn8 Lib "VISA32.DLL" Alias "#283" (ByVal vi As Long, ByVal accSpace As Integer, ByVal offset As Long, ByVal length As Long, buf8 As Byte) As Long
Declare Function viMoveOut8 Lib "VISA32.DLL" Alias "#284" (ByVal vi As Long, ByVal accSpace As Integer, ByVal offset As Long, ByVal length As Long, buf8 As Byte) As Long
Declare Function viMoveIn16 Lib "VISA32.DLL" Alias "#285" (ByVal vi As Long, ByVal accSpace As Integer, ByVal offset As Long, ByVal length As Long, buf16 As Integer) As Long
Declare Function viMoveOut16 Lib "VISA32.DLL" Alias "#286" (ByVal vi As Long, ByVal accSpace As Integer, ByVal offset As Long, ByVal length As Long, buf16 As Integer) As Long
Declare Function viMoveIn32 Lib "VISA32.DLL" Alias "#287" (ByVal vi As Long, ByVal accSpace As Integer, ByVal offset As Long, ByVal length As Long, buf32 As Long) As Long
Declare Function viMoveOut32 Lib "VISA32.DLL" Alias "#288" (ByVal vi As Long, ByVal accSpace As Integer, ByVal offset As Long, ByVal length As Long, buf32 As Long) As Long
Declare Function viMove Lib "VISA32.DLL" Alias "#200" (ByVal vi As Long, ByVal srcSpace As Integer, ByVal srcOffset As Long, ByVal srcWidth As Integer, ByVal destSpace As Integer, ByVal destOffset As Long, ByVal destWidth As Integer, ByVal srcLength As Long) As Long
Declare Function viMapAddress Lib "VISA32.DLL" Alias "#263" (ByVal vi As Long, ByVal mapSpace As Integer, ByVal mapOffset As Long, ByVal mapSize As Long, ByVal accMode As Integer, ByVal suggested As Long, address As Long) As Long
Declare Function viUnmapAddress Lib "VISA32.DLL" Alias "#264" (ByVal vi As Long) As Long
Declare Sub viPeek8 Lib "VISA32.DLL" Alias "#275" (ByVal vi As Long, ByVal address As Long, val8 As Byte)
Declare Sub viPoke8 Lib "VISA32.DLL" Alias "#276" (ByVal vi As Long, ByVal address As Long, ByVal val8 As Byte)
Declare Sub viPeek16 Lib "VISA32.DLL" Alias "#265" (ByVal vi As Long, ByVal address As Long, val16 As Integer)
Declare Sub viPoke16 Lib "VISA32.DLL" Alias "#266" (ByVal vi As Long, ByVal address As Long, ByVal val16 As Integer)
Declare Sub viPeek32 Lib "VISA32.DLL" Alias "#289" (ByVal vi As Long, ByVal address As Long, val32 As Long)
Declare Sub viPoke32 Lib "VISA32.DLL" Alias "#290" (ByVal vi As Long, ByVal address As Long, ByVal val32 As Long)

' - Shared Memory Operations ---------------------------------------------- 

Declare Function viMemAlloc Lib "VISA32.DLL" Alias "#291" (ByVal vi As Long, ByVal memSize As Long, offset As Long) As Long
Declare Function viMemFree Lib "VISA32.DLL" Alias "#292" (ByVal vi As Long, ByVal offset As Long) As Long

' - Interface Specific Operations ----------------------------------------- 

Declare Function viGpibControlREN Lib "VISA32.DLL" Alias "#208" (ByVal vi As Long, ByVal mode As Integer) As Long
Declare Function viGpibControlATN Lib "VISA32.DLL" Alias "#210" (ByVal vi As Long, ByVal mode As Integer) As Long
Declare Function viGpibSendIFC Lib "VISA32.DLL" Alias "#211" (ByVal vi As Long) As Long
Declare Function viGpibCommand Lib "VISA32.DLL" Alias "#212" (ByVal vi As Long, ByVal Buffer As String, ByVal count As Long, retCount As Long) As Long
Declare Function viGpibPassControl Lib "VISA32.DLL" Alias "#213" (ByVal vi As Long, ByVal primAddr As Integer, ByVal secAddr As Integer) As Long
Declare Function viVxiCommandQuery Lib "VISA32.DLL" Alias "#209" (ByVal vi As Long, ByVal mode As Integer, ByVal devCmd As Long, devResponse As Long) As Long
Declare Function viAssertUtilSignal Lib "VISA32.DLL" Alias "#214" (ByVal vi As Long, ByVal line As Integer) As Long
Declare Function viAssertIntrSignal Lib "VISA32.DLL" Alias "#215" (ByVal vi As Long, ByVal mode As Integer, ByVal statusID As Long) As Long
Declare Function viMapTrigger Lib "VISA32.DLL" Alias "#216" (ByVal vi As Long, ByVal trigSrc As Integer, ByVal trigDest As Integer, ByVal mode As Integer) As Long
Declare Function viUnmapTrigger Lib "VISA32.DLL" Alias "#217" (ByVal vi As Long, ByVal trigSrc As Integer, ByVal trigDest As Integer) As Long
Declare Function viUsbControlOut Lib "VISA32.DLL" Alias "#293" (ByVal vi As Long, ByVal bmRequestType As Integer, ByVal bRequest As Integer, ByVal wValue As Integer, ByVal wIndex As Integer, ByVal wLength As Integer, buf As Byte) As Long
Declare Function viUsbControlIn Lib "VISA32.DLL" Alias "#294" (ByVal vi As Long, ByVal bmRequestType As Integer, ByVal bRequest As Integer, ByVal wValue As Integer, ByVal wIndex As Integer, ByVal wLength As Integer, buf As Byte, retCnt As Integer) As Long

' - Attributes ------------------------------------------------------------

Global Const VI_ATTR_RSRC_CLASS                  = &HBFFF0001&
Global Const VI_ATTR_RSRC_NAME                   = &HBFFF0002&
Global Const VI_ATTR_RSRC_IMPL_VERSION           = &H3FFF0003&
Global Const VI_ATTR_RSRC_LOCK_STATE             = &H3FFF0004&
Global Const VI_ATTR_MAX_QUEUE_LENGTH            = &H3FFF0005&
Global Const VI_ATTR_USER_DATA                   = &H3FFF0007&
Global Const VI_ATTR_FDC_CHNL                    = &H3FFF000D&
Global Const VI_ATTR_FDC_MODE                    = &H3FFF000F&
Global Const VI_ATTR_FDC_GEN_SIGNAL_EN           = &H3FFF0011&
Global Const VI_ATTR_FDC_USE_PAIR                = &H3FFF0013&
Global Const VI_ATTR_SEND_END_EN                 = &H3FFF0016&
Global Const VI_ATTR_TERMCHAR                    = &H3FFF0018&
Global Const VI_ATTR_TMO_VALUE                   = &H3FFF001A&
Global Const VI_ATTR_GPIB_READDR_EN              = &H3FFF001B&
Global Const VI_ATTR_IO_PROT                     = &H3FFF001C&
Global Const VI_ATTR_DMA_ALLOW_EN                = &H3FFF001E&
Global Const VI_ATTR_ASRL_BAUD                   = &H3FFF0021&
Global Const VI_ATTR_ASRL_DATA_BITS              = &H3FFF0022&
Global Const VI_ATTR_ASRL_PARITY                 = &H3FFF0023&
Global Const VI_ATTR_ASRL_STOP_BITS              = &H3FFF0024&
Global Const VI_ATTR_ASRL_FLOW_CNTRL             = &H3FFF0025&
Global Const VI_ATTR_RD_BUF_OPER_MODE            = &H3FFF002A&
Global Const VI_ATTR_RD_BUF_SIZE                 = &H3FFF002B&
Global Const VI_ATTR_WR_BUF_OPER_MODE            = &H3FFF002D&
Global Const VI_ATTR_WR_BUF_SIZE                 = &H3FFF002E&
Global Const VI_ATTR_SUPPRESS_END_EN             = &H3FFF0036&
Global Const VI_ATTR_TERMCHAR_EN                 = &H3FFF0038&
Global Const VI_ATTR_DEST_ACCESS_PRIV            = &H3FFF0039&
Global Const VI_ATTR_DEST_BYTE_ORDER             = &H3FFF003A&
Global Const VI_ATTR_SRC_ACCESS_PRIV             = &H3FFF003C&
Global Const VI_ATTR_SRC_BYTE_ORDER              = &H3FFF003D&
Global Const VI_ATTR_SRC_INCREMENT               = &H3FFF0040&
Global Const VI_ATTR_DEST_INCREMENT              = &H3FFF0041&
Global Const VI_ATTR_WIN_ACCESS_PRIV             = &H3FFF0045&
Global Const VI_ATTR_WIN_BYTE_ORDER              = &H3FFF0047&
Global Const VI_ATTR_GPIB_ATN_STATE              = &H3FFF0057&
Global Const VI_ATTR_GPIB_ADDR_STATE             = &H3FFF005C&
Global Const VI_ATTR_GPIB_CIC_STATE              = &H3FFF005E&
Global Const VI_ATTR_GPIB_NDAC_STATE             = &H3FFF0062&
Global Const VI_ATTR_GPIB_SRQ_STATE              = &H3FFF0067&
Global Const VI_ATTR_GPIB_SYS_CNTRL_STATE        = &H3FFF0068&
Global Const VI_ATTR_GPIB_HS488_CBL_LEN          = &H3FFF0069&
Global Const VI_ATTR_CMDR_LA                     = &H3FFF006B&
Global Const VI_ATTR_VXI_DEV_CLASS               = &H3FFF006C&
Global Const VI_ATTR_MAINFRAME_LA                = &H3FFF0070&
Global Const VI_ATTR_MANF_NAME                   = &HBFFF0072&
Global Const VI_ATTR_MODEL_NAME                  = &HBFFF0077&
Global Const VI_ATTR_VXI_VME_INTR_STATUS         = &H3FFF008B&
Global Const VI_ATTR_VXI_TRIG_STATUS             = &H3FFF008D&
Global Const VI_ATTR_VXI_VME_SYSFAIL_STATE       = &H3FFF0094&
Global Const VI_ATTR_WIN_BASE_ADDR               = &H3FFF0098&
Global Const VI_ATTR_WIN_SIZE                    = &H3FFF009A&
Global Const VI_ATTR_ASRL_AVAIL_NUM              = &H3FFF00AC&
Global Const VI_ATTR_MEM_BASE                    = &H3FFF00AD&
Global Const VI_ATTR_ASRL_CTS_STATE              = &H3FFF00AE&
Global Const VI_ATTR_ASRL_DCD_STATE              = &H3FFF00AF&
Global Const VI_ATTR_ASRL_DSR_STATE              = &H3FFF00B1&
Global Const VI_ATTR_ASRL_DTR_STATE              = &H3FFF00B2&
Global Const VI_ATTR_ASRL_END_IN                 = &H3FFF00B3&
Global Const VI_ATTR_ASRL_END_OUT                = &H3FFF00B4&
Global Const VI_ATTR_ASRL_REPLACE_CHAR           = &H3FFF00BE&
Global Const VI_ATTR_ASRL_RI_STATE               = &H3FFF00BF&
Global Const VI_ATTR_ASRL_RTS_STATE              = &H3FFF00C0&
Global Const VI_ATTR_ASRL_XON_CHAR               = &H3FFF00C1&
Global Const VI_ATTR_ASRL_XOFF_CHAR              = &H3FFF00C2&
Global Const VI_ATTR_WIN_ACCESS                  = &H3FFF00C3&
Global Const VI_ATTR_RM_SESSION                  = &H3FFF00C4&
Global Const VI_ATTR_VXI_LA                      = &H3FFF00D5&
Global Const VI_ATTR_MANF_ID                     = &H3FFF00D9&
Global Const VI_ATTR_MEM_SIZE                    = &H3FFF00DD&
Global Const VI_ATTR_MEM_SPACE                   = &H3FFF00DE&
Global Const VI_ATTR_MODEL_CODE                  = &H3FFF00DF&
Global Const VI_ATTR_SLOT                        = &H3FFF00E8&
Global Const VI_ATTR_INTF_INST_NAME              = &HBFFF00E9&
Global Const VI_ATTR_IMMEDIATE_SERV              = &H3FFF0100&
Global Const VI_ATTR_INTF_PARENT_NUM             = &H3FFF0101&
Global Const VI_ATTR_RSRC_SPEC_VERSION           = &H3FFF0170&
Global Const VI_ATTR_INTF_TYPE                   = &H3FFF0171&
Global Const VI_ATTR_GPIB_PRIMARY_ADDR           = &H3FFF0172&
Global Const VI_ATTR_GPIB_SECONDARY_ADDR         = &H3FFF0173&
Global Const VI_ATTR_RSRC_MANF_NAME              = &HBFFF0174&
Global Const VI_ATTR_RSRC_MANF_ID                = &H3FFF0175&
Global Const VI_ATTR_INTF_NUM                    = &H3FFF0176&
Global Const VI_ATTR_TRIG_ID                     = &H3FFF0177&
Global Const VI_ATTR_GPIB_REN_STATE              = &H3FFF0181&
Global Const VI_ATTR_GPIB_UNADDR_EN              = &H3FFF0184&
Global Const VI_ATTR_DEV_STATUS_BYTE             = &H3FFF0189&
Global Const VI_ATTR_FILE_APPEND_EN              = &H3FFF0192&
Global Const VI_ATTR_VXI_TRIG_SUPPORT            = &H3FFF0194&
Global Const VI_ATTR_TCPIP_ADDR                  = &HBFFF0195&
Global Const VI_ATTR_TCPIP_HOSTNAME              = &HBFFF0196&
Global Const VI_ATTR_TCPIP_PORT                  = &H3FFF0197&
Global Const VI_ATTR_TCPIP_DEVICE_NAME           = &HBFFF0199&
Global Const VI_ATTR_TCPIP_NODELAY               = &H3FFF019A&
Global Const VI_ATTR_TCPIP_KEEPALIVE             = &H3FFF019B&
Global Const VI_ATTR_4882_COMPLIANT              = &H3FFF019F&
Global Const VI_ATTR_USB_SERIAL_NUM              = &HBFFF01A0&
Global Const VI_ATTR_USB_INTFC_NUM               = &H3FFF01A1&
Global Const VI_ATTR_USB_PROTOCOL                = &H3FFF01A7&
Global Const VI_ATTR_USB_MAX_INTR_SIZE           = &H3FFF01AF&

Global Const VI_ATTR_JOB_ID                      = &H3FFF4006&
Global Const VI_ATTR_EVENT_TYPE                  = &H3FFF4010&
Global Const VI_ATTR_SIGP_STATUS_ID              = &H3FFF4011&
Global Const VI_ATTR_RECV_TRIG_ID                = &H3FFF4012&
Global Const VI_ATTR_INTR_STATUS_ID              = &H3FFF4023&
Global Const VI_ATTR_STATUS                      = &H3FFF4025&
Global Const VI_ATTR_RET_COUNT                   = &H3FFF4026&
Global Const VI_ATTR_BUFFER                      = &H3FFF4027&
Global Const VI_ATTR_RECV_INTR_LEVEL             = &H3FFF4041&
Global Const VI_ATTR_OPER_NAME                   = &HBFFF4042&
Global Const VI_ATTR_GPIB_RECV_CIC_STATE         = &H3FFF4193&
Global Const VI_ATTR_RECV_TCPIP_ADDR             = &HBFFF4198&
Global Const VI_ATTR_USB_RECV_INTR_SIZE          = &H3FFF41B0&
Global Const VI_ATTR_USB_RECV_INTR_DATA          = &HBFFF41B1&

' - Event Types -----------------------------------------------------------

Global Const VI_EVENT_IO_COMPLETION              = &H3FFF2009&
Global Const VI_EVENT_TRIG                       = &HBFFF200A&
Global Const VI_EVENT_SERVICE_REQ                = &H3FFF200B&
Global Const VI_EVENT_CLEAR                      = &H3FFF200D&
Global Const VI_EVENT_EXCEPTION                  = &HBFFF200E&
Global Const VI_EVENT_GPIB_CIC                   = &H3FFF2012&
Global Const VI_EVENT_GPIB_TALK                  = &H3FFF2013&
Global Const VI_EVENT_GPIB_LISTEN                = &H3FFF2014&
Global Const VI_EVENT_VXI_VME_SYSFAIL            = &H3FFF201D&
Global Const VI_EVENT_VXI_VME_SYSRESET           = &H3FFF201E&
Global Const VI_EVENT_VXI_SIGP                   = &H3FFF2020&
Global Const VI_EVENT_VXI_VME_INTR               = &HBFFF2021&
Global Const VI_EVENT_TCPIP_CONNECT              = &H3FFF2036&
Global Const VI_EVENT_USB_INTR                   = &H3FFF2037&

Global Const VI_ALL_ENABLED_EVENTS               = &H3FFF7FFF&

' - Completion and Error Codes --------------------------------------------

Global Const VI_SUCCESS                          = &H0&
Global Const VI_SUCCESS_EVENT_EN                 = &H3FFF0002&
Global Const VI_SUCCESS_EVENT_DIS                = &H3FFF0003&
Global Const VI_SUCCESS_QUEUE_EMPTY              = &H3FFF0004&
Global Const VI_SUCCESS_TERM_CHAR                = &H3FFF0005&
Global Const VI_SUCCESS_MAX_CNT                  = &H3FFF0006&
Global Const VI_SUCCESS_DEV_NPRESENT             = &H3FFF007D&
Global Const VI_SUCCESS_TRIG_MAPPED              = &H3FFF007E&
Global Const VI_SUCCESS_QUEUE_NEMPTY             = &H3FFF0080&
Global Const VI_SUCCESS_NCHAIN                   = &H3FFF0098&
Global Const VI_SUCCESS_NESTED_SHARED            = &H3FFF0099&
Global Const VI_SUCCESS_NESTED_EXCLUSIVE         = &H3FFF009A&
Global Const VI_SUCCESS_SYNC                     = &H3FFF009B&

Global Const VI_WARN_QUEUE_OVERFLOW              = &H3FFF000C&
Global Const VI_WARN_CONFIG_NLOADED              = &H3FFF0077&
Global Const VI_WARN_NULL_OBJECT                 = &H3FFF0082&
Global Const VI_WARN_NSUP_ATTR_STATE             = &H3FFF0084&
Global Const VI_WARN_UNKNOWN_STATUS              = &H3FFF0085&
Global Const VI_WARN_NSUP_BUF                    = &H3FFF0088&
Global Const VI_WARN_EXT_FUNC_NIMPL              = &H3FFF00A9&

Global Const VI_ERROR_SYSTEM_ERROR               = &HBFFF0000&
Global Const VI_ERROR_INV_OBJECT                 = &HBFFF000E&
Global Const VI_ERROR_RSRC_LOCKED                = &HBFFF000F&
Global Const VI_ERROR_INV_EXPR                   = &HBFFF0010&
Global Const VI_ERROR_RSRC_NFOUND                = &HBFFF0011&
Global Const VI_ERROR_INV_RSRC_NAME              = &HBFFF0012&
Global Const VI_ERROR_INV_ACC_MODE               = &HBFFF0013&
Global Const VI_ERROR_TMO                        = &HBFFF0015&
Global Const VI_ERROR_CLOSING_FAILED             = &HBFFF0016&
Global Const VI_ERROR_INV_DEGREE                 = &HBFFF001B&
Global Const VI_ERROR_INV_JOB_ID                 = &HBFFF001C&
Global Const VI_ERROR_NSUP_ATTR                  = &HBFFF001D&
Global Const VI_ERROR_NSUP_ATTR_STATE            = &HBFFF001E&
Global Const VI_ERROR_ATTR_READONLY              = &HBFFF001F&
Global Const VI_ERROR_INV_LOCK_TYPE              = &HBFFF0020&
Global Const VI_ERROR_INV_ACCESS_KEY             = &HBFFF0021&
Global Const VI_ERROR_INV_EVENT                  = &HBFFF0026&
Global Const VI_ERROR_INV_MECH                   = &HBFFF0027&
Global Const VI_ERROR_HNDLR_NINSTALLED           = &HBFFF0028&
Global Const VI_ERROR_INV_HNDLR_REF              = &HBFFF0029&
Global Const VI_ERROR_INV_CONTEXT                = &HBFFF002A&
Global Const VI_ERROR_NENABLED                   = &HBFFF002F&
Global Const VI_ERROR_ABORT                      = &HBFFF0030&
Global Const VI_ERROR_RAW_WR_PROT_VIOL           = &HBFFF0034&
Global Const VI_ERROR_RAW_RD_PROT_VIOL           = &HBFFF0035&
Global Const VI_ERROR_OUTP_PROT_VIOL             = &HBFFF0036&
Global Const VI_ERROR_INP_PROT_VIOL              = &HBFFF0037&
Global Const VI_ERROR_BERR                       = &HBFFF0038&
Global Const VI_ERROR_IN_PROGRESS                = &HBFFF0039&
Global Const VI_ERROR_INV_SETUP                  = &HBFFF003A&
Global Const VI_ERROR_QUEUE_ERROR                = &HBFFF003B&
Global Const VI_ERROR_ALLOC                      = &HBFFF003C&
Global Const VI_ERROR_INV_MASK                   = &HBFFF003D&
Global Const VI_ERROR_IO                         = &HBFFF003E&
Global Const VI_ERROR_INV_FMT                    = &HBFFF003F&
Global Const VI_ERROR_NSUP_FMT                   = &HBFFF0041&
Global Const VI_ERROR_LINE_IN_USE                = &HBFFF0042&
Global Const VI_ERROR_NSUP_MODE                  = &HBFFF0046&
Global Const VI_ERROR_SRQ_NOCCURRED              = &HBFFF004A&
Global Const VI_ERROR_INV_SPACE                  = &HBFFF004E&
Global Const VI_ERROR_INV_OFFSET                 = &HBFFF0051&
Global Const VI_ERROR_INV_WIDTH                  = &HBFFF0052&
Global Const VI_ERROR_NSUP_OFFSET                = &HBFFF0054&
Global Const VI_ERROR_NSUP_VAR_WIDTH             = &HBFFF0055&
Global Const VI_ERROR_WINDOW_NMAPPED             = &HBFFF0057&
Global Const VI_ERROR_RESP_PENDING               = &HBFFF0059&
Global Const VI_ERROR_NLISTENERS                 = &HBFFF005F&
Global Const VI_ERROR_NCIC                       = &HBFFF0060&
Global Const VI_ERROR_NSYS_CNTLR                 = &HBFFF0061&
Global Const VI_ERROR_NSUP_OPER                  = &HBFFF0067&
Global Const VI_ERROR_INTR_PENDING               = &HBFFF0068&
Global Const VI_ERROR_ASRL_PARITY                = &HBFFF006A&
Global Const VI_ERROR_ASRL_FRAMING               = &HBFFF006B&
Global Const VI_ERROR_ASRL_OVERRUN               = &HBFFF006C&
Global Const VI_ERROR_TRIG_NMAPPED               = &HBFFF006E&
Global Const VI_ERROR_NSUP_ALIGN_OFFSET          = &HBFFF0070&
Global Const VI_ERROR_USER_BUF                   = &HBFFF0071&
Global Const VI_ERROR_RSRC_BUSY                  = &HBFFF0072&
Global Const VI_ERROR_NSUP_WIDTH                 = &HBFFF0076&
Global Const VI_ERROR_INV_PARAMETER              = &HBFFF0078&
Global Const VI_ERROR_INV_PROT                   = &HBFFF0079&
Global Const VI_ERROR_INV_SIZE                   = &HBFFF007B&
Global Const VI_ERROR_WINDOW_MAPPED              = &HBFFF0080&
Global Const VI_ERROR_NIMPL_OPER                 = &HBFFF0081&
Global Const VI_ERROR_INV_LENGTH                 = &HBFFF0083&
Global Const VI_ERROR_INV_MODE                   = &HBFFF0091&
Global Const VI_ERROR_SESN_NLOCKED               = &HBFFF009C&
Global Const VI_ERROR_MEM_NSHARED                = &HBFFF009D&
Global Const VI_ERROR_LIBRARY_NFOUND             = &HBFFF009E&
Global Const VI_ERROR_NSUP_INTR                  = &HBFFF009F&
Global Const VI_ERROR_INV_LINE                   = &HBFFF00A0&
Global Const VI_ERROR_FILE_ACCESS                = &HBFFF00A1&
Global Const VI_ERROR_FILE_IO                    = &HBFFF00A2&
Global Const VI_ERROR_NSUP_LINE                  = &HBFFF00A3&
Global Const VI_ERROR_NSUP_MECH                  = &HBFFF00A4&
Global Const VI_ERROR_INTF_NUM_NCONFIG           = &HBFFF00A5&
Global Const VI_ERROR_CONN_LOST                  = &HBFFF00A6&
Global Const VI_ERROR_MACHINE_NAVAIL             = &HBFFF00A7&
Global Const VI_ERROR_NPERMISSION                = &HBFFF00A8&

' - Other VISA Definitions ------------------------------------------------

Global Const VI_FIND_BUFLEN                      = 256

Global Const VI_NULL                             = 0
Global Const VI_TRUE                             = 1
Global Const VI_FALSE                            = 0

Global Const VI_INTF_GPIB                        = 1
Global Const VI_INTF_VXI                         = 2
Global Const VI_INTF_GPIB_VXI                    = 3
Global Const VI_INTF_ASRL                        = 4
Global Const VI_INTF_PXI                         = 5
Global Const VI_INTF_TCPIP                       = 6
Global Const VI_INTF_USB                         = 7

Global Const VI_PROT_NORMAL                      = 1
Global Const VI_PROT_FDC                         = 2
Global Const VI_PROT_HS488                       = 3
Global Const VI_PROT_4882_STRS                   = 4
Global Const VI_PROT_USBTMC_VENDOR               = 5

Global Const VI_FDC_NORMAL                       = 1
Global Const VI_FDC_STREAM                       = 2

Global Const VI_LOCAL_SPACE                      = 0
Global Const VI_A16_SPACE                        = 1
Global Const VI_A24_SPACE                        = 2
Global Const VI_A32_SPACE                        = 3
Global Const VI_OPAQUE_SPACE                     = &HFFFF

Global Const VI_UNKNOWN_LA                       = -1
Global Const VI_UNKNOWN_SLOT                     = -1
Global Const VI_UNKNOWN_LEVEL                    = -1

Global Const VI_QUEUE                            = 1
Global Const VI_ALL_MECH                         = &HFFFF

Global Const VI_TRIG_ALL                         = -2
Global Const VI_TRIG_SW                          = -1
Global Const VI_TRIG_TTL0                        = 0
Global Const VI_TRIG_TTL1                        = 1
Global Const VI_TRIG_TTL2                        = 2
Global Const VI_TRIG_TTL3                        = 3
Global Const VI_TRIG_TTL4                        = 4
Global Const VI_TRIG_TTL5                        = 5
Global Const VI_TRIG_TTL6                        = 6
Global Const VI_TRIG_TTL7                        = 7
Global Const VI_TRIG_ECL0                        = 8
Global Const VI_TRIG_ECL1                        = 9
Global Const VI_TRIG_PANEL_IN                    = 27
Global Const VI_TRIG_PANEL_OUT                   = 28

Global Const VI_TRIG_PROT_DEFAULT                = 0
Global Const VI_TRIG_PROT_ON                     = 1
Global Const VI_TRIG_PROT_OFF                    = 2
Global Const VI_TRIG_PROT_SYNC                   = 5

Global Const VI_READ_BUF                         = 1
Global Const VI_WRITE_BUF                        = 2
Global Const VI_READ_BUF_DISCARD                 = 4
Global Const VI_WRITE_BUF_DISCARD                = 8
Global Const VI_IO_IN_BUF                        = 16
Global Const VI_IO_OUT_BUF                       = 32
Global Const VI_IO_IN_BUF_DISCARD                = 64
Global Const VI_IO_OUT_BUF_DISCARD               = 128

Global Const VI_FLUSH_ON_ACCESS                  = 1
Global Const VI_FLUSH_WHEN_FULL                  = 2
Global Const VI_FLUSH_DISABLE                    = 3

Global Const VI_NMAPPED                          = 1
Global Const VI_USE_OPERS                        = 2
Global Const VI_DEREF_ADDR                       = 3

Global Const VI_TMO_IMMEDIATE                    = &H0&
Global Const VI_TMO_INFINITE                     = &HFFFFFFFF&

Global Const VI_NO_LOCK                          = 0
Global Const VI_EXCLUSIVE_LOCK                   = 1
Global Const VI_SHARED_LOCK                      = 2
Global Const VI_LOAD_CONFIG                      = 4

Global Const VI_NO_SEC_ADDR                      = &HFFFF

Global Const VI_ASRL_PAR_NONE                    = 0 
Global Const VI_ASRL_PAR_ODD                     = 1 
Global Const VI_ASRL_PAR_EVEN                    = 2 
Global Const VI_ASRL_PAR_MARK                    = 3 
Global Const VI_ASRL_PAR_SPACE                   = 4 

Global Const VI_ASRL_STOP_ONE                    = 10 
Global Const VI_ASRL_STOP_ONE5                   = 15
Global Const VI_ASRL_STOP_TWO                    = 20 

Global Const VI_ASRL_FLOW_NONE                   = 0 
Global Const VI_ASRL_FLOW_XON_XOFF               = 1 
Global Const VI_ASRL_FLOW_RTS_CTS                = 2 
Global Const VI_ASRL_FLOW_DTR_DSR                = 4

Global Const VI_ASRL_END_NONE                    = 0 
Global Const VI_ASRL_END_LAST_BIT                = 1 
Global Const VI_ASRL_END_TERMCHAR                = 2 
Global Const VI_ASRL_END_BREAK                   = 3 

Global Const VI_STATE_ASSERTED                   = 1
Global Const VI_STATE_UNASSERTED                 = 0
Global Const VI_STATE_UNKNOWN                    = -1
 
Global Const VI_BIG_ENDIAN                       = 0
Global Const VI_LITTLE_ENDIAN                    = 1
 
Global Const VI_DATA_PRIV                        = 0
Global Const VI_DATA_NPRIV                       = 1
Global Const VI_PROG_PRIV                        = 2
Global Const VI_PROG_NPRIV                       = 3
Global Const VI_BLCK_PRIV                        = 4
Global Const VI_BLCK_NPRIV                       = 5
Global Const VI_D64_PRIV                         = 6
Global Const VI_D64_NPRIV                        = 7
 
Global Const VI_WIDTH_8                          = 1
Global Const VI_WIDTH_16                         = 2
Global Const VI_WIDTH_32                         = 4

Global Const VI_GPIB_REN_DEASSERT                = 0
Global Const VI_GPIB_REN_ASSERT                  = 1
Global Const VI_GPIB_REN_DEASSERT_GTL            = 2
Global Const VI_GPIB_REN_ASSERT_ADDRESS          = 3
Global Const VI_GPIB_REN_ASSERT_LLO              = 4
Global Const VI_GPIB_REN_ASSERT_ADDRESS_LLO      = 5
Global Const VI_GPIB_REN_ADDRESS_GTL             = 6

Global Const VI_GPIB_ATN_DEASSERT                = 0
Global Const VI_GPIB_ATN_ASSERT                  = 1
Global Const VI_GPIB_ATN_DEASSERT_HANDSHAKE      = 2
Global Const VI_GPIB_ATN_ASSERT_IMMEDIATE        = 3

Global Const VI_GPIB_HS488_DISABLED              = 0
Global Const VI_GPIB_HS488_NIMPL                 = -1

Global Const VI_GPIB_UNADDRESSED                 = 0
Global Const VI_GPIB_TALKER                      = 1
Global Const VI_GPIB_LISTENER                    = 2

Global Const VI_VXI_CMD16                        = &H0200
Global Const VI_VXI_CMD16_RESP16                 = &H0202
Global Const VI_VXI_RESP16                       = &H0002
Global Const VI_VXI_CMD32                        = &H0400
Global Const VI_VXI_CMD32_RESP16                 = &H0402
Global Const VI_VXI_CMD32_RESP32                 = &H0404
Global Const VI_VXI_RESP32                       = &H0004

Global Const VI_ASSERT_SIGNAL                    = -1
Global Const VI_ASSERT_USE_ASSIGNED              = 0
Global Const VI_ASSERT_IRQ1                      = 1
Global Const VI_ASSERT_IRQ2                      = 2
Global Const VI_ASSERT_IRQ3                      = 3
Global Const VI_ASSERT_IRQ4                      = 4
Global Const VI_ASSERT_IRQ5                      = 5
Global Const VI_ASSERT_IRQ6                      = 6
Global Const VI_ASSERT_IRQ7                      = 7

Global Const VI_UTIL_ASSERT_SYSRESET             = 1
Global Const VI_UTIL_ASSERT_SYSFAIL              = 2
Global Const VI_UTIL_DEASSERT_SYSFAIL            = 3

Global Const VI_VXI_CLASS_MEMORY                 = 0
Global Const VI_VXI_CLASS_EXTENDED               = 1
Global Const VI_VXI_CLASS_MESSAGE                = 2
Global Const VI_VXI_CLASS_REGISTER               = 3
Global Const VI_VXI_CLASS_OTHER                  = 4

' - Backward Compatibility Macros -----------------------------------------

Global Const VI_ERROR_INV_SESSION                = &HBFFF000E&
Global Const VI_INFINITE                         = &HFFFFFFFF&

Global Const VI_NORMAL                           = 1
Global Const VI_FDC                              = 2
Global Const VI_HS488                            = 3
Global Const VI_ASRL488                          = 4

Global Const VI_ASRL_IN_BUF                      = 16
Global Const VI_ASRL_OUT_BUF                     = 32
Global Const VI_ASRL_IN_BUF_DISCARD              = 64
Global Const VI_ASRL_OUT_BUF_DISCARD             = 128
