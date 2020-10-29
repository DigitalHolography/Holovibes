//
// Include file for accessing the NI-488.2 API
//
// Contains function prototypes and useful defined constants for calling
// NI-488 and NI-488.2 routines from a C/C++ application.
//
// Copyright 2007 National Instruments Corporation, All rights reserved.
//

#ifndef ___NI4882_H___ // ensure we are only included once
#define ___NI4882_H___

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h> // for wchar_t
#if defined (_WIN32)
#define NI488CC __stdcall
#else
#define NI488CC
#endif

//
// Handy constants for use by application programs
//

//
// Command codes
//
#define UNL  0x3f  // GPIB unlisten command
#define UNT  0x5f  // GPIB untalk command
#define GTL  0x01  // GPIB go to local
#define SDC  0x04  // GPIB selected device clear
#define PPC  0x05  // GPIB parallel poll configure
#define GET  0x08  // GPIB group execute trigger
#define TCT  0x09  // GPIB take control
#define LLO  0x11  // GPIB local lock out
#define DCL  0x14  // GPIB device clear
#define PPU  0x15  // GPIB parallel poll unconfigure
#define SPE  0x18  // GPIB serial poll enable
#define SPD  0x19  // GPIB serial poll disable
#define PPE  0x60  // GPIB parallel poll enable
#define PPD  0x70  // GPIB parallel poll disable

//
// GPIB status bit vector: global variable ibsta and wait mask
//
#define ERR   (1<<15)  // Error detected
#define TIMO  (1<<14)  // Timeout
#define END   (1<<13)  // EOI or EOS detected
#define SRQI  (1<<12)  // SRQ detected by CIC
#define RQS   (1<<11)  // Device needs service
#define CMPL  (1<<8)   // I/O completed
#define LOK   (1<<7)   // Local lockout state
#define REM   (1<<6)   // Remote state
#define CIC   (1<<5)   // Controller-in-Charge
#define ATN   (1<<4)   // Attention asserted
#define TACS  (1<<3)   // Talker active
#define LACS  (1<<2)   // Listener active
#define DTAS  (1<<1)   // Device trigger state
#define DCAS  (1<<0)   // Device clear state

//
// Error messages returned in global variable iberr
//
#define EDVR   0  // System error
#define ECIC   1  // Function requires GPIB board to be CIC
#define ENOL   2  // Write function detected no Listeners
#define EADR   3  // Interface board not addressed correctly
#define EARG   4  // Invalid argument to function call
#define ESAC   5  // Function requires GPIB board to be SAC
#define EABO   6  // I/O operation aborted
#define ENEB   7  // Non-existent interface board
#define EDMA   8  // Error performing DMA
#define EOIP  10  // I/O operation started before previous operation completed
#define ECAP  11  // No capability for intended operation
#define EFSO  12  // File system operation error
#define EBUS  14  // Command error during device call
#define ESTB  15  // Serial poll status byte lost
#define ESRQ  16  // SRQ remains asserted
#define ETAB  20  // The return buffer is full
#define ELCK  21  // Address or board is locked
#define EARM  22  // The ibnotify Callback failed to rearm
#define EHDL  23  // The input handle is invalid
#define EWIP  26  // Wait already in progress on input ud
#define ERST  27  // The event notification was cancelled due to a reset of the interface
#define EPWR  28  // The system or board has lost power or gone to standby

//
// Warning messages returned in global variable iberr
//
#define WCFG  24    // Configuration warning
#define ECFG  WCFG

//
// EOS mode bits
//
#define BIN   (1<<12)  // Eight bit compare
#define XEOS  (1<<11)  // Send END with EOS byte
#define REOS  (1<<10)  // Terminate read on EOS

//
// Timeout values and meanings
//
#define TNONE    0   // Infinite timeout (disabled)
#define T10us    1   // Timeout of 10 us (ideal)
#define T30us    2   // Timeout of 30 us (ideal)
#define T100us   3   // Timeout of 100 us (ideal)
#define T300us   4   // Timeout of 300 us (ideal)
#define T1ms     5   // Timeout of 1 ms (ideal)
#define T3ms     6   // Timeout of 3 ms (ideal)
#define T10ms    7   // Timeout of 10 ms (ideal)
#define T30ms    8   // Timeout of 30 ms (ideal)
#define T100ms   9   // Timeout of 100 ms (ideal)
#define T300ms  10   // Timeout of 300 ms (ideal)
#define T1s     11   // Timeout of 1 s (ideal)
#define T3s     12   // Timeout of 3 s (ideal)
#define T10s    13   // Timeout of 10 s (ideal)
#define T30s    14   // Timeout of 30 s (ideal)
#define T100s   15   // Timeout of 100 s (ideal)
#define T300s   16   // Timeout of 300 s (ideal)
#define T1000s  17   // Timeout of 1000 s (ideal)

//
// IBLN Constants
//
#define NO_SAD    0
#define ALL_SAD  -1

//
// The following constants are used for the second parameter of the
// ibconfig function.  They are the "option" selection codes.
//
#define IbcPAD             0x0001  // Primary Address
#define IbcSAD             0x0002  // Secondary Address
#define IbcTMO             0x0003  // Timeout Value
#define IbcEOT             0x0004  // Send EOI with last data byte?
#define IbcPPC             0x0005  // Parallel Poll Configure
#define IbcREADDR          0x0006  // Repeat Addressing
#define IbcAUTOPOLL        0x0007  // Disable Auto Serial Polling
#define IbcSC              0x000A  // Board is System Controller?
#define IbcSRE             0x000B  // Assert SRE on device calls?
#define IbcEOSrd           0x000C  // Terminate reads on EOS
#define IbcEOSwrt          0x000D  // Send EOI with EOS character
#define IbcEOScmp          0x000E  // Use 7 or 8-bit EOS compare
#define IbcEOSchar         0x000F  // The EOS character
#define IbcPP2             0x0010  // Use Parallel Poll Mode 2
#define IbcTIMING          0x0011  // NORMAL, HIGH, or VERY_HIGH timing
#define IbcDMA             0x0012  // Use DMA for I/O
#define IbcSendLLO         0x0017  // Enable/disable the sending of LLO
#define IbcSPollTime       0x0018  // Set the timeout value for serial polls
#define IbcPPollTime       0x0019  // Set the parallel poll length period
#define IbcEndBitIsNormal  0x001A  // Remove EOS from END bit of IBSTA
#define IbcUnAddr          0x001B  // Enable/disable device unaddressing
#define IbcHSCableLength   0x001F  // Length of cable specified for high speed timing
#define IbcIst             0x0020  // Set the IST bit
#define IbcRsv             0x0021  // Set the RSV byte
#define IbcLON             0x0022  // Enter listen only mode
#define IbcEOS             0x0025  // Macro for ibeos

//
// Constants that can be used (in addition to the ibconfig constants)
// when calling the ibask() function.
//
#define IbaPAD             IbcPAD
#define IbaSAD             IbcSAD
#define IbaTMO             IbcTMO
#define IbaEOT             IbcEOT
#define IbaPPC             IbcPPC
#define IbaREADDR          IbcREADDR
#define IbaAUTOPOLL        IbcAUTOPOLL
#define IbaSC              IbcSC
#define IbaSRE             IbcSRE
#define IbaEOSrd           IbcEOSrd
#define IbaEOSwrt          IbcEOSwrt
#define IbaEOScmp          IbcEOScmp
#define IbaEOSchar         IbcEOSchar
#define IbaPP2             IbcPP2
#define IbaTIMING          IbcTIMING
#define IbaDMA             IbcDMA
#define IbaSendLLO         IbcSendLLO
#define IbaSPollTime       IbcSPollTime
#define IbaPPollTime       IbcPPollTime
#define IbaEndBitIsNormal  IbcEndBitIsNormal
#define IbaUnAddr          IbcUnAddr
#define IbaHSCableLength   IbcHSCableLength
#define IbaIst             IbcIst
#define IbaRsv             IbcRsv
#define IbaLON             IbcLON
#define IbaSerialNumber    0x0023
#define IbaEOS             IbcEOS
#define IbaBNA             0x0200  // A device's access board

//
// Values used by the Send 488.2 command
//
#define NULLend  0x00  // Do nothing at the end of a transfer
#define NLend    0x01  // Send NL with EOI after a transfer
#define DABend   0x02  // Send EOI with the last DAB

//
// Value used by the 488.2 Receive command
//
#define STOPend  0x0100

//
// Address type (for 488.2 calls)
//
typedef short Addr4882_t;  // System dependent: must be 16 bits

//
// This macro can be used to easily create an entry in address list
// that is required by many of the 488.2 functions. The primary address goes in the
// lower 8-bits and the secondary address goes in the upper 8-bits.
//
#define MakeAddr(pad, sad) ((Addr4882_t)(((pad)&0xFF) | ((sad)<<8)))

//
// This value is used to terminate an address list.  It should be
// assigned to the last entry.
//
#ifndef NOADDR
#define NOADDR  ((Addr4882_t)((unsigned short)0xFFFF))
#endif

//
// The following two macros are used to "break apart" an address list
// entry.  They take an unsigned integer and return either the primary
// or secondary address stored in the integer.
//
#define  GetPAD(val)    ((val) & 0xFF)
#define  GetSAD(val)    (((val) >> 8) & 0xFF)

//
// iblines constants
//
#define  ValidEOI   ((short)0x0080)
#define  ValidATN   ((short)0x0040)
#define  ValidSRQ   ((short)0x0020)
#define  ValidREN   ((short)0x0010)
#define  ValidIFC   ((short)0x0008)
#define  ValidNRFD  ((short)0x0004)
#define  ValidNDAC  ((short)0x0002)
#define  ValidDAV   ((short)0x0001)
#define  BusEOI     ((short)0x8000)
#define  BusATN     ((short)0x4000)
#define  BusSRQ     ((short)0x2000)
#define  BusREN     ((short)0x1000)
#define  BusIFC     ((short)0x0800)
#define  BusNRFD    ((short)0x0400)
#define  BusNDAC    ((short)0x0200)
#define  BusDAV     ((short)0x0100)

//
// typedef for ibnotify callback
//
typedef int (NI488CC * GpibNotifyCallback_t)(int, unsigned long, unsigned long, unsigned long, void *);

//
// Compatibility layer for code using the regular process globals
//
extern unsigned long ibsta;
extern unsigned long iberr;
extern unsigned long ibcnt;
extern unsigned long ibcntl;

#if defined(UNICODE)
   #define ibfind(udname)        ibfindW(udname)
   #define ibrdf(ud, filename)   ibrdfW(ud, filename)
   #define ibwrtf(ud, filename)  ibwrtfW(ud, filename)
#else
   #define ibfind(udname)        ibfindA(udname)
   #define ibrdf(ud, filename)   ibrdfA(ud, filename)
   #define ibwrtf(ud, filename)  ibwrtfA(ud, filename)
#endif

//
// NI-488.2 DLL Functions
//

//
// NI-488 Function Prototypes
//
unsigned long NI488CC ibask    (int ud, int option, int * v);
unsigned long NI488CC ibcac    (int ud, int v);
unsigned long NI488CC ibclr    (int ud);
unsigned long NI488CC ibcmd    (int ud, const void * buf, size_t cnt);
unsigned long NI488CC ibcmda   (int ud, const void * buf, size_t cnt);
unsigned long NI488CC ibconfig (int ud, int option, int v);
int           NI488CC ibdev    (int boardID, int pad, int sad, int tmo, int eot, int eos);
unsigned long NI488CC ibexpert (int ud, int option, const void * Input, void * Output);
int           NI488CC ibfindA  (const char * udname);
int           NI488CC ibfindW  (const wchar_t * udname);
unsigned long NI488CC ibgts    (int ud, int v);
unsigned long NI488CC iblck    (int ud, int v, unsigned int LockWaitTime, void * Reserved);
unsigned long NI488CC iblines  (int ud, short * result);
unsigned long NI488CC ibln     (int ud, int pad, int sad, short * listen);
unsigned long NI488CC ibloc    (int ud);
unsigned long NI488CC ibnotify (int ud, int mask, GpibNotifyCallback_t Callback, void * RefData);
unsigned long NI488CC ibonl    (int ud, int v);
unsigned long NI488CC ibpct    (int ud);
unsigned long NI488CC ibppc    (int ud, int v);
unsigned long NI488CC ibrd     (int ud, void * buf, size_t cnt);
unsigned long NI488CC ibrda    (int ud, void * buf, size_t cnt);
unsigned long NI488CC ibrdfA   (int ud, const char * filename);
unsigned long NI488CC ibrdfW   (int ud, const wchar_t * filename);
unsigned long NI488CC ibrpp    (int ud, char * ppr);
unsigned long NI488CC ibrsp    (int ud, char * spr);
unsigned long NI488CC ibsic    (int ud);
unsigned long NI488CC ibstop   (int ud);
unsigned long NI488CC ibtrg    (int ud);
unsigned long NI488CC ibwait   (int ud, int mask);
unsigned long NI488CC ibwrt    (int ud, const void * buf, size_t cnt);
unsigned long NI488CC ibwrta   (int ud, const void * buf, size_t cnt);
unsigned long NI488CC ibwrtfA  (int ud, const char * filename);
unsigned long NI488CC ibwrtfW  (int ud, const wchar_t * filename);

//
// Aliases for config functions
//
#define ibdma(ud, v)  ibconfig((ud), IbcDMA, (v))
#define ibeos(ud, v)  ibconfig((ud), IbcEOS, (v))
#define ibeot(ud, v)  ibconfig((ud), IbcEOT, (v))
#define ibist(ud, v)  ibconfig((ud), IbcIst, (v))
#define ibpad(ud, v)  ibconfig((ud), IbcPAD, (v))
#define ibrsc(ud, v)  ibconfig((ud), IbcSC,  (v))
#define ibrsv(ud, v)  ibconfig((ud), IbcRsv, (v))
#define ibsad(ud, v)  ibconfig((ud), IbcSAD, (v))
#define ibsre(ud, v)  ibconfig((ud), IbcSRE, (v))
#define ibtmo(ud, v)  ibconfig((ud), IbcTMO, (v))

//
// Function to access process-wide GPIB global variables
//
unsigned long NI488CC Ibsta(void);
unsigned long NI488CC Iberr(void);
unsigned long NI488CC Ibcnt(void);

//
// Functions to access Thread-local GPIB global variables
//
unsigned long NI488CC ThreadIbsta(void);
unsigned long NI488CC ThreadIberr(void);
unsigned long NI488CC ThreadIbcnt(void);
#define ThreadIbcntl() ThreadIbcnt()

//
//  NI-488.2 Function Prototypes
//
void NI488CC AllSpoll       (int boardID, const Addr4882_t * addrlist, short * results);
void NI488CC DevClear       (int boardID, Addr4882_t addr);
void NI488CC DevClearList   (int boardID, const Addr4882_t * addrlist);
void NI488CC EnableLocal    (int boardID, const Addr4882_t * addrlist);
void NI488CC EnableRemote   (int boardID, const Addr4882_t * addrlist);
void NI488CC FindLstn       (int boardID, const Addr4882_t * addrlist, Addr4882_t * results, size_t limit);
void NI488CC FindRQS        (int boardID, const Addr4882_t * addrlist, short * dev_stat);
void NI488CC PPoll          (int boardID, short * result);
void NI488CC PPollConfig    (int boardID, Addr4882_t addr, int dataLine, int lineSense);
void NI488CC PPollUnconfig  (int boardID, const Addr4882_t * addrlist);
void NI488CC PassControl    (int boardID, Addr4882_t addr);
void NI488CC RcvRespMsg     (int boardID, void * buffer, size_t cnt, int Termination);
void NI488CC ReadStatusByte (int boardID, Addr4882_t addr, short * result);
void NI488CC Receive        (int boardID, Addr4882_t addr, void * buffer, size_t cnt, int Termination);
void NI488CC ReceiveSetup   (int boardID, Addr4882_t addr);
void NI488CC ResetSys       (int boardID, const Addr4882_t * addrlist);
void NI488CC Send           (int boardID, Addr4882_t addr, const void * databuf, size_t datacnt, int eotMode);
void NI488CC SendCmds       (int boardID, const void * buffer, size_t cnt);
void NI488CC SendDataBytes  (int boardID, const void * buffer, size_t cnt, int eot_mode);
void NI488CC SendIFC        (int boardID);
void NI488CC SendLLO        (int boardID);
void NI488CC SendList       (int boardID, const Addr4882_t * addrlist, const void * databuf, size_t datacnt, int eotMode);
void NI488CC SendSetup      (int boardID, const Addr4882_t * addrlist);
void NI488CC SetRWLS        (int boardID, const Addr4882_t * addrlist);
void NI488CC TestSRQ        (int boardID, short * result);
void NI488CC TestSys        (int boardID, const Addr4882_t * addrlist, short * results);
void NI488CC Trigger        (int boardID, Addr4882_t addr);
void NI488CC TriggerList    (int boardID, const Addr4882_t * addrlist);
void NI488CC WaitSRQ        (int boardID, short * result);

#ifdef __cplusplus
}
#endif

#endif // ___NI4882_H___
