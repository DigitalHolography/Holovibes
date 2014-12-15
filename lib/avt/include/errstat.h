/*******************************************************************************
  Error status for FirePackage modules.
  c. kuehnel, intek, 10.10.2001
  Update 28.10.2010
*******************************************************************************/

#ifndef ERRSTAT_H
#define ERRSTAT_H

/* Lowest layer errors */
#define HALER_NOERROR           0
#define HALER_NOCARD            1               /* Card is not present */
#define HALER_NONTDEVICE        2               /* No logical Device */
#define HALER_NOMEM             3               /* Not enough memory */
#define HALER_MODE              4               /* Not allowed in this mode */
#define HALER_TIMEOUT           5               /* Timeout */
#define HALER_ALREADYSTARTED    6               /* Something is started */
#define HALER_NOTSTARTED        7               /* Not started */
#define HALER_BUSY              8               /* Busy at the moment */
#define HALER_NORESOURCES       9               /* No resources available */
#define HALER_NODATA           10               /* No data available */
#define HALER_NOACK            11               /* Didn't get acknowledge */
#define HALER_NOIRQ            12               /* Interruptinstallerror */
#define HALER_NOBUSRESET       13               /* Error waiting for busreset */
#define HALER_NOLICENSE        14               /* No license */
#define HALER_RCODEOTHER       15               /* RCode not RCODE_COMPLETE */
#define HALER_PENDING          16               /* Something still pending */
#define HALER_INPARMS          17               /* Input parameter range */
#define HALER_CHIPVERSION      18               /* Unrecognized chipversion */
#define HALER_HARDWARE         19               /* Hardware error */
#define HALER_NOTIMPLEMENTED   20               /* Not implemented */
#define HALER_CANCELLED        21               /* Cancelled */
#define HALER_NOTLOCKED        22               /* Memory is not locked */
#define HALER_GENERATIONCNT    23               /* Bus reset in between */
#define HALER_NOISOMANAGER     24               /* No IsoManager present */
#define HALER_NOBUSMANAGER     25               /* No BusManager present */
#define HALER_UNEXPECTED       26               /* Unexpected value */
#define HALER_REMOVED          27               /* Target was removed */
#define HALER_NOBUSRESOURCES   28               /* No ISO resources available */
#define HALER_DMAHALTED        29               /* DMA halted */
#define HALER_PHYSMEMNOT32     30               /* Physical memory > 2^32 */

/* Higher layer errors */
#define FCE_NOERROR               0             /* No Error */
#define FCE_ALREADYOPENED      1001             /* Something already opened */
#define FCE_NOTOPENED          1002             /* Need open before */
#define FCE_NODETAILS          1003             /* No details */
#define FCE_DRVNOTINSTALLED    1004             /* Driver not installed */
#define FCE_MISSINGBUFFERS     1005             /* Don't have buffers */
#define FCE_INPARMS            1006             /* Parameter error */
#define FCE_CREATEDEVICE       1007             /* Error creating WinDevice */
#define FCE_WINERROR           1008             /* Internal Windows error */
#define FCE_IOCTL              1009             /* Error DevIoCtl */
#define FCE_DRVRETURNLENGTH    1010             /* Wrong length return data */
#define FCE_INVALIDHANDLE      1011             /* Wrong handle */
#define FCE_NOTIMPLEMENTED     1012             /* Function not implemented */
#define FCE_DRVRUNNING         1013             /* Driver runs already */
#define FCE_STARTERROR         1014             /* Couldn't start */
#define FCE_INSTALLERROR       1015             /* Installation error */
#define FCE_DRVVERSION         1016             /* Driver has wrong version */
#define FCE_NODEADDRESS        1017             /* Wrong nodeaddress */
#define FCE_PARTIAL            1018             /* Partial info. copied */
#define FCE_NOMEM              1019             /* No memory */
#define FCE_NOTAVAILABLE       1020             /* Requested function not available */
#define FCE_NOTCONNECTED       1021             /* Not connected to target */
#define FCE_ADJUSTED           1022             /* A pararmeter had to be adjusted */

/* Error flags */
#define HALERF_RXHLTISO0         0x00000001     /* ISO-Channel 0 rx halted */
#define HALERF_RXHLTISO1         0x00000002     /* ISO-Channel 1 rx halted */
#define HALERF_RXHLTISO2         0x00000004     /* ISO-Channel 2 rx halted */
#define HALERF_RXHLTISO3         0x00000008     /* ISO-Channel 3 rx halted */
#define HALERF_RXHLTISO4         0x00000010     /* ISO-Channel 4 rx halted */
#define HALERF_RXHLTISO5         0x00000020     /* ISO-Channel 5 rx halted */
#define HALERF_RXHLTISO6         0x00000040     /* ISO-Channel 6 rx halted */
#define HALERF_RXHLTISO7         0x00000080     /* ISO-Channel 7 rx halted */
#define HALERF_ISORXACK          0x00000100     /* Error in iso rx ack code */
#define HALERF_ISORX             0x00004000     /* Unspecified ISO error */
#define HALERF_TXRESPONSE        0x00008000     /* Error sending response */
#define HALERF_ASYRX             0x00010000     /* Asynchronous receiption */
#define HALERF_ASYTX             0x00020000     /* Asynchronous transmission */
#define HALERF_PHYTIMEOUT        0x00040000     /* Timeout in phy */
#define HALERF_HDRERROR          0x00080000     /* Unknown header received */
#define HALERF_TCERROR           0x00100000     /* Transactioncode error */
#define HALERF_ATSTUCK           0x00200000     /* Asynchr. transm. stuck */
#define HALERF_GRFOVERFLOW       0x00400000     /* General rx fifo overflow */
#define HALERF_ITFUNDERFLOW      0x00800000     /* Isochr. Tx underflow */
#define HALERF_ATFUNDERFLOW      0x01000000     /* Asynchr. Tx underflow */
#define HALERF_PCIERROR          0x02000000     /* Error accessing PCI-Bus */
#define HALERF_ASYRXRESTART      0x04000000     /* Asy. Rx DMA was restarted */
#define HALERF_NOACCESSINFO      0x08000000     /* Ext. access not stored */
#define HALERF_SELFID            0x10000000     /* Error within SelfId packet */
#define HALERF_DMPORT            0x20000000     /* Data mover port error */
#define HALERF_ISOTX             0x40000000     /* Iso TX error */
#define HALERF_SNOOP             0x80000000     /* Snoop error */

/* License types */
#define HALLT_NOLICENSE          0              /* No license */
#define HALLT_CARDGUID           1              /* Firewire card */
#define HALLT_HARDDISK           2              /* Hard disk */
#define HALLT_ETHERNET           3              /* Ethernet adapter */
#define HALLT_VENDOR             4              /* Vendor global */
#define HALLT_DATELIMITED        5              /* Expires with date */
#define HALLT_DEVICE             6              /* Device specific */
#define HALLT_FRAMELIMITED       7              /* Limited by frames */

#endif

