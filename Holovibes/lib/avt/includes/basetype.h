/*******************************************************************************
  Headerfile for basetypes.
  c. kuehnel, intek, 29.11.2001
*******************************************************************************/

#ifndef BASETYPE_H
#define BASETYPE_H

#ifndef UINT64
#define UINT64 unsigned __int64 
#endif

#ifndef SINT64
#define SINT64 __int64 
#endif

#ifndef UINT32
#define UINT32 unsigned long
#endif

#ifndef SINT32
#define SINT32 long
#endif

#ifndef UINT16
#define UINT16 unsigned short
#endif

#ifndef SINT16
#define SINT16 short
#endif

#ifndef UINT8
#define UINT8 unsigned char
#endif

#ifndef SINT8
#define SINT8 char
#endif

#ifndef TRUE
#define TRUE            1
#endif

#ifndef FALSE
#define FALSE           0
#endif

#ifndef NULL
#define NULL 0
#endif

#ifndef NIL
#define NIL  ((void*)-1)
#endif

#ifndef FGHANDLE
#define FGHANDLE void*
#endif

#ifndef _UINT32HL
typedef struct
{
  UINT32        Low;
  UINT32        High;
}UINT32HL;
#define _UINT32HL
#endif

#ifndef USERINFO
typedef struct
{
  UINT16        YPos;
  UINT16        XPos;
  UINT16        YSize;
  UINT16        XSize;
  UINT16        ColorFormat;
  UINT16        ErrorCode;
}USERINFO;
#endif

#ifndef MAXUINT32
 #define MAXUINT32       0xFFFFFFFF
#endif

#endif
