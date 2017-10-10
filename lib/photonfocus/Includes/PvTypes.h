// *****************************************************************************
//
//     Copyright (c) 2007, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************
//
// File Name....: PvTypes.h
//
// *****************************************************************************

#ifndef __PVTYPES_H__
#define __PVTYPES_H__

typedef unsigned short		PvUInt16;
typedef int					PvInt32;
typedef unsigned int		PvUInt32;
typedef long long			PvInt64;
typedef unsigned long long	PvUInt64;
typedef unsigned char		PvUInt8;
typedef wchar_t				PvUnicodeChar;

#ifdef WIN32
#define PV_DEPRECATED_ALTERNATIVE(A) __declspec(deprecated("This method is deprecated. Consider using "##A))
#define PV_DEPRECATED __declspec(deprecated)
#define PV_DEPRECATED_MESSAGE(A) __declspec(deprecated(A))
#else
#define PV_DEPRECATED_MESSAGE(A)
#define PV_DEPRECATED_ALTERNATIVE(A)
#define PV_DEPRECATED
#endif

#endif // __PVTYPES_H__

