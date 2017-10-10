// *****************************************************************************
//
//     Copyright (c) 2010, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PV_GENTYPES_H__
#define __PV_GENTYPES_H__


typedef enum 
{
	PvGenVisibilityBeginner = 0,
	PvGenVisibilityExpert,
	PvGenVisibilityGuru,
	PvGenVisibilityInvisible,
	PvGenVisibilityUndefined = 999,

} PvGenVisibility;


typedef enum 
{
	PvGenTypeInteger = 0,
	PvGenTypeEnum,
	PvGenTypeBoolean,
	PvGenTypeString,
	PvGenTypeCommand,
	PvGenTypeFloat,
    PvGenTypeRegister,
	PvGenTypeUndefined = 999,

} PvGenType;

typedef enum 
{
    PvGenRepresentationLinear = 0,            
    PvGenRepresentationLogarithmic,   
    PvGenRepresentationBoolean,       
    PvGenRepresentationPureNumber,    
    PvGenRepresentationHexNumber,
	PvGenRepresentationUndefined = 999,

} PvGenRepresentation;

typedef enum 
{
	PvGenNameSpaceStandard = 0,
	PvGenNameSpaceCustom,
	PvGenNameSpaceUndefined = 999,

} PvGenNameSpace;

typedef enum
{
    PvGenAccessModeReadOnly = 0,
    PvGenAccessModeReadWrite = 1,
    PvGenAccessModeWriteOnly = 2,
    PvGenAccessModeNotImplemented = 3,
    PvGenAccessModeNotAvailable = 4,
    PvGenAccessModeUndefined = 999,

} PvGenAccessMode;


#endif // __PV_GENTYPES_H__

