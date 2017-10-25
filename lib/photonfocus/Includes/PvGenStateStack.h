// *****************************************************************************
//
//     Copyright (c) 2011, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVGENSTATESTACK_H__
#define __PVGENSTATESTACK_H__

#include <PvGenICamLib.h>
#include <PvGenParameterArray.h>


class PvGenParameterArray;
class PvGenParameter;

namespace PvGenICamLib
{
    class GenStateStack;
}


class PV_GENICAM_API PvGenStateStack
{
public:

    PvGenStateStack( PvGenParameterArray *aParameters );
    virtual ~PvGenStateStack();

	PvResult SetIntegerValue( const PvString &aName, PvInt64 aValue );
	PvResult SetFloatValue( const PvString &aName, double aValue );
	PvResult SetEnumValue( const PvString &aName, const PvString &aValue );
	PvResult SetEnumValue( const PvString &aName, PvInt64 aValue );
	PvResult SetBooleanValue( const PvString &aName, bool aValue );
	PvResult SetStringValue( const PvString &aName, const PvString &aValue );

private:

    // Not implemented
	PvGenStateStack( const PvGenStateStack & );
	const PvGenStateStack &operator=( const PvGenStateStack & );

    PvGenICamLib::GenStateStack *mThis;
};


#ifdef PV_DEBUG
    #include <PvGenICamLib/GenStateStack.h>
#endif // PV_DEBUG


#endif // __PVGENSTATESTACK_H__


