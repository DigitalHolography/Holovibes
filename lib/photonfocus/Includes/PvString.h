// *****************************************************************************
//
//     Copyright (c) 2007, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************
//
// File Name....: PvString.h
//
// *****************************************************************************

#ifndef __PVSTRING_H__
#define __PVSTRING_H__


#include <PvBaseLib.h>
#include <PvTypes.h>


namespace PtUtilsLib
{
    class String;
}

class PV_BASE_API PvString
{
public:

    PvString();
    PvString( const PvString & aValue );
    PvString( const char * aValue );
    PvString( const PvUnicodeChar * aValue );

    virtual ~PvString();

    const PvString &operator = ( const PvString & aValue );
    const PvString &operator += ( const PvString & aValue );

    bool operator == ( const char *aValue ) const;
    bool operator != ( const char *aValue ) const;

	bool operator == ( const PvUnicodeChar *aValue ) const;
    bool operator != ( const PvUnicodeChar *aValue ) const;

    bool operator == ( const PvString & aValue ) const;
    bool operator != ( const PvString & aValue ) const;

	operator const char *() const;
    operator const PvUnicodeChar *() const;

    const char *GetAscii() const;
    const PvUnicodeChar *GetUnicode() const;

    unsigned int GetLength() const;

private:

	mutable PtUtilsLib::String *mThis;
};


#ifdef PV_DEBUG
    #include <PtUtilsLib/String.h>
#endif // PV_DEBUG


#endif // __PVSTRING_H__
