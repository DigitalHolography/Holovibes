// *****************************************************************************
//
//     Copyright (c) 2012, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PTUTILSLIB_PTSTRING_H__
#define __PTUTILSLIB_PTSTRING_H__

#include <PtUtilsLib.h>
#include <PtTypes.h> 


// Forward declaration 
namespace PtUtilsLib
{
	class String; 
}


class PT_UTILS_LIB_API PtString
{
public:

    PtString();
    PtString( const PtString & aValue );
    PtString( const char * aValue );
    PtString( const PtUnicodeChar * aValue );

    virtual ~PtString();

    const PtString &operator = ( const PtString & aValue );
    const PtString &operator += ( const PtString & aValue );

    bool operator == ( const char *aValue ) const;
    bool operator != ( const char *aValue ) const;

	bool operator == ( const PtUnicodeChar *aValue ) const;
    bool operator != ( const PtUnicodeChar *aValue ) const;

    bool operator == ( const PtString & aValue ) const;
    bool operator != ( const PtString & aValue ) const;

	operator const char *() const;
    operator const PtUnicodeChar *() const;

    const char *GetAscii() const;
    const PtUnicodeChar *GetUnicode() const;

    unsigned int GetLength() const;

private:

	mutable PtUtilsLib::String *mThis;
};


#endif //__PTUTILSLIB_PTSTRING_H__
