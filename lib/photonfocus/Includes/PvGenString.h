// *****************************************************************************
//
//     Copyright (c) 2007, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PV_GENSTRING_H__
#define __PV_GENSTRING_H__

#include <PvGenICamLib.h>
#include <PvGenParameter.h>


class PvGenString : public PvGenParameter
{
public:

	PV_GENICAM_API PvResult SetValue( const PvString &aValue );
	PV_GENICAM_API PvResult GetValue( PvString &aValue ) const;

    PV_GENICAM_API PvResult GetMaxLength( PvInt64 &aMaxLength ) const;

protected:

	PvGenString();
	virtual ~PvGenString();
 
private:

    // Not implemented
    PvGenString( const PvGenString & );
	const PvGenString &operator=( const PvGenString & );
};


#endif // __PV_GENSTRING_H__

