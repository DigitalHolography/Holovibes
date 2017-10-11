// *****************************************************************************
//
//     Copyright (c) 2007, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PV_GENINTEGER_H__
#define __PV_GENINTEGER_H__

#include <PvGenICamLib.h>
#include <PvGenParameter.h>


class PvGenInteger : public PvGenParameter
{
public:

	PV_GENICAM_API PvResult SetValue( PvInt64 aValue );
	PV_GENICAM_API PvResult GetValue( PvInt64 &aValue ) const;

	PV_GENICAM_API PvResult GetMin( PvInt64 &aMin ) const;
	PV_GENICAM_API PvResult GetMax( PvInt64 &aMax ) const;
	PV_GENICAM_API PvResult GetIncrement( PvInt64 &aIncrement ) const;

	PV_GENICAM_API PvResult GetRepresentation( PvGenRepresentation &aRepresentation ) const;

protected:

	PvGenInteger();
	virtual ~PvGenInteger();

private:

    // Not implemented
	PvGenInteger( const PvGenInteger & );
	const PvGenInteger &operator=( const PvGenInteger & );
};


#endif // __PV_GENINTEGER_H__


