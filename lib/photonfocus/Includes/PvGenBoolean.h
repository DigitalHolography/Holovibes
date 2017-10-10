// *****************************************************************************
//
//     Copyright (c) 2007, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PV_GENBOOLEAN_H__
#define __PV_GENBOOLEAN_H__

#include <PvGenICamLib.h>
#include <PvGenParameter.h>


class PvGenBoolean : public PvGenParameter
{
public:

	PV_GENICAM_API PvResult SetValue( bool aValue );
	PV_GENICAM_API PvResult GetValue( bool &aValue ) const;

protected:

	PvGenBoolean();
	virtual ~PvGenBoolean();

private:

    // Not implemented
	PvGenBoolean( const PvGenBoolean & );
	const PvGenBoolean &operator=( const PvGenBoolean & );
};


#endif // __PV_GENBOOLEAN_H__
