// *****************************************************************************
//
//     Copyright (c) 2009, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PV_GENREGISTER_H__
#define __PV_GENREGISTER_H__

#include <PvGenICamLib.h>
#include <PvGenParameter.h>


class PvGenRegister : public PvGenParameter
{
public:

    PV_GENICAM_API PvResult Set( const PvUInt8 *aBuffer, PvInt64 aLength );
    PV_GENICAM_API PvResult Get( PvUInt8 *aBuffer, PvInt64 aLength ) const;

    PV_GENICAM_API PvResult GetLength( PvInt64 &aLength ) const;

protected:

	PvGenRegister();
	virtual ~PvGenRegister();

private:

    // Not implemented
	PvGenRegister( const PvGenRegister & );
	const PvGenRegister &operator=( const PvGenRegister & );
};


#endif // __PV_GENREGISTER_H__
