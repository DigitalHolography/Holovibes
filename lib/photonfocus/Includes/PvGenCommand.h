// *****************************************************************************
//
//     Copyright (c) 2007, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PV_GENCOMMAND_H__
#define __PV_GENCOMMAND_H__

#include <PvGenICamLib.h>
#include <PvGenParameter.h>

class PvGenCommand : public PvGenParameter
{
public:

	PV_GENICAM_API PvResult Execute();
	PV_GENICAM_API PvResult IsDone( bool &aDone );

protected:

	PvGenCommand();
	virtual ~PvGenCommand();

private:

    // Not implemented
	PvGenCommand( const PvGenCommand & );
	const PvGenCommand&operator=( const PvGenCommand & );
};


#endif // __PV_GENCOMMAND_H__
