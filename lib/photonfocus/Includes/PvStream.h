// *****************************************************************************
//
//     Copyright (c) 2007, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVSTREAM_H__
#define __PVSTREAM_H__


#include <PvBuffer.h>
#include <PvGenParameterArray.h>
#include <PvStreamBase.h>
#include <PvStreamLib.h>


class PV_STREAM_API PvStream : public PvStreamBase
{
public:
	
	PvStream();
	virtual ~PvStream();

	PvGenParameterArray *GetParameters();

private:

    PvGenParameterArray *mParameters;

	 // Not implemented
	PvStream( const PvStream & );
	const PvStream &operator=( const PvStream );

};


#endif // __PVSTREAM_H__

