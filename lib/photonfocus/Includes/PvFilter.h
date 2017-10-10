// *****************************************************************************
//
//     Copyright (c) 2008, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVFILTER_H__
#define __PVFILTER_H__


#include <PvBufferLib.h>
#include <PvBuffer.h>


class PV_BUFFER_API PvFilter
{

public:

    virtual PvResult Apply( PvBuffer *aBuffer );
    virtual PvResult Apply( const PvBuffer *aIn, PvBuffer *aOut );

protected:

    PvFilter();
    virtual ~PvFilter();

private:

    // Not implemented
    PvFilter( const PvFilter & );
	const PvFilter &operator=( const PvFilter & );
};


#endif // __PVFILTER_H__

