// *****************************************************************************
//
//     Copyright (c) 2009, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVSERIALPORT_H__
#define __PVSERIALPORT_H__


#include <PvResult.h>
#include <PvSerialLib.h>


class PV_SERIAL_API PvSerialPort
{
public:

    virtual PvResult Close() = 0;
    virtual PvResult Write( const PvUInt8 *aBuffer, PvUInt32 aSize, PvUInt32 &aBytesWritten ) = 0;
    virtual bool IsOpened() = 0;
    virtual PvResult Read( PvUInt8 *aBuffer, PvUInt32 aBufferSize, PvUInt32 &aBytesRead, PvUInt32 aTimeout = 0 ) = 0;
    virtual PvResult GetRxBytesReady( PvUInt32 &aBytes ) = 0; 
    virtual PvResult FlushRxBuffer() = 0;
    virtual PvResult GetRxBufferSize( PvUInt32 &aSize ) = 0;
    virtual PvResult SetRxBufferSize( PvUInt32 aSize ) = 0;
};


#endif // __PVSERIALPORT_H__
