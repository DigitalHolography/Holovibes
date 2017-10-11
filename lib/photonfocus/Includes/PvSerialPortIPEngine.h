// *****************************************************************************
//
//     Copyright (c) 2009, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVSERIALPORTIPENGINE_H__
#define __PVSERIALPORTIPENGINE_H__


#include <PvSerialLib.h>
#include <PvSerialPort.h>
#include <PvDevice.h>


namespace PvSerialLib
{
    class GenericIPEngine;
}


typedef enum 
{
    PvIPEngineSerialInvalid = -1,
    PvIPEngineSerial0 = 0,
    PvIPEngineSerial1 = 1,
    PvIPEngineSerialBulk0 = 2,
    PvIPEngineSerialBulk1 = 3,
    PvIPEngineSerialBulk2 = 4,
    PvIPEngineSerialBulk3 = 5,
    PvIPEngineSerialBulk4 = 6,
    PvIPEngineSerialBulk5 = 7,
    PvIPEngineSerialBulk6 = 8,
    PvIPEngineSerialBulk7 = 9

} PvIPEngineSerial;


class PV_SERIAL_API PvSerialPortIPEngine : public PvSerialPort
{
public:

    PvSerialPortIPEngine();
    virtual ~PvSerialPortIPEngine();

    PvResult Open( PvDevice *aDevice, PvIPEngineSerial aPort );
    PvResult Close();
    bool IsOpened();

    PvResult Write( const PvUInt8 *aBuffer, PvUInt32 aSize, PvUInt32 &aBytesWritten );
    PvResult Read( PvUInt8 *aBuffer, PvUInt32 aBufferSize, PvUInt32 &aBytesRead, PvUInt32 aTimeout = 0 );

    PvResult FlushRxBuffer();
    PvResult GetRxBytesReady( PvUInt32 &aBytes ); 
    PvResult GetRxBufferSize( PvUInt32 &aSize );
    PvResult SetRxBufferSize( PvUInt32 aSize );

    static bool IsSupported( PvDevice *aDevice, PvIPEngineSerial aPort );

private:

    // Not implemented
	PvSerialPortIPEngine( const PvSerialPortIPEngine & );
	const PvSerialPortIPEngine &operator=( const PvSerialPortIPEngine & );

    PvSerialLib::GenericIPEngine * mThis;
};


#ifdef PV_DEBUG
    #include <PvSerialLib/IPEngineSerial.h>
#endif // PV_DEBUG


#endif // __PVSERIALPORTIPENGINE_H__
