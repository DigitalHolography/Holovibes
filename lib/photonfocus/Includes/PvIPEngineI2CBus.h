
// *****************************************************************************
//
//     Copyright (c) 2009, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVIPENGINEI2CBUS_H__
#define __PVIPENGINEI2CBUS_H__

#include <PvSerialPortIPEngine.h>


namespace PvSerialLib
{
    class IPEngineI2CBus;
}; // namespace PvSerialLib


class PV_SERIAL_API PvIPEngineI2CBus
{
public:

    PvIPEngineI2CBus();
    ~PvIPEngineI2CBus();

    PvResult Open( PvDevice *aDevice,
       PvIPEngineSerial aPort = PvIPEngineSerialBulk0 );

    PvResult Close();

    bool IsOpened();

    static bool IsSupported( PvDevice *aDevice,
       PvIPEngineSerial aPort = PvIPEngineSerialBulk0 );

    PvResult BurstWrite(
       unsigned char aSlaveAddress,
       const unsigned char *aBuffer,
       PvUInt32 aBufferSize,
       bool aFastMode = true );

    PvResult IndirectBurstWrite(
       unsigned char aSlaveAddress,
       unsigned char aOffset,
       const unsigned char *aBuffer,
       PvUInt32 aBufferSize,
       bool aFastMode = true );

    PvResult BurstRead(
       unsigned char aSlaveAddress,
       unsigned char *aBuffer,
       PvUInt32 aBufferSize,
       PvUInt32 &aBytesRead,
       bool aFastMode = true );

    PvResult IndirectBurstRead(
       unsigned char aSlaveAddress,
       unsigned char aOffset,
       unsigned char *aBuffer,
       PvUInt32 aBufferSize,
       PvUInt32 &aBytesRead,
       bool aFastMode = true,
       bool aUseCombinedFormat = true );

    PvResult MasterTransmitter(
       PvUInt8 aSlaveAddress,
       const PvUInt8 *aBuffer,
       PvUInt32 aBufferSize,
       bool aFastMode = true,
       bool aGenerateStopCondition = true );

    PvResult MasterReceiverAfterFirstByte(
       PvUInt8 aSlaveAddress,
       PvUInt8 *aBuffer,
       PvUInt32 aBufferSize,
       PvUInt32 &aBytesRead,
       bool aFastMode = true,
       bool aGenerateStopCondition = true );

private:

    PvSerialLib::IPEngineI2CBus* mThis;

};


#ifdef PV_DEBUG
    #include <PvSerialLib/IPEngineI2CBus.h>
#endif // PV_DEBUG


#endif //__PVIPENGINEI2CBUS_H__

