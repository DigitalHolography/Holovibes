// *****************************************************************************
//
//     Copyright (c) 2012, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef _PV_GVSP_TRANSMITTER_RAW_H_
#define _PV_GVSP_TRANSMITTER_RAW_H_

#include <PvTransmitterRawLib.h>
#include <PvBuffer.h>

namespace PvTransmitterRawLib
{
    class TransmitterRaw;
};

class PV_TRANSMITTERRAW_API PvTransmitterRaw
{
public:

    PvTransmitterRaw();
    virtual ~PvTransmitterRaw();

    PvResult Open( PvString aDestinationIPAddress, PvUInt16 aDestinationPort, 
        PvString aSourceIPAddress = "", PvUInt16 aSourcePort = 0, bool aDontFrag = true,
        bool aExtendedID = false, PvUInt32 aBuffersCapacity = 64, bool aTimestampWhenSending = false );
    PvResult Close();
    bool IsOpen() const;
    PvResult LoadBufferPool( PvBuffer** aBuffers, PvUInt32 aBufferCount );
    PvResult QueueBuffer( PvBuffer* aBuffer );
    PvResult RetrieveFreeBuffer( PvBuffer ** aBuffer, PvUInt32 aTimeout = 0xFFFFFFFF );
    PvResult AbortQueuedBuffers( PvUInt32 aTimeout = 0xFFFFFFFF, bool* aPartialTransmission = NULL );

    PvUInt32 GetQueuedBufferCount();
    PvUInt32 GetPacketSize();
    PvResult SetPacketSize( PvUInt32 aPacketSize );

    float GetMaxPayloadThroughput();
    PvResult SetMaxPayloadThroughput( float aMaxPayloadThroughput );
    PvUInt16 GetSourcePort();
    PvUInt16 GetDestinationPort();

    PvString GetDestinationIPAddress();
    PvString GetSourceIPAddress();
    
    void ResetStats();
    PvUInt64 GetBlocksTransmitted() const;
    PvUInt64 GetSamplingTime() const;
    PvUInt64 GetPayloadBytesTransmitted() const;
    float GetInstantaneousPayloadThroughput() const;
    float GetAveragePayloadThroughput() const;
    float GetInstantaneousTransmissionRate() const;
    float GetAverageTransmissionRate() const;

    PvUInt32 GetUserModeTransmitterThreadPriority() const;
    PvResult SetUserModeTransmitterThreadPriority( PvUInt32 aPriority );
    PvUInt32 GetBufferPoolThreadPriority() const;
    PvResult SetBufferPoolThreadPriority( PvUInt32 aPriority );

private:
    PvTransmitterRawLib::TransmitterRaw *mThis;
};

#endif //_PV_GVSP_TRANSMITTER_RAW_H_
