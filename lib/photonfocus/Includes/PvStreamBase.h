// *****************************************************************************
//
//     Copyright (c) 2008, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVSTREAMBASE_H__
#define __PVSTREAMBASE_H__


#include <PvTypes.h>
#include <PvResult.h>
#include <PvString.h>
#include <PvBuffer.h>

#include <PvStreamRawLib.h>


namespace PvStreamRawLib
{
    class Stream;
};


class PvConfigurationWriter;
class PvConfigurationReader;
class PvPipeline;

class PV_STREAMRAW_API PvStreamBaseEventSink
{
public:

    PvStreamBaseEventSink();
    virtual ~PvStreamBaseEventSink();

    virtual void OnBufferQueued( PvBuffer *aBuffer );
    virtual void OnBufferRetrieved( PvBuffer* aBuffer );
};

class PV_STREAMRAW_API PvStreamBase
{
public:

    PvUInt32 GetQueuedBufferCount() const;
    PvUInt32 GetQueuedBufferMaximum() const;

	PvResult Open(
        const PvString & aIPAddress,
        PvUInt16 aLocalPort = 0,
        PvUInt16 aChannelID = 0,
        const PvString & aLocalIpAddress = PvString(), 
        PvUInt32 aBuffersCapacity = 64 );

    PvResult Open(
        const PvString & aIPAddress,
        const PvString & aMulticastAddr,
        PvUInt16 aDataPort,
        PvUInt16 aChannelID = 0,
        const PvString & aLocalIPAddress = PvString(), 
        PvUInt32 aBuffersCapacity = 64 );

    virtual PvResult Close();

    PvResult AbortQueuedBuffers();
    PvResult QueueBuffer( PvBuffer * aBuffer );
    PvResult RetrieveBuffer(
        PvBuffer ** aBuffer,
        PvResult * aOperationResult,
        PvUInt32 aTimeout = 0xFFFFFFFF );

    PvResult FlushPacketQueue();

    PvUInt16 GetLocalPort();
    PvString GetLocalIPAddress();
    PvString GetMulticastIPAddress();
    PvUInt16 GetChannelID();

    bool IsOpen() const;

    static PvResult IsDriverInstalled( PvString &aIPAddress, bool &aInstalled, const PvString & aLocalIPAddress = PvString() );

    PvResult RegisterEventSink( PvStreamBaseEventSink *aEventSink );
    PvResult UnregisterEventSink( PvStreamBaseEventSink *aEventSink );

    PvUInt32 GetUserModeDataReceiverThreadPriority() const;
    PvResult SetUserModeDataReceiverThreadPriority( PvUInt32 aPriority );

protected:

    friend class PvPipeline;
    friend class PvConfigurationWriter;
    friend class PvConfigurationReader;

    PvStreamBase();
    virtual ~PvStreamBase();

    PvStreamRawLib::Stream *mThis;

private:

	 // Not implemented
	PvStreamBase( const PvStreamBase & );
    const PvStreamBase &operator=( const PvStreamBase & );

};


#ifdef PV_DEBUG
    #include <PvStreamRawLib/Stream.h>
#endif // PV_DEBUG


#endif // __PVSTREAMBASE_H__
