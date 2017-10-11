// *****************************************************************************
//
//     Copyright (c) 2008, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVSTREAMRAW_H__
#define __PVSTREAMRAW_H__


#include <PvTypes.h>
#include <PvResult.h>
#include <PvStreamBase.h>
#include <PvStatistics.h>


class PV_STREAMRAW_API PvStreamRaw : public PvStreamBase
{
public:

    PvStreamRaw();
    virtual ~PvStreamRaw();

    PvUInt32 GetChannelID() const;
    PvUInt32 GetFirstPacketTimeout() const;
    PvString GetRemoteIpAddress() const;
    PV_DEPRECATED PvUInt32 GetPacketTimeout() const;
    PvUInt32 GetInterPacketTimeout() const;
    PvUInt32 GetPreemptiveResendTimeout() const;
    PvUInt32 GetRequestTimeout() const;
    bool GetResendPacketEnable() const;
    PvUInt32 GetDefaultBlockTimeout() const;
    PvUInt32 GetMaximumPendingResends() const;
	PvUInt32 GetLatencyLevel() const;
	PvUInt32 GetAutoResetOnLackOfResources() const;
    bool GetForceMissingPacketsAtNextBlockStart() const;
	PvUInt32 GetMaximumResendRequestRetryByPacket() const;
	PvUInt32 GetMaximumResendGroupSize() const;
  	PvUInt32 GetResendRequestTimeout() const;
    PvUInt32 GetResetOnIdle() const;
    PvUInt32 GetMaximumPreQueuedBuffers() const;
    PvUInt32 GetMaximumInternalRetrieveBuffers() const;
    bool GetEnableMissingPacketsList() const;
    PvUInt32 GetResendDelay() const;
    bool GetWaitForFirstPacketOfBlockToStart() const;

    PvResult SetFirstPacketTimeout( PvUInt32 aTimeout );
    PV_DEPRECATED PvResult SetPacketTimeout( PvUInt32 aTimeout );
    PvResult SetInterPacketTimeout( PvUInt32 aTimeout );
    PvResult SetPreemptiveResendTimeout( PvUInt32 aTimeout );
    PvResult SetRequestTimeout( PvUInt32 aTimeout );
    PvResult SetResendPacketEnable( bool aEnable );
    PvResult SetDefaultBlockTimeout( PvUInt32 aTimeout );
    PvResult SetMaximumPendingResends( PvUInt32 aMaximumPendingResends );
    PvResult SetLatencyLevel( PvUInt32 aLatencyLevel );
    PvResult SetAutoResetOnLackOfResources( PvUInt32 aNumberOfBuffers );
    PvResult SetForceMissingPacketsAtNextBlockStart( bool aEnable );
    PvResult SetMaximumResendRequestRetryByPacket( PvUInt32 aMaximumResendRequestRetryByPacket );
    PvResult SetMaximumResendGroupSize( PvUInt32 aMaximumResendGroupSize );
    PvResult SetResendRequestTimeout( PvUInt32 aResendRequestTimeout );
    PvResult SetResetOnIdle( PvUInt32 aResetOnIdle );
    PvResult SetMaximumPreQueuedBuffers( PvUInt32 aMaximumPreQueuedBuffers );
	PvResult SetMaximumInternalRetrieveBuffers( PvUInt32 aMaximumInternalRetrieveBuffers );
    PvResult SetEnableMissingPacketsList( bool aEnableMissingPacketsList );
    PvResult SetResendDelay( PvUInt32 aResendDelay );
    PvResult SetWaitForFirstPacketOfBlockToStart( bool aWaitForFirstPacketOfBlockToStart );

    PvStatistics *GetStatistics();

private:

	 // Not implemented
	PvStreamRaw( const PvStreamRaw & );
	const PvStreamRaw &operator=( const PvStreamRaw & );

};


#endif // __PVSTREAMRAW_H__
