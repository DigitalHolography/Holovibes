// *****************************************************************************
//
//     Copyright (c) 2007, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PV_DEVICE_H__
#define __PV_DEVICE_H__

#include <PvDeviceLib.h>
#include <PvDeviceInfo.h>
#include <PvGenParameterArray.h>


namespace PvDeviceLib
{
    class Device;

}; // namespace PvDeviceLib;


class PvDeviceEventSink;
class PvConfigurationWriter;
class PvConfigurationReader;


class PV_DEVICE_API PvDevice 
{
public:

	PvDevice();
	~PvDevice();

	PvResult Connect( const PvDeviceInfo *aDeviceInfo, PvAccessType aAccessType = PvAccessControl );
	PvResult Connect( const PvString &aInfo, PvAccessType aAccessType = PvAccessControl );
	PvResult Disconnect();

	bool IsConnected() const;
    bool IsIPEngine() const;

	PvResult DumpGenICamXML( const PvString &aFilename );
	PvResult GetDefaultGenICamXMLFilename( PvString &aFilename );

	PvGenParameterArray *GetGenParameters();
	PvGenParameterArray *GetGenLink();

    PvResult SetStreamDestination( const PvString &aIPAddress, PvUInt16 aDataPort, PvUInt32 aChannel = 0 ); 
    PvResult ResetStreamDestination( PvUInt32 aChannel = 0 );
    PvResult SetPacketSize( PvUInt32 aPacketSize, PvUInt32 aChannel = 0 );
    PvResult NegotiatePacketSize( PvUInt32 aChannel = 0, PvUInt32 aDefaultPacketSize = 0 );

	PvResult Reset();

	PvResult ReadRegister( PvUInt32 aAddress, PvUInt32 &aValue );
	PvResult WriteRegister( PvUInt32 aAddress, PvUInt32 aValue, bool aAcknowledge = true );
	PvResult ReadMemory( PvUInt32 aAddress, unsigned char *aDestination, PvUInt32 aByteCount );
	PvResult WriteMemory( PvUInt32 aAddress, const unsigned char *aSource, PvUInt32 aByteCount );

    PvResult WaitForMessagingChannelIdle( PvUInt32 aTimeout );

    static PvResult SetIPConfiguration( 
        const PvString &aMACAddress, 
        const PvString &aIP, 
        const PvString &aSubnetMask = PvString( "255.255.255.0" ), 
        const PvString &aGateway = PvString( "0.0.0.0" ) );

    // Notifications
    PvResult RegisterEventSink( PvDeviceEventSink *aEventSink );
    PvResult UnregisterEventSink( PvDeviceEventSink *aEventSink );

    PvUInt32 GetHeartbeatThreadPriority() const;
    PvResult SetHeartbeatThreadPriority( PvUInt32 aPriority );
    PvUInt32 GetInterruptLinkThreadPriority() const;
    PvResult SetInterruptLinkThreadPriority( PvUInt32 aPriority );
    PvUInt32 GetInterruptQueueThreadPriority() const;
    PvResult SetInterruptQueueThreadPriority( PvUInt32 aPriority );
    PvUInt32 GetRecoveryThreadPriority() const;
    PvResult SetRecoveryThreadPriority( PvUInt32 aPriority );

protected:
    
private:

    friend class PvConfigurationWriter;
    friend class PvConfigurationReader;

	 // Not implemented
	PvDevice( const PvDevice & );
	const PvDevice &operator=( const PvDevice & );

    PvDeviceLib::Device *mThis;

};


class PV_DEVICE_API PvDeviceEventSink
{
public:

    PvDeviceEventSink();
    virtual ~PvDeviceEventSink();

    // Notifications
    virtual void OnLinkDisconnected( PvDevice *aDevice );
    virtual void OnLinkReconnected( PvDevice *aDevice );

    // Messaging channel events (raw)
    virtual void OnEvent( PvDevice *aDevice, 
        PvUInt16 aEventID, PvUInt16 aChannel, PvUInt64 aBlockID, PvUInt64 aTimestamp, 
        const void *aData, PvUInt32 aDataLength );

    // Messaging channel events (GenICam)
    virtual void OnEventGenICam( PvDevice *aDevice,
        PvUInt16 aEventID, PvUInt16 aChannel, PvUInt64 aBlockID, PvUInt64 aTimestamp,
        PvGenParameterList *aData );

	// GigE Vision command link GenApi::IPort monitoring hooks
	virtual void OnCmdLinkRead( const void *aBuffer, PvInt64 aAddress, PvInt64 aLength );
	virtual void OnCmdLinkWrite( const void *aBuffer, PvInt64 aAddress, PvInt64 aLength );
};


#ifdef PV_DEBUG
    #include <PvDeviceLib/Device.h>
#endif // PV_DEBUG


#endif // __PV_DEVICE_H__


