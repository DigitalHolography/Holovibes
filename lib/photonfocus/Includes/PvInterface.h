// *****************************************************************************
//
//     Copyright (c) 2008, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PV_INTERFACE_H__
#define __PV_INTERFACE_H__

#include <PvDeviceLib.h>
#include <PvDeviceInfo.h>


namespace PvDeviceLib
{
    class Interface;
};


typedef enum
{
    PvInterfaceNetworkStack = 0,
    PvInterfaceEBus = 1

} PvInterfaceType;


class PV_DEVICE_API PvInterface
{
public:

    PvResult Find();

    PvUInt32 GetDeviceCount() const;
    PvDeviceInfo *GetDeviceInfo( PvUInt32 aIndex );
    PvString GetMACAddress() const;
    PvInterfaceType GetType() const;
    PvString GetIPAddress() const;
    PvString GetSubnetMask() const;
    PvString GetDefaultGateway() const;
    PvString GetDescription() const;
    PvString GetID() const;

    PV_DEPRECATED_ALTERNATIVE( "IsUsingSubnetLimitedBroadcasts" ) bool IsUsingSubnetLimitedBroadcasts() const;

protected:

    PvInterface();
    ~PvInterface();

    PvDeviceLib::Interface *mThis;

private:

	 // Not implemented
	PvInterface( const PvInterface & );
	const PvInterface&operator=( const PvInterface & );

};


#ifdef PV_DEBUG
    #include <PvDeviceLib/Interface.h>
#endif // PV_DEBUG


#endif // __PV_INTERFACE_H__
