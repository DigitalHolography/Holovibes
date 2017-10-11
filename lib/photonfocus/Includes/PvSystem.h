// *****************************************************************************
//
//     Copyright (c) 2008, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PV_SYSTEM_H__
#define __PV_SYSTEM_H__

#include <PvDeviceLib.h>
#include <PvInterface.h>


namespace PvDeviceLib
{
    class System;
};


class PvSystemEventSink;


class PV_DEVICE_API PvSystem
{
public:

    PvSystem();
    ~PvSystem();

    PvResult Find();

    PvUInt32 GetInterfaceCount();
    PvInterface *GetInterface( PvUInt32 aIndex );

    void SetDetectionTimeout( PvUInt32 aTimeout );
    PvUInt32 GetDetectionTimeout() const;

    PvUInt32 GetProtocolVersionMinor() const;
    PvUInt32 GetProtocolVersionMajor() const;

    PvResult RegisterEventSink( PvSystemEventSink *aEventSink );
    PvResult UnregisterEventSink( PvSystemEventSink *aEventSink );

    PvUInt32 GetDetectionThreadsPriority() const;
    PvResult SetDetectionThreadsPriority( PvUInt32 aPriority );

    // Attempts to detect a single device based on its IP, MAC or device name
    PvResult FindDevice( const PvString &aDeviceToFind, PvDeviceInfo **aDeviceInfo );

protected:

private:

	 // Not implemented
	PvSystem( const PvSystem & );
	const PvSystem &operator=( const PvSystem & );

    PvDeviceLib::System *mThis;
};


class PvSystemEventSink
{
public:

    virtual void OnDeviceFound( 
        PvInterface *aInterface, PvDeviceInfo *aDeviceInfo, 
        bool &aIgnore ) = 0;

};


#ifdef PV_DEBUG
    #include <PvDeviceLib/System.h>
#endif // PV_DEBUG


#endif // __PV_SYSTEM_H__

