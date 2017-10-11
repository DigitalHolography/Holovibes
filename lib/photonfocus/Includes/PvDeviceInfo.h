// *****************************************************************************
//
//     Copyright (c) 2007, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PV_DEVICEINFO_H__
#define __PV_DEVICEINFO_H__

#include <PvDeviceLib.h>


namespace PvDeviceLib
{
    class DeviceInfo;

}; // namespace PvDeviceLib


class PvInterface;
class PvDevice;


typedef enum
{
    PvAccessUnknown = -1,
    PvAccessOpen = 0,
    PvAccessControl = 1,
    PvAccessReadOnly = 2,
    PvAccessExclusive = 3

} PvAccessType;

typedef enum
{
    PvDeviceClassUnknown = -1,
    PvDeviceClassTransmitter = 0,
    PvDeviceClassReceiver = 1,
    PvDeviceClassTransceiver = 2,
    PvDeviceClassPeripheral = 3

} PvDeviceClass;


class PV_DEVICE_API PvDeviceInfo
{
public:

    PvString GetMACAddress() const;
    PvString GetIPAddress() const;
    PvString GetSubnetMask() const;
    PvString GetDefaultGateway() const;
    PvString GetVendor() const;
    PvString GetModel() const;
    PvAccessType GetAccessStatus() const;
    PvString GetManufacturerInfo() const;
    PvString GetVersion() const;
    PvString GetID() const;
    PvString GetSerialNumber() const;
    PvString GetUserDefinedName() const;
    PvUInt32 GetProtocolVersionMajor() const;
    PvUInt32 GetProtocolVersionMinor() const;
    bool IsIPConfigurationValid() const;
    bool IsLicenseValid() const;
    const PvInterface *GetInterface() const;
    PvDeviceClass GetClass() const;

protected:

	PvDeviceInfo();
	virtual ~PvDeviceInfo();

    PvDeviceLib::DeviceInfo *mThis;

    friend class PvDevice;

private:

	 // Not implemented
    PvDeviceInfo( const PvDeviceInfo & );
	const PvDeviceInfo &operator=( const PvDeviceInfo & );

};


#ifdef PV_DEBUG
    #include <PvDeviceLib/DeviceInfo.h>
#endif // PV_DEBUG


#endif // __PV_DEVICE_H__

