// *****************************************************************************
//
//     Copyright (c) 2011, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PV_VIRTUAL_DEVICE_H__
#define __PV_VIRTUAL_DEVICE_H__

#include <PvVirtualDeviceLib.h>

namespace PvVirtualDeviceLib
{
    class VirtualDevice;
};

class PV_VIRTUAL_DEVICE_API PvVirtualDevice
{
public:

    PvVirtualDevice();
    ~PvVirtualDevice();
    PvResult StartListening( PvString aInfo );
    void StopListening();

    PvUInt32 GetDevicePortThreadPriority() const;
    PvResult SetDevicePortThreadPriority( PvUInt32 aPriority );

private:
    PvVirtualDeviceLib::VirtualDevice* mThis;

};

#endif //__PV_VIRTUAL_DEVICE_H__

