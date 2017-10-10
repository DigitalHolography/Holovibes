// *****************************************************************************
//
//     Copyright (c) 2012, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __EBINSTALLERLIB_EBNETWORKADAPTER_H__
#define __EBINSTALLERLIB_EBNETWORKADAPTER_H__

#include <EbInstallerLib.h>
#include <EbDriver.h>

#include <PtTypes.h>


namespace EbInstallerLib
{
    class NetworkAdapter;
};

class EbInstaller;

class EB_INSTALLER_LIB_API EbNetworkAdapter
{
public:

    PtString GetName() const;
    PtString GetMACAddress() const;
    const EbDriver* GetCurrentDriver() const;
    bool IsDriverSupported( const EbDriver* aDriver ) const;
    bool IsRebootNeeded() const;

protected:

#ifndef DOXYGEN
    EbNetworkAdapter();
    EbNetworkAdapter( EbInstallerLib::NetworkAdapter* aNetworkAdapter );
    virtual ~EbNetworkAdapter();

    EbInstallerLib::NetworkAdapter* mThis;
#endif // DOXYGEN

    friend class EbInstaller;
};

#endif //__EBINSTALLERLIB_EBNETWORKADAPTER_H__