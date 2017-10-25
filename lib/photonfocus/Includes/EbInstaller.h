// *****************************************************************************
//
//     Copyright (c) 2012, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __EBINSTALLERLIB_EBINSTALLER_H__
#define __EBINSTALLERLIB_EBINSTALLER_H__

#include <EbInstallerLib.h>
#include <EbNetworkAdapter.h>
#include <EbInstallerEventSink.h>

#include <PtResult.h>
#include <PtTypes.h>

namespace EbInstallerLib
{
    class Installer;
    class EventSinkManager;
};

class EbInstallerEventSinkLib;

class EB_INSTALLER_LIB_API EbInstaller
{
public:
    
    EbInstaller();
    virtual ~EbInstaller();

    PtResult Initialize();

    PtUInt32 GetNetworkAdapterCount() const;
    const EbNetworkAdapter* GetNetworkAdapter( PtUInt32 aIndex ) const;

    PtUInt32 GetDriverCount() const;
    const EbDriver* GetDriver( PtUInt32 aIndex ) const;

    PtResult Install( const EbNetworkAdapter* aNetworkAdapter, const EbDriver* aDriver );
    PtResult UninstallAll();
    PtResult UpdateAll();

    PtResult RegisterEventSink( EbInstallerEventSink *aEventSink );
    PtResult UnregisterEventSink( EbInstallerEventSink *aEventSink );
  
    bool IsRebootNeeded() const;
    
private:

#ifndef DOXYGEN
    EbInstallerLib::Installer* mThis;
    EbInstallerLib::EventSinkManager* mEventSinkManager;
#endif // DOXYGEN
};

#endif //__EBINSTALLERLIB_EBINSTALLER_H__