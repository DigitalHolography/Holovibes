// *****************************************************************************
//
//     Copyright (c) 2012, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __EBINSTALLERLIB_EBDRIVER_H__
#define __EBINSTALLERLIB_EBDRIVER_H__

#include <EbInstallerLib.h>

#include <PtTypes.h>
#include <PtString.h>

namespace EbInstallerLib
{
    class Driver;
};

class EbInstaller;
class EbNetworkAdapter;

class EB_INSTALLER_LIB_API EbDriver
{
public:

    PtString GetName() const;
    PtString GetDisplayName() const;
    PtString GetDescription() const;
    PtString GetAvailableVersion() const;
    PtString GetInstalledVersion() const;
    bool IsRebootNeeded() const;

protected:

    EbDriver();
    virtual ~EbDriver();

#ifndef DOXYGEN
    EbInstallerLib::Driver* mThis;
#endif // DOXYGEN

    friend class EbInstaller;
    friend class EbNetworkAdapter;
};


#endif //__EBINSTALLERLIB_EBDRIVER_H__
