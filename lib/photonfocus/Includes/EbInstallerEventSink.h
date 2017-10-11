// *****************************************************************************
//
//     Copyright (c) 2012, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __EBINSTALLERLIB_EBINSTALLEREVENTSINK_H__
#define __EBINSTALLERLIB_EBINSTALLEREVENTSINK_H__

#include <EbInstallerLib.h>

#include <PtTypes.h>

namespace EbInstallerLib
{
    class EventSinkManager;
}

class EB_INSTALLER_LIB_API EbInstallerEventSink
{
public:

    EbInstallerEventSink();
    virtual ~EbInstallerEventSink();

protected:

    virtual void OnUpdateProgress( PtUInt32 aProgress );
    virtual void OnStatusMessage( PtString aMsg );

private:
    friend class EbInstallerLib::EventSinkManager;
};

#endif // __EBINSTALLERLIB_EBINSTALLEREVENTSINK_H__