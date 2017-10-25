// *****************************************************************************
//
//     Copyright (c) 2007, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVDEVICEFINDERWND_H__
#define __PVDEVICEFINDERWND_H__


#include <PvGUILib.h>
#include <PvWnd.h>
#include <PvDeviceInfo.h>


class DeviceFinderWnd;


class PV_GUI_API PvDeviceFinderWnd : public PvWnd
{
public:

	PvDeviceFinderWnd();
	virtual ~PvDeviceFinderWnd();

	PvDeviceInfo *GetSelected();
	virtual bool OnFound( PvDeviceInfo *aDI );

protected:

private:

    // Not implemented
	PvDeviceFinderWnd( const PvDeviceFinderWnd & );
	const PvDeviceFinderWnd &operator=( const PvDeviceFinderWnd & );

};


#endif // __PVDEVICEFINDERWND_H__

