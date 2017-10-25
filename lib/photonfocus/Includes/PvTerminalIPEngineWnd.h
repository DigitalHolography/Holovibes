// *****************************************************************************
//
//     Copyright (c) 2007, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVTERMINALIPENGINEWND_H___
#define __PVTERMINALIPENGINEWND_H___


#include <PvGUILib.h>
#include <PvWnd.h>
#include <PvSerialPortIPEngine.h>


class DeviceFinderWnd;


class PV_GUI_API PvTerminalIPEngineWnd : public PvWnd
{
public:

	PvTerminalIPEngineWnd();
	virtual ~PvTerminalIPEngineWnd();

    PvResult SetDevice( PvDevice *aDevice );

    PvResult SetSerialPort( PvIPEngineSerial aPort );
    PvIPEngineSerial GetSerialPort() const;

protected:

private:

    // Not implemented
	PvTerminalIPEngineWnd( const PvTerminalIPEngineWnd & );
	const PvTerminalIPEngineWnd &operator=( const PvTerminalIPEngineWnd & );

};


#endif // __PVTERMINALIPENGINEWND_H___



