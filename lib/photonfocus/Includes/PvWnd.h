// *****************************************************************************
//
//     Copyright (c) 2008, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PV_WND_H__
#define __PV_WND_H__

#include <PvGUILib.h>


class Wnd;


class PV_GUI_API PvWnd
{
public:

	void SetPosition( PvInt32  aPosX, PvInt32  aPosY, PvInt32  aSizeX, PvInt32  aSizeY );
	void GetPosition( PvInt32 &aPosX, PvInt32 &aPosY, PvInt32 &aSizeX, PvInt32 &aSizeY );

	PvResult ShowModal( PvWindowHandle aParentHwnd = 0 );
	PvResult ShowModeless( PvWindowHandle aParentHwnd = 0 );
	PvResult Create( PvWindowHandle aHwnd, PvUInt32 aID );

	PvString GetTitle() const;
	void SetTitle( const PvString &aTitle );

	PvResult Close();

	PvWindowHandle GetHandle();
    PvResult DoEvents();

protected:

    PvWnd();
	virtual ~PvWnd();

    Wnd *mThis;

private:

    // Not implemented
	PvWnd( const PvWnd & );
	const PvWnd &operator=( const PvWnd & );

};


#endif // __PV_WND_H__

