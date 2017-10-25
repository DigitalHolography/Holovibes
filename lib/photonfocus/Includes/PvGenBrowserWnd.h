// *****************************************************************************
//
//     Copyright (c) 2007, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVGENBROWSERWND_H__
#define __PVGENBROWSERWND_H__


#include <PvGUILib.h>
#include <PvWnd.h>
#include <PvGenParameterArray.h>
#include <PvPropertyList.h>


class GenBrowserWndBase;


class PV_GUI_API PvGenBrowserWnd : public PvWnd
{
public:

	PvGenBrowserWnd();
	virtual ~PvGenBrowserWnd();

	void SetGenParameterArray( PvGenParameterArray *aArray );
	virtual bool IsParameterDisplayed( PvGenParameter *aParameter );

    PvResult SetVisibility( PvGenVisibility aVisibility );
    PvGenVisibility GetVisibility();

    PvResult Save( PvPropertyList &aPropertyList );
    PvResult Load( PvPropertyList &aPropertyList );

    void Refresh();

protected:

private:

    // Not implemented
	PvGenBrowserWnd( const PvGenBrowserWnd & );
	const PvGenBrowserWnd &operator=( const PvGenBrowserWnd & );

};


#endif // __PVGENBROWSERWND_H__
