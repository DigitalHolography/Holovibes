// *****************************************************************************
//
//     Copyright (c) 2011, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVGENBROWSERTABBEDWND_H__
#define __PVGENBROWSERTABBEDWND_H__


#include <PvGUILib.h>
#include <PvWnd.h>
#include <PvGenParameterArray.h>
#include <PvPropertyList.h>


class GenBrowserWndBase;


class PV_GUI_API PvGenBrowserTabbedWnd : public PvWnd
{
public:

	PvGenBrowserTabbedWnd();
	virtual ~PvGenBrowserTabbedWnd();

    void Reset();
    PvResult AddGenParameterArray( PvGenParameterArray *aArray, const PvString &aName, PvUInt32 *aIndex = NULL );
    PvResult RemoveGenParameterArray( PvUInt32 aIndex );

    PvUInt32 GetParameterArrayCount() const;
    const PvGenParameterArray *GetGenParameterArray( PvUInt32 aIndex ) const;
    PvResult GetParameterArrayName( PvUInt32 aIndex, PvString &aName ) const;
    PvResult SetParameterArrayName( PvUInt32 aIndex, const PvString &aName );

	virtual bool IsParameterDisplayed( PvGenParameterArray *aArray, PvGenParameter *aParameter );

    PvResult SetVisibility( PvGenVisibility aVisibility );
    PvGenVisibility GetVisibility();

    PvResult Save( PvPropertyList &aPropertyList );
    PvResult Load( PvPropertyList &aPropertyList );

    void Refresh();
    void Refresh( PvUInt32 aIndex );

protected:

private:

    // Not implemented
	PvGenBrowserTabbedWnd( const PvGenBrowserTabbedWnd & );
	const PvGenBrowserTabbedWnd &operator=( const PvGenBrowserTabbedWnd & );

};


#endif // __PVGENBROWSERTABBEDWND_H__

