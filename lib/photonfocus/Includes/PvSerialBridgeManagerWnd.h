// *****************************************************************************
//
//     Copyright (c) 2011, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVSERIALBRIDGEMANAGERWND_H__
#define __PVSERIALBRIDGEMANAGERWND_H__


#include <PvGUILib.h>
#include <PvWnd.h>

#include <PvSerialBridge.h>
#include <PvGenParameterArray.h>
#include <PvPropertyList.h>
#include <PvStringList.h>


typedef enum 
{
    PvSerialBridgeTypeInvalid = -1,
    PvSerialBridgeTypeNone = 0,
    PvSerialBridgeTypeSerialCOMPort = 1,
    PvSerialBridgeTypeCameraLinkDLL = 2,
    PvSerialBridgeTypeCLProtocol = 3,

} PvSerialBridgeType;


class PV_GUI_API PvSerialBridgeManagerWnd : public PvWnd
{
public:

	PvSerialBridgeManagerWnd();
	virtual ~PvSerialBridgeManagerWnd();

    PvResult SetDevice( PvDevice *aDevice );

    PvUInt32 GetPortCount() const;
    PvString GetPortName( PvUInt32 aIndex ) const;
    PvGenParameterArray *GetGenParameterArray( PvUInt32 aIndex );
    PvString GetGenParameterArrayName( PvUInt32 aIndex );

    virtual void OnParameterArrayCreated( PvGenParameterArray *aArray, const PvString &aName );
    virtual void OnParameterArrayDeleted( PvGenParameterArray *aArray );

    PvResult Save( PvPropertyList &aPropertyList );
    PvResult Load( PvPropertyList &aPropertyList, PvStringList &aErrors );

    PvResult Recover();

protected:

private:

    // Not implemented
	PvSerialBridgeManagerWnd( const PvSerialBridgeManagerWnd & );
	const PvSerialBridgeManagerWnd &operator=( const PvSerialBridgeManagerWnd & );

};


#endif // __PVSERIALBRIDGEMANAGERWND_H__

