// *****************************************************************************
//
//     Copyright (c) 2011, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PV_ACQUISITION_STATE_MANAGER__
#define __PV_ACQUISITION_STATE_MANAGER__

#include <PvGUIUtilsLib.h>
#include <PvDevice.h>
#include <PvStreamBase.h>


namespace PvGUIUtilsLib
{
    class AcquisitionStateManager;

}; // namespace PvGUIUtilsLib


typedef enum
{
    PvAcquisitionStateUnknown = -1,
    PvAcquisitionStateUnlocked = 0,
    PvAcquisitionStateLocked

} PvAcquisitionState;


class PV_GUIUTILS_API PvAcquisitionStateEventSink
{
public:

	PvAcquisitionStateEventSink();
	virtual ~PvAcquisitionStateEventSink();

    virtual void OnAcquisitionStateChanged( PvDevice* aDevice, PvStreamBase* aStream, PvUInt32 aSource, PvAcquisitionState aState );

};


class PV_GUIUTILS_API PvAcquisitionStateManager
{
public:

    PvAcquisitionStateManager( PvDevice* aDevice, PvStreamBase* aStream = 0, PvUInt32 aSource = 0 );
    virtual ~PvAcquisitionStateManager();

    PvResult Start();
    PvResult Stop();

    PvAcquisitionState GetState() const;
    PvUInt32 GetSource() const;

    PvResult RegisterEventSink( PvAcquisitionStateEventSink* aEventSink );
    PvResult UnregisterEventSink( PvAcquisitionStateEventSink* aEventSink );

private:

    PvGUIUtilsLib::AcquisitionStateManager *mThis;

	 // Not implemented
	PvAcquisitionStateManager( const PvAcquisitionStateManager & );
	const PvAcquisitionStateManager &operator=( const PvAcquisitionStateManager & );

};


#endif //__PV_ACQUISITION_STATE_MANAGER__

