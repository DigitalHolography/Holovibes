// *****************************************************************************
//
//     Copyright (c) 2012, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PV_ACTIONCOMMAND_H__
#define __PV_ACTIONCOMMAND_H__

#include <PvDeviceLib.h>
#include <PvActionAckStatusEnum.h>

#include <PvResult.h>
#include <PvTypes.h>

namespace PvDeviceLib
{
    class ActionCommand;
};



class PV_DEVICE_API PvActionCommand
{
public:    
    PvActionCommand();
    ~PvActionCommand();

    PvUInt32 GetInterfaceCount() const;
    PvResult GetInterfaceMACAddress( PvUInt32 aIndex, PvString& aInterfaceMACAddress ) const;
    PvResult GetInterfaceIPAddress( PvUInt32 aIndex, PvString& aInterfaceIPAddress ) const;
    PvResult GetInterfaceDescription( PvUInt32 aIndex, PvString& aInterfaceDescription ) const;
    PvResult GetInterfaceEnabled( PvUInt32 aIndex, bool& aEnabled ) const;
    PvResult SetInterfaceEnabled( PvUInt32 aIndex, bool aEnabled );

    PvUInt32 GetDeviceKey() const;
    void SetDeviceKey( PvUInt32 aDeviceKey );
    PvUInt32 GetGroupKey() const;
    void SetGroupKey( PvUInt32 aGroupKey );
    PvUInt32 GetGroupMask() const;
    void SetGroupMask( PvUInt32 aGroupMask );
    bool GetScheduledTimeEnable() const;
    void SetScheduledTimeEnable( bool aEnabled );
    PvUInt64 GetScheduledTime() const;
    void SetScheduledTime( PvUInt64 aScheduledTime );

    PvResult Send( PvUInt32 aTimeout, PvUInt32 aDeviceCount = 0, bool aRequestAcknowledges = true );
    PvResult Resend( PvUInt32 aTimeout, PvUInt32 aDeviceCount = 0, bool aRequestAcknowledges = true );

    PvUInt32 GetAcknowledgeCount() const;
    PvResult GetAcknowledgeIPAddress( PvUInt32 aIndex, PvString& aIPAddress ) const;
    PvResult GetAcknowledgeStatus( PvUInt32 aIndex, PvActionAckStatusEnum& aStatus ) const;

    PvUInt32 GetActionAckStatusOKCount() const;
    PvUInt32 GetActionAckStatusLateCount() const;
    PvUInt32 GetActionAckStatusOverflowCount() const;
    PvUInt32 GetActionAckStatusNoRefTimeCount() const;
    void ResetStatistics();

private:
    PvDeviceLib::ActionCommand *mThis;
};


#endif // __PV_ACTIONCOMMAND_H__


