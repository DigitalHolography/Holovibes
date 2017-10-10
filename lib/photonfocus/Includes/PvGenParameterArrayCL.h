// *****************************************************************************
//
//     Copyright (c) 2011, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PV_GENPARAMETERARRAYCL_H__
#define __PV_GENPARAMETERARRAYCL_H__


#include <PvGenParameterArray.h>
#include <PvStringList.h>


class PvGenParameterArrayCL : public PvGenParameterArray
{
public:

	PV_GENICAM_API PvGenParameterArrayCL();
	PV_GENICAM_API virtual ~PvGenParameterArrayCL();

    PV_GENICAM_API static void GetPortIDs( PvStringList &aPortIDList );
    PV_GENICAM_API static void GetDeviceTemplates( PvStringList &aDeviceTemplateList );
    PV_GENICAM_API static PvString ProbeDevice( const PvString &aPortID, const PvString &aDeviceTemplate, PvUInt32 aSerialTimeout = 500 );
 
    PV_GENICAM_API PvResult Connect( const PvString &aPortID, const PvString &aDeviceID );
    PV_GENICAM_API PvResult GetXMLIDs( PvStringList &aXMLIDList );
    PV_GENICAM_API PvResult Build( const PvString &aXMLID );

    PV_GENICAM_API PvResult SetSerialTimeout( PvUInt32 aTimeout );
    PV_GENICAM_API PvResult GetSerialTimeout( PvUInt32 &aTimeout ) const;

    PV_GENICAM_API PvString GetPortID() const;
    PV_GENICAM_API PvString GetDeviceID() const;
    PV_GENICAM_API PvString GetXMLID() const;

protected:

private:

	 // Not implemented
	PvGenParameterArrayCL( const PvGenParameterArrayCL & );
	const PvGenParameterArrayCL &operator=( const PvGenParameterArrayCL & );

};


#endif // __PV_GENPARAMETERARRAYCL_H__

