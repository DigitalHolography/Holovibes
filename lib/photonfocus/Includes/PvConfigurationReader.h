// *****************************************************************************
//
//     Copyright (c) 2008, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVCONFIGURATIONREADER_H__
#define __PVCONFIGURATIONREADER_H__


#include <PvResult.h>
#include <PvDevice.h>
#include <PvStream.h>
#include <PvStringList.h>
#include <PvPropertyList.h>
#include <PvPersistenceLib.h>


namespace PvPersistenceLib
{
    class ConfigurationReader;

}; // namespace PvPersistenceLib


class PvConfigurationReader
{
public:
    
    PV_PERSISTENCE_API PvConfigurationReader();
    PV_PERSISTENCE_API ~PvConfigurationReader();
    
    PV_PERSISTENCE_API PvResult Load( const PvString &aFilename );
    
    PV_PERSISTENCE_API PvUInt32 GetDeviceCount();
    PV_PERSISTENCE_API PvResult GetDeviceName( PvUInt32 aIndex, PvString &aName );
    PV_PERSISTENCE_API PvResult Restore( const PvString &aName, PvDevice *aDevice );
    PV_PERSISTENCE_API PvResult Restore( PvUInt32 aIndex, PvDevice *aDevice );
      
    PV_PERSISTENCE_API PvUInt32 GetStreamCount();
    PV_PERSISTENCE_API PvResult GetStreamName( PvUInt32 aIndex, PvString &aName );
    PV_PERSISTENCE_API PvResult Restore( const PvString &aName, PvStream &aStream );
    PV_PERSISTENCE_API PvResult Restore( PvUInt32 aIndex, PvStream &aStream );
    
    PV_PERSISTENCE_API PvUInt32 GetStringCount();
    PV_PERSISTENCE_API PvResult GetStringName( PvUInt32 aIndex, PvString &aName );
    PV_PERSISTENCE_API PvResult Restore( const PvString &aKey, PvString &aValue );
    PV_PERSISTENCE_API PvResult Restore( PvUInt32 aIndex, PvString &aValue );

    PV_PERSISTENCE_API PvUInt32 GetGenParameterArrayCount();
    PV_PERSISTENCE_API PvResult GetGenParameterArrayName( PvUInt32 aIndex, PvString &aName );
    PV_PERSISTENCE_API PvResult Restore( const PvString &aKey, PvGenParameterArray &aParameterArray );
    PV_PERSISTENCE_API PvResult Restore( PvUInt32 aIndex, PvGenParameterArray &aParameterArray );

    PV_PERSISTENCE_API PvUInt32 GetPropertyListCount();
    PV_PERSISTENCE_API PvResult GetPropertyListName( PvUInt32 aIndex, PvString &aName );
    PV_PERSISTENCE_API PvResult Restore( const PvString &aKey, PvPropertyList &aPropertyList );
    PV_PERSISTENCE_API PvResult Restore( PvUInt32 aIndex, PvPropertyList &aPropertyList );

    PV_PERSISTENCE_API void SetErrorList( PvStringList *aList, const PvString &aPrefix );

private:

    PvPersistenceLib::ConfigurationReader *mThis;
    
    PvStringList *mErrorList;
    PvString mErrorPrefix;

	 // Not implemented
	PvConfigurationReader( const PvConfigurationReader& );
	const PvConfigurationReader &operator=( const PvConfigurationReader & );

};


#ifdef PV_DEBUG
    #include <PvPersistenceLib/ConfigurationReader.h>
#endif // PV_DEBUG


#endif // __PVCONFIGURATIONREADER_H__


