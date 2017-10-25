// *****************************************************************************
//
//     Copyright (c) 2008, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVCONFIGURATIONWRITER_H__
#define __PVCONFIGURATIONWRITER_H__


#include <PvResult.h>
#include <PvDevice.h>
#include <PvStream.h>
#include <PvStringList.h>
#include <PvPropertyList.h>
#include <PvPersistenceLib.h>


namespace PvPersistenceLib
{
    class ConfigurationWriter;

}; // namespace PvPersistenceLib


class PvConfigurationWriter
{
public:
    
    PV_PERSISTENCE_API PvConfigurationWriter();
    PV_PERSISTENCE_API ~PvConfigurationWriter();
    
    PV_PERSISTENCE_API PvResult Store( PvDevice *aDevice, PvString aName = PvString( "" ) );
    PV_PERSISTENCE_API PvResult Store( PvStream *aStream, PvString aName = PvString( "" ) );
    PV_PERSISTENCE_API PvResult Store( const PvString &aString, const PvString &aName );
    PV_PERSISTENCE_API PvResult Store( PvGenParameterArray *aGenParameterArray, PvString aName = PvString( "" ) );
    PV_PERSISTENCE_API PvResult Store( PvPropertyList *aPropertyList, PvString aName = PvString( "" ) );

    PV_PERSISTENCE_API PvResult Save( const PvString &aFilename );

    PV_PERSISTENCE_API void SetErrorList( PvStringList *aList, const PvString &aPrefix );

protected:

private:

    PvPersistenceLib::ConfigurationWriter *mThis;

    PvStringList *mErrorList;
    PvString mErrorPrefix;

    // Not implemented
	PvConfigurationWriter( const PvConfigurationWriter& );
	const PvConfigurationWriter &operator=( const PvConfigurationWriter & );

};


#ifdef PV_DEBUG
    #include <PvPersistenceLib/ConfigurationWriter.h>
#endif // PV_DEBUG


#endif // __PVCONFIGURATIONWRITER_H__
