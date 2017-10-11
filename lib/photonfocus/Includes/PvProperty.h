// *****************************************************************************
//
//     Copyright (c) 2011, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVPROPERTY_H__
#define __PVPROPERTY_H__

#include <PvPersistenceLib.h>
#include <PvString.h>
#include <PvResult.h>


class PV_PERSISTENCE_API PvProperty
{
public:
    
    PvProperty();
    PvProperty( const PvString &aName, const PvString &aValue );
    ~PvProperty();
    
	PvProperty( const PvProperty &aProperty );
	const PvProperty &operator=( const PvProperty &aProperty );

    void SetName( const PvString &aName );
    PvString GetName() const;

    void SetValue( const PvString &aValue );
    void SetValue( PvInt64 aValue );
    void SetValue( double aValue );

    PvString GetValue() const;
    PvResult GetValue( PvInt64 &aValue ) const;
    PvResult GetValue( double &aValue ) const;

private:

    PvString mName;
    PvString mValue;

};


#endif // __PVPROPERTY_H__

