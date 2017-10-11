// *****************************************************************************
//
//     Copyright (c) 2007, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PV_GENENUM_H__
#define __PV_GENENUM_H__

#include <PvGenICamLib.h>
#include <PvGenParameter.h>
#include <PvGenEnumEntry.h>


class PvGenEnum : public PvGenParameter
{
public:

	PV_GENICAM_API PvResult SetValue( PvInt64 aValue );
	PV_GENICAM_API PvResult SetValue( const PvString &aValue );
	PV_GENICAM_API PvResult GetValue( PvString &aValue ) const;
	PV_GENICAM_API PvResult GetValue( PvInt64 &aValue ) const;

	PV_GENICAM_API PvResult GetEntriesCount( PvInt64 &aCount ) const;
	PV_GENICAM_API PvResult GetEntryByName( const PvString &aEntryName, const PvGenEnumEntry **aEntry ) const;
	PV_GENICAM_API PvResult GetEntryByIndex( PvInt64 aIndex, const PvGenEnumEntry **aEntry ) const;
	PV_GENICAM_API PvResult GetEntryByValue( PvInt64 aValue, const PvGenEnumEntry **aEntry ) const;

protected:

	PvGenEnum();
	virtual ~PvGenEnum();

private:

    // Not implemented
	PvGenEnum( const PvGenEnum & );
	const PvGenEnum &operator=( const PvGenEnum & );
};


#endif // __PV_GENENUM_H__


