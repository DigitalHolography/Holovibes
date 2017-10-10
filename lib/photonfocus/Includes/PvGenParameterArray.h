// *****************************************************************************
//
//     Copyright (c) 2007, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PV_GENPARAMETERARRAY_H__
#define __PV_GENPARAMETERARRAY_H__


#include <PvGenICamLib.h>
#include <PvGenString.h>
#include <PvGenInteger.h>
#include <PvGenEnum.h>
#include <PvGenFloat.h>
#include <PvGenCommand.h>
#include <PvGenBoolean.h>
#include <PvGenRegister.h>
#include <PvGenCategory.h>


namespace PvGenICamLib
{
    class GenParameterArray;
    class GenParameterArrayManager;
}


namespace GenApi
{
    struct INodeMap;
}; // namespace GenApi


class PvGenFile;
class PvConfigurationWriter;
class PvConfigurationReader;


class PvGenParameterArray
{
public:

	PV_GENICAM_API PvGenParameterArray();
	PV_GENICAM_API virtual ~PvGenParameterArray();

	PV_GENICAM_API PvUInt32 GetCount() const;
	PV_GENICAM_API PvGenParameter *Get( PvUInt32 aIndex );
	PV_GENICAM_API PvGenParameter *Get( const PvString &aName );

	PV_GENICAM_API PvGenParameter *operator[]( PvUInt32 aIndex );
	PV_GENICAM_API PvGenParameter *operator[]( const PvString &aName );

	PV_GENICAM_API PvUInt32 GetCategoryCount() const;
	PV_GENICAM_API PvGenCategory *GetCategory( PvUInt32 aIndex );
	PV_GENICAM_API PvGenCategory *GetCategory( const PvString &aName );

    PV_GENICAM_API PvResult InvalidateCache();

	PV_GENICAM_API GenApi::INodeMap *GetNodeMap();

	// Helper methods for direct feature access. Returns NULL on type mismatch.
	PV_GENICAM_API PvGenInteger *GetInteger( const PvString &aName );
	PV_GENICAM_API PvGenFloat *GetFloat( const PvString &aName );
	PV_GENICAM_API PvGenEnum *GetEnum( const PvString &aName );
	PV_GENICAM_API PvGenBoolean *GetBoolean( const PvString &aName );
	PV_GENICAM_API PvGenCommand *GetCommand( const PvString &aName );
	PV_GENICAM_API PvGenString *GetString( const PvString &aName );
	PV_GENICAM_API PvGenRegister *GetRegister( const PvString &aName );

	// Helper methods for direct feature get/set
	PV_GENICAM_API PvResult GetIntegerValue( const PvString &aName, PvInt64 &aValue );
	PV_GENICAM_API PvResult SetIntegerValue( const PvString &aName, PvInt64 aValue );
	PV_GENICAM_API PvResult GetFloatValue( const PvString &aName, double &aValue );
	PV_GENICAM_API PvResult SetFloatValue( const PvString &aName, double aValue );
	PV_GENICAM_API PvResult GetEnumValue( const PvString &aName, PvString &aValue );
	PV_GENICAM_API PvResult GetEnumValue( const PvString &aName, PvInt64 &aValue );
	PV_GENICAM_API PvResult SetEnumValue( const PvString &aName, const PvString &aValue );
	PV_GENICAM_API PvResult SetEnumValue( const PvString &aName, PvInt64 aValue );
	PV_GENICAM_API PvResult GetBooleanValue( const PvString &aName, bool &aValue );
	PV_GENICAM_API PvResult SetBooleanValue( const PvString &aName, bool aValue );
	PV_GENICAM_API PvResult GetString( const PvString &aName, PvString &aValue );
	PV_GENICAM_API PvResult SetString( const PvString &aName, const PvString &aValue );
	PV_GENICAM_API PvResult ExecuteCommand( const PvString &aName );

    // Range accessors
    PV_GENICAM_API PvResult GetIntegerRange( const PvString &aName, PvInt64 &aMin, PvInt64 &aMax );
    PV_GENICAM_API PvResult GetFloatRange( const PvString &aName, double &aMin, double &aMax );

    // Chunks
    PV_GENICAM_API PvResult AttachDataChunks( PvUInt8 *aBuffer, PvUInt32 aBufferLength );
    PV_GENICAM_API PvResult DetachDataChunks();

    PV_GENICAM_API PvGenAccessMode GetAccessMode() const;

    PV_GENICAM_API PvResult Poll();

protected:

    PvGenICamLib::GenParameterArray *mThis;

private:

    friend class PvGenFile;
    friend class PvConfigurationWriter;
    friend class PvConfigurationReader;
    friend class PvGenICamLib::GenParameterArrayManager;

	 // Not implemented
	PvGenParameterArray( const PvGenParameterArray & );
	const PvGenParameterArray &operator=( const PvGenParameterArray & );

};


#endif // __PV_GENPARAMETERARRAY_H__
