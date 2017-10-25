// *****************************************************************************
//
//     Copyright (c) 2007, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PV_GENPARAMETER_H__
#define __PV_GENPARAMETER_H__

#include <PvString.h>
#include <PvResult.h>
#include <PvGenICamLib.h>
#include <PvGenTypes.h>
#include <PvGenParameterList.h>


class PvGenEventSink;
class PvGenParameterList;

namespace PvGenICamLib
{
    class GenParameterArray;
    class GenParameterInternal;
    class GenParameterArrayManager;

}; // namespace PvGenICamLib

namespace GenApi
{
    struct INode;
}; // namespace GenApi

#ifdef PV_DEBUG
    class PvGenBoolean;
    class PvGenCommand;
    class PvGenEnum;
    class PvGenFloat;
    class PvGenInteger;
    class PvGenRegister;
    class PvGenString;
#endif

class PvGenParameter
{
public:

	PV_GENICAM_API PvString GetName() const;
	PV_GENICAM_API PvResult GetName( PvString &aName ) const;
	PV_GENICAM_API PvResult GetType( PvGenType &aType ) const;
	PV_GENICAM_API PvResult GetCategory( PvString &aCategory ) const;
	PV_GENICAM_API PvResult GetToolTip( PvString &aToolTip ) const;
	PV_GENICAM_API PvResult GetDescription( PvString &aDescription ) const;
    PV_GENICAM_API PvResult GetDisplayName( PvString &aDisplayName ) const;
    PV_GENICAM_API PvResult GetNameSpace( PvGenNameSpace &aNameSpace ) const;

	PV_GENICAM_API PvResult IsValueCached( bool &aCached ) const;
	PV_GENICAM_API PvResult IsPersistent( bool &aPersistent ) const;
	PV_GENICAM_API PvResult IsAvailable( bool &aAvailable ) const;
	PV_GENICAM_API PvResult IsWritable( bool &aWritable ) const;
	PV_GENICAM_API PvResult IsReadable( bool &aReadable ) const;
	PV_GENICAM_API PvResult IsStreamable( bool &aStreamable ) const;

	PV_GENICAM_API bool IsValueCached() const;
	PV_GENICAM_API bool IsPersistent() const;
	PV_GENICAM_API bool IsAvailable() const;
	PV_GENICAM_API bool IsWritable() const;
	PV_GENICAM_API bool IsReadable() const;
	PV_GENICAM_API bool IsVisible( PvGenVisibility aCurrentVisibility ) const;
	PV_GENICAM_API bool IsStreamable() const;

	PV_GENICAM_API PvResult GetVisibility( PvGenVisibility &aVisibility ) const;
	PV_GENICAM_API PvResult IsVisible( PvGenVisibility aCurrentVisibility, bool &aVisible ) const;

    PV_GENICAM_API PvResult IsSelector( bool &aSelector ) const;
    PV_GENICAM_API PvResult GetSelectedParameters( PvGenParameterList &aList ) const;
    PV_GENICAM_API PvResult GetSelectingParameters( PvGenParameterList &aList ) const;

    PV_GENICAM_API PvResult RegisterEventSink( PvGenEventSink *aEventSink );
	PV_GENICAM_API PvResult UnregisterEventSink( PvGenEventSink *aEventSink );

	PV_GENICAM_API PvString ToString() const;
    PV_GENICAM_API PvResult ToString( PvString &aValue ) const;
    PV_GENICAM_API PvResult FromString( const PvString &aValue );

	PV_GENICAM_API GenApi::INode *GetNode();

    // Temporarily disable (or re-enable) parameter update notifications
	PV_GENICAM_API bool GetUpdatesEnabled();
	PV_GENICAM_API void SetUpdatesEnabled( bool aEnabled );

protected:

	PvGenParameter();
	virtual ~PvGenParameter();

#ifndef PV_GENERATING_DOXYGEN_DOC

    PvGenICamLib::GenParameterInternal *mThis;

    friend class PvGenICamLib::GenParameterArray;
    friend class PvGenICamLib::GenParameterArrayManager;

#endif // PV_GENERATING_DOXYGEN_DOC 

private:

    // Not implemented
	PvGenParameter( const PvGenParameter & );
	const PvGenParameter &operator=( const PvGenParameter & );
};


class PvGenEventSink
{
public:

	virtual void OnParameterUpdate( PvGenParameter *aParameter ) = 0;
};


#ifdef PV_DEBUG
    #include <PvGenICamLib/GenParameter.h>
#endif // PV_DEBUG


#endif // __PV_GENPARAMETER_H__

