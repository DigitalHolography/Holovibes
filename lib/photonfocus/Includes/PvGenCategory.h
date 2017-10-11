// *****************************************************************************
//
//     Copyright (c) 2011, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PV_GENCATEGORY_H__
#define __PV_GENCATEGORY_H__

#include <PvString.h>
#include <PvResult.h>
#include <PvGenICamLib.h>
#include <PvGenTypes.h>


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


class PvGenCategory
{
public:

	PV_GENICAM_API PvResult GetName( PvString &aName ) const;
	PV_GENICAM_API PvResult GetToolTip( PvString &aToolTip ) const;
	PV_GENICAM_API PvResult GetDescription( PvString &aDescription ) const;
    PV_GENICAM_API PvResult GetDisplayName( PvString &aDisplayName ) const;
    PV_GENICAM_API PvResult GetNameSpace( PvGenNameSpace &aNameSpace ) const;

	PV_GENICAM_API GenApi::INode *GetNode();

protected:

	PvGenCategory();
	virtual ~PvGenCategory();

#ifndef PV_GENERATING_DOXYGEN_DOC

    PvGenICamLib::GenParameterInternal *mThis;

    friend class PvGenICamLib::GenParameterArray;
    friend class PvGenICamLib::GenParameterArrayManager;

#endif // PV_GENERATING_DOXYGEN_DOC 

private:

    // Not implemented
	PvGenCategory( const PvGenCategory & );
	const PvGenCategory &operator=( const PvGenCategory & );
};


#ifdef PV_DEBUG
    #include <PvGenICamLib/GenParameter.h>
#endif // PV_DEBUG


#endif // __PV_GENCATEGORY_H__

