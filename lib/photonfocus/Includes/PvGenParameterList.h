// *****************************************************************************
//
//     Copyright (c) 2009, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PV_GENPARAMETERLIST_H__
#define __PV_GENPARAMETERLIST_H__

#include <PvGenICamLib.h>
#include <PvGenParameter.h>


class PvGenParameter;


namespace PvGenICamLib
{
    class GenParameterList;
}


class PV_GENICAM_API PvGenParameterList
{
public:

    PvGenParameterList();
    ~PvGenParameterList();

    void Clear();
    void Add( PvGenParameter *aParameter );

    PvUInt32 GetSize() const;
    PvGenParameter *GetItem( PvUInt32 aIndex );
	PvGenParameter *operator[]( PvUInt32 aIndex );
  
    PvGenParameter *GetFirst();
    PvGenParameter *GetNext();

protected:

private:

#ifndef PV_GENERATING_DOXYGEN_DOC

    PvGenICamLib::GenParameterList *mThis;

#endif // PV_GENERATING_DOXYGEN_DOC

    // Not implemented
	PvGenParameterList( const PvGenParameterList & );
	const PvGenParameterList &operator=( const PvGenParameterList & );
};


#endif // __PV_GENPARAMETERLIST_H__


