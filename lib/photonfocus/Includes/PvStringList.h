// *****************************************************************************
//
//     Copyright (c) 2011, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVSTRINGLIST_H__
#define __PVSTRINGLIST_H__

#include <PvBaseLib.h>
#include <PvString.h>


namespace PvBaseLib
{
    class StringList;
}


class PV_BASE_API PvStringList
{
public:

    PvStringList();
    ~PvStringList();

    void Clear();
    void Add( const PvString &aString );

    PvUInt32 GetSize() const;
    PvString *GetItem( PvUInt32 aIndex );
    PvString *operator[]( PvUInt32 aIndex );
  
    PvString *GetFirst();
    PvString *GetNext();

protected:

private:

#ifndef PV_GENERATING_DOXYGEN_DOC

    PvBaseLib::StringList *mThis;

#endif // PV_GENERATING_DOXYGEN_DOC

    // Not implemented
	PvStringList( const PvStringList & );
	const PvStringList &operator=( const PvStringList & );
};


#endif // __PVSTRINGLIST_H__


