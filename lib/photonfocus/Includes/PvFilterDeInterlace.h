// *****************************************************************************
//
//     Copyright (c) 200, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVFILTERDEINTERLACE_H__
#define __PVFILTERDEINTERLACE_H__


#include <PvBufferLib.h>
#include <PvFilter.h>


namespace PvBufferLib
{
    class FilterDeinterlace;
};


class PV_BUFFER_API PvFilterDeinterlace : public PvFilter
{

public:

    PvFilterDeinterlace();
    virtual ~PvFilterDeinterlace();

    PvResult Apply( const PvBuffer *aIn, PvBuffer *aOut );
    PvResult Apply( const PvBuffer *aInOdd, const PvBuffer *aInEven, PvBuffer *aOut );

    PvResult ApplyOdd( const PvBuffer *aIn, PvBuffer *aOut );
    PvResult ApplyEven( const PvBuffer *aIn, PvBuffer *aOut );

    bool GetFieldInversion() const;
    void SetFieldInversion( bool aInvert );

protected:

private:

    // Not implemented
	PvFilterDeinterlace( const PvFilterDeinterlace & );
	const PvFilterDeinterlace &operator=( const PvFilterDeinterlace & );

    PvBufferLib::FilterDeinterlace *mThis;
};


#ifdef PV_DEBUG
    #include <PvBufferLib/FilterDeInterlace.h>
#endif // PV_DEBUG


#endif // __PVFILTERDEINTERLACE_H__
