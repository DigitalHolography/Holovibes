// *****************************************************************************
//
//     Copyright (c) 2008, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVFILTERRGB_H__
#define __PVFILTERRGB_H__


#include <PvBufferLib.h>
#include <PvFilter.h>


namespace PvBufferLib
{
    class FilterRGB;
};


class PvBufferConverter;


class PV_BUFFER_API PvFilterRGB : public PvFilter
{

public:

    PvFilterRGB();
    virtual ~PvFilterRGB();

    PvResult Apply( PvBuffer *aBuffer );

    double GetGainR() const;
    double GetGainG() const;
    double GetGainB() const;

    void SetGainR( double aValue );
    void SetGainG( double aValue );
    void SetGainB( double aValue );

    PvInt32 GetOffsetR() const;
    PvInt32 GetOffsetG() const;
    PvInt32 GetOffsetB() const;
    
    void SetOffsetR( PvInt32 aValue );
    void SetOffsetG( PvInt32 aValue );
    void SetOffsetB( PvInt32 aValue );

    PvResult WhiteBalance( PvBuffer *aBuffer );
    void Reset();

protected:

private:

    friend class PvBufferConverter;

    // Not implemented
	PvFilterRGB( const PvFilterRGB & );
	const PvFilterRGB &operator=( const PvFilterRGB & );

    PvBufferLib::FilterRGB *mThis;
};


#ifdef PV_DEBUG
    #include <PvBufferLib/FilterRGB.h>
#endif // PV_DEBUG


#endif // __PVFILTERRGB_H__
