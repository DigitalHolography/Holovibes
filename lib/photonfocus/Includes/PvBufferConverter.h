// *****************************************************************************
//
//     Copyright (c) 2008, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVBUFFERCONVERTER_H__
#define __PVBUFFERCONVERTER_H__


#include <PvBufferLib.h>
#include <PvBuffer.h>
#include <PvFilterRGB.h>


namespace PvBufferLib
{
    class BufferConverter;
};


typedef enum 
{
    PvBayerFilterSimple = 1,
    PvBayerFilter3X3 = 2

} PvBayerFilterType;


class PV_BUFFER_API PvBufferConverter
{
public:

    PvBufferConverter( PvInt32 aMaxNumberOfThreads = -1 );
    virtual ~PvBufferConverter();

    bool IsConversionSupported( PvPixelType aSource, PvPixelType aDestination );

    PvResult Convert( const PvBuffer *aSource, PvBuffer *aDestination, bool aReallocIfNeeded = true );
    
    PvBayerFilterType GetBayerFilter() const;
    PvResult SetBayerFilter( PvBayerFilterType aFilter );

    PvResult ResetRGBFilter();
    PvResult SetRGBFilter( PvFilterRGB &aFilter );

    PvUInt32 GetConversionThreadsPriority() const;
    PvResult SetConversionThreadsPriority( PvUInt32 aPriority );

protected:

private:

    // Not implemented
	PvBufferConverter( const PvBufferConverter & );
	const PvBufferConverter &operator=( const PvBufferConverter & );

    PvBufferLib::BufferConverter *mThis;
};


#ifdef PV_DEBUG
    #include <PvBufferLib/BufferConverter.h>
#endif // PV_DEBUG


#endif // __PVBUFFERCONVERTER_H__

