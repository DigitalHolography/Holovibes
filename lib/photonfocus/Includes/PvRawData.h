// *****************************************************************************
//
//     Copyright (c) 2010, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVRAWDATA_H__
#define __PVRAWDATA_H__

#include <PvResult.h>
#include <PvTypes.h>
#include <PvPixelType.h>


namespace PvBufferLib
{
    class RawData;
    class Buffer;
}


class PV_BUFFER_API PvRawData
{
public:


protected:

	PvRawData( PvBufferLib::RawData *aRawData );
    virtual ~PvRawData();

public:

    PvUInt64 GetPayloadLength() const;

    PvResult Alloc( PvUInt64 aPayloadLength );
	void Free();

    PvResult Attach( void * aRawBuffer, PvUInt64 aPayloadLength );
	PvUInt8 *Detach();

private:

	friend class PvBufferLib::Buffer;

	// Not implemented
	PvRawData( const PvRawData & );
	const PvRawData &operator=( const PvRawData & );

    PvBufferLib::RawData *mThis;
};


#ifdef PV_DEBUG
    #include <PvBufferLib/Buffer.h>
#endif // PV_DEBUG


#endif // __PVRAWDATA_H__

