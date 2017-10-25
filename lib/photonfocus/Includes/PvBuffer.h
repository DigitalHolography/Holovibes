// *****************************************************************************
//
//     Copyright (c) 2007, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVBUFFER_H__
#define __PVBUFFER_H__


#include <PvResult.h>
#include <PvString.h>
#include <PvTypes.h>

#include <PvBufferLib.h>
#include <PvPixelType.h>
#include <PvPayloadType.h>
#include <PvImage.h>
#include <PvRawData.h>


namespace PvBufferLib
{
    class Buffer;
}

namespace PvStreamRawLib
{
	class Pipeline;
}

class PvPipeline;
class PvStreamBase;
class PvBufferConverter;
class PvFilterRGB;
class PvFilterDeinterlace;
class PvImage;
class PvTransmitterRaw;

class PV_BUFFER_API PvBuffer
{
public:

    PvBuffer( PvPayloadType aPayloadType = PvPayloadTypeImage );
    virtual ~PvBuffer();

	PvPayloadType GetPayloadType() const;
	
    PvImage *GetImage();
	const PvImage *GetImage() const;
    PvRawData *GetRawData();
	const PvRawData *GetRawData() const;

    const PvUInt8 * GetDataPointer() const;
    PvUInt8 * GetDataPointer();

    PvUInt64 GetID() const;
    void SetID( PvUInt64 aValue );

    bool IsExtendedID() const;

    PvUInt32 GetAcquiredSize() const;
    PvUInt32 GetRequiredSize() const;
	PvUInt32 GetSize() const;

    PvResult Reset( PvPayloadType aPayloadType = PvPayloadTypeImage );

    PvResult Alloc( PvUInt32 aSize );
    void Free();

    PvResult Attach( void * aBuffer, PvUInt32 aSize );
    PvUInt8 *Detach();

    PvUInt64 GetBlockID() const;
    PvResult GetOperationResult() const;
    PvUInt64 GetTimestamp() const;
    PvResult SetTimestamp( PvUInt64 aTimestamp );

    PvUInt32 GetPacketsRecoveredCount() const;
    PvUInt32 GetPacketsRecoveredSingleResendCount() const;
    PvUInt32 GetResendGroupRequestedCount() const;
    PvUInt32 GetResendPacketRequestedCount() const;
    PvUInt32 GetLostPacketCount() const;
    PvUInt32 GetIgnoredPacketCount() const;
    PvUInt32 GetRedundantPacketCount() const;
    PvUInt32 GetPacketOutOfOrderCount() const;

    PvResult GetMissingPacketIdsCount( PvUInt32& aCount );
    PvResult GetMissingPacketIds( PvUInt32 aIndex, PvUInt32& aPacketIdLow, PvUInt32& aPacketIdHigh );

    bool HasChunks() const;
    PvUInt32 GetChunkCount();
    PvResult GetChunkIDByIndex( PvUInt32 aIndex, PvUInt32 &aID );
    PvUInt32 GetChunkSizeByIndex( PvUInt32 aIndex );
    PvUInt32 GetChunkSizeByID( PvUInt32 aID );
    const PvUInt8 *GetChunkRawDataByIndex( PvUInt32 aIndex );
    const PvUInt8 *GetChunkRawDataByID( PvUInt32 aID );
    PvUInt32 GetPayloadSize() const;

    bool IsHeaderValid() const;
    bool IsTrailerValid() const;

private:

    // Not implemented
	PvBuffer( const PvBuffer & );
	const PvBuffer &operator=( const PvBuffer & );

	friend class PvStreamRawLib::Pipeline;
    friend class PvPipeline;
    friend class PvStreamBase;
    friend class PvBufferConverter;
    friend class PvFilterRGB;
    friend class PvFilterDeinterlace;
    friend class PvTransmitterRaw;

    PvBufferLib::Buffer * mThis;
};


#ifdef PV_DEBUG
    #include <PvBufferLib/Buffer.h>
#endif // PV_DEBUG


#endif // __PVBUFFER_H__
