// *****************************************************************************
//
//     Copyright (c) 2010, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVIMAGE_H__
#define __PVIMAGE_H__

#include <PvResult.h>
#include <PvTypes.h>
#include <PvPixelType.h>


class PvBuffer;

namespace PvBufferLib
{
    class Image;
    class Buffer;
}


class PV_BUFFER_API PvImage
{
public:

    const PvUInt8 *GetDataPointer() const;
    PvUInt8 *GetDataPointer();

    PvUInt32 GetWidth() const;
    PvUInt32 GetHeight() const;
    PvPixelType GetPixelType() const;
    PvUInt32 GetBitsPerPixel() const;
    static PvUInt32 GetPixelSize( PvPixelType aPixelType );

    PvUInt32 GetRequiredSize() const;
    PvUInt32 GetImageSize() const;
    PvUInt32 GetEffectiveImageSize() const;

    PvUInt32 GetOffsetX() const;
    PvUInt32 GetOffsetY() const;
    PvUInt16 GetPaddingX() const;
    PvUInt16 GetPaddingY() const;

    PvResult Alloc( PvUInt32 aSizeX, PvUInt32 aSizeY, PvPixelType aPixelType, PvUInt16 aPaddingX = 0, PvUInt16 aPaddingY = 0 );
	void Free();

    PvResult Attach( void * aRawBuffer, PvUInt32 aSizeX, PvUInt32 aSizeY, PvPixelType aPixelType, PvUInt16 aPaddingX = 0, PvUInt16 aPaddingY = 0 );
	PvUInt8 *Detach();

    bool IsPartialLineMissing() const;
    bool IsFullLineMissing() const;
    bool IsEOFByLineCount() const;
    bool IsInterlacedEven() const;
    bool IsInterlacedOdd() const;
    bool IsImageDropped() const;
    bool IsDataOverrun() const;

	PvBuffer *GetBuffer();

protected:

    PvImage( PvBufferLib::Image *aImage );
    virtual ~PvImage();

private:

	friend class PvBufferLib::Buffer;

    PvBufferLib::Image * mThis;

	// Not implemented
	PvImage( const PvImage & );
	const PvImage &operator=( const PvImage & );
};


#ifdef PV_DEBUG
    #include <PvBufferLib/Buffer.h>
#endif // PV_DEBUG


#endif // __PvImage_H__

