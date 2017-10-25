// *****************************************************************************
//
//     Copyright (c) 2008, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVBUFFERWRITER_H__
#define __PVBUFFERWRITER_H__


#include <PvBufferLib.h>
#include <PvBuffer.h>
#include <PvBufferConverter.h>
#include <PvBufferFormatType.h>


namespace PvBufferLib
{
    class BufferWriter;
}


class PV_BUFFER_API PvBufferWriter
{

public:

    PvBufferWriter();
    virtual ~PvBufferWriter();

    PvResult Store( PvBuffer* aBuffer, const PvString& aFilename, PvBufferFormatType aType = PvBufferFormatBMP );

    PvBufferConverter &GetConverter();

protected:

private:

    // Not implemented
	PvBufferWriter( const PvBufferWriter & );
	const PvBufferWriter &operator=( const PvBufferWriter & );

    PvBufferLib::BufferWriter *mThis;
};


#ifdef PV_DEBUG
    #include <PvBufferLib/BufferWriter.h>
#endif // PV_DEBUG


#endif // __PVBUFFERWRITER_H__
