// *****************************************************************************
//
//     Copyright (c) 2012, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PV_STREAM_INFO_H__
#define __PV_STREAM_INFO_H__

#include <PvGUIUtilsLib.h>
#include <PvStream.h>


namespace PvGUIUtilsLib
{
    class StreamInfo;

}; // namespace PvGUIUtilsLib


class PV_GUIUTILS_API PvStreamInfo
{
public:

    PvStreamInfo( PvStream *aStream );
	~PvStreamInfo();

	PvString GetStatistics( PvUInt32 aDisplayFrameRate );
	PvString GetErrors();
	PvString GetWarnings( bool aPipelineReallocated );

private:

    PvGUIUtilsLib::StreamInfo *mThis;

	 // Not implemented
	PvStreamInfo( const PvStreamInfo & );
	const PvStreamInfo &operator=( const PvStreamInfo & );

};


#endif //__PV_STREAM_INFO_H__

