// *****************************************************************************
//
//     Copyright (c) 2008, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVPIPELINE_H__
#define __PVPIPELINE_H__


#include <PvTypes.h>
#include <PvResult.h>
#include <PvStreamBase.h>


namespace PvStreamRawLib
{
    class Pipeline;

}; // namespace PvStreamLib


class PvPipelineEventSink;


class PV_STREAMRAW_API PvPipeline
{

public:

    PvPipeline( PvStreamBase *aStream );
    virtual ~PvPipeline();

    PvUInt32 GetBufferSize() const;
    PvUInt32 GetBufferCount() const;
    PvUInt32 GetOutputQueueSize() const;
    PvUInt32 GetDefaultBufferSize() const;
    bool GetHandleBufferTooSmall() const;

    bool IsStarted();

    void SetDefaultBufferSize( PvUInt32 aSize );
    void SetBufferSize( PvUInt32 aSize );
    PvResult SetBufferCount( PvUInt32 aBufferCount );
    void SetHandleBufferTooSmall( bool aValue );

    PvResult RetrieveNextBuffer(
        PvBuffer ** aBuffer,
        PvUInt32 aTimeout = 0xFFFFFFFF,
		PvResult * aOperationResult = NULL );

    PvResult ReleaseBuffer( PvBuffer * aBuffer );

    PvResult Start();
    PvResult Stop();
    PvResult Reset();

	// Notifications
    PvResult RegisterEventSink( PvPipelineEventSink *aEventSink );
    PvResult UnregisterEventSink( PvPipelineEventSink *aEventSink );

    PvUInt32 GetBufferHandlingThreadPriority() const;
    PvResult SetBufferHandlingThreadPriority( PvUInt32 aPriority );

protected:

private:

    PvStreamRawLib::Pipeline * mThis;

	 // Not implemented
	PvPipeline( const PvPipeline& );
	const PvPipeline &operator=( const PvPipeline & );

};

class PV_STREAMRAW_API PvPipelineEventSink
{
public:

    PvPipelineEventSink();
    virtual ~PvPipelineEventSink();

    // Notifications
    virtual void OnBufferCreated( PvPipeline *aPipeline, PvBuffer *aBuffer );
    virtual void OnBufferDeleted( PvPipeline *aPipeline, PvBuffer *aBuffer );
    virtual void OnStart( PvPipeline *aPipeline );
    virtual void OnStop( PvPipeline *aPipeline );
    virtual void OnReset( PvPipeline *aPipeline );
    virtual void OnBufferTooSmall( PvPipeline *aPipeline, bool *aReallocAll, bool *aResetStats );

};


#ifdef PV_DEBUG
    #include <PvStreamRawLib/Pipeline.h>
#endif // PV_DEBUG


#endif // __PVPIPELINE_H__

