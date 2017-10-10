// *****************************************************************************
//
//     Copyright (c) 2010, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVDISPLAYTHREAD_H__
#define __PVDISPLAYTHREAD_H__


#include <PvGUIUtilsLib.h>
#include <PvTypes.h>
#include <PvResult.h>
#include <PvPropertyList.h>


namespace PvGUIUtilsLib
{
    class DisplayThread;
    class DisplayThreadProxy;

}; // namespace PvGUIUtilsLib

class PvBuffer;
class PvPipeline;
class PvGenParameterArray;

typedef enum
{
    PvDeinterlacingDisabled = 0,
    PvDeinterlacingWeaving = 1

} PvDeinterlacingType;


class PV_GUIUTILS_API PvDisplayThread
{
public:

    PvDisplayThread();
    ~PvDisplayThread();

    PvResult Start( PvPipeline *aPipeline, PvGenParameterArray *aParameters );
    PvResult Stop( bool aWait );
    PvResult WaitComplete();
    bool IsRunning() const;

    PvUInt32 GetPriority() const;
    PvResult SetPriority( PvUInt32 aPriority );

    PvBuffer *RetrieveLatestBuffer();
    void ReleaseLatestBuffer();

    bool GetKeepPartialImagesEnabled() const;
    void SetKeepPartialImagesEnabled( bool aEnabled );

    bool GetBufferLogErrorEnabled() const;
    void SetBufferLogErrorEnabled( bool aValue );

    bool GetBufferLogAllEnabled() const;
    void SetBufferLogAllEnabled( bool aValue );

    PvDeinterlacingType GetDeinterlacing() const;
    void SetDeinterlacing( PvDeinterlacingType aValue );

    PvUInt32 GetFPS() const;
    PvUInt32 GetTargetFPS() const;
    void SetTargetFPS( PvUInt32 aValue );

    bool GetVSyncEnabled() const;
    void SetVSyncEnabled( bool aEnabled );

    bool GetDisplayChunkDataEnabled() const;
    void SetDisplayChunkDataEnabled( bool aEnabled );

    void ResetStatistics();

    virtual PvResult Save( PvPropertyList &aPropertyList );
    virtual PvResult Load( PvPropertyList &aPropertyList );

protected:

    virtual void OnBufferRetrieved( PvBuffer *aBuffer );
    virtual void OnBufferDisplay( PvBuffer *aBuffer );
    virtual void OnBufferDone( PvBuffer *aBuffer );
    virtual void OnBufferLog( const PvString &aLog );
    virtual void OnBufferTextOverlay( const PvString &aText );

private:

    PvGUIUtilsLib::DisplayThread *mThis;
    friend class PvGUIUtilsLib::DisplayThreadProxy;

	 // Not implemented
	PvDisplayThread( const PvDisplayThread & );
	const PvDisplayThread &operator=( const PvDisplayThread & );

};


#endif // __PVDISPLAYTHREAD_H__

