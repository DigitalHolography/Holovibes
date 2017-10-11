// *****************************************************************************
//
//     Copyright (c) 2008, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVSTATISTICS_H__
#define __PVSTATISTICS_H__


#include <PvTypes.h>
#include <PvResult.h>
#include <PvString.h>
#include <PvStreamRawLib.h>


namespace PvStreamRawLib
{
    class Statistics;
};


typedef enum
{
	PvTimestampSourceHardware = 0,
	PvTimestampSourceSoftware = 1

} PvTimestampSourceType;


class PV_STREAMRAW_API PvStatistics
{
public:

    void Reset();

    PvUInt64 GetBlocksCount() const;
    PvUInt64 GetSamplingTime() const;
    PvUInt64 GetBytesCount() const;
    float GetBandwidth() const;
    float GetBandwidthAverage() const;
    float GetAcquisitionRate() const;
    float GetAcquisitionRateAverage() const;
    PvResult GetLastError() const;
    PvUInt32 GetErrorCount() const;

    bool GetTimeoutCountedAsError() const;
    void SetTimeoutCountedAsError( bool aCounted );
    bool GetAbortCountedAsError() const;
    void SetAbortCountedAsError( bool aCounted );

	PvTimestampSourceType GetTimestampSourceEffective() const;
	PvTimestampSourceType GetTimestampSourcePreferred() const;
	void SetTimestampSourcePreferred( PvTimestampSourceType aSource );

    PvUInt32 GetExpectedResendCount() const;
	PvUInt32 GetUnexpectedResendCount() const;
    PvUInt32 GetLostPacketCount() const;
    PvUInt32 GetIgnoredPacketCount() const;

    PvUInt64 GetBlockIDsMissing() const;
    PvUInt32 GetPipelineBlocksDropped() const;

    PvUInt32 GetResultImageError() const;
    PvUInt32 GetResultMissingPackets() const;
    PvUInt32 GetResultBufferTooSmall() const;
    PvUInt32 GetResultAborted() const;
    PvUInt32 GetResultTimeout() const;
    PvUInt32 GetResultStateError() const;
    PvUInt32 GetResultTooManyResends() const;
    PvUInt32 GetResultResendsFailure() const;
    PvUInt32 GetResultInvalidDataFormat() const;
    PvUInt32 GetResultTooManyConsecutiveResends() const;
    PvUInt32 GetResultAutoAborted() const;
    PvUInt32 GetResultNotInitialized() const;


    PvUInt32 GetResendGroupRequested() const;
    PvUInt32 GetResendPacketRequested() const;
    PvUInt32 GetExpectedSingleResend() const;
    PvUInt32 GetRedundantPacket() const;
    PvUInt32 GetPacketOutOfOrder() const;
    PvUInt32 GetPacketUnavaliable() const;

    PvUInt32 GetStatusUnknown() const;
    PvUInt32 GetStatusLocalProblem() const;

    // Only for Pleora IP Engine devices
    PvUInt32 GetDataOverrunCount() const;
    PvUInt64 GetBlocksDroppedCount() const;
    PvUInt32 GetPartialLineMissing() const;
    PvUInt32 GetFullLineMissing() const;
    PvUInt32 GetEOFByLineCount() const;
    PvUInt32 GetInterlacedEven() const;
    PvUInt32 GetInterlacedOdd() const;

protected:

    PvStatistics( PvStreamRawLib::Statistics *aThis );
    virtual ~PvStatistics();

    PvStreamRawLib::Statistics *mThis;

private:

	 // Not implemented
	PvStatistics( const PvStatistics & );
	const PvStatistics &operator=( const PvStatistics & );

};


#ifdef PV_DEBUG
    #include <PvStreamRawLib/Statistics.h>
#endif // PV_DEBUG


#endif // __PVSTATISTICS_H__
