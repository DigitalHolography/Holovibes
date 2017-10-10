// *****************************************************************************
//
//     Copyright (c) 2007, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************
//
// File Name....: PvResult.h
//
// *****************************************************************************

#ifndef __PVRESULT_H__
#define __PVRESULT_H__


#include <PvBaseLib.h>
#include <PvTypes.h>
#include <PvString.h>


class PV_BASE_API PvResult
{

public:

    PvResult();
    PvResult( PvUInt32 aCode );
    PvResult( PvUInt32 aCode, PvUInt32 aOSCode );
    PvResult( PvUInt32 aCode, const PvString & aDescription );
    PvResult( PvUInt32 aCode, PvUInt32 aOSCode, const PvString & aDescription );

    // copy constructor
    PvResult( const PvResult& aResult );

    // The destructor is not virtual to make as much efficient as possible using
    // the object as return value.
	~PvResult();

    operator const char  * () const;
    PvResult& operator = (const PvResult & aB);
    bool operator == ( const PvResult & aB ) const;
    bool operator == ( const PvUInt32 aCode ) const;
    bool operator != ( const PvResult & aB ) const;
    bool operator != ( const PvUInt32 aCode ) const;
	const PvResult & operator |= ( const PvResult & aB );

	void SetCode( PvUInt32 aIn );
    PvUInt32 GetCode() const;
    PvString GetCodeString() const;
    PvString GetDescription() const;
    void SetDescription( const PvString & aDescription );

    bool IsFailure() const;
    bool IsOK() const;
	bool IsPending() const;
    bool IsSuccess() const;

    // Can be used to retrieve internal diagnostic information
    PvUInt32 GetInternalCode() const;
    PvUInt32 GetOSCode() const;

	struct PV_BASE_API Code
	{
		static const PvUInt32 OK;
		static const PvUInt32 NOT_INITIALIZED;       
		static const PvUInt32 NOT_FOUND;           
		static const PvUInt32 CANNOT_OPEN_FILE;         
		static const PvUInt32 NOT_CONNECTED;            
		static const PvUInt32 STATE_ERROR;
		static const PvUInt32 INVALID_DATA_FORMAT;   
		static const PvUInt32 ABORTED;
		static const PvUInt32 NOT_ENOUGH_MEMORY;
		static const PvUInt32 GENERIC_ERROR;
		static const PvUInt32 INVALID_PARAMETER;
		static const PvUInt32 CANCEL;
		static const PvUInt32 PENDING;
        static const PvUInt32 TIMEOUT;
        static const PvUInt32 NO_LICENSE;
        static const PvUInt32 GENICAM_XML_ERROR;
        static const PvUInt32 NOT_IMPLEMENTED;
        static const PvUInt32 NOT_SUPPORTED;
        static const PvUInt32 FILE_ERROR;
        static const PvUInt32 ERR_OVERFLOW;
        static const PvUInt32 IMAGE_ERROR;
        static const PvUInt32 MISSING_PACKETS;
        static const PvUInt32 BUFFER_TOO_SMALL;
        static const PvUInt32 TOO_MANY_RESENDS;
        static const PvUInt32 RESENDS_FAILURE;
        static const PvUInt32 TOO_MANY_CONSECUTIVE_RESENDS;
        static const PvUInt32 AUTO_ABORTED;
        static const PvUInt32 BAD_VERSION;
        static const PvUInt32 NO_MORE_ENTRY;
        static const PvUInt32 NO_AVAILABLE_DATA;
        static const PvUInt32 NETWORK_ERROR;
        static const PvUInt32 THREAD_ERROR;
        static const PvUInt32 NO_MORE_ITEM;

	};

private:

	PvUInt32 mCode;
    PvUInt32 mInternalCode;
    PvUInt32 mOSCode;
    PvString* mDescription;

};


//
// Direct #defines or the PvResult::Code - typically used to solve
// delay-loading issues
//

#define PV_OK ( 0 )
#define PV_NOT_INITIALIZED ( 0x0605 )
#define PV_NOT_FOUND ( 0x0019 )
#define PV_CANNOT_OPEN_FILE (0x0006 )
#define PV_NO_MORE_ITEM ( 0x0015 )
#define PV_NOT_CONNECTED ( 0x0017 )         
#define PV_STATE_ERROR ( 0x001c )
#define PV_THREAD_ERROR ( 0x001d )
#define PV_INVALID_DATA_FORMAT ( 0x0501 )
#define PV_ABORTED ( 0x0001 )
#define PV_NOT_ENOUGH_MEMORY ( 0x0018 )
#define PV_GENERIC_ERROR ( 0x4000 )
#define PV_INVALID_PARAMETER ( 0x4001 )
#define PV_CANCEL ( 0x4002 )
#define PV_PENDING ( 0xffff )
#define PV_TIMEOUT ( 0x001e )
#define PV_NO_LICENSE ( 0x0602 )
#define PV_GENICAM_XML_ERROR ( 0x0904 )
#define PV_NOT_IMPLEMENTED ( 0x0604 )
#define PV_NOT_SUPPORTED ( 0x001a )
#define PV_FILE_ERROR ( 0x0010 )
#define PV_ERR_OVERFLOW ( 0x001b )
#define PV_IMAGE_ERROR ( 0x0025 )
#define PV_MISSING_PACKETS ( 0x0027 )
#define PV_BUFFER_TOO_SMALL ( 0x0004 )
#define PV_TOO_MANY_RESENDS ( 0x0b00 )
#define PV_RESENDS_FAILURE ( 0x0b01 )
#define PV_TOO_MANY_CONSECUTIVE_RESENDS ( 0x0b03 )
#define PV_AUTO_ABORTED ( 0x0b02 )
#define PV_BAD_VERSION ( 0x0201 )
#define PV_NO_MORE_ENTRY ( 0x0603 )
#define PV_NO_AVAILABLE_DATA ( 0x0014 )
#define PV_NETWORK_ERROR ( 0x0013 )
#define PV_NO_MORE_ITEM ( 0x0015 )


#endif // __PVRESULT_H__
