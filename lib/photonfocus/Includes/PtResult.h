// *****************************************************************************
//
//     Copyright (c) 2012, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PTUTLISLIB_PTRESULT_H__
#define __PTUTLISLIB_PTRESULT_H__

#include <PtUtilsLib.h>
#include <PtTypes.h>
#include <PtString.h>


class PT_UTILS_LIB_API PtResult
{

public:

    PtResult();
    PtResult( PtUInt32 aCode );
    PtResult( PtUInt32 aCode, const PtString & aDescription );

    // copy constructor
    PtResult( const PtResult& aResult );

    // The destructor is not virtual to make as much efficient as possible using
    // the object as return value.
	~PtResult();

    operator const char  * () const;
    PtResult& operator = (const PtResult & aB);
    bool operator == ( const PtResult & aB ) const;
    bool operator == ( const PtUInt32 aCode ) const;
    bool operator != ( const PtResult & aB ) const;
    bool operator != ( const PtUInt32 aCode ) const;
	const PtResult & operator |= ( const PtResult & aB );

	void SetCode( PtUInt32 aIn );
    PtUInt32 GetCode() const;
    PtString GetCodeString() const;
    PtString GetDescription() const;
    void SetDescription( const PtString & aDescription );

    bool IsFailure() const;
    bool IsOK() const;
	bool IsPending() const;
    bool IsSuccess() const;

	struct PT_UTILS_LIB_API Code
	{
		static const PtUInt32 OK;
		static const PtUInt32 NOT_INITIALIZED;       
		static const PtUInt32 NOT_FOUND;           
		static const PtUInt32 CANNOT_OPEN_FILE;         
		static const PtUInt32 NOT_CONNECTED;            
		static const PtUInt32 STATE_ERROR;
		static const PtUInt32 INVALID_DATA_FORMAT;   
		static const PtUInt32 ABORTED;
		static const PtUInt32 NOT_ENOUGH_MEMORY;
		static const PtUInt32 GENERIC_ERROR;
		static const PtUInt32 INVALID_PARAMETER;
		static const PtUInt32 CANCEL;
		static const PtUInt32 PENDING;
        static const PtUInt32 TIMEOUT;
        static const PtUInt32 NO_DRIVER;
        static const PtUInt32 NO_LICENSE;
        static const PtUInt32 GENICAM_XML_ERROR;
        static const PtUInt32 NOT_IMPLEMENTED;
        static const PtUInt32 NOT_SUPPORTED;
        static const PtUInt32 FILE_ERROR;
        static const PtUInt32 ERR_OVERFLOW;
        static const PtUInt32 IMAGE_ERROR;
        static const PtUInt32 MISSING_PACKETS;
        static const PtUInt32 BUFFER_TOO_SMALL;
        static const PtUInt32 TOO_MANY_RESENDS;
        static const PtUInt32 RESENDS_FAILURE;
        static const PtUInt32 TOO_MANY_CONSECUTIVE_RESENDS;
        static const PtUInt32 AUTO_ABORTED;
        static const PtUInt32 BAD_VERSION;
        static const PtUInt32 NO_MORE_ENTRY;
        static const PtUInt32 NO_AVAILABLE_DATA;
        static const PtUInt32 NETWORK_ERROR;
        static const PtUInt32 REBOOT_NEEDED;
        static const PtUInt32 REBOOT_AND_RECALL;
	};

private:

	PtUInt32 mCode;
    PtString* mDescription;
};


#endif // __PTUTLISLIB_PTRESULT_H__

