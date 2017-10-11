// *****************************************************************************
//
//     Copyright (c) 2009, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVGENFILE_H__
#define __PVGENFILE_H__

#include <PvGenICamLib.h>
#include <PvGenParameterArray.h>


typedef enum 
{
	PvGenOpenModeWrite = 0,
	PvGenOpenModeRead = 1,
	PvGenOpenModeUndefined = 999,

} PvGenOpenMode;


namespace PvGenICamLib
{
    class GenFile;
}


class PV_GENICAM_API PvGenFile
{
public:

    PvGenFile();
    virtual ~PvGenFile();

    PvResult Open( PvGenParameterArray *aArray, const PvString &aFilename, PvGenOpenMode aMode );
    PvResult Close();

    bool IsOpened() const;

    PvResult WriteFrom( const PvString &aLocalFilename );
    PvResult ReadTo( const PvString &aLocalFilename );

    PvResult Write( const PvUInt8 *aBuffer, PvInt64 aLength, PvInt64 &aBytesWritten );
    PvResult Read( PvUInt8 *aBuffer, PvInt64 aLength, PvInt64 &aBytesRead );

    PvResult GetStatus( PvString &aStatus );
    PvString GetLastErrorMessage() const;

    static bool IsSupported( PvGenParameterArray* aArray );

private:

    // Not implemented
	PvGenFile( const PvGenFile & );
	const PvGenFile &operator=( const PvGenFile & );

    PvGenICamLib::GenFile *mThis;
};


#ifdef PV_DEBUG
    #include <PvGenICamLib/GenFile.h>
#endif // PV_DEBUG


#endif // __PVGENFILE_H__


