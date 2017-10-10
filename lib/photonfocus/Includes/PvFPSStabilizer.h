// *****************************************************************************
//
//     Copyright (c) 2012, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PV_FPS_STABILIZER_H__
#define __PV_FPS_STABILIZER_H__

#include <PvGUIUtilsLib.h>
#include <PvTypes.h>


namespace PvGUIUtilsLib
{
    class FPSStabilizer;

}; // namespace PvGUIUtilsLib


class PV_GUIUTILS_API PvFPSStabilizer
{
public:

    PvFPSStabilizer();
    ~PvFPSStabilizer();

	bool IsTimeToDisplay( PvUInt32 aTargetFPS );
    PvUInt32 GetAverage();

	void Reset();

private:

    PvGUIUtilsLib::FPSStabilizer *mThis;

	 // Not implemented
	PvFPSStabilizer( const PvFPSStabilizer & );
	const PvFPSStabilizer &operator=( const PvFPSStabilizer & );

};


#endif //__PV_FPS_STABILIZER_H__


