// CameraPCOPixelfly.cpp : Defines the exported functions for the DLL application.
//

#include "CameraPCOPixelfly.h"


// This is an example of an exported variable
CAMERAPCOPIXELFLY_API int nCameraPCOPixelfly=0;

// This is an example of an exported function.
CAMERAPCOPIXELFLY_API int fnCameraPCOPixelfly(void)
{
	return 42;
}

// This is the constructor of a class that has been exported.
// see CameraPCOPixelfly.h for the class definition
CCameraPCOPixelfly::CCameraPCOPixelfly()
{
	return;
}
