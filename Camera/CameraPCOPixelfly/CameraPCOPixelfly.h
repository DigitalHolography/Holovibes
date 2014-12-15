// The following ifdef block is the standard way of creating macros which make exporting 
// from a DLL simpler. All files within this DLL are compiled with the CAMERAPCOPIXELFLY_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see 
// CAMERAPCOPIXELFLY_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef CAMERAPCOPIXELFLY_EXPORTS
#define CAMERAPCOPIXELFLY_API __declspec(dllexport)
#else
#define CAMERAPCOPIXELFLY_API __declspec(dllimport)
#endif

// This class is exported from the CameraPCOPixelfly.dll
class CAMERAPCOPIXELFLY_API CCameraPCOPixelfly {
public:
	CCameraPCOPixelfly(void);
	// TODO: add your methods here.
};

extern CAMERAPCOPIXELFLY_API int nCameraPCOPixelfly;

CAMERAPCOPIXELFLY_API int fnCameraPCOPixelfly(void);
