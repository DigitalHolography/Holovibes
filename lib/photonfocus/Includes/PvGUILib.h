// *****************************************************************************
//
//     Copyright (c) 2007, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVGUILIB_H__
#define __PVGUILIB_H__


#ifndef PV_GUI_DOTNET

	#ifdef PV_GUI_EXPORTS

		// Define the export symbol
		  #ifndef PT_LIB_STATIC
				#define PV_GUI_API __declspec( dllexport )
		  #else
				#define PV_GUI_API
		  #endif // PT_LIB_STATIC

	#else // PV_GUI_EXPORTS

		// Define the import symbol
		#ifndef PT_LIB_STATIC
			#define PV_GUI_API __declspec( dllimport )
		#else
			#define PV_GUI_API
		#endif // PT_LIB_STATIC

		#if defined ( PV_DEBUG )
			#define _PT_DEBUG_
			#define PT_DEBUG_ENABLED
		#endif // PV_DEBUG

		// Define the suffix used for the static version of the library file
		#ifdef PT_LIB_STATIC
			#define PT_SUFFIX_STATIC "_s"
		#else // PT_LIB_STATIC
			#define PT_SUFFIX_STATIC
		#endif // PT_LIB_STATIC

		// Define the suffix used for the debug version of the library file
		#if defined( _PT_DEBUG_ ) && defined( PT_DEBUG_ENABLED )
			#define PT_SUFFIX_DEBUG "_Dbg"
		#else // defined( _PT_DEBUG_ ) && defined( PT_DEBUG_ENABLED )
			#define PT_SUFFIX_DEBUG
		#endif // defined( _PT_DEBUG_ ) && defined( PT_DEBUG_ENABLED )

		// Define the suffix used for the 64-bit version of the library file
        #if defined( _PT_64_ ) || defined( _WIN64 )
			#define PT_SUFFIX_64 "64"
		#else
			#define PT_SUFFIX_64
		#endif

		#if(_MSC_VER >= 1600)
			// VC 10.0 (aka 2010)
			#define PT_SUFFIX_COMPILER "_VC10"
		#elif(_MSC_VER >= 1500)
			// VC 9.0 (aka 2008)
			#define PT_SUFFIX_COMPILER "_VC9"
		#elif(_MSC_VER >= 1400)
			// VC 8.0 (aka 2005)
			#define PT_SUFFIX_COMPILER "_VC8"
		#else
			#ifdef WIN32
				#pragma message ( "Warning: Your compiler is not officially supported by the PureGEV SDK. Currently supported compiler versions on Windows include Visual C++ 2005 and Visual C++ 2008." )
			#endif //WIN32
			#define PT_SUFFIX_COMPILER
		#endif

		// Ask the compiler to link the required version of the library
		#pragma comment( lib, "PvGUI" PT_SUFFIX_64 PT_SUFFIX_STATIC PT_SUFFIX_COMPILER PT_SUFFIX_DEBUG ".lib" )

		// Undefined the symbols used to link the required library file
		#undef PT_SUFFIX_STATIC
		#undef PT_SUFFIX_DEBUG
		#undef PT_SUFFIX_64

	#endif // PV_GUI_EXPORTS


	#include <PvTypes.h>
	#include <PvString.h>
	#include <PvResult.h>

    #include <Windows.h>

	typedef HWND PvWindowHandle;

#endif // PV_GUI_DOTNET


#endif // __PVGUILIB_H__


