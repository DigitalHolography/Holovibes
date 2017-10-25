// *****************************************************************************
//
//     Copyright (c) 2012, Pleora Technologies Inc., All rights reserved.
//
// -----------------------------------------------------------------------------
//
// Header file containing #include, #define and some inline code shared by
// all eBUS SDK samples.
//
// *****************************************************************************

#ifndef __PV_SAMPLEUTILS_H__
#define __PV_SAMPLEUTILS_H__


#ifdef WIN32

    #include <windows.h>
    #include <process.h>
    #include <conio.h>

    #pragma comment(linker, "/manifestdependency:\"type='win32' " \
        "name='Microsoft.Windows.Common-Controls' " \
        "version='6.0.0.0' " \
        "processorArchitecture='*' " \
        "publicKeyToken='6595b64144ccf1df' " \
        "language='*'\"")

#endif // WIN32

#include <stdio.h>
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>

#include <PvTypes.h>
#include <PvDevice.h>
#include <PvSystem.h>

using namespace std;

#ifdef _UNIX_

    #include <stdlib.h>
    #include <string.h>
    #include <termios.h>
    #include <unistd.h>
    #include <fcntl.h>
    #include <signal.h>
    #include <sys/time.h>

    #define PV_INIT_SIGNAL_HANDLER()                                    \
        void CatchCtrlC( int aSig ) { extern int gStop; gStop = 1; }    \
        int InitHandler() { signal( SIGINT, CatchCtrlC ); return 1; }   \
        int gInit = InitHandler();                                      \
        int gStop = 0;                                                  \

    inline int PvKbHit(void)
    {
        struct termios oldt, newt;
        int ch;
        int oldf;
        extern int gStop;

        if( gStop )
        {
            return 1;
        }

        tcgetattr(STDIN_FILENO, &oldt);
        newt = oldt;
        newt.c_lflag &= ~(ICANON | ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &newt);
        oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
        fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

        ch = getchar();

        tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
        fcntl(STDIN_FILENO, F_SETFL, oldf);

        if(ch != EOF)
        {
            ungetc(ch, stdin);
            return 1;
        }

        return 0;
    }

    inline int PvGetChar()
    {
        extern int gStop;
        if( gStop )
        {
            return 0;
        }

        return getchar();
    }

    inline void PvWaitForKeyPress()
    {
        while ( !PvKbHit() )
        {
            usleep( 10000 );
        }

        PvGetChar(); // Flush key buffer for next stop
    }

    inline PvUInt64 PvGetTickCountMs()
    {
        timeval ts;
        gettimeofday( &ts, 0 );

        PvUInt64 lTickCount = (PvInt64)(ts.tv_sec * 1000LL + ( ts.tv_usec / 1000LL ) );

        return lTickCount;
    }

    inline void PvSleepMs( PvUInt32 aSleepTime )
    {
        usleep( aSleepTime * 1000 );
    }

    #define PvScanf ( scanf )

#endif // _UNIX_

#ifdef WIN32

    #define PV_INIT_SIGNAL_HANDLER()   

    inline int PvKbHit( void )
    {
        return _kbhit();
    }

    inline int PvGetChar()
    {
        return _getch();
    }

    inline void PvWaitForKeyPress()
    {
        while ( !PvKbHit() )
        {
            ::Sleep( 10 );
        }

        PvGetChar(); // Flush key buffer for next stop
    }

    inline PvUInt64 PvGetTickCountMs()
    {
        return ::GetTickCount();
    }

    inline void PvSleepMs( PvUInt32 aSleepTime )
    {
        ::Sleep( aSleepTime );
    }

    #define PvScanf ( scanf_s )

#endif // WIN32

inline void PvFlushKeyboard()
{
    int c;
    while( ( c = PvGetChar() ) != '\n' && c != EOF );
}


// These defines ensures no old/deprecated pixel types are used in the samples
#ifndef PV_NO_GEV1X_PIXEL_TYPES
    #define PV_NO_GEV1X_PIXEL_TYPES
#endif // PV_NO_GEV1X_PIXEL_TYPES
#ifndef PV_NO_DEPRECATED_PIXEL_TYPES
    #define PV_NO_DEPRECATED_PIXEL_TYPES
#endif // PV_NO_DEPRECATED_PIXEL_TYPES


inline PvDeviceInfo* PvSelectDevice( PvSystem& aSystem )
{
    PvResult lResult;
    PvDeviceInfo *lSelectedDI = NULL;
    int lCount;
#ifdef _UNIX_
    extern int gStop;
#endif

    printf( "\nDetecting devices.\n" );


    while( 1 )
    {

#ifdef _UNIX_
        if( gStop )
        { 
            return NULL;
        }
#endif
        aSystem.Find();
	    
        // Detect, select device
        std::vector<PvDeviceInfo *> lDIVector;
	    for ( PvUInt32 i = 0; i < aSystem.GetInterfaceCount(); i++ )
	    {
		    PvInterface *lInterface = aSystem.GetInterface( i );
		    printf( "   %s\n", lInterface->GetID().GetAscii() );

		    for ( PvUInt32 j = 0; j < lInterface->GetDeviceCount(); j++ )
		    {
			    PvDeviceInfo *lDI = lInterface->GetDeviceInfo( j );
			    lDIVector.push_back( lDI );
			    printf( "[%i]\t%s\n", lDIVector.size() - 1, lDI->GetID().GetAscii() );
		    }
	    }

        if( lDIVector.size() == 0)
        {
            printf( "No device found!\n" );
        }

        printf( "[%i] to abort\n", lDIVector.size());
        printf( "[%i] to search again\n\n", lDIVector.size() + 1);
        
	    printf( "Enter your action or device selection?\n" );
	    printf( ">" );

	    // Read device selection, optional new IP address
	    size_t lIndex = -1;
	    int lCount = PvScanf( "%i", &lIndex );
        PvFlushKeyboard();
        if( lCount )
        {
            if( lIndex == lDIVector.size() )
            {
                // We abort the selection process
                return NULL;
            }
            else if ( ( lIndex >=0 ) && ( lIndex < lDIVector.size() ) )
            {
                // The device is selected
                lSelectedDI = lDIVector[ lIndex ];
                break;
            }
        }
        // Otherwise, the finder will be run again
    }

	// If the IP Address valid?
	if ( lSelectedDI->IsIPConfigurationValid() )
	{
        printf( "\n" );
        return lSelectedDI;
    }

    // Ask the user for a new IP address
	printf( "The IP configuration of the device is not valid.\n" );
	printf( "Which IP address should be assigned to the device?\n" );
	printf( ">" );

	// Read new IP address
	char lNewIPAddress[ 1024 ] = { 0 };
	lCount = PvScanf( "%s", lNewIPAddress );
    PvFlushKeyboard();
	if ( ( lCount <= 0 ) || ( lNewIPAddress == NULL ) )
	{
		return NULL;
	}

	// Force new IP address
	lResult = PvDevice::SetIPConfiguration( lSelectedDI->GetMACAddress().GetAscii(), lNewIPAddress, 
         lSelectedDI->GetSubnetMask().GetAscii(), lSelectedDI->GetDefaultGateway().GetAscii() );

	// Wait for the device to come back on the network
    int lTimeout;
    while( 1 )
    {
#ifdef _UNIX_
        if( gStop )
        { 
            return NULL;
        }
#endif
        lTimeout = 10;
        while( lTimeout )
        {
#ifdef _UNIX_
            if( gStop )
            { 
                return NULL;
            }
#endif

            aSystem.Find();

	        std::vector<const PvDeviceInfo *> lDIVector;
	        for ( PvUInt32 i = 0; i < aSystem.GetInterfaceCount(); i++ )
	        {
		        PvInterface *lInterface = aSystem.GetInterface( i );
		        for ( PvUInt32 j = 0; j < lInterface->GetDeviceCount(); j++ )
		        {
			        PvDeviceInfo *lDI = lInterface->GetDeviceInfo( j );
			        if( strcmp( lDI->GetIPAddress().GetAscii(), lNewIPAddress ) == 0 )
                    {
                    	printf( "\n" );
                        return lDI;
                    }
		        }
	        }
            PvSleepMs( 1000 );

            lTimeout--;
        }

        printf( "The device %s was not locate. Do you want to continue waiting? yes or no\n ", lNewIPAddress );
	    printf( ">" );
    	char lAnswer[ 1024 ] = { 0 };
        lCount = PvScanf( "%s", lAnswer );
        if( lCount 
            && ( ( strcmp( lAnswer, "n") == 0 ) 
                 || strcmp( lAnswer, "no") ) )
        {
            break;
        }
    }
    
	printf( "\n" );
    return NULL;
}

inline bool ParseOptionFlag( int aCount, const char ** aArgs, const char *aOption, bool *aValue = NULL )
{
    std::string lOption = aOption;
    for ( int i = 1; i < aCount; i++ )
	{
		std::string lString = aArgs[i];
        size_t lPos = lString.find( aOption );
        if ( lPos != std::string::npos )
        {
            if ( aValue != NULL )
            {
                *aValue = true;
            }

            return true;
        }
    }    

    return false;
}

template <class T>
inline bool ParseOption( int aCount, const char ** aArgs, const char *aOption, T &aValue )
{
    std::string lOption = aOption;
    lOption += "=";
    for ( int i = 1; i < aCount; i++ )
	{
		std::string lString = aArgs[i];

        size_t lPos = lString.find( aOption );
        if ( lPos != std::string::npos )
        {
            if ( lString.size() > lOption.size() )
            {
                std::string lParameter = lString.substr( lOption.size(), ( lString.size() - lOption.size() ) + 1 );
                std::istringstream iss( lParameter, std::istringstream::in );
                iss >> aValue;
                return true;
            }
        }
    }

    return false;
}

#endif // __PV_SAMPLEUTILS_H__

