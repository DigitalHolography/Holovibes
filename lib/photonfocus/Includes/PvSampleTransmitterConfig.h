// *****************************************************************************
//
//     Copyright (c) 2012, Pleora Technologies Inc., All rights reserved.
//
// -----------------------------------------------------------------------------
//
// Common configuration for all transmitter samples.
//
// *****************************************************************************

#ifndef __PV_TRANSMITTER_SAMPLE_CONFIG_H__
#define __PV_TRANSMITTER_SAMPLE_CONFIG_H__

#include <PvSampleUtils.h>
#include <PvSystem.h>


// Default values
#define DEFAULT_DESTINATION_ADDRESS ( "239.192.1.1" )
#define DEFAULT_DESTINATION_PORT ( 1042 )
#define DEFAULT_SOURCE_PORT ( 0 )
#define DEFAULT_BUFFER_COUNT ( 4 )
#define DEFAULT_PACKET_SIZE ( 1440 )
#define DEFAULT_SILENT ( false )
#define DEFAULT_FPS ( 30 )


#ifndef PV_GENERATING_DOXYGEN_DOC


// Application config
class PvSampleTransmitterConfig
{
public:

    PvSampleTransmitterConfig()
    {
    }

    ~PvSampleTransmitterConfig()
    {
    }

    const char *GetSourceAddress() const { return mSourceAddress.c_str(); }
    PvUInt16 GetSourcePort() const { return mSourcePort; }
    const char *GetDestinationAddress() const { return mDestinationAddress.c_str(); }
    PvUInt16 GetDestinationPort() const { return mDestinationPort; }
    float GetFPS() const { return mFPS; }
    PvUInt32 GetPacketSize() const { return mPacketSize; }
    PvUInt32 GetBufferCount() const { return mBufferCount; }
    bool GetSilent() const { return mSilent; }

    void SetDefaults()
    {
        PvSystem lSystem;
        lSystem.Find();

        // Find default source address
        bool lFound = false;
        for ( PvUInt32 i = 0; i < lSystem.GetInterfaceCount() && !lFound; i++ )
        {
            if ( strcmp( "0.0.0.0", lSystem.GetInterface( i )->GetIPAddress().GetAscii() ) != 0 )
            {
                mSourceAddress = lSystem.GetInterface( i )->GetIPAddress().GetAscii();
                lFound = true;
            }
        }
        if ( !lFound )
        {
            cout << "No valid interfaces found." << endl;
            exit( 1 );
        }

        // Set static defaults
        mDestinationAddress = DEFAULT_DESTINATION_ADDRESS;
        mDestinationPort = DEFAULT_DESTINATION_PORT;
        mSourcePort = DEFAULT_SOURCE_PORT;
        mPacketSize = DEFAULT_PACKET_SIZE;
        mBufferCount = DEFAULT_BUFFER_COUNT;
        mSilent = DEFAULT_SILENT;
        mFPS = DEFAULT_FPS;
    }

    void ParseCommandLine( int aCount, const char **aArgs )
    {
        ParseOption<float>( aCount, aArgs, "--fps", mFPS );
        ParseOption<PvUInt32>( aCount, aArgs, "--packetsize", mPacketSize );
        ParseOption<string>( aCount, aArgs, "--destinationaddress", mDestinationAddress );
        ParseOption<PvUInt16>( aCount, aArgs, "--destinationport", mDestinationPort );
        ParseOption<string>( aCount, aArgs, "--sourceaddress", mSourceAddress );
        ParseOption<PvUInt16>( aCount, aArgs, "--sourceport", mSourcePort );
        ParseOption<PvUInt32>( aCount, aArgs, "--buffercount", mBufferCount );
        ParseOptionFlag( aCount, aArgs, "--silent", &mSilent );

        if ( mDestinationPort == 0 )
        {
            cout << "Please enter a destination port." << endl;
            cin >> mDestinationPort;
        }
    }

    void PrintHelp()
    {
        cout << "Optional command line arguments:" << endl << endl;

        cout << "--help " << endl << "Print this help message." << endl << endl;

        cout << "--packetsize=<maximimum size of streaming packets>" << endl;
        cout << "Default: 1440 For best results, set \"Jumbo Frames\" property on your NIC and increase this value accordingly." << endl << endl;

        cout << "--destinationaddress=<destination address in the form 123.456.789.101>" << endl;
        cout << "Default: " << DEFAULT_DESTINATION_ADDRESS << endl << endl;

        cout << "--destinationport=<destination port>" << endl;
        cout << "Default: " << DEFAULT_DESTINATION_PORT << endl << endl;

        cout << "--sourceaddress=<source address in the form 123.456.789.101>" << endl;
        cout << "Default: first valid address encountered while enumerating interfaces" << endl << endl;

        cout << "--sourceport=<source port>" << endl;
        cout << "Default: " << DEFAULT_SOURCE_PORT << " - a port is automatically assigned when the socket is opened" << endl << endl;

        cout << "--buffercount=<number of transmit buffers to use>" << endl;
        cout << "Default: " << DEFAULT_BUFFER_COUNT << " - increase this number when sending smaller images at high frame rates." << endl << endl;

        cout << "--silent" << endl;
        cout << "Don't wait for a key press." << endl;
        cout << "By default, the system waits for a key press before it begins transmitting. " << endl << endl;
    }

private:

    string mSourceAddress;
    PvUInt16 mSourcePort;

    string mDestinationAddress;
    PvUInt16 mDestinationPort;

    float mFPS;

    PvUInt32 mPacketSize;
    PvUInt32 mBufferCount;

    bool mSilent;
};


#endif // PV_GENERATING_DOXYGEN_DOC


#endif // __PV_TRANSMITTER_SAMPLE_CONFIG_H__

