// *****************************************************************************
//
//     Copyright (c) 2011, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVSERIALBRIDGE_H__
#define __PVSERIALBRIDGE_H__


#include <PvResult.h>
#include <PvSerialLib.h>

#include <PvSerialPortIPEngine.h>
#include <PvSerialPortConfiguration.h>


namespace PvSerialLib
{
    class Bridge;
}


class PV_SERIAL_API PvSerialBridge
{
public:

    PvSerialBridge();
    ~PvSerialBridge();

    // Start a serial COM port bridge
    PvResult Start( const PvString &aSerialPort, PvSerialPortConfiguration aSerialPortConfiguration, 
        PvDevice *aDevice, PvIPEngineSerial aDevicePort );

    // Start, stops a Camera Link DLL bridge
    PvResult Start( const PvString &aName, PvDevice *aDevice, PvIPEngineSerial aDevicePort );
    PvResult Stop();

    // Stats
    PvUInt64 GetBytesSentToDevice() const;
    PvUInt64 GetBytesReceivedFromDevice() const;
    void ResetStatistics();

    // IP Engine serial port in use
    PvIPEngineSerial GetDevicePort() const;

    // Retrieve supported/selected IP Engine port baud rates
    PvUInt32 GetSupportedBaudRateCount() const;
    PvUInt32 GetSupportedBaudRate( PvUInt32 aIndex ) const;
    PvUInt32 GetBaudRate() const;

    // Gets, sets the IP Engine hard-coded port baud rate (if cannot be read/written)
    PvUInt32 GetHardCodedBaudRate() const;
    PvResult SetHardCodedBaudRate( PvUInt32 aBaudRate );

    // Serial COM port bridge configuration
    PvString GetSerialPort() const;
    PvSerialPortConfiguration GetSerialPortConfiguration() const;
    PvResult SetSerialPortConfiguration( PvSerialPortConfiguration aSerialPortConfiguration );

    // Camera Link DLL bridge configuration
    PvString GetName() const;

    // Closes and re-opens the device serial port
    PvResult Recover();

private:

    // Not implemented
	PvSerialBridge( const PvSerialBridge & );
	const PvSerialBridge &operator=( const PvSerialBridge & );

    PvSerialLib::Bridge * mThis;
};


#endif // __PVSERIALBRIDGE_H__


