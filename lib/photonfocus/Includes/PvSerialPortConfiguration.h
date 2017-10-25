// *****************************************************************************
//
//     Copyright (c) 2011, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVSERIALPORTCONFIGURATION_H__
#define __PVSERIALPORTCONFIGURATION_H__


#include <PvResult.h>
#include <PvSerialLib.h>


typedef enum 
{
    PvParityInvalid = -1,
    PvParityNone = 0,
    PvParityEven = 1,
    PvParityOdd = 2

} PvParity;


class PV_SERIAL_API PvSerialPortConfiguration
{
public:

    PvSerialPortConfiguration();
    PvSerialPortConfiguration( PvUInt32 aBaudRate, PvParity aParity, PvUInt32 aByteSize, PvUInt32 aStopBits );
    ~PvSerialPortConfiguration();

    PvResult IsValid() const;
    void MakeInvalid();

    PvUInt32 mBaudRate;
    PvParity mParity;
    PvUInt32 mByteSize;
    PvUInt32 mStopBits;

private:

};


#endif // __PVSERIALPORTCONFIGURATION_H__





