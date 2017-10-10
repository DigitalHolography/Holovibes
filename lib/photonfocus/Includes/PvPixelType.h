// *****************************************************************************
//
//     Copyright (c) 2008, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVPIXELTYPE_H__
#define __PVPIXELTYPE_H__

#include <PvTypes.h>
#include <PvBufferLib.h>


//
// Color
//

#define PVPIXELMONO                  ( 0x01000000 )
#define PVPIXELRGB                   ( 0x02000000 ) // Pre GEV 1.1, kept for bw compatibility
#define PVPIXELCOLOR                 ( 0x02000000 ) // GEV 1.1
#define PVPIXELCUSTOM                ( 0x80000000 )
#define PVPIXELCOLORMASK             ( 0xFF000000 )

//
// Effective number of bits per pixel (including padding)
//

#define PVPIXEL1BIT                  ( 0x00010000 )
#define PVPIXEL2BIT                  ( 0x00020000 )
#define PVPIXEL4BIT                  ( 0x00040000 )
#define PVPIXEL8BIT                  ( 0x00080000 )
#define PVPIXEL12BIT                 ( 0x000C0000 )
#define PVPIXEL16BIT                 ( 0x00100000 )
#define PVPIXEL24BIT                 ( 0x00180000 )
#define PVPIXEL32BIT                 ( 0x00200000 )
#define PVPIXEL36BIT                 ( 0x00240000 )
#define PVPIXEL48BIT                 ( 0x00300000 )
#define PVPIXEL64BIT                 ( 0x00400000 )
#define PVBITSPERPIXELMASK           ( 0x00FF0000 )

//
// Pixel type ID
//

#define PVPIXELIDMASK                ( 0x0000FFFF )


typedef enum 
{

    PvPixelUndefined =               ( 0 ),
    PvPixelMono1p =                  ( PVPIXELMONO  | PVPIXEL1BIT   | 0x0037 ), // GEV 2.0
    PvPixelMono2p =                  ( PVPIXELMONO  | PVPIXEL2BIT   | 0x0038 ), // GEV 2.0
    PvPixelMono4p =                  ( PVPIXELMONO  | PVPIXEL4BIT   | 0x0039 ), // GEV 2.0
    PvPixelMono8 =                   ( PVPIXELMONO  | PVPIXEL8BIT   | 0x0001 ),
    PvPixelMono8s =                  ( PVPIXELMONO  | PVPIXEL8BIT   | 0x0002 ),
    PvPixelMono10 =                  ( PVPIXELMONO  | PVPIXEL16BIT  | 0x0003 ),
    PvPixelMono10Packed =            ( PVPIXELMONO  | PVPIXEL12BIT  | 0x0004 ),
    PvPixelMono12 =                  ( PVPIXELMONO  | PVPIXEL16BIT  | 0x0005 ),
    PvPixelMono12Packed =            ( PVPIXELMONO  | PVPIXEL12BIT  | 0x0006 ),
    PvPixelMono14 =                  ( PVPIXELMONO  | PVPIXEL16BIT  | 0x0025 ),
    PvPixelMono16 =                  ( PVPIXELMONO  | PVPIXEL16BIT  | 0x0007 ),
    PvPixelBayerGR8 =                ( PVPIXELMONO  | PVPIXEL8BIT   | 0x0008 ),
    PvPixelBayerRG8 =                ( PVPIXELMONO  | PVPIXEL8BIT   | 0x0009 ),
    PvPixelBayerGB8 =                ( PVPIXELMONO  | PVPIXEL8BIT   | 0x000A ),
    PvPixelBayerBG8 =                ( PVPIXELMONO  | PVPIXEL8BIT   | 0x000B ),
    PvPixelBayerGR10 =               ( PVPIXELMONO  | PVPIXEL16BIT  | 0x000C ),
    PvPixelBayerRG10 =               ( PVPIXELMONO  | PVPIXEL16BIT  | 0x000D ),
    PvPixelBayerGB10 =               ( PVPIXELMONO  | PVPIXEL16BIT  | 0x000E ),
    PvPixelBayerBG10 =               ( PVPIXELMONO  | PVPIXEL16BIT  | 0x000F ),
    PvPixelBayerGR12 =               ( PVPIXELMONO  | PVPIXEL16BIT  | 0x0010 ),
    PvPixelBayerRG12 =               ( PVPIXELMONO  | PVPIXEL16BIT  | 0x0011 ),
    PvPixelBayerGB12 =               ( PVPIXELMONO  | PVPIXEL16BIT  | 0x0012 ),
    PvPixelBayerBG12 =               ( PVPIXELMONO  | PVPIXEL16BIT  | 0x0013 ),
    PvPixelBayerGR10Packed =         ( PVPIXELMONO  | PVPIXEL12BIT  | 0x0026 ), // GEV 1.1
    PvPixelBayerRG10Packed =         ( PVPIXELMONO  | PVPIXEL12BIT  | 0x0027 ), // GEV 1.1
    PvPixelBayerGB10Packed =         ( PVPIXELMONO  | PVPIXEL12BIT  | 0x0028 ), // GEV 1.1
    PvPixelBayerBG10Packed =         ( PVPIXELMONO  | PVPIXEL12BIT  | 0x0029 ), // GEV 1.1
    PvPixelBayerGR12Packed =         ( PVPIXELMONO  | PVPIXEL12BIT  | 0x002A ), // GEV 1.1
    PvPixelBayerRG12Packed =         ( PVPIXELMONO  | PVPIXEL12BIT  | 0x002B ), // GEV 1.1
    PvPixelBayerGB12Packed =         ( PVPIXELMONO  | PVPIXEL12BIT  | 0x002C ), // GEV 1.1
    PvPixelBayerBG12Packed =         ( PVPIXELMONO  | PVPIXEL12BIT  | 0x002D ), // GEV 1.1
    PvPixelBayerGR16 =               ( PVPIXELMONO  | PVPIXEL16BIT  | 0x002E ), // GEV 1.1
    PvPixelBayerRG16 =               ( PVPIXELMONO  | PVPIXEL16BIT  | 0x002F ), // GEV 1.1
    PvPixelBayerGB16 =               ( PVPIXELMONO  | PVPIXEL16BIT  | 0x0030 ), // GEV 1.1
    PvPixelBayerBG16 =               ( PVPIXELMONO  | PVPIXEL16BIT  | 0x0031 ), // GEV 1.1
    PvPixelRGB8 =                    ( PVPIXELCOLOR | PVPIXEL24BIT  | 0x0014 ), // New name in 2.0
    PvPixelBGR8 =                    ( PVPIXELCOLOR | PVPIXEL24BIT  | 0x0015 ), // New name in 2.0
    PvPixelRGBa8 =                   ( PVPIXELCOLOR | PVPIXEL32BIT  | 0x0016 ), // New name in 2.0
    PvPixelBGRa8 =                   ( PVPIXELCOLOR | PVPIXEL32BIT  | 0x0017 ), // New name in 2.0
    PvPixelRGB10 =                   ( PVPIXELCOLOR | PVPIXEL48BIT  | 0x0018 ), // New name in 2.0
    PvPixelBGR10 =                   ( PVPIXELCOLOR | PVPIXEL48BIT  | 0x0019 ), // New name in 2.0
    PvPixelRGB12 =                   ( PVPIXELCOLOR | PVPIXEL48BIT  | 0x001A ), // New name in 2.0
    PvPixelBGR12 =                   ( PVPIXELCOLOR | PVPIXEL48BIT  | 0x001B ), // New name in 2.0
    PvPixelRGB16 =                   ( PVPIXELCOLOR | PVPIXEL48BIT  | 0x0033 ), // GEV 1.1
    PvPixelRGB10V1Packed =           ( PVPIXELCOLOR | PVPIXEL32BIT  | 0x001C ), // Renamed from BGR to RGB, corrects error
    PvPixelRGB10p32 =                ( PVPIXELCOLOR | PVPIXEL32BIT  | 0x001D ), // New name in 2.0
    PvPixelRGB12V1Packed =           ( PVPIXELCOLOR | PVPIXEL36BIT  | 0x0034 ), // GEV 1.1
    PvPixelRGB565p =                 ( PVPIXELCOLOR | PVPIXEL16BIT  | 0x0035 ), // GEV 2.0
    PvPixelBGR565p =                 ( PVPIXELCOLOR | PVPIXEL16BIT  | 0x0036 ), // GEV 2.0
    PvPixelYUV411_8_UYYVYY =         ( PVPIXELCOLOR | PVPIXEL12BIT  | 0x001E ), // New name in 2.0
    PvPixelYUV422_8_UYVY =           ( PVPIXELCOLOR | PVPIXEL16BIT  | 0x001F ), // New name in 2.0
    PvPixelYUV422_8 =                ( PVPIXELCOLOR | PVPIXEL16BIT  | 0x0032 ), // GEV 1.1, new name in 2.0
    PvPixelYUV8_UYV =                ( PVPIXELCOLOR | PVPIXEL24BIT  | 0x0020 ), // New name in 2.0
    PvPixelYCbCr8_CbYCr =            ( PVPIXELCOLOR | PVPIXEL24BIT  | 0x003A ), // GEV 2.0
    PvPixelYCbCr422_8 =              ( PVPIXELCOLOR | PVPIXEL16BIT  | 0x003B ), // GEV 2.0
    PvPixelYCbCr422_8_CbYCrY =       ( PVPIXELCOLOR | PVPIXEL16BIT  | 0x0043 ), // GEV 2.0
    PvPixelYCbCr411_8_CbYYCrYY =     ( PVPIXELCOLOR | PVPIXEL12BIT  | 0x003C ), // GEV 2.0
    PvPixelYCbCr601_8_CbYCr =        ( PVPIXELCOLOR | PVPIXEL24BIT  | 0x003D ), // GEV 2.0
    PvPixelYCbCr601_422_8 =          ( PVPIXELCOLOR | PVPIXEL16BIT  | 0x003E ), // GEV 2.0
    PvPixelYCbCr601_422_8_CbYCrY =   ( PVPIXELCOLOR | PVPIXEL16BIT  | 0x0044 ), // GEV 2.0
    PvPixelYCbCr601_411_8_CbYYCrYY = ( PVPIXELCOLOR | PVPIXEL12BIT  | 0x003F ), // GEV 2.0
    PvPixelYCbCr709_8_CbYCr =        ( PVPIXELCOLOR | PVPIXEL24BIT  | 0x0040 ), // GEV 2.0
    PvPixelYCbCr709_422_8 =          ( PVPIXELCOLOR | PVPIXEL16BIT  | 0x0041 ), // GEV 2.0
    PvPixelYCbCr709_422_8_CbYCrY =   ( PVPIXELCOLOR | PVPIXEL16BIT  | 0x0045 ), // GEV 2.0
    PvPixelYCbCr709_411_8_CbYYCrYY = ( PVPIXELCOLOR | PVPIXEL12BIT  | 0x0042 ), // GEV 2.0
    PvPixelRGB8_Planar =             ( PVPIXELCOLOR | PVPIXEL24BIT  | 0x0021 ), // Added _ to name
    PvPixelRGB10_Planar =            ( PVPIXELCOLOR | PVPIXEL48BIT  | 0x0022 ), // Added _ to name
    PvPixelRGB12_Planar =            ( PVPIXELCOLOR | PVPIXEL48BIT  | 0x0023 ), // Added _ to name
    PvPixelRGB16_Planar =            ( PVPIXELCOLOR | PVPIXEL48BIT  | 0x0024 ), // Added _ to name

} PvPixelType;

// Mapping to Windows pixel types (MFC, .NET, DirectX, Windows Bitmap, etc.)
#define PV_PIXEL_WIN_RGB32 ( PvPixelBGRa8 )
#define PV_PIXEL_WIN_RGB24 ( PvPixelBGR8 )
#define PV_PIXEL_WIN_RGB16 ( PvPixelRGB565p )

// Mapping to Qt pixel types
#define PV_PIXEL_QT_RGB32 ( PvPixelBGRa8 )
#define PV_PIXEL_QT_RGB888 ( PvPixelBGR8 )
#define PV_PIXEL_QT_RGB565 ( PvPixelRGB565p )

// Mapping to OpenGL
#define PV_PIXEL_OPENGL_RGB32 ( PvPixelRGBa8 )
#define PV_PIXEL_OPENGL_RGB24 ( PvPixelRGB8 )
#define PV_PIXEL_OPENGL_BGR32 ( PvPixelBGRa8 )
#define PV_PIXEL_OPENGL_BGR24 ( PvPixelBGR8 )

// Pre GEV 2.0 pixel types
#ifndef PV_NO_GEV1X_PIXEL_TYPES
    #define PvPixelMono8Signed ( PvPixelMono8s )
    #define PvPixelRGB8Packed ( PvPixelRGB8 )
    #define PvPixelBGR8Packed ( PvPixelBGR8 )
    #define PvPixelRGBA8Packed ( PvPixelRGBa8 )
    #define PvPixelBGRA8Packed ( PvPixelBGRa8 )
    #define PvPixelRGB10Packed ( PvPixelRGB10 )
    #define PvPixelBGR10Packed ( PvPixelBGR10 )
    #define PvPixelRGB12Packed ( PvPixelRGB12 )
    #define PvPixelBGR12Packed ( PvPixelBGR12 )
    #define PvPixelRGB16Packed ( PvPixelRGB16 )
    #define PvPixelBGR10V1Packed ( PvPixelRGB10V1Packed )
    #define PvPixelBGR10V2Packed ( PvPixelRGB10p32 )
    #define PvPixelYUV411Packed ( PvPixelYUV411_8_UYYVYY )
    #define PvPixelYUV422Packed ( PvPixelYUV422_8_UYVY )
    #define PvPixelYUV422YUYVPacked ( PvPixelYUV422_8 )
    #define PvPixelYUV444Packed ( PvPixelYUV8_UYV )
    #define PvPixelRGB8Planar ( PvPixelRGB8_Planar )
    #define PvPixelRGB10Planar ( PvPixelRGB10_Planar )
    #define PvPixelRGB12Planar ( PvPixelRGB12_Planar )
    #define PvPixelRGB16Planar ( PvPixelRGB16_Planar )
#endif // PV_NO_GEV1X_PIXEL_TYPES
    
// Deprecated pixel types, for backward compatibility
#ifndef PV_NO_DEPRECATED_PIXEL_TYPES
    #define PvPixelWinRGB16 ( PvPixelRGB565p )
    #define PvPixelWinRGB32 ( PvPixelBGRa8 )
    #define PvPixelWinRGB24 ( PvPixelBGR8 )
    #define PvPixelWinBGR32 ( PvPixelRGBa8 )
    #define PvPixelWinBGR24 ( PvPixelRGB8 )
#endif // PV_NO_DEPRECATED_PIXEL_TYPES

PV_BUFFER_API PvUInt32 PvGetPixelBitCount( PvPixelType aType );


#endif // __PVPIXELTYPE_H__
