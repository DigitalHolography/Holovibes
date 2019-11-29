/****************************************************************************************************************
 * COPYRIGHT � 2010 PixeLINK CORPORATION.  ALL RIGHTS RESERVED.                                                 *
 *                                                                                                              *
 * Copyright Notice and Disclaimer of Liability:                                                                *
 *                                                                                                              *
 *                                                                                                              *
 * PixeLINK Corporation is henceforth referred to as PixeLINK or PixeLINK Corporation.                          *
 * Purchaser henceforth refers to the original purchaser(s) of the equipment, and/or any legitimate user(s).    *
 *                                                                                                              *
 * PixeLINK hereby explicitly prohibits any form of reproduction (with the express strict exception for backup  *
 * and archival purposes, which are allowed as stipulated within the License Agreement for PixeLINK Corporation *
 * Software), modification, and/or distribution of this software and/or its associated documentation unless     *
 * explicitly specified in a written agreement signed by both parties.                                          *
 *                                                                                                              *
 * To the extent permitted by law, PixeLINK disclaims all other warranties or conditions of any kind, either    *
 * express or implied, including but not limited to all warranties or conditions of merchantability and         *
 * fitness for a particular purpose and those arising by statute or otherwise in law or from a course of        *
 * dealing or usage of trade. Other written or oral statements by PixeLINK, its representatives, or others do   *
 * not constitute warranties or conditions of PixeLINK.                                                         *
 *                                                                                                              *
 * PixeLINK makes no guarantees or representations, including but not limited to: the use of, or the result(s)  *
 * of the use of: the software and associated documentation in terms of correctness, accuracy, reliability,     *
 * being current, or otherwise. The Purchaser hereby agree to rely on the software, associated hardware and     *
 * documentation and results stemming from the use thereof solely at their own risk.                            *
 *                                                                                                              *
 * By using the products(s) associated with the software, and/or the software, the Purchaser and/or user(s)     *
 * agree(s) to abide by the terms and conditions set forth within this document, as well as, respectively,      *
 * any and all other documents issued by PixeLINK in relation to the product(s).                                *
 *                                                                                                              *
 * PixeLINK is hereby absolved of any and all liability to the Purchaser, and/or a third party, financial or    *
 * otherwise, arising from any subsequent loss, direct and indirect, and damage, direct and indirect,           *
 * resulting from intended and/or unintended usage of its software, product(s) and documentation, as well       *
 * as additional service(s) rendered by PixeLINK, such as technical support, unless otherwise explicitly        *
 * specified in a written agreement signed by both parties. Under no circumstances shall the terms and          *
 * conditions of such an agreement apply retroactively.                                                         *
 *                                                                                                              *
 ****************************************************************************************************************/
#ifndef PIXELINK_COM_PIXELINKAPI_H
#define PIXELINK_COM_PIXELINKAPI_H

/**
 * Specify our calling convention and visibility 
 */
#ifdef PIXELINK_LINUX
#define PXL_APICALL
#ifdef PXLAPI40_EXPORTS
#define PXL_API  __attribute__(( visibility("default") ))
#else
#define PXL_API
#endif

#else
/* Windows */
#define _WINSOCKAPI_  // Stop windows from including winsock.h
#include <windows.h>
#define PXL_APICALL __stdcall
#ifdef PXLAPI40_EXPORTS
#define PXL_API __declspec(dllexport) PXL_APICALL
#else
#define PXL_API __declspec(dllimport) PXL_APICALL 
#endif

#endif

#include "PixeLINKTypes.h"
#include "PixeLINKCodes.h"

#ifdef __cplusplus
extern "C"
{
#endif


/* 
 * Enumerating cameras
 */
PXL_RETURN_CODE 
PXL_API
PxLGetNumberCameras(
    OUT U32*    pSerialNumbers,
    IN OUT U32* pNumberSerialNumbers);

PXL_RETURN_CODE 
PXL_API
PxLGetNumberCamerasEx(
    OUT CAMERA_ID_INFO* pCameraIdInfo,
    IN OUT U32*         pNumberCameraIdInfos);

/* 
 * Enumerating external controllers
 */
PXL_RETURN_CODE 
PXL_API
PxLGetNumberControllers(
        OUT PCONTROLLER_INFO pControllerInfo,
        IN  ULONG     sizeofControllerInfo,
        IN OUT PULONG pNumberControllerInfo);


/* 
 * Configuring an IP Camera 
 */
PXL_RETURN_CODE 
PXL_API
PxLSetCameraIpAddress(
    IN PXL_MAC_ADDRESS const * pCameraMac,
    IN PXL_IP_ADDRESS  const * pCameraIp,
    IN PXL_IP_ADDRESS  const * pCameraSubnetMask,
    IN PXL_IP_ADDRESS  const * pCameraDefaultGateway,
    IN BOOL32           bPersistent);


/* 
 * Connecting to and disconnecting from a camera
 */
PXL_RETURN_CODE 
PXL_API
PxLInitialize( 
    IN U32      serialNumber,
    OUT HANDLE* phCamera);

PXL_RETURN_CODE 
PXL_API
PxLInitializeEx( 
    IN U32      serialNumber,
    OUT HANDLE* phCamera,
    IN U32      flags);

PXL_RETURN_CODE 
PXL_API
PxLUninitialize(
    IN HANDLE hCamera);

/* 
 * Assigning and unassigning external controllers to a camera
 */
PXL_RETURN_CODE 
PXL_API
PxLAssignController( 
        IN HANDLE hCamera,
        IN ULONG  controllerSerialNum);

PXL_RETURN_CODE 
PXL_API
PxLUnassignController( 
        IN HANDLE hCamera,
        IN ULONG  controllerSerialNum);

/*
 * Functions for getting information about a camera
 */
PXL_RETURN_CODE 
PXL_API
PxLGetCameraInfo( 
    IN HANDLE        hCamera,
    OUT CAMERA_INFO* pInformation);

PXL_RETURN_CODE 
PXL_API
PxLGetCameraInfoEx( 
    IN HANDLE        hCamera,
    OUT CAMERA_INFO* pInformation,
    IN U32           informationSize);

PXL_RETURN_CODE
PXL_API
PxLGetCameraFeatures( 
    IN HANDLE            hCamera,
    IN U32               featureId,
    OUT CAMERA_FEATURES* pFeatureInfo,
    IN OUT U32*          pBufferSize);

PXL_RETURN_CODE 
PXL_API
PxLGetFeature( 
    IN HANDLE   hCamera,
    IN U32      featureId,
    OUT U32*    pFlags,
    IN OUT U32* pNumberOfParams,
    OUT F32*    pParams);

PXL_RETURN_CODE 
PXL_API
PxLSetFeature(
    IN HANDLE hCamera,
    IN U32    featureId,
    IN U32    flags,
    IN U32    numberOfParams,
    IN F32 const *   pParams);

PXL_RETURN_CODE 
PXL_API
PxLGetCurrentTimestamp(
    IN HANDLE   hCamera,
    OUT double* pCurrentTimestamp);

PXL_RETURN_CODE 
PXL_API
PxLSetCameraName(
    IN HANDLE hCamera,
    IN LPCSTR  pCameraName);

PXL_RETURN_CODE 
PXL_API
PxLGetErrorReport(
    IN  HANDLE        hCamera,
    OUT ERROR_REPORT* pErrorReport);


/*
 * Factory- and User-Settings
 */
PXL_RETURN_CODE 
PXL_API
PxLSaveSettings(
    IN HANDLE hCamera,
    IN U32    channelNumber);

PXL_RETURN_CODE 
PXL_API
PxLLoadSettings(
    IN HANDLE hCamera,
    IN U32    channelNumber);


/*
 * Camera Descriptor Control
 */
#ifndef PIXELINK_LINUX
PXL_RETURN_CODE 
PXL_API
PxLCreateDescriptor( 
    IN HANDLE   hCamera,
    OUT HANDLE* pDescriptorHandle,
    IN U32      updateMode);
#endif /* PIXELINK_LINUX */

#ifndef PIXELINK_LINUX
PXL_RETURN_CODE 
PXL_API
PxLRemoveDescriptor( 
    IN HANDLE hCamera,
    IN HANDLE hDescriptor);
#endif /* PIXELINK_LINUX */

#ifndef PIXELINK_LINUX
PXL_RETURN_CODE 
PXL_API
PxLUpdateDescriptor(
    IN HANDLE hCamera,
    IN HANDLE hDescriptor,
    IN U32    updateMode);
#endif /* PIXELINK_LINUX */


/* 
 * Streaming and frames
 */
PXL_RETURN_CODE 
PXL_API
PxLSetStreamState(
    IN HANDLE hCamera,
    IN U32    streamState);

PXL_RETURN_CODE 
PXL_API
PxLGetNextFrame( 
    IN HANDLE       hCamera,
    IN U32          bufferSize,
    OUT LPVOID      pFrame,
    OUT FRAME_DESC* pFrameDesc);


PXL_RETURN_CODE 
PXL_API
PxLFormatImage( 
    IN void const *       pSrcFrame,
    IN FRAME_DESC const * pSrcFrameDesc,
    IN U32         outputFormat,
    OUT LPVOID     pDestBuffer,
    IN OUT U32*    pDestBufferSize);


/*
 * Callbacks
 */
PXL_RETURN_CODE 
PXL_API
PxLSetCallback ( 
    IN HANDLE hCamera,
    IN U32    callbackType,
    IN LPVOID pContext,
    IN U32 (PXL_APICALL* DataProcessFunction)(
            IN HANDLE             hCamera,
            IN OUT LPVOID         pFrameData,
            IN U32                dataFormat, 
            IN FRAME_DESC const * pFrameDescr,
            IN LPVOID             pContext));


/*
 * Clips, and PixeLINK Data Streams
 */
#ifndef PIXELINK_LINUX
PXL_RETURN_CODE 
PXL_API
PxLGetClip(
    IN HANDLE hCamera,
    IN U32    numberOfFramesToCapture,
    IN LPCSTR  pFileName,
    IN U32 (PXL_APICALL * TerminationFunction)(
        IN HANDLE          hCamera,
        IN U32             numberOfFramesCaptured,
        IN PXL_RETURN_CODE returnCode));
#endif /* PIXELINK_LINUX */

PXL_RETURN_CODE 
PXL_API
PxLGetEncodedClip(
    IN HANDLE hCamera,
    IN U32    numberOfFramesToCapture,
    IN LPCSTR pFileName,
    IN PCLIP_ENCODING_INFO  pClipInfo,
    IN U32 (PXL_APICALL * TerminationFunction)(
        IN HANDLE          hCamera,
        IN U32             numberOfFrameBlocksStreamed,
        IN PXL_RETURN_CODE returnCode));

#ifndef PIXELINK_LINUX
PXL_RETURN_CODE
PXL_API
PxLFormatClip( 
    IN LPCSTR pInputFileName,
    IN LPCSTR pOutputFileName,
    IN U32   outputFormat);
#endif /* PIXELINK_LINUX */

PXL_RETURN_CODE
PXL_API
PxLFormatClipEx( 
    IN LPCSTR pInputFileName,
    IN LPCSTR pOutputFileName,
    IN U32   inputFormat,
    IN U32   outputFormat);


/*
 * Previewing
 */
PXL_RETURN_CODE 
PXL_API
PxLSetPreviewState(
    IN HANDLE hCamera,
    IN U32    previewState,
    OUT HWND* pHWnd);

PXL_RETURN_CODE 
PXL_API
PxLSetPreviewStateEx(
    IN HANDLE hCamera,
    IN U32    previewState,
    OUT HWND* pHWnd,
    IN LPVOID pContext,
    IN U32 (PXL_APICALL * ChangeFunction)(
        IN HANDLE         hCamera,
        IN U32            changeCode,
        IN LPVOID         pContext));

#ifdef __cplusplus
PXL_RETURN_CODE 
PXL_API
PxLSetPreviewSettings(
    IN HANDLE hCamera,
    IN LPCSTR  pTitle  = "PixeLINK Preview",
#ifndef PIXELINK_LINUX
    IN U32    style   = WS_OVERLAPPEDWINDOW|WS_VISIBLE,
    IN U32    left    = CW_USEDEFAULT, 
    IN U32    top     = CW_USEDEFAULT,
    IN U32    width   = CW_USEDEFAULT,
    IN U32    height  = CW_USEDEFAULT,
#else
    IN U32    style   = 0,
    IN U32    left    = 0, 
    IN U32    top     = 0,
    IN U32    width   = 0,
    IN U32    height  = 0,
#endif /* PIXELINK_LINUX */
    IN HWND   hParent = NULL,
    IN U32    childId = 0);
#else
PXL_RETURN_CODE 
PXL_API
PxLSetPreviewSettings(
    IN HANDLE hCamera,
    IN LPCSTR  pTitle,
    IN U32    style,
    IN U32    left,
    IN U32    top,
    IN U32    width,
    IN U32    height,
    IN HWND   hParent,
    IN U32    childId);
#endif

PXL_RETURN_CODE
PXL_API
PxLResetPreviewWindow(
    IN HANDLE hCamera);


/*
 * Functions for use by PixeLINK only
 */
#ifndef PIXELINK_LINUX
PXL_RETURN_CODE 
PXL_API
PxLDebug(
    IN HANDLE  hCamera,
    IN U16     requestType,
    IN U32     offsetHigh,
    IN U32     offsetLow,
    IN U32     bufferSize,
    IN OUT U8* pBuffer);
#endif /* PIXELINK_LINUX */

PXL_RETURN_CODE
PXL_API
PxLCameraRead (
    IN HANDLE  hCamera,
    IN U32     bufferSize,
    IN OUT U8* pBuffer);

PXL_RETURN_CODE
PXL_API
PxLCameraWrite (
    IN HANDLE    hCamera,
    IN U32       bufferSize,
    IN const U8* pBuffer);

PXL_RETURN_CODE
PXL_API
PxLPrivateCmd (
    IN HANDLE   hCamera,
    IN U32      bufferSize,
    IN OUT U32* pBuffer);



#ifdef __cplusplus
}
#endif 

#endif /* PIXELINK_COM_PIXELINKAPI_H */
