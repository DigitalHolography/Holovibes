/****************************************************************************************************************
 * COPYRIGHT © 2009 PixeLINK CORPORATION.  ALL RIGHTS RESERVED.                                                 *
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

#ifndef PIXELINK_COM_PIXELINKCODES_H
#define PIXELINK_COM_PIXELINKCODES_H

#include "PixeLINKTypes.h"
 
#define API_SUCCESS(rc) (!((rc) & (0x80000000)) )  

#define ApiSuccess                   ((PXL_RETURN_CODE)0x00000000)  /* The function completed successfully */
#define ApiSuccessParametersChanged  ((PXL_RETURN_CODE)0x00000001)  /* Indicates that a feature was set successfully, but one or more parameters had to be changed */
#define ApiSuccessAlreadyRunning     ((PXL_RETURN_CODE)0x00000002)  /* The stream is already started */
#define ApiSuccessLowMemory          ((PXL_RETURN_CODE)0x00000003)  /* There is not as much memory as needed for optimum performance. Performance may be affected. */
#define ApiSuccessParameterWarning   ((PXL_RETURN_CODE)0x00000004)  /* The operation completed successfully, but one or more of the parameters are  */
                                                                    /* suspect (e.g. using an invalid IP address) */
#define ApiSuccessReducedSpeedWarning ((PXL_RETURN_CODE)0x00000005) /* The operation completed sucessfully, but the camera is operating at a speed less than it's maximum */
#define ApiSuccessExposureAdjustmentMade  ((PXL_RETURN_CODE)0x00000006) /* The operation completed sucessfully, but the camera had to adjust exposure to accomodate */
#define ApiSuccessWhiteBalanceTooDark     ((PXL_RETURN_CODE)0x00000007) /* The Auto WhiteBalance algorithm could not achieve good results because the auto-adjust area is too dark */
#define ApiSuccessWhiteBalanceTooBright   ((PXL_RETURN_CODE)0x00000008) /* The Auto WhiteBalance algorithm could not achieve good results because the auto-adjust area has too many saturated pxels */
#define ApiSuccessWithFrameLoss      ((PXL_RETURN_CODE)0x00000009)  /* The operation completed sucessfully, but some frame loss occurred (the system could not keep pace with the camera */
#define ApiSuccessGainIneffectiveWarning  ((PXL_RETURN_CODE)0x0000000A) /* Gain operation was successful, but gain is curretly not applicable */


#define ApiUnknownError              ((PXL_RETURN_CODE)0x80000001)  /* Unknown error */
#define ApiInvalidHandleError        ((PXL_RETURN_CODE)0x80000002)  /* The handle parameter is invalid */
#define ApiInvalidParameterError     ((PXL_RETURN_CODE)0x80000003)  /* Invalid parameter */
#define ApiBufferTooSmall            ((PXL_RETURN_CODE)0x80000004)  /* A buffer passed as parameter is too small. */
#define ApiInvalidFunctionCallError  ((PXL_RETURN_CODE)0x80000005)  /* The function cannot be called at this time  */
#define ApiNotSupportedError         ((PXL_RETURN_CODE)0x80000006)  /* The API cannot complete the request */
#define ApiCameraInUseError          ((PXL_RETURN_CODE)0x80000007)  /* The camera is already being used by another application */
#define ApiNoCameraError             ((PXL_RETURN_CODE)0x80000008)  /* There is no response from the camera */
#define ApiHardwareError             ((PXL_RETURN_CODE)0x80000009)  /* The camera responded with an error */
#define ApiCameraUnknownError        ((PXL_RETURN_CODE)0x8000000A)  /* The API does not recognize the camera */
#define ApiOutOfBandwidthError       ((PXL_RETURN_CODE)0x8000000B)  /* There is not enough 1394 bandwidth to start the stream */
#define ApiOutOfMemoryError          ((PXL_RETURN_CODE)0x8000000C)  /* The API cannot allocate the required memory */
#define ApiOSVersionError            ((PXL_RETURN_CODE)0x8000000D)  /* The API cannot run on the current operating system */
#define ApiNoSerialNumberError       ((PXL_RETURN_CODE)0x8000000E)  /* The serial number could not be obtained from the camera */
#define ApiInvalidSerialNumberError  ((PXL_RETURN_CODE)0x8000000F)  /* A camera with that serial number could not be found */
#define ApiDiskFullError             ((PXL_RETURN_CODE)0x80000010)  /* Not enough disk space to complete an IO operation */
#define ApiIOError                   ((PXL_RETURN_CODE)0x80000011)  /* An error occurred during an IO operation */
#define ApiStreamStopped             ((PXL_RETURN_CODE)0x80000012)  /* Application requested streaming termination */
#define ApiNullPointerError          ((PXL_RETURN_CODE)0x80000013)  /* A pointer parameter is NULL */
#define ApiCreatePreviewWndError     ((PXL_RETURN_CODE)0x80000014)  /* Error creating the preview window */
#define ApiOutOfRangeError           ((PXL_RETURN_CODE)0x80000016)  /* Indicates that a feature set value is out of range  */
#define ApiNoCameraAvailableError    ((PXL_RETURN_CODE)0x80000017)  /* There is no camera available */
#define ApiInvalidCameraName         ((PXL_RETURN_CODE)0x80000018)  /* Indicates that the name specified is not a valid camera name */
#define ApiGetNextFrameBusy          ((PXL_RETURN_CODE)0x80000019)  /* PxLGetNextFrame() cannot be called at this time a FRAME_OVERLAY callback function is */
                                                                    /* currently being called. */
#define ApiFrameInUseError           ((PXL_RETURN_CODE)0x8000001A)  /* A frame was still in use when the buffers were deallocated. */


#define ApiStreamExistingError       ((PXL_RETURN_CODE)0x90000001)  
#define ApiEnumDoneError             ((PXL_RETURN_CODE)0x90000002)   
#define ApiNotEnoughResourcesError   ((PXL_RETURN_CODE)0x90000003)
#define ApiBadFrameSizeError         ((PXL_RETURN_CODE)0x90000004)
#define ApiNoStreamError             ((PXL_RETURN_CODE)0x90000005)
#define ApiVersionError              ((PXL_RETURN_CODE)0x90000006) 
#define ApiNoDeviceError             ((PXL_RETURN_CODE)0x90000007)
#define ApiCannotMapFrameError       ((PXL_RETURN_CODE)0x90000008)
#define ApiLinkDriverError           ((PXL_RETURN_CODE)0x90000009)  /* The driver for the link to the device, reported an error */
#define ApiInvalidIoctlParameter     ((PXL_RETURN_CODE)0x9000000A) 
#define ApiInvalidOhciDriverError    ((PXL_RETURN_CODE)0x9000000B) 
#define ApiCameraTimeoutError        ((PXL_RETURN_CODE)0x9000000C)  /* Timeout waiting for the camera to respond. */
#define ApiInvalidFrameReceivedError ((PXL_RETURN_CODE)0x9000000D)  /* The camera returned an invalid frame (image) */
#define ApiOSServiceError            ((PXL_RETURN_CODE)0x9000000E)  /* An operating system service returned an error */
#define ApiTimeoutError              ((PXL_RETURN_CODE)0x9000000F)
#define ApiRequiresControlAccess     ((PXL_RETURN_CODE)0x90000010)  /* Camera opertion not permitted because it requires control access */
#define ApiGevInitializationError    ((PXL_RETURN_CODE)0x90000011)  /* error attempting to initialize for communication to GEV cameras. */
#define ApiIpServicesError           ((PXL_RETURN_CODE)0x90000012)  /* Error within the IP Stack while attempting communications with a GEV camera */
#define ApiIpAddressingError         ((PXL_RETURN_CODE)0x90000013)  /* The camera's IP address is not reachable on this host */
#define ApiDriverCommunicationError  ((PXL_RETURN_CODE)0x90000014)  /* Could not cummunicate properly with the driver */
#define ApiInvalidXmlError           ((PXL_RETURN_CODE)0x90000015)  /* An error was encountered when accessing the (GEV) cameras XML file */
#define ApiCameraRejectedValueError  ((PXL_RETURN_CODE)0x90000016)  /* Communications with the camera are good -- but the camera did not like requested value */
#define ApiSuspectedFirewallBlockError ((PXL_RETURN_CODE)0x90000017)  /* Timeout hearing from the (GEV) camera -- suspected firewall issue */
#define ApiIncorrectLinkSpeed        ((PXL_RETURN_CODE)0x90000018)  /* Connected camera link is notsufficient for this camaera (probably too slow) */
#define ApiCameraNotReady            ((PXL_RETURN_CODE)0x90000019)  /* Camera is not in a state to perform this operation */
#define ApiInconsistentConfiguration ((PXL_RETURN_CODE)0x9000001A)  /* Camera configuration is inconsistent */
#define ApiNotPermittedWhileStreaming ((PXL_RETURN_CODE)0x9000001B)  /* You must stop the stream to perform this operation */
#define ApiOSAccessDeniedError       ((PXL_RETURN_CODE)0x9000001C)  /* OS returned ERROR_ACCESS_DENIED error -- trying running as adminstrator */
#define ApiInvalidAutoRoiError       ((PXL_RETURN_CODE)0x9000001D)  /* An 'auto' operation was attempted, but the auto ROI is not within the cameras ROI */
#define ApiGpiHardwareTriggerConflict ((PXL_RETURN_CODE)0x9000001E) /* GPI and Hardware trigger are mutually exclusive; they both can't be enabled simultaneously */
#define ApiGpioConfigurationError    ((PXL_RETURN_CODE)0x9000001F)  /* Invalid configuration -- Only GPIO #1 can be used as a GPI */
#define ApiUnsupportedPixelFormatError ((PXL_RETURN_CODE)0x90000020) /* Attempt to perfrom an operation while using a pixel format that is not supported */
#define ApiUnsupportedClipEncoding   ((PXL_RETURN_CODE)0x90000021)  /* Attempt to capture a video clip using an unsupported video encoding sheme */
#define ApiH264EncodingError         ((PXL_RETURN_CODE)0x90000022)  /* The h264 compression libraries reported an error while captureing video */
#define ApiH264FrameTooLargeError    ((PXL_RETURN_CODE)0x90000023)  /* The h264 compression is limited to 9 MegaPixels */
#define ApiH264InsufficientDataError ((PXL_RETURN_CODE)0x90000024)  /* There was not enough data to encode */
#define ApiNoControllerError         ((PXL_RETURN_CODE)0x90000025)  /* Attempted to perform a controller operation on a non-existing controller */
#define ApiControllerAlreadyAssignedError ((PXL_RETURN_CODE)0x90000026)  /* Attempted to assign a controller to a camera that is already assigned to a differnt camera */
#define ApiControllerInaccessibleError    ((PXL_RETURN_CODE)0x90000027)  /* Cannot access the specified controller */
#define ApiControllerCommunicationError   ((PXL_RETURN_CODE)0x90000028)  /* A error occurred while attempting to communicate with a controller */
#define ApiControllerTimeoutError         ((PXL_RETURN_CODE)0x90000029)  /* The controller timed out responding to a command */
#define ApiBufferTooSmallForInterleavedError ((PXL_RETURN_CODE)0x9000002A)  /* Simiar to ApiBufferTooSmall, but unique to Interleaved HDR mode */

#define ApiOhciDriverError                 ApiLinkDriverError /* Defined for backwards comaptibility */
#define ApiNotPermittedWhileStreamingError ApiNotPermittedWhileStreaming /* Defined for backwards comaptibility, use the shorter one */

#endif
