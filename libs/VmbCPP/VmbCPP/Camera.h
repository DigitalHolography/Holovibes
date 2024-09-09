/*=============================================================================
  Copyright (C) 2012 - 2022 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        Camera.h

  Description: Definition of class VmbCPP::Camera.

-------------------------------------------------------------------------------

  THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR IMPLIED
  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF TITLE,
  NON-INFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A PARTICULAR  PURPOSE ARE
  DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, 
  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED  
  AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR 
  TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=============================================================================*/

#ifndef VMBCPP_CAMERA_H
#define VMBCPP_CAMERA_H

/**
* \file  Camera.h
*
* \brief Definition of class VmbCPP::Camera.
*/

#include <string>
#include <vector>

#include <VmbC/VmbC.h>

#include "Frame.h"
#include "IFrameObserver.h"
#include "Interface.h"
#include "LocalDevice.h"
#include "PersistableFeatureContainer.h"
#include "SharedPointerDefines.h"
#include "Stream.h"
#include "UniquePointer.hpp"
#include "VmbCPPCommon.h"

namespace VmbCPP {

/**
 * \brief A type alias for a vector of shared pointers to frames.
 */
typedef std::vector<FramePtr> FramePtrVector;

/**
 * \brief A type alias for a vector of shared pointers to streams. 
 */
typedef std::vector<StreamPtr> StreamPtrVector;

/**
 * \brief A class for accessing camera related functionality.
 * 
 * This object corresponds to the GenTL remote device.
 */
class Camera : public PersistableFeatureContainer, public ICapturingModule
{
  public:

    /**
    * \brief    Creates an instance of class Camera given the interface info object received from
    *           the Vmb C API.
    * 
    * If "IP_OR_MAC@" occurs in ::VmbCameraInfo::cameraIdString of \p cameraInfo, the camera id used to
    * identify the camera is the substring starting after the first occurence of this string with the
    * next occurence of the same string removed, should it exist. Otherwise
    * ::VmbCameraInfo::cameraIdExtended is used to identify the camera.
    * 
    * If ::VmbCameraInfo::cameraIdExtended is used, it needs to match the extended id retrieved from
    * the VmbC API.
    * 
    * Any strings in \p cameraInfo that are null are treated as the empty string.
    *
    * \param[in ] cameraInfo    The struct containing the information about the camera.
    * \param[in ] pInterface    The shared pointer to the interface providing the camera
    * 
    * \exception std::bad_alloc   The memory available is insufficient to allocate the storage
    *                             required to store the data.
    */
    IMEXPORT Camera(const VmbCameraInfo_t& cameraInfo,
                    const InterfacePtr& pInterface);

    /**
    * 
    * \brief    Destroys an instance of class Camera
    * 
    * Destroying a camera implicitly closes it beforehand.
    * 
    */ 
    IMEXPORT virtual ~Camera();

    /**
     * 
     * \brief     Opens the specified camera.
     * 
     * A camera may be opened in a specific access mode. This mode determines
     * the level of control you have on a camera.
     * 
     * \param[in ]  accessMode      Access mode determines the level of control you have on the camera
     * 
     * \returns ::VmbErrorType
     * 
     * \retval ::VmbErrorSuccess            The call was successful
     *
     * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
     *
     * \retval ::VmbErrorInvalidCall        If called from frame callback or chunk access callback
     * 
     * \retval ::VmbErrorNotFound           The designated camera cannot be found
     */   
    IMEXPORT virtual VmbErrorType Open(VmbAccessModeType accessMode);

    /**
    * 
    * \brief     Closes the specified camera.
    * 
    * Depending on the access mode this camera was opened in, events are killed,
    * callbacks are unregistered, the frame queue is cleared, and camera control is released.
    * 
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess         If no error
    * 
    * \retval ::VmbErrorDeviceNotOpen   Camera was not opened before the current command
    * 
    * \retval ::VmbErrorInUse           The camera is currently in use with ::VmbChunkDataAccess
    * 
    * \retval ::VmbErrorBadHandle       The handle does not correspond to an open camera
    *
    * \retval ::VmbErrorInvalidCall     If called from frame callback or chunk access callback
    */ 
    IMEXPORT virtual VmbErrorType Close();

    /**
    * 
    * \brief     Gets the ID of a camera.
    * 
    * The id is the id choosen by the transport layer. There's no guarantee it's human readable.
    * 
    * The id same id may be used by multiple cameras provided by different interfaces.
    * 
    * \note      This information remains static throughout the object's lifetime
    * 
    * \param[out]   cameraID         The string the camera id is written to
    * 
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess     If no error
    * 
    * \retval ::VmbErrorResources   The attempt to allocate memory for storing the output failed.
    */ 
    VmbErrorType GetID(std::string& cameraID) const noexcept;

    /**
    *
    * \brief     Gets the extenden ID of a camera (globally unique identifier)
    * 
    * The extended id is unique for the camera. The same physical camera may be listed multiple times
    * with different extended ids, if multiple ctis or multiple interfaces provide access to the device.
    * 
    * \note      This information remains static throughout the object's lifetime
    *
    * \param[out]   extendedID      The the extended id is written to
    *
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess     If no error
    * 
    * \retval ::VmbErrorResources   The attempt to allocate memory for storing the output failed.
    */
    VmbErrorType GetExtendedID(std::string& extendedID) const noexcept;

    /**
    * \brief     Gets the display name of a camera.
    * 
    * \note      This information remains static throughout the object's lifetime
    * 
    * \param[out]   name         The string the name of the camera is written to
    * 
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess     If no error
    * 
    * \retval ::VmbErrorResources   The attempt to allocate memory for storing the output failed.
    */ 
    VmbErrorType GetName(std::string& name) const noexcept;
    
    /**
    * 
    * \brief     Gets the model name of a camera.
    *
    * \note      This information remains static throughout the object's lifetime
    * 
    * \param[out]   model         The string the model name of the camera is written to
    * 
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess     If no error
    * 
    * \retval ::VmbErrorResources   The attempt to allocate memory for storing the output failed.
    */ 
    VmbErrorType GetModel(std::string& model) const noexcept;

    /**
    * 
    * \brief     Gets the serial number of a camera.
    *
    * \note      This information remains static throughout the object's lifetime
    * 
    * \param[out]   serialNumber    The string to write the serial number of the camera to
    * 
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess     If no error
    * 
    * \retval ::VmbErrorResources   The attempt to allocate memory for storing the output failed.
    */ 
    VmbErrorType GetSerialNumber(std::string& serialNumber) const noexcept;

    /**
    * 
    * \brief     Gets the interface ID of a camera.
    * 
    * \note      This information remains static throughout the object's lifetime
    * 
    * \param[out]   interfaceID     The string to write the interface ID to
    * 
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess     If no error
    * 
    * \retval ::VmbErrorResources   The attempt to allocate memory for storing the output failed.
    */ 
    VmbErrorType GetInterfaceID(std::string& interfaceID) const;

    /**
    * 
    * \brief     Gets the type of the interface the camera is connected to. And therefore the type of the camera itself.
    * 
    * \param[out]   interfaceType   A reference to the interface type variable to write the output to
    * 
    * \returns ::VmbErrorType
    *
    * \retval ::VmbErrorSuccess         If no error
    * 
    * \retval ::VmbErrorNotAvailable    No interface is currently associated with this object
    */ 
    IMEXPORT VmbErrorType GetInterfaceType(VmbTransportLayerType& interfaceType) const;

    /**
    *
    * \brief     Gets the shared pointer to the interface providing the camera.
    *
    * \param[out]   pInterface    The shared pointer to assign the interface assigned to
    *
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess         If no error
    * 
    * \retval ::VmbErrorNotAvailable    No interface is currently associated with this object
    */
    IMEXPORT VmbErrorType GetInterface(InterfacePtr& pInterface) const;

    /**
    *
    * \brief     Gets the shared pointer to the local device module associated with this camera.
    * 
    * This information is only available for open cameras.
    *
    * \param[out]   pLocalDevice    The shared pointer the local device is assigned to
    *
    * \returns ::VmbErrorType
    *
    * \retval ::VmbErrorSuccess           If no error
    * 
    * \retval ::VmbErrorDeviceNotOpen     The camera is currently not opened
    */
    IMEXPORT VmbErrorType GetLocalDevice(LocalDevicePtr& pLocalDevice);

    /**
    * \brief     Gets the pointer of the related transport layer.
    *
    * \param[out]   pTransportLayer     The shared pointer the transport layer is assigned to
    *
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess         If no error
    * 
    * \retval ::VmbErrorNotAvailable    No interface is currently associated with
    *                                   this object or the interface is not associated
    *                                   with a transport layer
    */
    IMEXPORT VmbErrorType GetTransportLayer(TransportLayerPtr& pTransportLayer) const;

    /**
    * \brief     Gets the vector with the available streams of the camera.
    *
    * \param[out]   streams     The vector the available streams of the camera are written to
    *
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess     If no error
    * 
    * \retval ::VmbErrorResources      The attempt to allocate memory for storing the output failed.
    * \retval ::VmbErrorDeviceNotOpen  The camera is currently not open
    * \retval ::VmbErrorNotAvailable   The camera does not provide any streams
    */
    VmbErrorType GetStreams(StreamPtrVector& streams) noexcept;

    /**
     * 
     * \brief     Gets the access modes of a camera.
     * 
     * \param[out]  permittedAccess     The possible access modes of the camera
     * 
     * \returns ::VmbErrorType
     * 
     * \retval ::VmbErrorSuccess            If no error
     * 
     * \retval ::VmbErrorInvalidCall        If called from a chunk access callback
     * 
     * \retval ::VmbErrorNotFound           No camera with the given id is found
        */ 
    IMEXPORT VmbErrorType GetPermittedAccess(VmbAccessModeType& permittedAccess) const;

    /**
    * \brief     Reads a block of memory. The number of bytes to read is determined by the size of the provided buffer.
    * 
    * \param[in ]   address    The address to read from
    * \param[out]   buffer     The returned data as vector
    * 
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess         If all requested bytes have been read
    * 
    * \retval ::VmbErrorBadParameter    Vector \p buffer is empty.
    *
    * \retval ::VmbErrorIncomplete      If at least one, but not all bytes have been read. See overload `ReadMemory(const VmbUint64_t&, UcharVector&, VmbUint32_t&) const`.
    * 
    * \retval ::VmbErrorInvalidCall     If called from a chunk access callback
    *
    * \retval ::VmbErrorInvalidAccess   Operation is invalid with the current access mode
    */ 
    VmbErrorType ReadMemory(const VmbUint64_t& address, UcharVector& buffer) const noexcept;

    /**
    * 
    * \brief     Same as `ReadMemory(const Uint64Vector&, UcharVector&) const`, but returns the number of bytes successfully read in case of an error ::VmbErrorIncomplete.
    * 
    * \param[in]    address        The address to read from
    * \param[out]   buffer         The returned data as vector
    * \param[out]   completeReads  The number of successfully read bytes
    * 
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess         If all requested bytes have been read
    * 
    * \retval ::VmbErrorBadParameter    Vector \p buffer is empty.
    *
    * \retval ::VmbErrorIncomplete      If at least one, but not all bytes have been read. See overload `ReadMemory(const VmbUint64_t&, UcharVector&, VmbUint32_t&) const`.
    * 
    * \retval ::VmbErrorInvalidCall     If called from a chunk access callback
    *
    * \retval ::VmbErrorInvalidAccess   Operation is invalid with the current access mode
    */ 
    VmbErrorType ReadMemory(const VmbUint64_t& address, UcharVector& buffer, VmbUint32_t& completeReads) const noexcept;

    /**
    * 
    * \brief     Writes a block of memory. The number of bytes to write is determined by the size of the provided buffer.
    * 
    * \param[in]    address    The address to write to
    * \param[in]    buffer     The data to write as vector
    * 
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess         If all requested bytes have been written
    * 
    * \retval ::VmbErrorBadParameter    Vector \p buffer is empty.
    * 
    * \retval ::VmbErrorIncomplete      If at least one, but not all bytes have been written. See overload `WriteMemory(const VmbUint64_t&, const UcharVector&, VmbUint32_t&)`.
    * 
    * \retval ::VmbErrorInvalidCall     If called from a chunk access callback
    *
    * \retval ::VmbErrorInvalidAccess   Operation is invalid with the current access mode
    */ 
    VmbErrorType WriteMemory(const VmbUint64_t& address, const UcharVector& buffer) noexcept;

    /**
    * \brief     Same as WriteMemory(const Uint64Vector&, const UcharVector&), but returns the number of bytes successfully written in case of an error ::VmbErrorIncomplete.
    * 
    * \param[in]    address        The address to write to
    * \param[in]    buffer         The data to write as vector
    * \param[out]   sizeComplete   The number of successfully written bytes
    * 
    * \returns ::VmbErrorType
    *
    * \retval ::VmbErrorSuccess         If all requested bytes have been written
    * 
    * \retval ::VmbErrorBadParameter    Vector \p buffer is empty.
    * 
    * \retval ::VmbErrorIncomplete      If at least one, but not all bytes have been written.
    * 
    * \retval ::VmbErrorInvalidCall     If called from a chunk access callback
    *
    * \retval ::VmbErrorInvalidAccess   Operation is invalid with the current access mode
    */ 
    VmbErrorType WriteMemory(const VmbUint64_t& address, const UcharVector& buffer, VmbUint32_t& sizeComplete) noexcept;

    /**
    * \brief     Gets one image synchronously.
    * 
    * \param[out]   pFrame          The frame that gets filled
    * \param[in ]   timeout         The time in milliseconds to wait until the frame got filled
    * \param[in ]   allocationMode  The frame allocation mode
    * 
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess         If no error
    * 
    * \retval ::VmbErrorBadParameter    \p pFrame is null.
    * 
    * \retval ::VmbErrorInvalidCall     If called from a chunk access callback
    * 
    * \retval ::VmbErrorInUse           If the frame was queued with a frame callback
    * 
    * \retval ::VmbErrorTimeout         Call timed out
    * 
    * \retval ::VmbErrorDeviceNotOpen   The camera is currently not open
    * 
    * \retval ::VmbErrorNotAvailable    The camera does not provide any streams
    */ 
    IMEXPORT VmbErrorType AcquireSingleImage(FramePtr& pFrame, VmbUint32_t timeout, FrameAllocationMode allocationMode = FrameAllocation_AnnounceFrame);

    /**
    * 
    * \brief     Gets a certain number of images synchronously.
    * 
    * The size of the frame vector determines the number of frames to use.
    * 
    * \param[in,out]    frames          The frames that get filled
    * \param[in ]       timeout         The time in milliseconds to wait until one frame got filled
    * \param[in ]       allocationMode  The frame allocation mode
    * 
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess         If no error
    * 
    * \retval ::VmbErrorInternalFault   Filling all the frames was not successful.
    * 
    * \retval ::VmbErrorBadParameter    Vector \p frames is empty or one of the frames is null.
    * 
    * \retval ::VmbErrorInvalidCall     If called from a chunk access callback
    * 
    * \retval ::VmbErrorInUse           If the frame was queued with a frame callback
    * 
    * \retval ::VmbErrorDeviceNotOpen   The camera is currently not open
    * 
    * \retval ::VmbErrorNotAvailable    The camera does not provide any streams
    * 
    */ 
    VmbErrorType AcquireMultipleImages(FramePtrVector& frames, VmbUint32_t timeout, FrameAllocationMode allocationMode = FrameAllocation_AnnounceFrame);

    /**
    * 
    * \brief    Same as `AcquireMultipleImages(FramePtrVector&, VmbUint32_t, FrameAllocationMode)`, but returns the number of frames that were filled completely.
    * 
    * The size of the frame vector determines the number of frames to use.
    * On return, \p numFramesCompleted holds the number of frames actually filled.
    * 
    * \param[in,out]    frames              The frames to fill
    * \param[in ]       timeout             The time in milliseconds to wait until one frame got filled
    * \param[out]       numFramesCompleted  The number of frames that were filled completely
    * \param[in ]       allocationMode      The frame allocation mode
    * 
    * \returns ::VmbErrorType

    * \retval ::VmbErrorInternalFault   Filling all the frames was not successful.
    * 
    * \retval ::VmbErrorBadParameter    Vector \p frames is empty or one of the frames is null.
    * 
    * \retval ::VmbErrorInvalidCall     If called from a chunk access callback
    * 
    * \retval ::VmbErrorInUse           If the frame was queued with a frame callback
    * 
    * \retval ::VmbErrorDeviceNotOpen   The camera is currently not open
    * 
    * \retval ::VmbErrorNotAvailable    The camera does not provide any streams
    */ 
    VmbErrorType AcquireMultipleImages(FramePtrVector& frames, VmbUint32_t timeout, VmbUint32_t& numFramesCompleted, FrameAllocationMode allocationMode = FrameAllocation_AnnounceFrame);

    /**
    * 
    * \brief     Starts streaming and allocates the needed frames
    * 
    * \param[in ]   bufferCount    The number of frames to use
    * \param[out]   pObserver      The observer to use on arrival of acquired frames
    * \param[in ]   allocationMode The frame allocation mode
    * 
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess         If no error
    * \retval ::VmbErrorDeviceNotOpen   The camera is currently not open
    * \retval ::VmbErrorNotAvailable    The camera does not provide any streams
    * \retval ::VmbErrorApiNotStarted   VmbStartup() was not called before the current command
    * \retval ::VmbErrorBadHandle       The given handle is not valid
    * \retval ::VmbErrorInvalidAccess   Operation is invalid with the current access mode
    */ 
    IMEXPORT VmbErrorType StartContinuousImageAcquisition(int bufferCount, const IFrameObserverPtr& pObserver, FrameAllocationMode allocationMode = FrameAllocation_AnnounceFrame);

    /**
    * 
    * \brief     Stops streaming and deallocates the frames used
    * 
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess        If no error
    * \retval ::VmbErrorApiNotStarted  VmbStartup() was not called before the current command
    * \retval ::VmbErrorDeviceNotOpen  The camera is currently not open
    */ 
    IMEXPORT VmbErrorType StopContinuousImageAcquisition();

    /**
      * \brief     Get the necessary payload size for buffer allocation.
      *
      * \param[in ]  nPayloadSize   The variable to write the payload size to
      *
      * \returns ::VmbErrorType
      * \retval ::VmbErrorSuccess        If no error
      * \retval ::VmbErrorApiNotStarted  VmbStartup() was not called before the current command
      * \retval ::VmbErrorBadHandle      The given handle is not valid
      */
    IMEXPORT VmbErrorType GetPayloadSize(VmbUint32_t& nPayloadSize) noexcept;

    /**
    * 
    * \brief     Announces a frame to the API that may be queued for frame capturing later.
    * 
    * The frame is announced for the first stream.
    * 
    * Allows some preparation for frames like DMA preparation depending on the transport layer.
    * The order in which the frames are announced is not taken in consideration by the API.
    * 
    * \param[in ]  pFrame         Shared pointer to a frame to announce
    * 
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess        If no error
    * \retval ::VmbErrorApiNotStarted  VmbStartup() was not called before the current command
    * \retval ::VmbErrorBadHandle      The given handle is not valid
    * \retval ::VmbErrorBadParameter   \p pFrame is null.
    * \retval ::VmbErrorStructSize     The given struct size is not valid for this version of the API
    * \retval ::VmbErrorDeviceNotOpen  The camera is currently not open
    * \retval ::VmbErrorNotAvailable   The camera does not provide any streams
    */ 
    IMEXPORT virtual VmbErrorType AnnounceFrame(const FramePtr& pFrame) override;
    
    /**
    * 
    * \brief     Revoke a frame from the API.
    * 
    * The frame is revoked for the first stream.
    * 
    * The referenced frame is removed from the pool of frames for capturing images.
    * 
    * A call to FlushQueue may be required for the frame to be revoked successfully.
    * 
    * \param[in ]  pFrame         Shared pointer to a frame that is to be removed from the list of announced frames
    * 
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess        If no error
    * \retval ::VmbErrorApiNotStarted  VmbStartup() was not called before the current command
    * \retval ::VmbErrorBadHandle      The given frame pointer is not valid
    * \retval ::VmbErrorBadParameter   \p pFrame is null.
    * \retval ::VmbErrorStructSize     The given struct size is not valid for this version of the API
    * \retval ::VmbErrorDeviceNotOpen  The camera is currently not open
    * \retval ::VmbErrorNotAvailable   The camera does not provide any streams
    */ 
    IMEXPORT virtual VmbErrorType RevokeFrame(const FramePtr& pFrame) override;

    /**
    * \brief     Revoke all frames announced for the first stream of this camera.
    * 
    * A call to FlushQueue may be required to be able to revoke all frames.
    * 
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess        If no error
    * \retval ::VmbErrorApiNotStarted  VmbStartup() was not called before the current command
    * \retval ::VmbErrorBadHandle      The given handle is not valid
    * \retval ::VmbErrorDeviceNotOpen  The camera is currently not open
    * \retval ::VmbErrorNotAvailable   The camera does not provide any streams
    */ 
    IMEXPORT virtual VmbErrorType RevokeAllFrames() override;
    
    /**
    * 
    * \brief     Queues a frame that may be filled during frame capturing.
    * 
    * The frame is queued for the first stream.
    *
    * The given frame is put into a queue that will be filled sequentially.
    * The order in which the frames are filled is determined by the order in which they are queued.
    * If the frame was announced with AnnounceFrame() before, the application
    * has to ensure that the frame is also revoked by calling RevokeFrame() or RevokeAll()
    * when cleaning up.
    * 
    * \param[in ]  pFrame    A shared pointer to a frame
    * 
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess        If no error
    * \retval ::VmbErrorDeviceNotOpen  The camera is currently not open
    * \retval ::VmbErrorNotAvailable   The camera does not provide any streams  
    * \retval ::VmbErrorApiNotStarted  VmbStartup() was not called before the current command
    * \retval ::VmbErrorBadHandle      The given frame is not valid
    * \retval ::VmbErrorBadParameter   \p pFrame is null.
    * \retval ::VmbErrorStructSize     The given struct size is not valid for this version of the API
    * \retval ::VmbErrorInvalidCall    StopContinuousImageAcquisition is currently running in another thread
    */ 
    IMEXPORT virtual VmbErrorType QueueFrame(const FramePtr& pFrame) override;

    /**
    * 
    * \brief     Flushes the capture queue.
    * 
    * Works with the first available stream.
    *            
    * All currently queued frames will be returned to the user, leaving no frames in the input queue.
    * After this call, no frame notification will occur until frames are queued again.
    * 
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess        If no error
    * \retval ::VmbErrorApiNotStarted  VmbStartup() was not called before the current command
    * \retval ::VmbErrorBadHandle      The given handle is not valid
    * \retval ::VmbErrorDeviceNotOpen  The camera is currently not open
    * \retval ::VmbErrorNotAvailable   The camera does not provide any streams
    */ 
    IMEXPORT virtual VmbErrorType FlushQueue() override;
    
    /**
    * 
    * \brief     Prepare the API for incoming frames from this camera.
    *            Works with the first available stream.
    * 
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess        If no error
    * \retval ::VmbErrorApiNotStarted  VmbStartup() was not called before the current command
    * \retval ::VmbErrorBadHandle      The given handle is not valid
    * \retval ::VmbErrorDeviceNotOpen  The camera is currently not open
    * \retval ::VmbErrorNotAvailable   The camera does not provide any streams
    * \retval ::VmbErrorInvalidAccess  Operation is invalid with the current access mode
    */ 
    IMEXPORT virtual VmbErrorType StartCapture() override;

    /**
    * 
    * \brief     Stops the API from being able to receive frames from this camera. The frame callback will not be called any more.
    *            Works with the first stream of the camera.
    * 
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess        If no error
    * \retval ::VmbErrorApiNotStarted  VmbStartup() was not called before the current command
    * \retval ::VmbErrorBadHandle      The given handle is not valid
    * \retval ::VmbErrorDeviceNotOpen  The camera is currently not open
    * \retval ::VmbErrorNotAvailable   The camera does not provide any streams
    */ 
    IMEXPORT virtual VmbErrorType EndCapture() override;

    /**
     * \brief Checks if the extended id of this object matches a string.
     * 
     * \param[in]   extendedId  the id to to compare the extended id of this object with
     * 
     * \return true, if \p extendedId is non-null and matches the extended id of this object,
     *         false otherwise.
     */
    IMEXPORT bool ExtendedIdEquals(char const* extendedId) noexcept;

    Camera() = delete;

    /**
     * \brief The object is non-copyable
     */
    Camera (const Camera&) = delete;

    /**
     * \brief The object is non-copyable
     */
    Camera& operator=(const Camera&) = delete;

    /**
    * \brief    Retrieve the necessary buffer alignment size in bytes (equals 1 if Data Stream has no such restriction)
    *
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess         If no error
    * \retval ::VmbErrorApiNotStarted   VmbStartup() was not called before the current command
    */
    IMEXPORT virtual VmbErrorType GetStreamBufferAlignment(VmbUint32_t& nBufferAlignment) override;

  private:

        struct Impl;
        UniquePointer<Impl> m_pImpl;

    //  Array functions to pass data across DLL boundaries
    IMEXPORT VmbErrorType GetID(char* const pID, VmbUint32_t& length, bool extended = false) const noexcept;
    IMEXPORT VmbErrorType GetName(char* const pName, VmbUint32_t& length) const noexcept;
    IMEXPORT VmbErrorType GetModel(char* const pModelName, VmbUint32_t& length) const noexcept;
    IMEXPORT VmbErrorType GetSerialNumber(char* const pSerial, VmbUint32_t& length) const noexcept;
    IMEXPORT VmbErrorType AcquireMultipleImages(FramePtr* pFrames, VmbUint32_t size, VmbUint32_t nTimeout, VmbUint32_t* pNumFramesCompleted, FrameAllocationMode allocationMode);
    IMEXPORT VmbErrorType ReadMemory(VmbUint64_t address, VmbUchar_t* pBuffer, VmbUint32_t bufferSize, VmbUint32_t* pSizeComplete) const noexcept;
    IMEXPORT VmbErrorType WriteMemory(VmbUint64_t address, const VmbUchar_t* pBuffer, VmbUint32_t bufferSize, VmbUint32_t* pSizeComplete) noexcept;
    IMEXPORT VmbErrorType GetStreams(StreamPtr* pStreams, VmbUint32_t& rnSize) noexcept;

};

} // namespace VmbCPP

#include "Camera.hpp"

#endif
