/*=============================================================================
  Copyright (C) 2022 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        Stream.h

  Description: Definition of class VmbCPP::Stream.

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

#ifndef VMBCPP_STREAM_H
#define VMBCPP_STREAM_H

/**
* \file             Stream.h
*
* \brief            Definition of class VmbCPP::Stream.
*/

#include <string>

#include <VmbC/VmbC.h>

#include "ICapturingModule.h"
#include "PersistableFeatureContainer.h"
#include "UniquePointer.hpp"
#include "VmbCPPCommon.h"


namespace VmbCPP {

/**
 * \brief A class providing access to a single stream of a single camera.
 * 
 * The class provides functionality for acquiring data via the stream. Furthermore it
 * provides access to information about the corresponding GenTL stream module.
 */
class Stream : public PersistableFeatureContainer, public ICapturingModule
{
public:
  
    /**  
    *  \brief     Creates an instance of class Stream
    * 
    * \param[in]    streamHandle    Handle to the stream
    * \param[in]    deviceIsOpen    Sets the internal status to know if the camera device is open or not
    */  
    IMEXPORT Stream(VmbHandle_t streamHandle, bool deviceIsOpen);

    /**
    * \brief Object is not default constructible 
    */
    Stream() = delete;

    /**
    * \brief Object is not copyable 
    */
    Stream(const Stream&) = delete;

    /**
    * \brief Object is not copyable
    */
    Stream& operator=(const Stream&) = delete;

    /**
    *
    * \brief     Destroys an instance of class Stream.
    *            Destroying a stream implicitly closes it beforehand.
    */
    IMEXPORT virtual ~Stream();

    /**
    *
    * \brief     Opens the specified stream.
    *
    * \returns VmbErrorType
    *
    * \retval ::VmbErrorSuccess        If no error
    * \retval ::VmbErrorDeviceNotOpen  VmbSystem or Camera was not opened before the current command
    */
    IMEXPORT virtual VmbErrorType Open();

    /**
    *
    * \brief     Closes the specified stream.
    *
    * \returns VmbErrorType
    *
    * \retval ::VmbErrorSuccess        If no error
    * \retval ::VmbErrorDeviceNotOpen  Camera was not opened before the current command
    */
    IMEXPORT virtual VmbErrorType Close();

    /**
    * \brief     Announces a frame to the API that may be queued for frame capturing later.
    *            Allows some preparation for frames like DMA preparation depending on the transport layer.
    *            The order in which the frames are announced is not taken in consideration by the API.
    *
    * \param[in ]  pFrame              Shared pointer to a frame to announce
    *
    * \returns ::VmbErrorType
    * \retval ::VmbErrorSuccess        If no error
    * \retval ::VmbErrorDeviceNotOpen  Camera was not opened before the current command
    * \retval ::VmbErrorBadHandle      The given handle is not valid
    * \retval ::VmbErrorBadParameter   "pFrame" is null.
    * \retval ::VmbErrorStructSize     The given struct size is not valid for this version of the API
    */
    IMEXPORT virtual VmbErrorType AnnounceFrame(const FramePtr& pFrame) override;


    /**
    * \brief     Revoke a frame from the API.
    *            The referenced frame is removed from the pool of frames for capturing images.
    *
    * \param[in ]  pFrame             Shared pointer to a frame that is to be removed from the list of announced frames
    *
    * \returns VmbErrorType
    * 
    * \retval ::VmbErrorSuccess        If no error
    * \retval ::VmbErrorDeviceNotOpen  Camera was not opened before the current command
    * \retval ::VmbErrorBadHandle      The given frame pointer is not valid
    * \retval ::VmbErrorBadParameter   "pFrame" is null.
    * \retval ::VmbErrorStructSize     The given struct size is not valid for this version of the API
    */
    IMEXPORT virtual VmbErrorType RevokeFrame(const FramePtr& pFrame) override;


    /**
    * \brief     Revoke all frames assigned to this certain camera.
    *
    * \returns VmbErrorType
    * 
    * \retval ::VmbErrorSuccess        If no error
    * \retval ::VmbErrorDeviceNotOpen  Camera was not opened before the current command
    * \retval ::VmbErrorBadHandle      The given handle is not valid
    */
    IMEXPORT virtual VmbErrorType RevokeAllFrames() override;


    /**
    * \brief     Queues a frame that may be filled during frame capturing.
    *
    *  The given frame is put into a queue that will be filled sequentially.
    *  The order in which the frames are filled is determined by the order in which they are queued.
    *  If the frame was announced with AnnounceFrame() before, the application
    *  has to ensure that the frame is also revoked by calling RevokeFrame() or RevokeAll()
    *  when cleaning up.
    *
    * \param[in ]  pFrame             A shared pointer to a frame
    *
    * \returns VmbErrorType
    * 
    * \retval ::VmbErrorSuccess        If no error
    * \retval ::VmbErrorDeviceNotOpen  Camera was not opened before the current command
    * \retval ::VmbErrorBadHandle      The given frame is not valid
    * \retval ::VmbErrorBadParameter   "pFrame" is null.
    * \retval ::VmbErrorStructSize     The given struct size is not valid for this version of the API
    * \retval ::VmbErrorInvalidCall    StopContinuousImageAcquisition is currently running in another thread
    */
    IMEXPORT virtual VmbErrorType QueueFrame(const FramePtr& pFrame) override;


    /**
    * \brief     Flushes the capture queue.
    *
    * All currently queued frames will be returned to the user, leaving no frames in the input queue.
    * After this call, no frame notification will occur until frames are queued again.
    *
    * \returns VmbErrorType
    * 
    * \retval ::VmbErrorSuccess        If no error
    * \retval ::VmbErrorDeviceNotOpen  Camera was not opened before the current command
    * \retval ::VmbErrorBadHandle      The given handle is not valid
    */
    IMEXPORT virtual VmbErrorType FlushQueue() override;


    /**
    * \brief     Prepare the API for incoming frames from this camera.
    *
    * \returns ::VmbErrorType
    * \retval ::VmbErrorSuccess        If no error
    * \retval ::VmbErrorBadHandle      The given handle is not valid
    * \retval ::VmbErrorDeviceNotOpen  Camera was not opened for usage
    * \retval ::VmbErrorInvalidAccess  Operation is invalid with the current access mode
    */
    IMEXPORT virtual VmbErrorType StartCapture() noexcept override;


    /**
    * \brief     Stop the API from being able to receive frames from this camera.
    *
    * Consequences of VmbCaptureEnd():
    *    - The frame queue is flushed
    *    - The frame callback will not be called any more
    *
    * \returns VmbErrorType
    * \retval ::VmbErrorSuccess       If no error
    * \retval ::VmbErrorDeviceNotOpen Camera was not opened before the current command
    * \retval ::VmbErrorBadHandle     The given handle is not valid
    */
    IMEXPORT virtual VmbErrorType EndCapture() noexcept override;

    /**
    * \brief    Retrieve the necessary buffer alignment size in bytes (equals 1 if Data Stream has no such restriction)
    *
    * \returns ::VmbErrorType
    *
    * \retval ::VmbErrorSuccess         If no error
    * \retval ::VmbErrorApiNotStarted   VmbStartup() was not called before the current command
    * \retval ::VmbErrorDeviceNotOpen  Camera was not opened before the current command
    */
    IMEXPORT virtual VmbErrorType GetStreamBufferAlignment(VmbUint32_t& nBufferAlignment) override;
    
private:
    
    struct Impl;
    UniquePointer<Impl> m_pImpl;
};

    
} // namespace VmbCPP

#endif