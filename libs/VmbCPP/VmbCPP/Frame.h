/*=============================================================================
  Copyright (C) 2012-2022 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        Frame.h

  Description: Definition of class VmbCPP::Frame.

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

#ifndef VMBCPP_FRAME_H
#define VMBCPP_FRAME_H

/**
* \file  Frame.h
*
* \brief Definition of class VmbCPP::Frame.
*/

#include <functional>
#include <vector>

#include <VmbC/VmbCommonTypes.h>

#include "FeatureContainer.h"
#include "IFrameObserver.h"
#include "SharedPointerDefines.h"
#include "UniquePointer.hpp"
#include "VmbCPPCommon.h"

struct VmbFrame;

namespace VmbCPP {

/**
 * \brief UniquePointer to a FeatureContainer for Chunk access
 */
using ChunkFeatureContainerPtr = UniquePointer<FeatureContainer>;

// forward declarations for befriending
class Camera;
class Stream;

/**
 * \brief An object representing a data buffer that can be filled by acquiring data from a camera. 
 */
class Frame 
{
  friend class Stream;
  friend class Camera;

  public:

    /**
    * \brief Type for an std::function for accessing ChunkData via a FeatureContainer
    */
    typedef std::function<VmbErrorType(ChunkFeatureContainerPtr&)> ChunkDataAccessFunction;

    /** 
    * 
    * \brief     Creates an instance of class Frame of a certain size and memory alignment
    * 
    * \param[in ]   bufferSize          The size of the underlying buffer
    * \param[in ]   allocationMode      Indicates if announce frame or alloc and announce frame is used
    * \param[in ]   bufferAlignment     The alignment that needs to be satisfied for the frame buffer allocation
    */ 
    IMEXPORT explicit Frame( VmbInt64_t bufferSize, FrameAllocationMode allocationMode = FrameAllocation_AnnounceFrame, VmbUint32_t bufferAlignment = 1);

    /** 
    * 
    * \brief     Creates an instance of class Frame with the given user buffer of the given size
    * 
    * \param[in ]   pBuffer     A pointer to an allocated buffer
    * \param[in ]   bufferSize  The size of the underlying buffer
    */ 
    IMEXPORT Frame( VmbUchar_t *pBuffer, VmbInt64_t bufferSize );

    /** 
    * 
    * \brief     Destroys an instance of class Frame
    */ 
    IMEXPORT ~Frame();

    /** 
    * 
    * \brief     Registers an observer that will be called whenever a new frame arrives.
    *            As new frames arrive, the observer's FrameReceived method will be called.
    *            Only one observer can be registered.
    * 
    * \param[in ]   pObserver   An object that implements the IObserver interface
    * 
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess       If no error
    * \retval ::VmbErrorBadParameter  \p pObserver is null.
    * \retval ::VmbErrorResources     The observer was in use
    */ 
    IMEXPORT VmbErrorType RegisterObserver( const IFrameObserverPtr &pObserver );

    /** 
    * 
    * \brief     Unregisters the observer that was called whenever a new frame arrived
    */ 
    IMEXPORT VmbErrorType UnregisterObserver();

    /** 
    * 
    * \brief     Returns the complete buffer including image and chunk data
    * 
    * \param[out]   pBuffer        A pointer to the buffer
    * 
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess       If no error
    */ 
    IMEXPORT VmbErrorType GetBuffer( VmbUchar_t* &pBuffer );

    /** 
    * 
    * \brief     Returns the complete buffer including image and chunk data
    * 
    * \param[out]   pBuffer  A pointer to the buffer
    * 
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess       If no error
    */ 
    IMEXPORT VmbErrorType GetBuffer( const VmbUchar_t* &pBuffer ) const;

    /** 
    * 
    * \brief     Returns only the image data
    * 
    * \param[out]   pBuffer     A pointer to the buffer
    * 
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess       If no error
    */ 
    IMEXPORT VmbErrorType GetImage( VmbUchar_t* &pBuffer );

    /** 
    * \brief     Returns the pointer to the first byte of the image data
    * 
    * \param[out]   pBuffer    A pointer to the buffer
    * 
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess       If no error
    */ 
    IMEXPORT VmbErrorType GetImage( const VmbUchar_t* &pBuffer ) const;

    /**
    * \brief     Returns the receive status of a frame
    * 
    * \param[out]   status    The receive status
    * 
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess       If no error
    */ 
    IMEXPORT VmbErrorType GetReceiveStatus( VmbFrameStatusType &status ) const;

    /**
    * \brief     Returns the payload type of a frame
    *
    * \param[out]   payloadType    The payload type
    *
    * \returns ::VmbErrorType
    *
    * \retval ::VmbErrorSuccess       If no error
    */
    IMEXPORT VmbErrorType GetPayloadType(VmbPayloadType& payloadType) const;

    /**
    * \brief     Returns the memory size of the frame buffer holding
    * 
    * both the image data and the chunk data
    * 
    * \param[out]   bufferSize      The size in bytes
    * 
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess       If no error
    */ 
    IMEXPORT VmbErrorType GetBufferSize( VmbUint32_t &bufferSize ) const;

    /**
    * \brief     Returns the GenICam pixel format
    * 
    * \param[out]   pixelFormat    The GenICam pixel format
    * 
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess       If no error
    */ 
    IMEXPORT VmbErrorType GetPixelFormat( VmbPixelFormatType &pixelFormat ) const;

    /**
    * \brief     Returns the width of the image
    * 
    * \param[out]   width       The width in pixels
    * 
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess       If no error
    */ 
    IMEXPORT VmbErrorType GetWidth( VmbUint32_t &width ) const;

    /**
    * \brief     Returns the height of the image
    * 
    * \param[out]   height       The height in pixels
    * 
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess       If no error
    */ 
    IMEXPORT VmbErrorType GetHeight( VmbUint32_t &height ) const;

    /** 
    * \brief     Returns the X offset of the image
    * 
    * \param[out]   offsetX     The X offset in pixels
    * 
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess       If no error
    */ 
    IMEXPORT VmbErrorType GetOffsetX( VmbUint32_t &offsetX ) const;

    /** 
    * \brief     Returns the Y offset of the image
    * 
    * \param[out]   offsetY     The Y offset in pixels
    * 
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess       If no error
    */ 
    IMEXPORT VmbErrorType GetOffsetY( VmbUint32_t &offsetY ) const;

    /** 
    * \brief     Returns the frame ID
    * 
    * \param[out]   frameID    The frame ID
    * 
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess       If no error
    */ 
    IMEXPORT VmbErrorType GetFrameID( VmbUint64_t &frameID ) const;

    /** 
    * \brief     Returns the timestamp
    * 
    * \param[out]   timestamp  The timestamp
    * 
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess       If no error
    */ 
    IMEXPORT VmbErrorType GetTimestamp( VmbUint64_t &timestamp ) const;

    /**
    * \brief     Returns true if the frame contains Chunk Data
    *
    * \param[out]   containsChunkData  true if the frame contains Chunk Data
    *
    * \returns ::VmbErrorType
    *
    * \retval ::VmbErrorSuccess       If no error
    * \retval ::VmbErrorNotAvailable  If the underlying Transport Layer does not provide this information
    */
    IMEXPORT VmbErrorType ContainsChunkData(VmbBool_t& containsChunkData) const;

    /**
    * \brief     Access the frame's chunk data via a FeatureContainerPtr.
    * 
    * \note Chunk data can be accessed only in the scope of the given ChunkDataAccessFunction.
    *
    * \param[in]   chunkAccessFunction  Callback for Chunk data access
    *
    * \returns ::VmbErrorType
    *
    * \retval ::VmbErrorSuccess       If no error
    */
    VmbErrorType AccessChunkData(ChunkDataAccessFunction chunkAccessFunction);

    /**
     * \brief Getter for the frame observer.
     * 
     * \param[out] observer     the frame observer pointer to write the retrieved observer to.
     * 
     * \return True, if there was a non-null observer to retrieve, false otherwise.
     */
    bool GetObserver( IFrameObserverPtr &observer ) const;

    ///  No default ctor
    Frame() = delete;
    ///  No copy ctor
    Frame(Frame&) = delete;
    ///  No assignment operator
    Frame& operator=(const Frame&) = delete;

  private:
    struct Impl;
    UniquePointer<Impl> m_pImpl;

    static VmbError_t InternalChunkDataAccessCallback(VmbHandle_t featureAccessHandle, void* userContext);
    IMEXPORT VmbErrorType GetFrameStruct(VmbFrame*& frame);
    IMEXPORT VmbErrorType ChunkDataAccess(const VmbFrame* frame, VmbChunkAccessCallback chunkAccessCallback, void* userContext);
};

} // namespace VmbCPP

#include <VmbCPP/Frame.hpp>

#endif
