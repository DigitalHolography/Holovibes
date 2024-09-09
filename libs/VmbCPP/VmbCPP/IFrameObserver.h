/*=============================================================================
  Copyright (C) 2012 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        IFrameObserver.h

  Description: Definition of interface VmbCPP::IFrameObserver.

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

#ifndef VMBCPP_IFRAMEOBSERVER_H
#define VMBCPP_IFRAMEOBSERVER_H

/**
*
* \file  IFrameObserver.h
*
* \brief Definition of interface VmbCPP::IFrameObserver.
*/

#include <VmbCPP/VmbCPPCommon.h>
#include <VmbCPP/SharedPointerDefines.h>
#include <VmbCPP/Frame.h>


namespace VmbCPP {

/**
 * \brief The base class for observers listening for acquired frames.
 * 
 * A derived class must implement the FrameReceived function.
 */
class IFrameObserver 
{
public:
    /**
    * \brief     The event handler function that gets called whenever
    *            a new frame is received
    *
    * \param[in]     pFrame                  The frame that was received
    */
    IMEXPORT virtual void FrameReceived( const FramePtr pFrame ) = 0;

    /**
    * \brief     Destroys an instance of class IFrameObserver
    */
    IMEXPORT virtual ~IFrameObserver() {}

    /**
     * \brief frame observers are not intended to be default constructed
     */
    IFrameObserver() = delete;

protected:
    /**
     * \brief A pointer storing the camera pointer passed in teh constructor.
     */
    CameraPtr m_pCamera;

    /**
     * \brief A pointer to the stream pointer passed in the constructor.
     *
     * If IFrameObserver(CameraPtr) is used, this is the first stream of the camera, should it exist.
     *
     */
    StreamPtr m_pStream;

    /**
     * \brief Creates an observer initializing both m_pCamera and m_pStream
     * 
     * \param[in] pCamera   the camera pointer to store in m_pCamera
     * \param[in] pStream   the stream pointer to store in m_pStream
     */
    IMEXPORT IFrameObserver(CameraPtr pCamera, StreamPtr pStream);

    /**
     * \brief Creates an observer initializing m_pStream with the first stream of the camera provided.
     * 
     * \param[in] pCamera   the camera pointer to store in m_pCamera
     */
    IMEXPORT IFrameObserver(CameraPtr pCamera);

    /**
     * \brief copy constructor for use by a derived class
     */
    IMEXPORT IFrameObserver( const IFrameObserver& other)
        : m_pCamera(other.m_pCamera),
        m_pStream(other.m_pStream)
    {
    }

    /**
     * \brief copy assignment operator for use by a derived class
     */
    IMEXPORT IFrameObserver& operator=( IFrameObserver const& other)
    {
        m_pCamera = other.m_pCamera;
        m_pStream = other.m_pStream;
        return *this;
    }
};

} // namespace VmbCPP

#endif
