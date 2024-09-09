/*=============================================================================
  Copyright (C) 2012-2022 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        ICameraListObserver.h

  Description: Definition of interface VmbCPP::ICameraListObserver.

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

#ifndef VMBCPP_ICAMERALISTOBSERVER_H
#define VMBCPP_ICAMERALISTOBSERVER_H

/**
* \file  ICameraListObserver.h
*
* \brief Definition of interface VmbCPP::ICameraListObserver.
*/

#include <VmbCPP/VmbCPPCommon.h>
#include <VmbCPP/SharedPointerDefines.h>
#include <VmbCPP/Camera.h>
#include <vector>



namespace VmbCPP {

/**
 * \brief A base class for listeners observing the list of available cameras. 
 */
class ICameraListObserver 
{
public:
    /**  
    * \brief     The event handler function that gets called whenever
    *            an ICameraListObserver is triggered. This occurs most
    *            likely when a camera was plugged in or out.
    * 
    * \param[out]    pCam                    The camera that triggered the event
    * \param[out]    reason                  The reason why the callback routine was triggered
    *                                                       (e.g., a new camera was plugged in)
    */ 
    IMEXPORT virtual void CameraListChanged( CameraPtr pCam, UpdateTriggerType reason ) = 0;

    /** 
    * \brief     Destroys an instance of class ICameraListObserver
    */ 
    IMEXPORT virtual ~ICameraListObserver() {}

protected:
    /**
     * \brief default constructor for use by derived classes. 
     */
    IMEXPORT ICameraListObserver() { }

    /**
     * \brief Copy constructor for use by derived classes. 
     */
    IMEXPORT ICameraListObserver( const ICameraListObserver&)
    {
    }

    /**
     * \brief Copy assignment operator for use by derived classes. 
     */
    IMEXPORT ICameraListObserver& operator=( const ICameraListObserver&)
    {
        return *this;
    }
};

} // namespace VmbCPP

#endif
