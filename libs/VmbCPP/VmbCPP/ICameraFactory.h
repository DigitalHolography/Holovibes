/*=============================================================================
  Copyright (C) 2012-2022 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        ICameraFactory.h

  Description: Definition of interface VmbCPP::ICameraFactory.

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

#ifndef VMBCPP_ICAMERAFACTORY_H
#define VMBCPP_ICAMERAFACTORY_H

/**
* \file  ICameraFactory.h
*
* \brief Definition of interface VmbCPP::ICameraFactory.
*/

#include <VmbC/VmbC.h>
#include <VmbCPP/VmbCPPCommon.h>
#include <VmbCPP/SharedPointerDefines.h>
#include <VmbCPP/Interface.h>
#include <VmbCPP/Camera.h>


namespace VmbCPP {

/**
 * \brief A interface for creating camera objects.
 *
 * A object of an implementing class can be registered at VmbSystem level before startup of the API
 * to customze the type of Camera objects created by the API.
 */
class ICameraFactory 
{
  public:
    /** 
     * \brief     Factory method to create a camera that extends the Camera class
     *
     *            The ID of the camera may be, among others, one of the following: "169.254.12.13",
     *            "000f31000001", a plain serial number: "1234567890", or the device ID 
     *            of the underlying transport layer.
     *
     * \param[in ]    cameraInfo         Reference to the camera info struct
     * \param[in ]    pInterface         The shared pointer to the interface camera is connected to
     */ 
    IMEXPORT virtual CameraPtr CreateCamera(const VmbCameraInfo_t& cameraInfo,
                                            const InterfacePtr& pInterface) = 0;

    /**
    * \brief     Destroys an instance of class Camera
    */ 
    IMEXPORT virtual ~ICameraFactory() {}

};

} // namespace VmbCPP

#endif
