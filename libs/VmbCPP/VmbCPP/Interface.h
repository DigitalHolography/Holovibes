/*=============================================================================
  Copyright (C) 2012-2022 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        Interface.h

  Description: Definition of class VmbCPP::Interface.

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

#ifndef VMBCPP_INTERFACE_H
#define VMBCPP_INTERFACE_H

/**
* \file      Interface.h
*
*  \brief    Definition of class VmbCPP::Interface.
*/

#include <functional>
#include <vector>

#include <VmbC/VmbC.h>

#include "PersistableFeatureContainer.h"
#include "SharedPointerDefines.h"
#include "UniquePointer.hpp"
#include "VmbCPPCommon.h"

namespace VmbCPP {

using CameraPtrVector = std::vector<CameraPtr>;

/**
 * \brief An object representing the GenTL interface.
 */
class Interface : public PersistableFeatureContainer
{
public:

    /**
     *\brief Object is not default constructible
     */
    Interface() = delete;

    /**
     *\brief Object is not copyable
     */
    Interface(const Interface&) = delete;

    /**
     *\brief Object is not copyable
     */
    Interface& operator=(const Interface&) = delete;

    /**
    * \brief Type for an std::function to retrieve an Interface's cameras
    */
    using GetCamerasByInterfaceFunction = std::function<VmbErrorType(const Interface* pInterface, CameraPtr* pCameras, VmbUint32_t& size)>;

    /**
     * \brief Create an interface given the interface info and info about related objects.
     * 
     * \param[in] interfaceInfo             the information about the interface
     * \param[in] pTransportLayerPtr        the pointer to the transport layer providing this interface
     * \param[in] getCamerasByInterface     the function for retrieving the cameras of this interface
     */
    Interface(const VmbInterfaceInfo_t& interfaceInfo,
              const TransportLayerPtr& pTransportLayerPtr,
              GetCamerasByInterfaceFunction getCamerasByInterface);

    virtual ~Interface();

    /**
    * \brief      Gets the ID of an interface.
    *
    * \param[out]   interfaceID          The ID of the interface
    *
    * \returns ::VmbErrorType
    * \retval ::VmbErrorSuccess        If no error
    *
    * \details    This information remains static throughout the object's lifetime
    */
    VmbErrorType GetID(std::string &interfaceID) const noexcept;

    /**
    * \brief     Gets the type, e.g. GigE or USB of an interface.
    *
    * \param[out]   type        The type of the interface
    *
    * \returns ::VmbErrorType
    * \retval ::VmbErrorSuccess        If no error
    *
    * \details    This information remains static throughout the object's lifetime
    */
    IMEXPORT VmbErrorType GetType(VmbTransportLayerType& type) const noexcept;

    /**
    * \brief     Gets the name of an interface.
    *
    * \param[out]   name        The name of the interface
    *
    * \returns ::VmbErrorType
    * \retval ::VmbErrorSuccess        If no error
    *
    * Details:     This information remains static throughout the object's lifetime
    */
    VmbErrorType GetName(std::string& name) const noexcept;

    /**
    * \brief     Gets the pointer of the related transport layer.
    *
    * \param[out]   pTransportLayer     The pointer of the related transport layer.
    *
    * \returns ::VmbErrorType
    * \retval ::VmbErrorSuccess        If no error
    */
    IMEXPORT VmbErrorType GetTransportLayer(TransportLayerPtr& pTransportLayer) const;

    /**
    * \brief     Get all cameras related to this transport layer.
    *
    * \param[out]  cameras         Returned list of related cameras
    *
    * \returns ::VmbErrorType
    * \retval ::VmbErrorSuccess        If no error
    * \retval ::VmbErrorBadHandle      The handle is not valid
    * \retval ::VmbErrorResources      Resources not available (e.g. memory)
    * \retval ::VmbErrorInternalFault  An internal fault occurred
    */
    VmbErrorType GetCameras(CameraPtrVector& cameras);

  private:

    struct Impl;
    UniquePointer<Impl> m_pImpl;

    // Array functions to pass data across DLL boundaries
    IMEXPORT VmbErrorType GetID(char* const pID, VmbUint32_t& length) const noexcept;
    IMEXPORT VmbErrorType GetName(char* const pName, VmbUint32_t& length) const noexcept;
    IMEXPORT VmbErrorType GetCameras(CameraPtr* pCameras, VmbUint32_t& size);
};

} // namespace VmbCPP

#include "Interface.hpp"

#endif
