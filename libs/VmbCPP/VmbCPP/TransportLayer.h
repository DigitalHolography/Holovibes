/*=============================================================================
  Copyright (C) 2022 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        TransportLayer.h

  Description: Definition of class VmbCPP::TransportLayer.

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

#ifndef VMBCPP_TRANSPORTLAYER_H
#define VMBCPP_TRANSPORTLAYER_H

/**
* \file      TransportLayer.h
*
* \brief     Definition of class VmbCPP::TransportLayer.
*/

#include <functional>
#include <vector>

#include <VmbC/VmbC.h>

#include "Camera.h"
#include "PersistableFeatureContainer.h"
#include "UniquePointer.hpp"
#include "VmbCPPCommon.h"

namespace VmbCPP {

/**
 * \brief An alias for a vector of shared pointers to Camera.
 */
using CameraPtrVector = std::vector<CameraPtr>;

/**
 * \brief An alias for a vector of shared pointers to Interface.
 */
using InterfacePtrVector = std::vector<InterfacePtr>;

/**
 * \brief A class providing access to GenTL system module specific functionality and information.
 */
class TransportLayer : public PersistableFeatureContainer
{
public:

    /**
    * \brief Type for an std::function to retrieve a Transport Layer's interfaces
    */
    typedef std::function<VmbErrorType(const TransportLayer* pTransportLayer, InterfacePtr* pInterfaces, VmbUint32_t& size)> GetInterfacesByTLFunction;

    /**
    * \brief Type for an std::function to retrieve a Transport Layer's cameras
    */
    typedef std::function<VmbErrorType(const TransportLayer* pTransportLayer, CameraPtr* pCameras, VmbUint32_t& size)> GetCamerasByTLFunction;
    
    /**
    * \brief Transport Layer constructor.
    * 
    * \param[out]   transportLayerInfo      The transport layer info struct
    * \param[in]    getInterfacesByTL       The function used to find interfaces
    * \param[in]    getCamerasByTL          The function used to find transport layers
    * 
    * \exception std::bad_alloc   not enough memory is available to store the data for the object constructed
    */
    TransportLayer(const VmbTransportLayerInfo_t& transportLayerInfo, GetInterfacesByTLFunction getInterfacesByTL, GetCamerasByTLFunction getCamerasByTL);

    /**
     * \brief the class is non-default-constructible
     */
    TransportLayer() = delete;

    /**
     * \brief the class is non-copyable
     */
    TransportLayer(const TransportLayer&) = delete;

    /**
      * \brief the class is non-copyable
      */
    TransportLayer& operator=(const TransportLayer&) = delete;

    virtual ~TransportLayer() noexcept;

    /**
    * \brief     Get all interfaces related to this transport layer.
    *
    * \param[out]  interfaces         Returned list of related interfaces
    *
    * \returns ::VmbErrorType
    * \retval ::VmbErrorSuccess        If no error
    * \retval ::VmbErrorBadHandle      The handle is not valid
    * \retval ::VmbErrorResources      Resources not available (e.g. memory)
    * \retval ::VmbErrorInternalFault  An internal fault occurred
    */
    VmbErrorType GetInterfaces(InterfacePtrVector& interfaces);

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
    
    /**
    * \brief     Gets the transport layer ID.
    *
    * \note    This information remains static throughout the object's lifetime
    *
    * \param[out]   transportLayerID          The ID of the transport layer
    *
    * \returns
    * \retval ::VmbErrorSuccess        If no error
    */
    VmbErrorType GetID(std::string& transportLayerID) const noexcept;

    /**
    * \brief     Gets the transport layer name.
    *
    * \note    This information remains static throughout the object's lifetime
    * 
    * \param[out]   name          The name of the transport layer
    *
    * \returns
    * \retval ::VmbErrorSuccess        If no error
    */
    VmbErrorType GetName(std::string& name) const noexcept;

   /**
   * \brief     Gets the model name of the transport layer.
   *
   * \note    This information remains static throughout the object's lifetime
   * 
   * \param[out]   modelName        The model name of the transport layer
   *
   * \returns
   * \retval ::VmbErrorSuccess        If no error
   */
    VmbErrorType GetModelName(std::string& modelName) const noexcept;

   /**
   * \brief     Gets the vendor of the transport layer.
   *
   * \note    This information remains static throughout the object's lifetime
   *
   * \param[out]   vendor           The vendor of the transport layer
   *
   * \returns
   * \retval ::VmbErrorSuccess        If no error
   */
    VmbErrorType GetVendor(std::string& vendor) const noexcept;

   /**
   * \brief     Gets the version of the transport layer.
   *
   * \note    This information remains static throughout the object's lifetime
   *
   * \param[out]   version          The version of the transport layer
   *
   * \returns
   * \retval ::VmbErrorSuccess        If no error
   */
    VmbErrorType GetVersion(std::string& version) const noexcept;

   /**
   * \brief     Gets the full path of the transport layer.
   *
   * \note    This information remains static throughout the object's lifetime
   *
   * \param[out]   path             The full path of the transport layer
   *
   * \returns
   * \retval ::VmbErrorSuccess        If no error
   */
    VmbErrorType GetPath(std::string& path) const noexcept;

    /**
    * \brief     Gets the type, e.g. GigE or USB of the transport layer.
    * 
    * \note    This information remains static throughout the object's lifetime
    *
    * \param[out]   type        The type of the transport layer
    *
    * \returns ::VmbErrorType
    * \retval ::VmbErrorSuccess        If no error
    */
    IMEXPORT VmbErrorType GetType(VmbTransportLayerType& type) const noexcept;
  private:

    struct Impl;
    UniquePointer<Impl> m_pImpl;
   
    // Array functions to pass data across DLL boundaries
    IMEXPORT VmbErrorType GetID(char* const pID, VmbUint32_t& length) const noexcept;
    IMEXPORT VmbErrorType GetName(char* const name, VmbUint32_t& length) const noexcept;
    IMEXPORT VmbErrorType GetModelName(char* const modelName, VmbUint32_t& length) const noexcept;
    IMEXPORT VmbErrorType GetVendor(char* const vendor, VmbUint32_t& length) const noexcept;
    IMEXPORT VmbErrorType GetVersion(char* const version, VmbUint32_t& length) const noexcept;
    IMEXPORT VmbErrorType GetPath(char* const path, VmbUint32_t& length) const noexcept;
    IMEXPORT VmbErrorType GetInterfaces(InterfacePtr* pInterfaces, VmbUint32_t& size);
    IMEXPORT VmbErrorType GetCameras(CameraPtr* pCameras, VmbUint32_t& size);
};

} // namespace VmbCPP

#include "TransportLayer.hpp"

#endif
