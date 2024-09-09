/*=============================================================================
  Copyright (C) 2012 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------
 
  File:        VmbSystem.h

  Description: Definition of class VmbCPP::VmbSystem.

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

#ifndef VMBCPP_SYSTEM_H
#define VMBCPP_SYSTEM_H

/**
* \file  VmbSystem.h
*
* \brief Definition of class VmbCPP::VmbSystem.
*/

#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <VmbC/VmbC.h>

#include "Camera.h"
#include "ICameraFactory.h"
#include "ICameraListObserver.h"
#include "IInterfaceListObserver.h"
#include "Interface.h"
#include "LoggerDefines.h"
#include "SharedPointerDefines.h"
#include "StringLike.hpp"
#include "TransportLayer.h"
#include "UniquePointer.hpp"
#include "VmbCPPCommon.h"


namespace VmbCPP {

/**
 * \brief An alias for a vector of shared pointers to transport layers.
 */
using TransportLayerPtrVector = std::vector<TransportLayerPtr>;

/**
 * \brief A class providing access to functionality and information about the Vmb API itself.
 * 
 * A singleton object is provided by the GetInstance function.
 * 
 * Access to any information other than the version of the VmbCPP API can only be accessed
 * after calling Startup and before calling Shutdown.
 * 
 * If a custom camera factory is used, this must be set before calling Startup.
 */
class VmbSystem : public FeatureContainer
{
public:
    /**
     * \brief the class is not copyable 
     */
    VmbSystem(const VmbSystem&) = delete;

    /**
     * \brief the class is not copyable 
     */
    VmbSystem& operator=(const VmbSystem& system) = delete;

    /**
    * \brief     Returns a reference to the System singleton.
    * 
    * \retval VmbSystem&
    */
    IMEXPORT static VmbSystem& GetInstance() noexcept;

    /**
    * \brief   Retrieve the version number of VmbCPP.
    * 
    * This function can be called at any time, even before the API is
    * initialized. All other version numbers can be queried via feature access
    * 
    * \param[out]  version      Reference to the struct where version information is copied
    *
    * \returns ::VmbErrorType
    * \retval ::VmbErrorSuccess        always returned
    */
    IMEXPORT VmbErrorType QueryVersion( VmbVersionInfo_t &version ) const noexcept;
    
    /**
     * \brief Initialize the VmbCPP module.
     * 
     * On successful return, the API is initialized; this is a necessary call.
     * This method must be called before any other VmbCPP function is run.
     * 
     * \param[in] pathConfiguration     A string containing the semicolon separated list of paths. The paths contain directories to search for .cti files,
     *                                  paths to .cti files and optionally the path to a configuration xml file. If null is passed the parameter is considered to contain the values
     *                                  from the GENICAM_GENTLXX_PATH environment variable
     * \returns
     * \retval ::VmbErrorSuccess        If no error
     * \retval ::VmbErrorInternalFault  An internal fault occurred
     */
    IMEXPORT VmbErrorType Startup(const VmbFilePathChar_t* pathConfiguration);

    /**
     * \brief Initialize the VmbCPP module (overload, without starting parameter pathConfiguration)
     *
     * On successful return, the API is initialized; this is a necessary call.
     * This method must be called before any other VmbCPP function is run.
     *
     * \returns
     * \retval ::VmbErrorSuccess        If no error
     * \retval ::VmbErrorInternalFault  An internal fault occurred
     */
    IMEXPORT VmbErrorType Startup();

    /**
    * \brief   Perform a shutdown of the API module.
    *          This will free some resources and deallocate all physical resources if applicable.
    *
    * \returns ::VmbErrorType
    * \retval ::VmbErrorSuccess        always returned
    */
    IMEXPORT VmbErrorType Shutdown();

    /**
    * \brief   List all the interfaces currently visible to VmbCPP.
    * 
    * All the interfaces known via a GenTL are listed by this command and filled into the vector provided.
    * If the vector is not empty, new elements will be appended.
    * Interfaces can be adapter cards or frame grabber cards, for instance.
    * 
    * \param[out]  interfaces       Vector of shared pointer to Interface object
    *
    * \returns ::VmbErrorType
    * \retval ::VmbErrorSuccess        If no error
    * \retval ::VmbErrorApiNotStarted  VmbStartup() was not called before the current command
    * \retval ::VmbErrorStructSize     The given struct size is not valid for this API version
    * \retval ::VmbErrorMoreData       More data were returned than space was provided
    * \retval ::VmbErrorInternalFault  An internal fault occurred
    */
    VmbErrorType GetInterfaces( InterfacePtrVector &interfaces );

    /**
    * \brief   Gets a specific interface identified by an ID.
    *
    * An interface known via a GenTL is listed by this command and filled into the pointer provided.
    * Interface can be an adapter card or a frame grabber card, for instance.
    * 
    * \param[in ]  pID                 The ID of the interface to get (returned by GetInterfaces())
    * \param[out]  pInterface          Shared pointer to Interface object
    *
    * \returns ::VmbErrorType
    * \retval ::VmbErrorSuccess           If no error
    * \retval ::VmbErrorApiNotStarted     VmbStartup() was not called before the current command
    * \retval ::VmbErrorBadParameter      \p pID is null.
    * \retval ::VmbErrorStructSize        The given struct size is not valid for this API version
    * \retval ::VmbErrorMoreData          More data were returned than space was provided
    */
    IMEXPORT VmbErrorType GetInterfaceByID( const char *pID, InterfacePtr &pInterface );

    /**
     * \brief null is not allowed as interface id parameter 
     */
    VmbErrorType GetInterfaceByID(std::nullptr_t, InterfacePtr&) = delete;

    /**
     * \brief Convenience function for calling GetInterfaceByID(char const*, InterfacePtr&)
     *        with \p id converted to `const char*`.
     *
     * This is a convenience function for calling `GetInterfaceById(CStringLikeTraits<T>::%ToString(id), pInterface)`.
     * 
     * Types other than std::string may be passed to this function, if CStringLikeTraits is specialized
     * before including this header.
     * 
     * \tparam T a type that is considered to be a c string like
     */
    template<class T>
    typename std::enable_if<CStringLikeTraits<T>::IsCStringLike, VmbErrorType>::type GetInterfaceByID(const T& id, InterfacePtr& pInterface);

    /**
    * \brief   Retrieve a list of all cameras.
    * 
    * \param[out]  cameras            Vector of shared pointer to Camera object
    *                                 A camera known via a GenTL is listed by this command and filled into the pointer provided.
    *
    * \returns ::VmbErrorType
    * \retval ::VmbErrorSuccess        If no error
    * \retval ::VmbErrorApiNotStarted  VmbStartup() was not called before the current command
    * \retval ::VmbErrorStructSize     The given struct size is not valid for this API version
    * \retval ::VmbErrorMoreData       More data were returned than space was provided
    */
    VmbErrorType GetCameras( CameraPtrVector &cameras );

    /**
    * \brief   Gets a specific camera identified by an ID. The returned camera is still closed.
    *
    * A camera known via a GenTL is listed by this command and filled into the pointer provided.
    * Only static properties of the camera can be fetched until the camera has been opened.
    * "pID" can be one of the following:
    *  - "169.254.12.13" for an IP address,
    *  - "000F314C4BE5" for a MAC address or
    *  - "DEV_1234567890" for an ID as reported by VmbCPP
    *
    * \param[in ]  pID                 The ID of the camera to get
    * \param[out]  pCamera             Shared pointer to camera object
    *
    * \returns ::VmbErrorType
    * \retval ::VmbErrorSuccess           If no error
    * \retval ::VmbErrorApiNotStarted     VmbStartup() was not called before the current command
    * \retval ::VmbErrorBadParameter      \p pID is null.
    * \retval ::VmbErrorStructSize        The given struct size is not valid for this API version
    * \retval ::VmbErrorMoreData          More data were returned than space was provided
    */
    IMEXPORT VmbErrorType GetCameraByID( const char *pID, CameraPtr &pCamera );

    /**
     * \brief It's not possible to identify a camera given null as id.
     */
    VmbErrorType GetCameraByID(std::nullptr_t, CameraPtr&) = delete;

    /**
     * \brief Convenience function for calling GetCameraByID(char const*, CameraPtr&)
     *        with \p id converted to `const char*`.
     *
     * This is a convenience function for calling `GetCameraByID(CStringLikeTraits<T>::%ToString(id), pCamera)`.
     *
     * Types other than std::string may be passed to this function, if CStringLikeTraits is specialized
     * before including this header.
     *
     * \tparam IdType a type that is considered to be a c string like
     */
    template<class IdType>
    typename std::enable_if<CStringLikeTraits<IdType>::IsCStringLike, VmbErrorType>::type
    GetCameraByID(
        const IdType& id,
        VmbAccessModeType eAccessMode,
        CameraPtr& pCamera);
    
    /**
    * \brief     Gets a specific camera identified by an ID. The returned camera is already open.
    *
    * A camera can be opened if camera-specific control is required, such as I/O pins
    * on a frame grabber card. Control is then possible via feature access methods.
    * "pID" can be one of the following: 
    *  - "169.254.12.13" for an IP address,
    *  - "000F314C4BE5" for a MAC address or 
    *  - "DEV_1234567890" for an ID as reported by VmbCPP
    * 
    * \param[in ]   pID                 The unique ID of the camera to get
    * \param[in ]   eAccessMode         The requested access mode
    * \param[out]   pCamera             A shared pointer to the camera
    *
    * \returns ::VmbErrorType
    * \retval ::VmbErrorSuccess           If no error
    * \retval ::VmbErrorApiNotStarted     VmbStartup() was not called before the current command
    * \retval ::VmbErrorNotFound          The designated interface cannot be found
    * \retval ::VmbErrorBadParameter      \p pID is null.
    */
    IMEXPORT VmbErrorType OpenCameraByID( const char *pID, VmbAccessModeType eAccessMode, CameraPtr &pCamera );

    /**
     * \brief It's not possible to identify a camera given null as id.
     */
    VmbErrorType OpenCameraByID(std::nullptr_t, VmbAccessModeType, CameraPtr& ) = delete;


    /**
     * \brief Convenience function for calling OpenCameraByID(char const*, VmbAccessModeType, CameraPtr&)
     *        with \p id converted to `const char*`.
     *
     * This is a convenience function for calling
     * `OpenCameraByID(CStringLikeTraits<T>::%ToString(id), eAccessMode, pCamera)`.
     *
     * Types other than std::string may be passed to this function, if CStringLikeTraits is specialized
     * before including this header.
     *
     * \tparam IdType a type that is considered to be a c string like
     */
    template<class IdType>
    typename std::enable_if<CStringLikeTraits<IdType>::IsCStringLike, VmbErrorType>::type
        OpenCameraByID(
            const IdType& id,
            VmbAccessModeType eAccessMode,
            CameraPtr& pCamera);

    /**
    * \brief     Registers an instance of camera observer whose CameraListChanged() method gets called
    *            as soon as a camera is plugged in, plugged out, or changes its access status
    * 
    * \param[in ]       pObserver   A shared pointer to an object derived from ICameraListObserver
    *
    * \returns ::VmbErrorType
    * \retval ::VmbErrorSuccess       If no error
    * \retval ::VmbErrorBadParameter  \p pObserver is null.
    * \retval ::VmbErrorInvalidCall   If the very same observer is already registered
    */
    IMEXPORT VmbErrorType RegisterCameraListObserver( const ICameraListObserverPtr &pObserver );

    /**
    * \brief     Unregisters a camera observer
    * 
    * \param[in ]       pObserver   A shared pointer to an object derived from ICameraListObserver
    *
    * \returns ::VmbErrorType
    * \retval ::VmbErrorSuccess       If no error
    * \retval ::VmbErrorNotFound      If the observer is not registered
    * \retval ::VmbErrorBadParameter  \p pObserver is null.
    */
    IMEXPORT VmbErrorType UnregisterCameraListObserver( const ICameraListObserverPtr &pObserver );

    /**
    * \brief     Registers an instance of interface observer whose InterfaceListChanged() method gets called
    *            as soon as an interface is plugged in, plugged out, or changes its access status
    * 
    * \param[in ]       pObserver   A shared pointer to an object derived from IInterfaceListObserver
    *
    * \returns ::VmbErrorType
    * \retval ::VmbErrorSuccess       If no error
    * \retval ::VmbErrorBadParameter  \p pObserver is null.
    * \retval ::VmbErrorInvalidCall   If the very same observer is already registered
    */
    IMEXPORT VmbErrorType RegisterInterfaceListObserver( const IInterfaceListObserverPtr &pObserver );

    /**
    * \brief     Unregisters an interface observer
    * 
    * \param[in ]       pObserver   A shared pointer to an object derived from IInterfaceListObserver
    *
    * \returns ::VmbErrorType
    * \retval ::VmbErrorSuccess       If no error
    * \retval ::VmbErrorNotFound      If the observer is not registered
    * \retval ::VmbErrorBadParameter  \p pObserver is null.
    */
    IMEXPORT VmbErrorType UnregisterInterfaceListObserver( const IInterfaceListObserverPtr &pObserver );

    /**
    * \brief     Registers an instance of camera factory. When a custom camera factory is registered, all instances of type camera
    *            will be set up accordingly.
    * 
    * \param[in ]   pCameraFactory  A shared pointer to an object derived from ICameraFactory
    *
    * \returns ::VmbErrorType
    * \retval ::VmbErrorSuccess       If no error
    * \retval ::VmbErrorBadParameter  \p pCameraFactory is null.
    */
    IMEXPORT VmbErrorType RegisterCameraFactory( const ICameraFactoryPtr &pCameraFactory );

    /**
    * \brief     Unregisters the camera factory. After unregistering the default camera class is used.
    *
    * \returns ::VmbErrorType
    * \retval ::VmbErrorSuccess       If no error
    */
    IMEXPORT VmbErrorType UnregisterCameraFactory();

    
    /**
    * \brief        Retrieve a list of all transport layers.
    *
    * \param[out]  transportLayers      Vector of shared pointer to TransportLayer object
    *
    * \details     All transport layers known via GenTL are listed by this command and filled into the pointer provided.
    * 
    * \returns ::VmbErrorType
    * \retval ::VmbErrorSuccess        If no error
    * \retval ::VmbErrorApiNotStarted  VmbStartup() was not called before the current command
    * \retval ::VmbErrorStructSize     The given struct size is not valid for this API version
    * \retval ::VmbErrorMoreData       More data were returned than space was provided
    *
    */
    VmbErrorType GetTransportLayers( TransportLayerPtrVector &transportLayers );
    
    /**
    * \brief       Gets a specific transport layer identified by an ID.
    * 
    * \details     An interface known via a GenTL is listed by this command and filled into the pointer provided.
    *              Interface can be an adapter card or a frame grabber card, for instance.
    *
    * \param[in ]  pID                     The ID of the interface to get (returned by GetInterfaces())
    * \param[out]  pTransportLayer         Shared pointer to Transport Layer object
    * 
    * \returns ::VmbErrorType
    * \retval ::VmbErrorSuccess           If no error
    * \retval ::VmbErrorApiNotStarted     VmbStartup() was not called before the current command
    * \retval ::VmbErrorBadParameter      \p pID is null.
    * \retval ::VmbErrorStructSize        The given struct size is not valid for this API version
    * \retval ::VmbErrorMoreData          More data were returned than space was provided
    *
    */
    IMEXPORT VmbErrorType GetTransportLayerByID(const char* pID, TransportLayerPtr& pTransportLayer);

    /**
     * \brief Convenience function for calling GetTransportLayerByID(char const*, TransportLayerPtr&)
     *        with \p id converted to `const char*`.
     *
     * This is a convenience function for calling
     * `GetTransportLayerByID(CStringLikeTraits<T>::%ToString(id), pTransportLayer)`.
     *
     * Types other than std::string may be passed to this function, if CStringLikeTraits is specialized
     * before including this header.
     *
     * \tparam T a type that is considered to be a c string like
     */
    template<class T>
    typename std::enable_if<CStringLikeTraits<T>::IsCStringLike, VmbErrorType>::type
        GetTransportLayerByID(T const& id, TransportLayerPtr& pTransportLayer);

    /**
     * \brief the transport layer cannot retrieved given null as id. 
     */
    VmbErrorType GetTransportLayerByID(std::nullptr_t, TransportLayerPtr&) = delete;
    
    /// Mapping of handle to CameraPtr
    CameraPtr GetCameraPtrByHandle( const VmbHandle_t handle ) const;

    /**
     * \brief get the logger for the VmbCPP Api
     *
     * \return A pointer to the logger or null
     */
    Logger* GetLogger() const noexcept;

  private:
    /// Singleton.
    static VmbSystem _instance;
    VmbSystem();
    ~VmbSystem() noexcept;
    
    struct Impl;
    UniquePointer<Impl> m_pImpl;

    IMEXPORT VmbErrorType GetCameras( CameraPtr *pCameras, VmbUint32_t &size );
    IMEXPORT VmbErrorType GetInterfaces( InterfacePtr *pInterfaces, VmbUint32_t &size );
    IMEXPORT VmbErrorType GetTransportLayers( TransportLayerPtr *pTransportLayers, VmbUint32_t &size ) noexcept;
};

} // namespace VmbCPP

#include "VmbSystem.hpp"

#endif
