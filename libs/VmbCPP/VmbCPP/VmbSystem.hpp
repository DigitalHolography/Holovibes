#include "VmbSystem.h"
/*=============================================================================
  Copyright (C) 2012 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

  -----------------------------------------------------------------------------

  File:        VmbSystem.hpp

  Description: Inline wrapper functions for class VmbCPP::VmbSystem.

  -----------------------------------------------------------------------------

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

#ifndef VMBCPP_VMBSYSTEM_HPP
#define VMBCPP_VMBSYSTEM_HPP

/**
* \file  VmbSystem.hpp
*
* \brief Inline wrapper functions for class VmbCPP::VmbSystem
*        that allocate memory for STL objects in the application's context
*        and to pass data across DLL boundaries using arrays
*/

namespace VmbCPP {

inline VmbErrorType VmbSystem::GetInterfaces( InterfacePtrVector &rInterfaces )
{
    VmbErrorType    res;
    VmbUint32_t     nSize;

    res = GetInterfaces(nullptr, nSize );
    if ( VmbErrorSuccess == res )
    {
        if( 0 != nSize)
        {
            try
            {
                InterfacePtrVector tmpInterfaces( nSize );
                res = GetInterfaces( &tmpInterfaces[0], nSize );
                if( VmbErrorSuccess == res )
                {
                    rInterfaces.swap( tmpInterfaces);
                }
            }
            catch(...)
            {
                return VmbErrorResources;
            }
        }
        else
        {
            rInterfaces.clear();
        }
    }

    return res;
}

inline VmbErrorType VmbSystem::GetCameras( CameraPtrVector &rCameras )
{
    VmbErrorType    res;
    VmbUint32_t     nSize;

    res = GetCameras(nullptr, nSize );
    if (    VmbErrorSuccess == res)
    {
        if( 0 != nSize)
        {
            try
            {
                CameraPtrVector tmpCameras( nSize );
                res = GetCameras( &tmpCameras[0], nSize );
                if( VmbErrorSuccess == res )
                {
                    if( nSize < tmpCameras.size() )
                    {
                        tmpCameras.resize( nSize);
                    }
                    rCameras.swap( tmpCameras );
                }
            }
            catch(...)
            {
                return VmbErrorResources;
            }
        }
        else
        {
            rCameras.clear();
        }
    }

    return res;
}

inline VmbErrorType VmbSystem::GetTransportLayers( TransportLayerPtrVector &rTransportLayers )
{
    VmbErrorType    res;
    VmbUint32_t     nSize;

    res = GetTransportLayers(nullptr, nSize );
    if ( VmbErrorSuccess == res )
    {
        if( 0 != nSize)
        {
            try
            {
                TransportLayerPtrVector tmpTransportLayers( nSize );
                res = GetTransportLayers( &tmpTransportLayers[0], nSize );
                if( VmbErrorSuccess == res )
                {
                    rTransportLayers.swap( tmpTransportLayers);
                }
            }
            catch(...)
            {
                return VmbErrorResources;
            }
        }
        else
        {
            rTransportLayers.clear();
        }
    }

    return res;
}

template<class T>
inline typename std::enable_if<CStringLikeTraits<T>::IsCStringLike, VmbErrorType>::type VmbSystem::GetInterfaceByID(const T& id, InterfacePtr& pInterface)
{
    static_assert(std::is_same<char const*, decltype(CStringLikeTraits<T>::ToString(id))>::value, "CStringLikeTraits<T>::ToString(const T&) does not return char const*");
    return GetInterfaceByID(CStringLikeTraits<T>::ToString(id), pInterface);
}

template<class IdType>
inline
typename std::enable_if<CStringLikeTraits<IdType>::IsCStringLike, VmbErrorType>::type
VmbSystem::OpenCameraByID(
    const IdType& id,
    VmbAccessModeType eAccessMode,
    CameraPtr& pCamera)
{
    static_assert(std::is_same<char const*, decltype(CStringLikeTraits<IdType>::ToString(id))>::value,
                  "CStringLikeTraits<IdType>::ToString(const IdType&) does not return char const*");
    return OpenCameraByID(CStringLikeTraits<IdType>::ToString(id), eAccessMode, pCamera);
}

template<class T>
inline typename std::enable_if<CStringLikeTraits<T>::IsCStringLike, VmbErrorType>::type VmbSystem::GetTransportLayerByID(T const& id, TransportLayerPtr& pTransportLayer)
{
    static_assert(std::is_same<char const*, decltype(CStringLikeTraits<T>::ToString(id))>::value,
                  "CStringLikeTraits<T>::ToString(const IdType&) does not return char const*");
    return GetTransportLayerByID(CStringLikeTraits<T>::ToString(id), pTransportLayer);
}

template<class IdType>
inline typename std::enable_if<CStringLikeTraits<IdType>::IsCStringLike, VmbErrorType>::type VmbSystem::GetCameraByID(const IdType& id, VmbAccessModeType eAccessMode, CameraPtr& pCamera)
{
    static_assert(std::is_same<char const*, decltype(CStringLikeTraits<IdType>::ToString(id))>::value,
                  "CStringLikeTraits<IdType>::ToString(const IdType&) does not return char const*");
    return GetCameraByID(CStringLikeTraits<IdType>::ToString(id), pCamera);
}

}  // namespace VmbCPP

#endif
