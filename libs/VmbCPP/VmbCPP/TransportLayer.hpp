/*=============================================================================
  Copyright (C) 2022 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        TransportLayer.hpp

  Description: Inline wrapper functions for class VmbCPP::TransportLayer.

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


#ifndef VMBCPP_TRANSPORTLAYER_HPP
#define VMBCPP_TRANSPORTLAYER_HPP

/**
* \file  TransportLayer.hpp
*
* \brief Inline wrapper functions for class VmbCPP::TransportLayer
*        that allocate memory for STL objects in the application's context
*        and to pass data across DLL boundaries using arrays
*/
#include <utility>

#include "CopyHelper.hpp"

namespace VmbCPP {

inline VmbErrorType TransportLayer::GetInterfaces( InterfacePtrVector &rInterfaces )
{
    VmbErrorType    res;
    VmbUint32_t     nSize;

    res = GetInterfaces(nullptr, nSize);
    if (VmbErrorSuccess == res)
    {
        if (0 != nSize)
        {
            try
            {
                InterfacePtrVector tmpInterfaces(nSize);
                res = GetInterfaces(&tmpInterfaces[0], nSize);
                if (VmbErrorSuccess == res)
                {
                    tmpInterfaces.resize(nSize);
                    rInterfaces = std::move(tmpInterfaces);
                }
            }
            catch (...)
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

inline VmbErrorType TransportLayer::GetCameras( CameraPtrVector &rCameras )
{
    VmbErrorType    res;
    VmbUint32_t     nSize;

    res = GetCameras(nullptr, nSize);
    if (VmbErrorSuccess == res)
    {
        if (0 != nSize)
        {
            try
            {
                CameraPtrVector tmpCameras(nSize);
                res = GetCameras(&tmpCameras[0], nSize);
                if (VmbErrorSuccess == res)
                {
                    tmpCameras.resize(nSize);
                    rCameras = std::move(tmpCameras);
                }
            }
            catch (...)
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

inline VmbErrorType TransportLayer::GetID( std::string &rStrID ) const noexcept
{
    return impl::ArrayGetHelper(*this, rStrID, &TransportLayer::GetID);
}

inline VmbErrorType TransportLayer::GetName(std::string& rName) const noexcept
{
    return impl::ArrayGetHelper(*this, rName, &TransportLayer::GetName);
}

inline VmbErrorType TransportLayer::GetModelName(std::string& rModelName) const noexcept
{
    return impl::ArrayGetHelper(*this, rModelName, &TransportLayer::GetModelName);
}

inline VmbErrorType TransportLayer::GetVendor(std::string& rVendor) const noexcept
{
    return impl::ArrayGetHelper(*this, rVendor, &TransportLayer::GetVendor);
}

inline VmbErrorType TransportLayer::GetVersion(std::string& rVersion) const noexcept
{
    return impl::ArrayGetHelper(*this, rVersion, &TransportLayer::GetVersion);
}

inline VmbErrorType TransportLayer::GetPath(std::string& rPath) const noexcept
{
    return impl::ArrayGetHelper(*this, rPath, &TransportLayer::GetPath);
}

}  // namespace VmbCPP

#endif
