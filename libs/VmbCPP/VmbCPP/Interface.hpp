/*=============================================================================
  Copyright (C) 2012-2022 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        Interface.hpp

  Description: Inline wrapper functions for class VmbCPP::Interface.

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

#ifndef VMBCPP_INTERFACE_HPP
#define VMBCPP_INTERFACE_HPP

/**
* \file  Interface.hpp
*
* \brief Inline wrapper functions for class VmbCPP::Interface
*        that allocate memory for STL objects in the application's context
*        and to pass data across DLL boundaries using arrays
*/

#include "CopyHelper.hpp"

namespace VmbCPP {

// HINT: This information remains static throughout the object's lifetime
inline VmbErrorType Interface::GetID( std::string &rStrID ) const noexcept
{
    return impl::ArrayGetHelper(*this, rStrID, &Interface::GetID);
}

// HINT: This information remains static throughout the object's lifetime
inline VmbErrorType Interface::GetName( std::string &rStrName ) const noexcept
{
    return impl::ArrayGetHelper(*this, rStrName, &Interface::GetName);
}

inline VmbErrorType Interface::GetCameras(CameraPtrVector& rCameras)
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

}  // namespace VmbCPP

#endif
