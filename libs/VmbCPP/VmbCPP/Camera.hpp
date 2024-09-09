/*=============================================================================
  Copyright (C) 2012 - 2016 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        Camera.hpp

  Description: Inline wrapper functions for class VmbCPP::Camera.

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

#ifndef VMBCPP_CAMERA_HPP
#define VMBCPP_CAMERA_HPP

/**
* \file Camera.hpp
*
* \brief Inline wrapper functions for class VmbCPP::Camera
*        that allocate memory for STL objects in the application's context
*        and to pass data across DLL boundaries using arrays
*/

#include <utility>

#include "CopyHelper.hpp"

namespace VmbCPP {

// HINT: This information remains static throughout the object's lifetime
inline VmbErrorType Camera::GetID( std::string &rStrID ) const noexcept
{
    constexpr bool extended = false;
    return impl::ArrayGetHelper(*this, rStrID, &Camera::GetID, extended);
}

inline VmbErrorType Camera::GetExtendedID(std::string& rStrID) const noexcept
{
    constexpr bool extended = true;
    return impl::ArrayGetHelper(*this, rStrID, &Camera::GetID, extended);
}

// HINT: This information remains static throughout the object's lifetime
inline VmbErrorType Camera::GetName( std::string &rStrName ) const noexcept
{
    return impl::ArrayGetHelper(*this, rStrName, &Camera::GetName);
}

// HINT: This information remains static throughout the object's lifetime
inline VmbErrorType Camera::GetModel( std::string &rStrModel ) const noexcept
{
    return impl::ArrayGetHelper(*this, rStrModel, &Camera::GetModel);
}

// HINT: This information remains static throughout the object's lifetime
inline VmbErrorType Camera::GetSerialNumber( std::string &rStrSerial ) const noexcept
{
    return impl::ArrayGetHelper(*this, rStrSerial, &Camera::GetSerialNumber);
}

// HINT: This information remains static throughout the object's lifetime
inline VmbErrorType Camera::GetInterfaceID( std::string &rStrInterfaceID ) const
{
    InterfacePtr pInterface;
    VmbErrorType res = GetInterface(pInterface);
    if (VmbErrorSuccess != res || SP_ISNULL(pInterface))
    {
        return VmbErrorNotAvailable;
    }
    return SP_ACCESS(pInterface)->GetID(rStrInterfaceID);
}

inline VmbErrorType Camera::AcquireMultipleImages( FramePtrVector &rFrames, VmbUint32_t nTimeout, FrameAllocationMode allocationMode )
{
    VmbErrorType res;
    VmbUint32_t i;
    res = AcquireMultipleImages( rFrames, nTimeout, i, allocationMode );
    if ( rFrames.size() != i )
    {
        res = VmbErrorInternalFault;
    }
    
    return res;
}
inline VmbErrorType Camera::AcquireMultipleImages( FramePtrVector &rFrames, VmbUint32_t nTimeout, VmbUint32_t &rNumFramesCompleted, FrameAllocationMode allocationMode )
{
    if ( rFrames.empty() )
    {
        return VmbErrorBadParameter;
    }

    return AcquireMultipleImages( &rFrames[0], (VmbUint32_t)rFrames.size(), nTimeout, &rNumFramesCompleted, allocationMode );
}

// HINT: Size of buffer determines how many bytes to read.
inline VmbErrorType Camera::ReadMemory( const VmbUint64_t &rAddress, UcharVector &rBuffer ) const noexcept
{
    VmbUint32_t i;
    return ReadMemory( rAddress, rBuffer, i );
}

inline VmbErrorType Camera::ReadMemory( const VmbUint64_t &rAddress, UcharVector &rBuffer, VmbUint32_t &rCompletedReads ) const noexcept
{
    if ( rBuffer.empty() )
    {
        return VmbErrorBadParameter;
    }

    return ReadMemory( rAddress, &rBuffer[0], (VmbUint32_t)rBuffer.size(), &rCompletedReads );
}

// HINT: Size of buffer determines how many bytes to write.
inline VmbErrorType Camera::WriteMemory( const VmbUint64_t &rAddress, const UcharVector &rBuffer ) noexcept
{
    VmbUint32_t i;
    return WriteMemory( rAddress, rBuffer, i );
}

inline VmbErrorType Camera::WriteMemory(
    const VmbUint64_t &rAddress,
    const UcharVector &rBuffer,
    VmbUint32_t &rCompletedWrites) noexcept
{
    if ( rBuffer.empty() )
    {
        return VmbErrorBadParameter;
    }

    return WriteMemory( rAddress, &rBuffer[0], (VmbUint32_t)rBuffer.size(), &rCompletedWrites );
}

inline VmbErrorType Camera::GetStreams(StreamPtrVector& rStreams) noexcept
{
    return impl::ArrayGetHelper(*this, rStreams, &Camera::GetStreams);
}

}  // namespace VmbCPP

#endif
