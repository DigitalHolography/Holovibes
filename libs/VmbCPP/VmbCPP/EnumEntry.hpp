/*=============================================================================
  Copyright (C) 2012-2022 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        EnumEntry.hpp

  Description: Inline wrapper functions for class VmbCPP::EnumEntry.

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

#ifndef VMBCPP_ENUMENTRY_HPP
#define VMBCPP_ENUMENTRY_HPP

/**
* \file  EnumEntry.hpp
*
* \brief Inline wrapper functions for class VmbCPP::EnumEntry
*        that allocate memory for STL objects in the application's context
*        and to pass data across DLL boundaries using arrays
*/

#include "CopyHelper.hpp"

namespace VmbCPP {

inline VmbErrorType EnumEntry::GetName( std::string &rStrName ) const noexcept
{
    return impl::ArrayGetHelper(*this, rStrName, &EnumEntry::GetName);
}

inline VmbErrorType EnumEntry::GetDisplayName( std::string &rStrDisplayName ) const noexcept
{
    return impl::ArrayGetHelper(*this, rStrDisplayName, &EnumEntry::GetDisplayName);
}

inline VmbErrorType EnumEntry::GetDescription( std::string &rStrDescription ) const noexcept
{
    return impl::ArrayGetHelper(*this, rStrDescription, &EnumEntry::GetDescription);
}

inline VmbErrorType EnumEntry::GetTooltip( std::string &rStrTooltip ) const noexcept
{
    return impl::ArrayGetHelper(*this, rStrTooltip, &EnumEntry::GetTooltip);
}

inline VmbErrorType EnumEntry::GetSFNCNamespace( std::string &rStrNamespace ) const noexcept
{
    return impl::ArrayGetHelper(*this, rStrNamespace, &EnumEntry::GetSFNCNamespace);
}

}  // namespace VmbCPP

#endif
