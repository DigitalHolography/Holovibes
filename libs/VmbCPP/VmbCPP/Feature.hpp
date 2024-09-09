/*=============================================================================
  Copyright (C) 2012-2022 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        Feature.hpp

  Description: Inline wrapper functions for class VmbCPP::Feature.

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

#ifndef VMBCPP_FEATURE_HPP
#define VMBCPP_FEATURE_HPP

/**
* \file  Feature.hpp
*
* \brief Inline wrapper functions for class VmbCPP::Feature
*        that allocate memory for STL objects in the application's context
*        and to pass data across DLL boundaries using arrays
*/
#include <utility>

#include "CopyHelper.hpp"

namespace VmbCPP {

inline VmbErrorType Feature::GetValues( StringVector &rValues ) noexcept
{
    VmbErrorType    res;
    VmbUint32_t     nSize;

    res = GetValues(static_cast<const char **>(nullptr), nSize);
    if (VmbErrorSuccess == res)
    {
        if ( 0 != nSize)
        {
            try
            {
                std::vector<const char*> data(nSize);
                res = GetValues(&data[0], nSize);

                if (VmbErrorSuccess == res)
                {
                    data.resize(nSize);
                    StringVector tmpValues(data.size());
                    std::copy(data.begin(), data.end(), tmpValues.begin());
                    rValues = std::move(tmpValues);
                }
            }
            catch(...)
            {
                return VmbErrorResources;
            }
        }
        else
        {
            rValues.clear();
        }
    }

    return res;
}

inline VmbErrorType Feature::GetEntries( EnumEntryVector &rEntries ) noexcept
{
    VmbUint32_t     nSize;

    VmbErrorType res = GetEntries(static_cast<EnumEntry*>(nullptr), nSize);
    if ( VmbErrorSuccess == res )
    {
        if( 0 != nSize )
        {
            try
            {
                EnumEntryVector tmpEntries( nSize );
                res = GetEntries( &tmpEntries[0], nSize );
                if( VmbErrorSuccess == res)
                {
                    tmpEntries.resize(nSize);
                    rEntries = std::move(tmpEntries);
                }
            }
            catch(...)
            {
                return VmbErrorResources;
            }
        }
        else
        {
            rEntries.clear();
        }
    }

    return res;
}

inline VmbErrorType Feature::GetValues(Int64Vector &rValues) noexcept
{
    return impl::ArrayGetHelper(*this, rValues, &Feature::GetValues);
}

inline VmbErrorType Feature::GetValue( std::string &rStrValue ) const noexcept
{
    std::string tmpStr;
    VmbErrorType res = impl::ArrayGetHelper(*this, tmpStr, &Feature::GetValue);
    rStrValue = tmpStr.c_str();
    return res;
}

inline VmbErrorType Feature::GetValue( UcharVector &rValue ) const noexcept
{
    VmbUint32_t i;
    return GetValue( rValue, i );
}

inline VmbErrorType Feature::GetValue( UcharVector &rValue, VmbUint32_t &rnSizeFilled ) const noexcept
{
    VmbErrorType    res;
    VmbUint32_t     nSize;

    res = GetValue( nullptr, nSize, rnSizeFilled );
    if ( VmbErrorSuccess == res )
    {
        if( 0 != nSize)
        {
            try
            {
                UcharVector tmpValue( nSize );
                res = GetValue( &tmpValue[0], nSize, rnSizeFilled );
                if( VmbErrorSuccess == res )
                {
                    rValue.swap( tmpValue);
                }
            }
            catch(...)
            {
                return VmbErrorResources;
            }
        }
        else
        {
            rValue.clear();
        }
    }

    return res;
}

template<class IntegralType>
inline
typename std::enable_if<std::is_integral<IntegralType>::value && !std::is_same<IntegralType, VmbInt64_t>::value, VmbErrorType>::type
Feature::SetValue(IntegralType value) noexcept
{
    return SetValue(static_cast<VmbInt64_t>(value));
}

template<class EnumType>
inline
typename std::enable_if<!std::is_same<bool, typename impl::UnderlyingTypeHelper<EnumType>::type>::value, VmbErrorType>::type
Feature::SetValue(EnumType value) noexcept
{
    return SetValue(static_cast<VmbInt64_t>(value));
}

inline VmbErrorType Feature::SetValue( const UcharVector &rValue ) noexcept
{
    if ( rValue.empty() )
    {
        return VmbErrorBadParameter;
    }
    return SetValue(&rValue[0], static_cast<VmbUint32_t>(rValue.size()));
}

inline VmbErrorType Feature::GetName( std::string &rStrName ) const noexcept
{
    return impl::ArrayGetHelper(*this, rStrName, &Feature::GetName);
}

inline VmbErrorType Feature::GetDisplayName( std::string &rStrDisplayName ) const noexcept
{
    return impl::ArrayGetHelper(*this, rStrDisplayName, &Feature::GetDisplayName);
}

inline VmbErrorType Feature::GetCategory( std::string &rStrCategory ) const noexcept
{
    return impl::ArrayGetHelper(*this, rStrCategory, &Feature::GetCategory);
}

inline VmbErrorType Feature::GetUnit( std::string &rStrUnit ) const noexcept
{
    return impl::ArrayGetHelper(*this, rStrUnit, &Feature::GetUnit);
}

inline VmbErrorType Feature::GetRepresentation( std::string &rStrRepresentation ) const noexcept
{
    return impl::ArrayGetHelper(*this, rStrRepresentation, &Feature::GetRepresentation);
}

inline VmbErrorType Feature::GetToolTip( std::string &rStrToolTip ) const noexcept
{
    return impl::ArrayGetHelper(*this, rStrToolTip, &Feature::GetToolTip);
}

inline VmbErrorType Feature::GetDescription( std::string &rStrDescription ) const noexcept
{
    return impl::ArrayGetHelper(*this, rStrDescription, &Feature::GetDescription);
}

inline VmbErrorType Feature::GetSFNCNamespace( std::string &rStrSFNCNamespace ) const noexcept
{
    return impl::ArrayGetHelper(*this, rStrSFNCNamespace, &Feature::GetSFNCNamespace);
}

inline VmbErrorType Feature::GetValidValueSet(Int64Vector& validValues) const noexcept
{
    VmbUint32_t nSize = 0;
    VmbErrorType res = GetValidValueSet(nullptr, 0, &nSize);
    
    if (VmbErrorSuccess == res)
    {
        if (0 != nSize)
        {
            try
            {
                Int64Vector tmpValues(nSize);
                res = GetValidValueSet(&tmpValues[0], static_cast<VmbUint32_t>(tmpValues.size()), &nSize);
                if (VmbErrorSuccess == res)
                {
                    validValues.swap(tmpValues);
                }
            }
            catch (...)
            {
                return VmbErrorResources;
            }
        }
        else
        {
            validValues.clear();
        }
    }

    return res;
}

inline VmbErrorType Feature::GetSelectedFeatures( FeaturePtrVector &rSelectedFeatures ) noexcept
{
    VmbErrorType    res;
    VmbUint32_t     nSize;

    res = GetSelectedFeatures( nullptr, nSize );
    if ( VmbErrorSuccess == res )
    {
        if( 0 != nSize )
        {
            try
            {
                FeaturePtrVector tmpSelectedFeatures( nSize );
                res = GetSelectedFeatures( &tmpSelectedFeatures[0], nSize );
                if( VmbErrorSuccess == res )
                {
                    rSelectedFeatures.swap ( tmpSelectedFeatures );
                }
            }
            catch(...)
            {
                return VmbErrorResources;
            }
        }
        else
        {
            rSelectedFeatures.clear();
        }
    }

    return res;
}

}  // namespace VmbCPP

#endif
