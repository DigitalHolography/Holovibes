/*=============================================================================
  Copyright (C) 2022 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        CopyHelper.hpp

  Description: Helper functionality for copying data to a standard library
               container.

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

#ifndef VMBCPP_COPYHELPER_HPP
#define VMBCPP_COPYHELPER_HPP

#include <utility>
#include <type_traits>

#include <VmbC/VmbCommonTypes.h>

/**
* \file  CopyHelper.hpp
*
* \brief Inline function for copying data to standard library containers given a function
*        with signature `VmbErrorType f(char*, VmbUint32_t&, ...)`
*/

namespace VmbCPP
{

/**
 * \brief namespace used internally for helpers of inline functions
 */
namespace impl
{

/**
 * \tparam Container the type of container to copy to (std::string, std::vector<SomeType>, ...)
 * \tparam T the class the getter is a member of
 */
template<class T, class Container, class ...Args>
inline VmbErrorType ArrayGetHelper(T& obj,
                                    Container& out,
                                    typename std::conditional<
                                    std::is_const<T>::value,
                                    VmbErrorType(T::*)(typename Container::value_type*, VmbUint32_t&, Args...) const noexcept,
                                    VmbErrorType(T::*)(typename Container::value_type*, VmbUint32_t&, Args...) noexcept
                                    >::type Getter,
                                    Args ...args) noexcept
{
    VmbUint32_t nLength;

    VmbErrorType res = (obj.*Getter)(nullptr, nLength, args...);
    if (VmbErrorSuccess == res)
    {
        if (0 != nLength)
        {
            try
            {
                Container tmpName(static_cast<typename Container::size_type>(nLength), typename Container::value_type{});
                res = (obj.*Getter)(&tmpName[0], nLength, args...);
                if (VmbErrorSuccess == res)
                {
                    out = std::move(tmpName);
                }
            }
            catch (...)
            {
                res = VmbErrorResources;
            }
        }
        else
        {
            out.clear();
        }
    }

    return res;
}

}}  // namespace VmbCPP::impl

#endif
