/*=============================================================================
  Copyright (C) 2022 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        StringLike.hpp

  Description: Helper functionality for types convertible to char const*.

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

#ifndef VMBCPP_STRINGLIKE_HPP
#define VMBCPP_STRINGLIKE_HPP

/**
* \file  StringLike.hpp
*
* \brief Helper functionality for types convertible to char const*
*/
#include <string>
#include <type_traits>

namespace VmbCPP {

/**
 * \brief Traits helping to determine, if a reference to a const object of type T
 *        is convertible to `char const*`.
 * 
 * Specializations must provide a static constexpr bool member variable IsCStringLike.
 * If this member variable evaluates to `true`, a static ToString function needs to be
 * provided that takes `const T&` and returns `char const*`
 * 
 */
template<class T>
struct CStringLikeTraits
{
    /**
     * \brief Marks objects as non-stringlike by default 
     */
    static constexpr bool IsCStringLike = false;
};

/**
 * \brief CStringLikeTraits specialization for std::string 
 */
template<>
struct CStringLikeTraits<std::string>
{
    /**
     * \brief marks this type of object as stringlike 
     */
    static constexpr bool IsCStringLike = true;

    /**
     * \brief Conversion function for the stringlike object to `const char*`
     * 
     * \param[in]   str the object to retrieve the data from.
     * 
     * \return a c string pointing to the data of \p str
     */
    static char const* ToString(const std::string& str) noexcept
    {
        return str.c_str();
    }
};

}  // namespace VmbCPP

#endif
