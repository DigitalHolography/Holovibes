/*=============================================================================
  Copyright (C) 2012 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        UserSharedPointerDefines.h

  Description: Definition of macros for using different shared pointer 
               implementations.

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

#ifndef VMBCPP_USERSHAREDPOINTERDEFINES_H
#define VMBCPP_USERSHAREDPOINTERDEFINES_H

/**
* \file UserSharedPointerDefines.h
* \brief Definition of macros for using different shared pointer implementations
* 
* VmbCPP does not necessarily rely on VmbCPP::shared_ptr. You might want to use your own shared pointer type or the one that ships with your
* implementation of the C++ standard.
* To use a custom shared pointer implementation simply add the define USER_SHARED_POINTER to your project / compiler settings and complete this header file.
*
* Set the calls for your implementation of the shared pointer functions
* + Declaration
* + Reset with argument
* + Reset without argument
* + == operator
* + null test
* + Access to underlying raw pointer
* + Dynamic cast of shared pointer
*
* Add all your required shared pointer implementation headers here.
* HINT: `#include <memory>` is used for std::shared_ptr
*/

#include <memory>
#include <type_traits>

namespace VmbCPP {



/// This is the define for a declaration.
template<class T>
using SharedPointer = std::shared_ptr<T>;

/**
 * This is the define for setting an existing shared pointer.
 * 
 * The definition may also reside in the namespace of the shared pointer.
 */
template<class T, class U, typename std::enable_if<std::is_assignable<T*&, U*>::value, int>::type = 0>
void SP_SET(std::shared_ptr<T>& sp, U* rawPtr)
{
    sp.reset(rawPtr);
}

/**
 * This is the define for resetting without an argument to decrease the ref count.
 *
 * The definition may also reside in the namespace of the shared pointer.
 */
template<class T>
void SP_RESET(std::shared_ptr<T>& sp) noexcept
{
    sp.reset();
}

/**
 * This is the define for the equal operator. Shared pointers are usually considered equal when the raw pointers point to the same address.
 *
 * The definition may also reside in the namespace of the shared pointer.
 */
template<class T, class U>
bool SP_ISEQUAL(const std::shared_ptr<T>& sp1, const std::shared_ptr<U>& sp2) noexcept
{
    return sp1 == sp2;
}

/**
 * This is the define for the null check.
 *
 * The definition may also reside in the namespace of the shared pointer.
 */
template<class T>
bool SP_ISNULL(const std::shared_ptr<T>& sp)
{
    return nullptr == sp;
}

/**
 * This is the define for the raw pointer access. This is usually accomplished through the dereferencing operator (->).
 *
 * The definition may also reside in the namespace of the shared pointer.
 */
template<class T>
T* SP_ACCESS(const std::shared_ptr<T>& sp) noexcept
{
    return sp.get();
}

/**
 * This is the define for the dynamic cast of the pointer.
 *
 * The definition may also reside in the namespace of the shared pointer.
 */
template<class T, class U>
std::shared_ptr<T> SP_DYN_CAST(const std::shared_ptr<U>& sp) noexcept
{
    return std::dynamic_pointer_cast<T>(sp);
}

} // namespace VmbCPP


#endif //VMBCPP_USERSHAREDPOINTERDEFINES_H
