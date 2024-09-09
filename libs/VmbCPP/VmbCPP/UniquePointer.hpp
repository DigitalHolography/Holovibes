/*=============================================================================
  Copyright (C) 2022 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        UniquePointer.hpp

  Description: Definition of a class ensuring destruction of a stored object

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

#ifndef VMBCPP_UNIQUE_POINTER_H
#define VMBCPP_UNIQUE_POINTER_H

/**
* \file      UniquePointer.hpp
*
* \brief     Definition of a smart pointer class that can be 
*            used with VmbCPP.
*/

#include <cassert>
#include <cstddef>
#include <type_traits>

namespace VmbCPP {

/**
 * \brief Smart pointer with sole ownership of the wrapped object.
 * 
 * \tparam the type of the object owned by the smart pointer
 */
template<class T>
class UniquePointer
{
public:
    /**
     * \brief The type of object owned by this type this smart pointer type 
     */
    using ValueType = T;

    /**
     * \brief Constructor taking ownership of \p ptr
     *
     * \param[in]   ptr     The object to take ownership of; may be null
     */
    explicit UniquePointer(ValueType* ptr) noexcept
        : m_ptr(ptr)
    {
    }

    UniquePointer(UniquePointer const&) = delete;
    UniquePointer& operator=(UniquePointer const&) = delete;

    /**
     * \brief Destroy this object freeing the owned object, if it exists. 
     */
    ~UniquePointer()
    {
        delete m_ptr;
    }

    /**
     * \brief Change the value of this pointer to a new one.
     *
     * Frees a previously owned object
     */
    void reset(ValueType* newPtr)
    {
        delete m_ptr;
        m_ptr = newPtr;
    }
    
    /**
     * \brief Member access to this pointer
     *
     * \note This function yields an assertion error, if the pointer is null 
     */
    ValueType* operator->() const noexcept
    {
        assert(m_ptr != nullptr);
        return m_ptr;
    }

    /**
     * \brief Dereferences this pointer
     *
     * \note This function yields an assertion error, if the pointer is null
     */
    ValueType& operator*() const noexcept
    {
        assert(m_ptr != nullptr);
        return *m_ptr;
    }

    /**
     * \brief Checks, if this pointer contains null
     */
    bool operator==(std::nullptr_t) const noexcept
    {
        return m_ptr == nullptr;
    }

    /**
     * \brief Checks, if \p ptr contains null
     * 
     * \param[in] ptr the unique pointer to check for null
     */
    friend bool operator==(std::nullptr_t, UniquePointer const& ptr) noexcept
    {
        return ptr.m_ptr == nullptr;
    }

    /**
     * \brief Checks, if this pointer contains something other than null
     */
    bool operator!=(std::nullptr_t) const noexcept
    {
        return m_ptr != nullptr;
    }

    /**
     * \brief Checks, if \p ptr contains something other than null
     * 
     * \param[in] ptr   the pointer to check for null
     */
    friend bool operator!=(std::nullptr_t, UniquePointer const& ptr) noexcept
    {
        return ptr.m_ptr != nullptr;
    }
private:
    ValueType* m_ptr;
};

} //namespace VmbCPP

#endif // VMBCPP_UNIQUE_POINTER_H
