/*=============================================================================
  Copyright (C) 2012 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        SharedPointer.h

  Description: Definition of an example shared pointer class that can be 
               used with VmbCPP.
               (This include file contains example code only.)

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

#ifndef VMBCPP_SHAREDPOINTER_H
#define VMBCPP_SHAREDPOINTER_H

/**
* \file      SharedPointer.h
*
* \brief     Definition of an example shared pointer class that can be 
*            used with VmbCPP.
* \note      (This include file contains example code only.)
*/

#include <cstddef>
#include <type_traits>

namespace VmbCPP {

    //Coding style of this file is different to mimic the shared_ptr classes of std and boost.

    struct dynamic_cast_tag
    {
    };

    class ref_count_base;

    /**
     * \brief A custom shared pointer implementation used by default by the VmbCPP API.
     *
     * \tparam T    The object type partially owned by this object.
     */
    template <class T>
    class shared_ptr final
    {
    private:
        class ref_count;

        typedef shared_ptr<T> this_type;
        
        template<class T2>
        friend class shared_ptr;

        VmbCPP::ref_count_base  *m_pRefCount;
        T                       *m_pObject;

        template <class T2>
        static void swap(T2 &rValue1, T2 &rValue2);

    public:
        shared_ptr() noexcept;

        /**
         * \brief Create a shared pointer object given a raw pointer to take ownership of
         *
         * \param[in] pObject   the raw pointer to take ownership of
         * 
         * \tparam T2   The pointer type passed as parameter.
         */
        template <class T2>
        explicit shared_ptr(T2 *pObject);

        /**
         * \brief Creates a shared pointer object not owning any object. 
         */
        explicit shared_ptr(std::nullptr_t) noexcept;

        /**
         * \brief copy constructor for a shared pointer 
         */
        shared_ptr(const shared_ptr &rSharedPointer);

        /**
         * \brief Constructor for taking shared ownership of a object owned by a shared pointer
         * to a different shared pointer type.
         * 
         * The raw pointers are converted using implicit conversion.
         * 
         * \param[in] rSharedPointer the pointer the ownership is shared with
         */
        template <class T2>
        shared_ptr(const shared_ptr<T2> &rSharedPointer);

        /**
         * \brief Constructor for taking shared ownership of a object owned by a shared pointer
         * to a different shared pointer type using dynamic_cast to attempt to convert
         * the raw pointers.
         * 
         * \param[in] rSharedPointer
         */
        template <class T2>
        shared_ptr(const shared_ptr<T2> &rSharedPointer, dynamic_cast_tag);

        ~shared_ptr();

        /**
         * \brief copy assignment operator
         * 
         * \param[in]   rSharedPointer  the pointer to copy
         * 
         * \return a reference to this object
         */
        shared_ptr& operator=(const shared_ptr &rSharedPointer);

        /**
         * \brief assignment operator using implicit conversion to convert the raw pointers.
         * 
         * \param[in]   rSharedPointer  the pointer to copy
         * 
         * \tparam T2 the pointer type of the source of the assignment
         * 
         * \return a reference to this object
         */
        template <class T2>
        shared_ptr<T>& operator=(const shared_ptr<T2> &rSharedPointer);
        
        /**
         * \brief reset this pointer to null 
         */
        void reset();

        /**
         * \brief replace the object owned by the object provided
         * 
         * The raw pointers are convererted using implicit conversion
         * 
         * \param[in]   pObject the pointer to the object to take ownership of.
         * 
         * \tparam T2 the type of the pointer
         */
        template <class T2>
        void reset(T2 *pObject);

        /**
         * \brief Getter for the raw pointer to the object owned by this object.
         *
         * \return the raw pointer to the object owned
         */
        T* get() const noexcept;

        /**
         * \brief Dereference operator for this object
         *
         * \note This operator will result in an assertion error, if this object is a null pointer.
         * 
         * \return a reference to the owned object.
         */
        T& operator * () const noexcept;

        /**
         * \brief Operator for member access of the owned object
         *
         * \note This operator will result in an assertion error, if this object is a null pointer.
         * 
         * \return the pointer to the object to access.
         */
        T* operator -> () const noexcept;

        /**
         * \brief Get the number of shared pointers currently sharing ownership of this object.
         * 
         * The result includes this object in the count.
         * 
         * \return the number of shared pointers sharing ownership of the object pointed to.
         */
        long use_count() const;

        /**
         * \brief Checks, if this object is currently the sole owner of the object pointed to.
         * 
         * \return true if and only if the object is currently owned by this object exclusively.
         */
        bool unique() const;

        /**
         * \brief Checks, if this object is not a null pointer.
         * 
         * \return true if and only if this object is not a null pointer.
         */
        operator bool() const noexcept
        {
            return m_pObject != nullptr;
        }

        /**
         * \brief Exchange the objects owned by this pointer and the pointer provided
         *
         * \param[in,out]   rSharedPointer  the pointer to exchange the owned objects with.
         */
        void swap(shared_ptr &rSharedPointer) noexcept;
    };

    /**
     * \brief Convert from one shared pointer type to another using dynamic_cast.
     *
     * \param[in] rSharedPointer    the shared pointer object that should be converted.
     * 
     * \tparam T    The target type of the conversion
     * \tparam T2   The source type of the conversion
     * 
     * \return A shared pointer sharing the reference counter of \p rSharedPointer, if the conversion was successful
     *         and a null pointer otherwise.
     */
    template<class T, class T2>
    shared_ptr<T> dynamic_pointer_cast(const shared_ptr<T2> &rSharedPointer);

    /**
     * \brief Operator checking, if the shared pointers point to the same object.
     *
     * \param[in] sp1   one of the shared pointers to compare
     * \param[in] sp2   the other shared pointer to compare
     *
     * \tparam T1   The type of the first shared pointer
     * \tparam T2   The type of the second shared pointer
     *
     * \return true, if the pointers point to the same object, false otherwise
     */
    template<class T1, class T2>
    bool operator==(const shared_ptr<T1>& sp1, const shared_ptr<T2>& sp2);

    /**
     * \brief Operator checking, if the shared pointers point to different objects.
     *
     * \param[in] sp1   one of the shared pointers to compare
     * \param[in] sp2   the other shared pointer to compare
     *
     * \tparam T1   The type of the first shared pointer
     * \tparam T2   The type of the second shared pointer
     *
     * \return false, if the pointers point to the same object, false otherwise
     */
    template<class T1, class T2>
    bool operator!=(const shared_ptr<T1>& sp1, const shared_ptr<T2>& sp2);

    /**
     * \brief Operator checking, a shared pointer for null
     *
     * \param[in] sp   the shared pointer to check for null
     *
     * \tparam T    The type of the shared pointer
     *
     * \return true, if and only if \p sp contains null
     */
    template<class T>
    bool operator==(const shared_ptr<T>& sp, std::nullptr_t);

    /**
     * \brief Operator checking, a shared pointer for null
     *
     * \param[in] sp   the shared pointer to check for null
     *
     * \tparam T    The type of the shared pointer
     *
     * \return true, if and only if \p sp contains null
     */
    template<class T>
    bool operator==(std::nullptr_t, const shared_ptr<T>& sp);

    /**
     * \brief Operator checking, a shared pointer for null
     *
     * \param[in] sp   the shared pointer to check for null
     *
     * \tparam T    The type of the shared pointer
     *
     * \return false, if and only if \p sp contains null
     */
    template<class T>
    bool operator!=(const shared_ptr<T>& sp, std::nullptr_t);

    /**
     * \brief Operator checking, a shared pointer for null
     *
     * \param[in] sp   the shared pointer to check for null
     *
     * \tparam T    The type of the shared pointer
     *
     * \return false, if and only if \p sp contains null
     */
    template<class T>
    bool operator!=(std::nullptr_t, const shared_ptr<T>& sp);

    /**
     * \defgroup AccessFunctions Functions for accessing shared pointer objects
     * \{
     */

    template<class T>
    using SharedPointer = shared_ptr<T>;

    /**
     * \brief The function used for assigning ownership of a raw pointer to a shared pointer object.
     * 
     * \param[out]  target  the shared pointer to assign the ownership to.
     * \param[in]   rawPtr  the raw pointer \p target should receive ownership of.
     * 
     * \tparam T    the type of shared pointer receiving the ownership.
     * \tparam U    the type of the raw pointer; `U*` must be assignable to `T*`
     */
    template<class T, class U, typename std::enable_if<std::is_assignable<T*&, U*>::value, int>::type = 0>
    void SP_SET(shared_ptr<T>& target, U* rawPtr);

    /**
     * \brief Function for resetting a shared pointer to null.
     *
     * \param[out] target   the shared pointer to set to null
     *
     * \tparam T    type the pointer points to
     */
    template<class T>
    void SP_RESET(shared_ptr<T>& target);

    /**
     * \brief A function used for checking, if to shared pointers point to the same object.
     * 
     * \param[in]   lhs the first pointer to compare
     * \param[in]   rhs the second pointer to compare
     * 
     * \tparam T    The first pointer type
     * \tparam U    The second pointer type
     * 
     * \return true if and only if the pointers point to the same object.
     */
    template<class T, class U>
    bool SP_ISEQUAL(const shared_ptr<T>& lhs, const shared_ptr<U>& rhs);

    /**
     * \brief A function used to check a shared pointer for null
     *
     * \param[in]   sp  the shared pointer to check for null
     *
     * \tparam T    The type of pointer
     *
     * \return true if and only if the pointer points to null
     */
    template<class T>
    bool SP_ISNULL(const shared_ptr<T>& sp);

    /**
     * \brief a function for accessing the raw pointer of the shared pointer.
     *
     * \param[in] sp the shared pointer to get the raw pointer from
     * 
     * \tparam T    the type of the pointer
     * 
     * \return a raw pointer to the object owned by \p sp
     */
    template<class T>
    T* SP_ACCESS(const shared_ptr<T>& sp);

    /**
     * \brief Convert from one shared pointer type to another using dynamic_cast.
     *
     * \param[in] sp    the shared pointer object that should be converted.
     *
     * \tparam T    The target type of the conversion
     * \tparam U    The source type of the conversion
     *
     * \return A shared pointer sharing the reference counter of \p sp, if the conversion was successful
     *         and a null pointer otherwise.
     */
    template<class T, class U>
    shared_ptr<T> SP_DYN_CAST(shared_ptr<U>& sp);

    /**
     * \}
     */

} //namespace VmbCPP

#include <VmbCPP/SharedPointer_impl.h>

#endif //VMBCPP_SHAREDPOINTER_H
