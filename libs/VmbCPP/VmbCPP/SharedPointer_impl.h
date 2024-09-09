/*=============================================================================
  Copyright (C) 2012 - 2022 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        SharedPointer_impl.h

  Description: Implementation of an example shared pointer class for VmbCPP.
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

#ifndef VMBCPP_SHAREDPOINTER_IMPL_H
#define VMBCPP_SHAREDPOINTER_IMPL_H

/**
* \file        SharedPointer_impl.h
*
* \brief       Implementation of an example shared pointer class for the VmbCPP.
* \note        (This include file contains example code only.)
*/

#include <cassert>
#include <cstddef>
#include <stdexcept>

#include <VmbCPP/Mutex.h>

namespace VmbCPP {

    /**
     * \brief The base class of pointer references for use by shared_ptr. 
     */
    class ref_count_base
    {
    public:
        virtual ~ref_count_base() = default;

        /**
         * \brief Increment the reference count. 
         */
        virtual void inc() = 0;

        /**
         * \brief decrement the reference count 
         */
        virtual void dec() = 0;

        /**
         * \brief Get the current reference count.
         *
         * \return the current reference count.
         */
        virtual long use_count() const = 0;
    };

    /**
     * \brief Reference counter implementation for shared_ptr.
     *
     * \tparam T    The type of object the shared pointer refers to
     */
    template <class T>
    class shared_ptr<T>::ref_count : public ref_count_base
    {
    private:
        T* m_pObject;
        long            m_nCount;
        Mutex           m_Mutex;

    public:
        /**
         * \brief constructor creating a reference counter for a given raw pointer
         *
         * \param[in]   pObject     a pointer to the object the created object counts references for
         */
        explicit ref_count(T* pObject)
            : m_pObject(pObject),
            m_nCount(1)
        {
        }

        virtual ~ref_count()
        {
            if (nullptr != m_pObject)
            {
                delete m_pObject;
            }

            m_Mutex.Unlock();
        }

        ref_count(const ref_count& rRefCount) = delete;
        ref_count& operator=(const ref_count& rRefCount) = delete;

        virtual void inc() override
        {
            m_Mutex.Lock();

            m_nCount++;

            m_Mutex.Unlock();
        }

        virtual void dec() override
        {
            m_Mutex.Lock();
            if (m_nCount == 0)
            {
                throw std::logic_error("shared pointer, used incorrectly");
            }
            if (m_nCount > 1)
            {
                m_nCount--;

                m_Mutex.Unlock();
            }
            else
            {
                // m_Mutex will be unlocked in d'tor
                delete this;
            }
        }

        virtual long use_count() const override
        {
            return m_nCount;
        }
    };

    template <class T>
    template <class T2>
    void shared_ptr<T>::swap(T2 &rValue1, T2 &rValue2)
    {
        T2 buffer = rValue1;
        rValue1 = rValue2;
        rValue2 = buffer;
    }

    template <class T>
    shared_ptr<T>::shared_ptr() noexcept
        :   m_pRefCount(nullptr)
        ,   m_pObject(nullptr)
    {
    }
    
    template <class T>
    template <class T2>
    shared_ptr<T>::shared_ptr(T2 *pObject)
        :   m_pRefCount(nullptr)
        ,   m_pObject(nullptr)
    {
        m_pRefCount = new typename shared_ptr<T2>::ref_count(pObject);
        if(nullptr == m_pRefCount)
        {
            delete pObject;

            throw std::bad_alloc();
        }

        m_pObject = pObject;
    }

    template <class T>
    shared_ptr<T>::shared_ptr(std::nullptr_t) noexcept
        : shared_ptr(static_cast<T*>(nullptr))
    {
    }
    
    template <class T>
    template <class T2>
    shared_ptr<T>::shared_ptr(const shared_ptr<T2> &rSharedPointer)
        :   m_pRefCount(nullptr)
        ,   m_pObject(nullptr)
    {
        if(nullptr != rSharedPointer.m_pRefCount)
        {
            rSharedPointer.m_pRefCount->inc();

            m_pRefCount = rSharedPointer.m_pRefCount;
            m_pObject = rSharedPointer.m_pObject;
        }
    }

    template <class T>
    template <class T2>
    shared_ptr<T>::shared_ptr(const shared_ptr<T2> &rSharedPointer, dynamic_cast_tag)
        :   m_pRefCount(nullptr)
        ,   m_pObject(nullptr)
    {
        if(nullptr != rSharedPointer.m_pRefCount)
        {
            T *pObject = dynamic_cast<T*>(rSharedPointer.m_pObject);
            if(nullptr != pObject)
            {
                rSharedPointer.m_pRefCount->inc();

                m_pRefCount = rSharedPointer.m_pRefCount;
                m_pObject = pObject;
            }
        }
    }

    template <class T>
    shared_ptr<T>::shared_ptr(const shared_ptr &rSharedPointer)
        :   m_pRefCount(nullptr)
        ,   m_pObject(nullptr)
    {
        if(nullptr != rSharedPointer.m_pRefCount)
        {
            rSharedPointer.m_pRefCount->inc();

            m_pRefCount = rSharedPointer.m_pRefCount;
            m_pObject = rSharedPointer.m_pObject;
        }
    }

    template <class T>
    shared_ptr<T>::~shared_ptr()
    {
        if(nullptr != m_pRefCount)
        {
            m_pRefCount->dec();
            m_pRefCount = nullptr;
            m_pObject = nullptr;
        }
    }

    template <class T>
    template <class T2>
    shared_ptr<T>& shared_ptr<T>::operator=(const shared_ptr<T2> &rSharedPointer)
    {
        shared_ptr(rSharedPointer).swap(*this);

        return *this;
    }

    template <class T>
    shared_ptr<T>& shared_ptr<T>::operator=(const shared_ptr &rSharedPointer)
    {
        shared_ptr(rSharedPointer).swap(*this);

        return *this;
    }

    template <class T>
    void shared_ptr<T>::reset()
    {
        shared_ptr().swap(*this);
    }
    
    template <class T>
    template <class T2>
    void shared_ptr<T>::reset(T2 *pObject)
    {
        shared_ptr(pObject).swap(*this);
    }

    template <class T>
    T* shared_ptr<T>::get() const noexcept
    {
        return m_pObject;
    }
    
    template <class T>
    T& shared_ptr<T>::operator * () const noexcept
    {
        assert(m_pObject != nullptr);
        return *m_pObject;
    }
    
    template <class T>
    T* shared_ptr<T>::operator -> () const noexcept
    {
        assert(m_pObject != nullptr);
        return m_pObject;
    }
    
    template <class T>
    long shared_ptr<T>::use_count() const
    {
        if(nullptr == m_pRefCount)
        {
            return 0;
        }

        return m_pRefCount->use_count();
    }
    
    template <class T>
    bool shared_ptr<T>::unique() const
    {
        return (use_count() == 1);
    }

    template <class T>
    void shared_ptr<T>::swap(shared_ptr &rSharedPointer) noexcept
    {
        swap(m_pObject, rSharedPointer.m_pObject);
        swap(m_pRefCount, rSharedPointer.m_pRefCount);
    }

    template<class T, class T2>
    shared_ptr<T> dynamic_pointer_cast(const shared_ptr<T2> &rSharedPointer)
    {
        return shared_ptr<T>(rSharedPointer, dynamic_cast_tag());
    }

    template <class T1, class T2>
    bool operator==(const shared_ptr<T1>& sp1, const shared_ptr<T2>& sp2)
    {
        return sp1.get() == sp2.get();
    }

    template <class T1, class T2>
    bool operator!=(const shared_ptr<T1>& sp1, const shared_ptr<T2>& sp2)
    {
        return sp1.get() != sp2.get();
    }

    template<class T>
    bool operator==(const shared_ptr<T>& sp, std::nullptr_t)
    {
        return sp.get() == nullptr;
    }

    template<class T>
    bool operator==(std::nullptr_t, const shared_ptr<T>& sp)
    {
        return sp.get() == nullptr;
    }

    template<class T>
    bool operator!=(const shared_ptr<T>& sp, std::nullptr_t)
    {
        return sp.get() != nullptr;
    }

    template<class T>
    bool operator!=(std::nullptr_t, const shared_ptr<T>& sp)
    {
        return sp.get() != nullptr;
    }

    template<class T, class U, typename std::enable_if<std::is_assignable<T*&, U*>::value, int>::type>
    inline void SP_SET(shared_ptr<T>& target, U* rawPtr)
    {
        return target.reset(rawPtr);
    }

    template<class T>
    inline void SP_RESET(shared_ptr<T>& target)
    {
        return target.reset();
    }

    template<class T, class U>
    inline bool SP_ISEQUAL(const shared_ptr<T>& lhs, const shared_ptr<U>& rhs)
    {
        return lhs == rhs;
    }

    template<class T>
    inline bool SP_ISNULL(const shared_ptr<T>& sp)
    {
        return nullptr == sp.get();
    }

    template<class T>
    inline T* SP_ACCESS(const shared_ptr<T>& sp)
    {
        return sp.get();
    }

    template<class T, class U>
    inline shared_ptr<T> SP_DYN_CAST(shared_ptr<U>& sp)
    {
        return dynamic_pointer_cast<T>(sp);
    }

} //namespace VmbCPP

#endif //VMBCPP_SHAREDPOINTER_IMPL_H
