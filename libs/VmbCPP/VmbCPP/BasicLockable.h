/*=============================================================================
  Copyright (C) 2012-2022 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        BasicLockable.h

  Description: Definition of class VmbCPP::BasicLockable.

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

#ifndef VMBCPP_BASICLOCKABLE
#define VMBCPP_BASICLOCKABLE

/**
* \file  BasicLockable.h
*
* \brief Definition of class VmbCPP::BasicLockable.
*/

#include <VmbCPP/VmbCPPCommon.h>
#include <VmbCPP/SharedPointerDefines.h>
#include <VmbCPP/Mutex.h>


namespace VmbCPP {

class BaseFeature;
class Condition;
class ConditionHelper;
class MutexGuard;

/**
 * \brief A class providing lock and unlock functionality implemented via mutex. 
 */
class BasicLockable
{
public:
    IMEXPORT BasicLockable();

    /**
     * \brief Constructor copying the mutex pointer.
     */
    IMEXPORT BasicLockable( MutexPtr pMutex );

    /**
     * \brief Constructor taking ownership of the mutex pointer.
     */
    IMEXPORT BasicLockable( MutexPtr&& pMutex ) noexcept;

    IMEXPORT virtual ~BasicLockable();

    /**
     * \brief Acquire the lock.
     * 
     * The call blocks, until the lock is available.
     */
    void Lock()
    {
        SP_ACCESS(m_pMutex)->Lock();
    }

    /**
     * \brief Release the lock. 
     */
    void Unlock()
    {
        SP_ACCESS(m_pMutex)->Unlock();
    }
private:
    friend class BaseFeature;
    friend class Condition;
    friend class ConditionHelper;
    friend class MutexGuard;

    MutexPtr& GetMutex();
    const MutexPtr& GetMutex() const;

    MutexPtr m_pMutex;
};

} //namespace VmbCPP

#endif 