/*=============================================================================
  Copyright (C) 2012-2022 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        Mutex.h

  Description: Definition of class VmbCPP::Mutex.

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

#ifndef VMBCPP_MUTEX
#define VMBCPP_MUTEX

/**
* \file    Mutex.h
*
* \brief   Definition of class VmbCPP::Mutex.
*/

#include <VmbCPP/VmbCPPCommon.h>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <pthread.h>
#endif


namespace VmbCPP {

/**
 * \brief a mutex implementation 
 */
class Mutex
{
public:
    /**
     * \brief creates a mutex that may be locked initially 
     */
    IMEXPORT explicit Mutex( bool bInitLock = false );

    /**
     * \brief mutexes are non-copyable 
     */
    Mutex& operator=(const Mutex&) = delete;

    /**
     * \brief mutexes are not copy-assignable. 
     */
    Mutex(const Mutex&) = delete;

    /**
     * \brief destroys the mutex
     */
    IMEXPORT ~Mutex();

    /**
     * \brief Lock this mutex.
     *
     * The call blocks, until the mutex is available
     */
    IMEXPORT void Lock();

    /**
     * \brief Release the lock on this mutex .
     */
    IMEXPORT void Unlock();

protected:
#ifdef _WIN32
      /**
       * \brief windows handle for the mutex 
       */
    HANDLE          m_hMutex;
#else
    /**
     * \brief the pthread handle for the mutex 
     */
    pthread_mutex_t m_Mutex;
#endif
};

} //namespace VmbCPP

#endif //VMBCPP_MUTEX
