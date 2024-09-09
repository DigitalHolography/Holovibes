/*=============================================================================
  Copyright (C) 2012 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        LoggerDefines.h

  Description: Definition of macros for logging.

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

#ifndef VMBCPP_LOGGERDEFINES_H
#define VMBCPP_LOGGERDEFINES_H

/**
* \file      LoggerDefines.h
*
* \brief     Definition of macros for logging.
*/

#include <utility>

#ifndef __APPLE__
    #include "VmbCPPConfig/config.h"
#else
    #include <VmbCPP/config.h>
#endif
#ifndef USER_LOGGER

#include "FileLogger.h"

namespace VmbCPP {

    /**
     * \brief A type alias determining the logger type to be used by the VmbCPP API.
     */
    using Logger = FileLogger;

    /**
     * \brief The used to pass the log info on to the logger object.
     *
     * \param[in] logger        a pointer to the logger object; may be null resulting in the log message being dropped
     * \param[in] loggingInfo   the info that should be logged
     *
     * \tparam LoggingInfoType  the type of information to be forwarded to the logger.
     */
    template<typename LoggingInfoType>
    inline void LOGGER_LOG(Logger* logger, LoggingInfoType&& loggingInfo)
    {
        if (nullptr != logger)
        {
            logger->Log(std::forward<LoggingInfoType>(loggingInfo));
        }
    }

    /**
     * \brief Create a file logger object.
     *
     * The created logger appends log entries to VmbCPP.log in the temporary directory.
     *
     * \return a raw pointer to the newly created file object.
     */
    inline Logger* CreateLogger()
    {
        return new FileLogger("VmbCPP.log", true);
    }
}

#endif

#include <VmbCPP/VmbSystem.h>

/**
 * \brief Macro for logging the provided text.
 *
 * The function using the macro is logged too.
 *
 * \note May throw std::bad_alloc, if there is insufficient memory to create
 * log message string.
 */
#define LOG_FREE_TEXT( txt )                                                \
{                                                                           \
    std::string strExc( txt );                                              \
    strExc.append( " in function: " );                                      \
    strExc.append( __FUNCTION__ );                                          \
    LOGGER_LOG(VmbSystem::GetInstance().GetLogger(), std::move(strExc));    \
}

#define LOG_ERROR( txt, errCode )                                           \
{                                                                           \
    std::string strExc( txt );                                              \
    strExc.append( " in function: " );                                      \
    strExc.append( __FUNCTION__ );                                          \
    strExc.append( ", VmbErrorType: ");                                            \
    strExc.append( std::to_string(errCode) );                               \
    LOGGER_LOG(VmbSystem::GetInstance().GetLogger(), std::move(strExc));    \
}

#endif
