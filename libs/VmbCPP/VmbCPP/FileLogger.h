/*=============================================================================
  Copyright (C) 2012-2022 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        FileLogger.h

  Description: Definition of class VmbCPP::FileLogger.

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

#ifndef VMBCPP_FILELOGGER_H
#define VMBCPP_FILELOGGER_H

/**
* \file        FileLogger.h
*
* \brief Definition of class VmbCPP::FileLogger.
*/

#include <cstddef>
#include <string>
#include <fstream>

#ifndef __APPLE__
    #include "VmbCPPConfig/config.h"
#else
    #include <VmbCPP/config.h>
#endif

#include <VmbCPP/SharedPointerDefines.h>
#include <VmbCPP/Mutex.h>


namespace VmbCPP {

/**
 * \brief A logger implementation logging to a file.
 */
class FileLogger
{
public:
    /**
     * \param[in] pFileName     the file name of the log file
     * \param[in] append        determines, if the contents of an existing file are kept or not
     */
    FileLogger( const char *pFileName, bool append = true );

    /**
     * \brief Object is not copyable
     */
    FileLogger(const FileLogger&) = delete;

    /**
     * \brief Object is not copyable
     */
    FileLogger& operator=(const FileLogger&) = delete;

    /**
     * \brief null is not allowed as file name
     */
    FileLogger(std::nullptr_t, bool) = delete;

    virtual ~FileLogger();

    /**
     * \brief Log the time and \p StrMessage to the file
     *
     * \param[in] StrMessage the message to log
     */
    void Log( const std::string &StrMessage );

private:
    std::ofstream   m_File;
    MutexPtr        m_pMutex;

    std::string GetTemporaryDirectoryPath();
};

} //namespace VmbCPP

#endif
