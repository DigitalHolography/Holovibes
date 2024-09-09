/*=============================================================================
  Copyright (C) 2012-2022 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        EnumEntry.h

  Description:  Definition of class VmbCPP::EnumEntry.
                An EnumEntry consists of
                Name
                DisplayName
                Value
                of one particular enumeration

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

#ifndef VMBCPP_ENUMENTRY_H
#define VMBCPP_ENUMENTRY_H

/**
* \file          EnumEntry.h
*
* \brief         Definition of class VmbCPP::EnumEntry
*                An EnumEntry consists of:
*                   - Name
*                   - DisplayName
*                   - Value
*                
*                of one particular enumeration.
*/

#include <string>

#include <VmbC/VmbC.h>

#include "SharedPointerDefines.h"
#include "UniquePointer.hpp"
#include "VmbCPPCommon.h"


namespace VmbCPP {

/**
 * \brief A class holding information about a single enum entry of a feature of type enumeration. 
 */
class EnumEntry final
{
public:
    /**
    * 
    * \brief     Creates an instance of class EnumEntry
    * 
    * \param[in ]    pName           The name of the enum
    * \param[in ]    pDisplayName    The declarative name of the enum
    * \param[in ]    pDescription    The description of the enum
    * \param[in ]    pTooltip        A tooltip that can be used by a GUI
    * \param[in ]    pSNFCNamespace  The SFNC namespace of the enum
    * \param[in ]    visibility      The visibility of the enum
    * \param[in ]    value           The integer value of the enum
    */ 
    EnumEntry(  const char              *pName,
                const char              *pDisplayName,
                const char              *pDescription,
                const char              *pTooltip,
                const char              *pSNFCNamespace,
                VmbFeatureVisibility_t  visibility,
                VmbInt64_t              value);

    /**
    * \brief     Creates an instance of class EnumEntry
    */ 
    IMEXPORT EnumEntry();

    /**
    * \brief Creates a copy of class EnumEntry
    */ 
    IMEXPORT EnumEntry(const EnumEntry &other);
    
    /**
     * \brief assigns EnumEntry to existing instance
     */ 
    IMEXPORT EnumEntry& operator=(const EnumEntry &other);

    /**
    * \brief     Destroys an instance of class EnumEntry
    */ 
    IMEXPORT ~EnumEntry() noexcept;

    /**
    * \brief     Gets the name of an enumeration
    * 
    * \param[out]    name   The name of the enumeration
    *
    * \returns ::VmbErrorType
    */   
    VmbErrorType GetName( std::string &name ) const noexcept;

    /**
    * \brief     Gets a more declarative name of an enumeration
    * 
    * \param[out]    displayName    The display name of the enumeration
    *
    * \returns ::VmbErrorType
    */    
    VmbErrorType GetDisplayName( std::string &displayName ) const noexcept;

    /**
    * \brief     Gets the description of an enumeration
    * 
    * \param[out]    description    The description of the enumeration
    *
    * \returns ::VmbErrorType
    */    
    VmbErrorType GetDescription( std::string &description ) const noexcept;

    /**
    * \brief     Gets a tooltip that can be used as pop up help in a GUI
    * 
    * \param[out]    tooltip    The tooltip as string
    *
    * \returns ::VmbErrorType
    */    
    VmbErrorType GetTooltip( std::string &tooltip ) const noexcept;

    /**
    * \brief     Gets the integer value of an enumeration
    * 
    * \param[out]    value   The integer value of the enumeration
    *
    * \returns ::VmbErrorType
    */    
    IMEXPORT    VmbErrorType GetValue( VmbInt64_t &value ) const noexcept;

    /**
    * \brief     Gets the visibility of an enumeration
    * 
    * \param[out]    value   The visibility of the enumeration
    *
    * \returns ::VmbErrorType
    */    
    IMEXPORT    VmbErrorType GetVisibility( VmbFeatureVisibilityType &value ) const noexcept;

    /**
    * \brief     Gets the standard feature naming convention namespace of the enumeration
    * 
    * \param[out]    sFNCNamespace    The feature's SFNC namespace
    *
    * \returns ::VmbErrorType
    */    
    VmbErrorType GetSFNCNamespace( std::string &sFNCNamespace ) const noexcept;

private:
    struct PrivateImpl;
    UniquePointer<PrivateImpl>  m_pImpl;

    //  Array functions to pass data across DLL boundaries
    
    IMEXPORT VmbErrorType GetName( char * const pName, VmbUint32_t &size ) const noexcept;
    IMEXPORT VmbErrorType GetDisplayName( char * const pDisplayName, VmbUint32_t &size ) const noexcept;
    IMEXPORT VmbErrorType GetTooltip( char * const pStrTooltip, VmbUint32_t &size ) const noexcept;
    IMEXPORT VmbErrorType GetDescription( char * const pStrDescription, VmbUint32_t &size ) const noexcept;
    IMEXPORT VmbErrorType GetSFNCNamespace( char * const pStrNamespace, VmbUint32_t &size ) const noexcept;

};

} // namespace VmbCPP

#include "EnumEntry.hpp"

#endif
