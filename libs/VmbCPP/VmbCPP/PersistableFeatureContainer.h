/*=============================================================================
  Copyright (C) 2022 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        PersistableFeatureContainer.h

  Description: Definition of class VmbCPP::PersistableFeatureContainer.

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

#ifndef VMBCPP_PERSISTABLEFEATURECONTAINER_H
#define VMBCPP_PERSISTABLEFEATURECONTAINER_H

/**
* \file             PersistableFeatureContainer.h
*
* \brief            Definition of class VmbCPP::PersistableFeatureContainer.
*/

#include <cstddef>
#include <string>

#include <VmbC/VmbC.h>
#include <VmbCPP/VmbCPPCommon.h>
#include <VmbCPP/FeatureContainer.h>

namespace VmbCPP {

/**
 * \brief An interface providing access and persistance functionality for features.
 */
class PersistableFeatureContainer : public FeatureContainer
{
public:
    /**
    *  \brief     Creates an instance of class FeatureContainer
    */
    IMEXPORT PersistableFeatureContainer();
  
    /**
     * \brief Object is not copyable
     */
    PersistableFeatureContainer(const PersistableFeatureContainer&) = delete;
    
    /**
     * \brief Object is not copyable
     */
    PersistableFeatureContainer& operator=(const PersistableFeatureContainer&) = delete;
  
    /**
    * 
    * \brief     Saves the current module setup to an XML file
    * 
    * \param[in ]   filePath        Path of the XML file
    * \param[in ]   pSettings       Pointer to settings struct
    * 
    * \returns ::VmbErrorType
    * \retval ::VmbErrorSuccess            If no error
    * \retval ::VmbErrorInvalidCall        If called from a chunk access callback
    * \retval ::VmbErrorBadParameter       If \p filePath is or the settings struct is invalid
    * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
    * \retval ::VmbErrorBadHandle          The object handle is not valid
    * \retval ::VmbErrorNotFound           The object handle is insufficient to identify the module that should be saved
    * \retval ::VmbErrorInvalidAccess      Operation is invalid with the current access mode
    * \retval ::VmbErrorIO                 There was an issue writing the file.
    */
    IMEXPORT VmbErrorType SaveSettings(const VmbFilePathChar_t* filePath, VmbFeaturePersistSettings_t* pSettings = nullptr) const noexcept;

    /**
     * \brief Settings cannot be saved given null as file path
     */
    VmbErrorType SaveSettings(std::nullptr_t, VmbFeaturePersistSettings_t* pSettings = nullptr) const noexcept = delete;

    /**
    * 
    * \brief     Loads the current module setup from an XML file into the camera
    * 
    * \param[in] filePath        Name of the XML file
    * \param[in] pSettings       Pointer to settings struct
    * 
    * \return An error code indicating success or the type of error that occured.
    * \retval ::VmbErrorSuccess            If no error
    * \retval ::VmbErrorApiNotStarted      ::VmbStartup() was not called before the current command
    * \retval ::VmbErrorInvalidCall        If called from a chunk access callback
    * \retval ::VmbErrorBadHandle          The object handle is not valid
    * \retval ::VmbErrorAmbiguous          The module to restore the settings for cannot be uniquely identified based on the information available
    * \retval ::VmbErrorNotFound           The object handle is insufficient to identify the module that should be restored
    * \retval ::VmbErrorRetriesExceeded    Some or all of the features could not be restored with the max iterations specified
    * \retval ::VmbErrorInvalidAccess      Operation is invalid with the current access mode
    * \retval ::VmbErrorBadParameter       If \p filePath is null or the settings struct is invalid
    * \retval ::VmbErrorIO                 There was an issue with reading the file.
    */ 
    IMEXPORT VmbErrorType LoadSettings(const VmbFilePathChar_t* const filePath, VmbFeaturePersistSettings_t* pSettings = nullptr) const noexcept;

    /**
     * \brief Loading settings requires a non-null path 
     */
    VmbErrorType LoadSettings(std::nullptr_t, VmbFeaturePersistSettings_t* pSettings= nullptr) const noexcept = delete;
private:

};

    
} // namespace VmbCPP

#endif