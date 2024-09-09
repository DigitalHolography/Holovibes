/*=============================================================================
  Copyright (C) 2012-2022 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        FeatureContainer.h

  Description: Definition of class VmbCPP::FeatureContainer.

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

#ifndef VMBCPP_FEATURECONTAINER_H
#define VMBCPP_FEATURECONTAINER_H

/**
* \file  FeatureContainer.h
*
* \brief Definition of class VmbCPP::FeatureContainer.
*/

#include <cstddef>

#include <VmbC/VmbCommonTypes.h>

#include "BasicLockable.h"
#include "Feature.h"
#include "SharedPointerDefines.h"
#include "UniquePointer.hpp"
#include "VmbCPPCommon.h"


namespace VmbCPP {

/**
 * \brief A entity providing access to a set of features
 */
class FeatureContainer : protected virtual BasicLockable
{
public:

    /**  
    *  \brief     Creates an instance of class FeatureContainer
    */  
    IMEXPORT FeatureContainer();

    /**
     * \brief Object is non-copyable
     */
    FeatureContainer( const FeatureContainer& ) = delete;

    /**
     * \brief Object is non-copyable
     */
    FeatureContainer& operator=( const FeatureContainer& ) = delete;

    /**
    *  \brief     Destroys an instance of class FeatureContainer
    */  
    IMEXPORT ~FeatureContainer();

    /**
    *  \brief     Gets one particular feature of a feature container (e.g. a camera)
    *  
    *  \param[in ]    pName               The name of the feature to get
    *  \param[out]    pFeature            The queried feature
    *  
    *  \returns ::VmbErrorType
    *  
    *  \retval ::VmbErrorSuccess           If no error
    *  \retval ::VmbErrorDeviceNotOpen     Base feature class (e.g. Camera) was not opened.
    *  \retval ::VmbErrorBadParameter      \p pName is null.
    */  
    IMEXPORT VmbErrorType GetFeatureByName( const char *pName, FeaturePtr &pFeature );

    /**
     * \brief the feature name must be non-null 
     */
    VmbErrorType GetFeatureByName(std::nullptr_t, FeaturePtr&) = delete;
    
    /**
    *  \brief     Gets all features of a feature container (e.g. a camera)
    * 
    * Once queried, this information remains static throughout the object's lifetime
    *  
    *  \param[out]    features        The container for all queried features
    *  
    *  \returns ::VmbErrorType
    *  
    *  \retval ::VmbErrorSuccess        If no error
    *  \retval ::VmbErrorBadParameter   \p features is empty.
    */  
    VmbErrorType GetFeatures( FeaturePtrVector &features );

    /**
     * \brief Gets the handle used for this container by the Vmb C API. 
     */
    VmbHandle_t GetHandle() const noexcept;

  protected:
    /**
    * \brief    Sets the C handle of a feature container
    */
    IMEXPORT void SetHandle( const VmbHandle_t handle );
    
    /**
    * \brief    Sets the C handle of a feature container to null
    */
    void RevokeHandle() noexcept;
    
    /**
    * \brief    Sets the back reference to feature container that each feature holds to null and resets all known features
    */
    void Reset();
  private:
    struct Impl;
    UniquePointer<Impl> m_pImpl;

    IMEXPORT VmbErrorType GetFeatures( FeaturePtr *pFeatures, VmbUint32_t &size );

};


} // namespace VmbCPP

#include "FeatureContainer.hpp"

#endif
