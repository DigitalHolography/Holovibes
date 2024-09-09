/*=============================================================================
  Copyright (C) 2012-2022 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        IFeatureObserver.h

  Description: Definition of interface VmbCPP::IFeatureObserver.

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

#ifndef VMBCPP_IFEATUREOBSERVER_H
#define VMBCPP_IFEATUREOBSERVER_H

/**
* \file  IFeatureObserver.h
*
* \brief Definition of interface VmbCPP::IFeatureObserver.
*/

#include <VmbCPP/SharedPointerDefines.h>

namespace VmbCPP {

/**
 * \brief The base class to derive feature invalidation listeners from.
 * 
 * Derived classes must implement IFeatureObserver::FeatureChanged .
 */
class IFeatureObserver 
{
  public:
    /**
    * \brief     The event handler function that gets called whenever
    *            a feature has changed
    *
    * \param[in]    pFeature    The feature that has changed
    */
    IMEXPORT virtual void FeatureChanged( const FeaturePtr &pFeature ) = 0;

    /**
    * \brief     Destroys an instance of class IFeatureObserver
    */
    IMEXPORT virtual ~IFeatureObserver() {}

protected:

    /**
     * \brief Default constructor for use by derived classes. 
     */
    IMEXPORT IFeatureObserver() {}
    
    /**
     * \brief Copy constructor for use by derived classes.
     */
    IMEXPORT IFeatureObserver( const IFeatureObserver& ) { }

    /**
     * \brief Copy assignment operator for use by derived classes.
     */
    IMEXPORT IFeatureObserver& operator=( const IFeatureObserver& )
    {
        return *this;
    }
};


} // namespace VmbCPP

#endif
