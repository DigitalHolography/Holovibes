/*=============================================================================
  Copyright (C) 2012-2022 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        IInterfaceListObserver.h

  Description: Definition of interface VmbCPP::IInterfaceListObserver.

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

#ifndef VMBCPP_IINTERFACELISTOBSERVER_H
#define VMBCPP_IINTERFACELISTOBSERVER_H

/**
* \file  IInterfaceListObserver.h
*
* \brief Definition of interface VmbCPP::IInterfaceListObserver.
*/

#include <VmbCPP/VmbCPPCommon.h>
#include <VmbCPP/SharedPointerDefines.h>
#include <VmbCPP/Interface.h>
#include <vector>


namespace VmbCPP {

/**
 * \brief Base class for Observers of the list of interfaces 
 */
class IInterfaceListObserver 
{
  public:
    /**
    * \brief     The event handler function that gets called whenever
    *            an IInterfaceListObserver is triggered.
    * 
    * \param[out]    pInterface              The interface that triggered the event
    * \param[out]    reason                  The reason why the callback routine was triggered
    */ 
    IMEXPORT virtual void InterfaceListChanged( InterfacePtr pInterface, UpdateTriggerType reason ) = 0;

    /**
    * \brief     Destroys an instance of class IInterfaceListObserver
    */ 
    IMEXPORT virtual ~IInterfaceListObserver() {}

  protected:
    /**
     * \brief Constructor for use of derived classes
     */
    IMEXPORT IInterfaceListObserver() {}

    /**
     * \brief Copy constructor for use by derived classes.
     */
    IMEXPORT IInterfaceListObserver( const IInterfaceListObserver& ) {}

    /**
     * \brief copy assignment operator for use by derived classes. 
     */
    IMEXPORT IInterfaceListObserver& operator=( const IInterfaceListObserver& ) { return *this; }
    
};

} // namespace VmbCPP

#endif
