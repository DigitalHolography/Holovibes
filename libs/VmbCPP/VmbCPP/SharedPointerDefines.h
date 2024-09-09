/*=============================================================================
  Copyright (C) 2012 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        SharedPointerDefines.h

  Description: Definition of macros for using the standard shared pointer
               (std::tr1) for VmbCPP.

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

#ifndef VMBCPP_SHAREDPOINTERDEFINES_H
#define VMBCPP_SHAREDPOINTERDEFINES_H

/**
* \file SharedPointerDefines.h
* \brief Definition of macros for using the standard shared pointer (std::tr1) for VmbCPP.
*
* \note If your version of STL does not provide a shared pointer implementation please see UserSharedPointerDefines.h for information on
* how to use another shared pointer than std::shared_ptr.
*
*/

// include the implementation of the shared pointer
#ifndef USER_SHARED_POINTER
    #include <VmbCPP/SharedPointer.h>
#else
// expected to specialize ::VmbCPP::UsedSharedPointerDefinitions
    #ifndef __APPLE__
        #include "VmbCPPConfig/config.h"
    #else
        #include <VmbCPP/config.h>
    #endif
#endif

namespace VmbCPP {

// These are all uses of a SharedPointer shared_ptr type alias
class BasicLockable;

/**
 * \brief An alias for a shared pointer to a BasicLockable.
 */
using BasicLockablePtr = SharedPointer<BasicLockable>;

class Camera;

/**
 * \brief An alias for a shared pointer to a Camera.
 */
using CameraPtr = SharedPointer<Camera>;

class Feature;

/**
 * \brief An alias for a shared pointer to a Feature.
 */
using FeaturePtr = SharedPointer<Feature>;

class FeatureContainer;

/**
 * \brief An alias for a shared pointer to a FeatureContainer.
 */
using FeatureContainerPtr = SharedPointer<FeatureContainer>;

class Frame;

/**
 * \brief An alias for a shared pointer to a Frame.
 */
using FramePtr = SharedPointer<Frame>;

class FrameHandler;

/**
 * \brief An alias for a shared pointer to a FrameHandler.
 */
using FrameHandlerPtr = SharedPointer<FrameHandler>;

class ICameraFactory;

/**
 * \brief An alias for a shared pointer to a camera factory.
 */
using ICameraFactoryPtr = SharedPointer<ICameraFactory>;

class ICameraListObserver;

/**
 * \brief An alias for a shared pointer to a camera list observer.
 */
using ICameraListObserverPtr = SharedPointer<ICameraListObserver>;

class IFeatureObserver;

/**
 * \brief An alias for a shared pointer to a feature observer.
 */
using IFeatureObserverPtr = SharedPointer<IFeatureObserver>;

class IFrameObserver;

/**
 * \brief An alias for a shared pointer to a frame observer.
 */
using IFrameObserverPtr = SharedPointer<IFrameObserver>;

class Interface;

/**
 * \brief An alias for a shared pointer to an Interface.
 */
using InterfacePtr = SharedPointer<Interface>;

class IInterfaceListObserver;

/**
 * \brief An alias for a shared pointer to an interface list observer.
 */
using IInterfaceListObserverPtr = SharedPointer<IInterfaceListObserver>;

class LocalDevice;

/**
 * \brief An alias for a shared pointer to a LocalDevice.
 */
using LocalDevicePtr = SharedPointer<LocalDevice>;

class Mutex;

/**
 * \brief An alias for a shared pointer to a Mutex.
 */
using MutexPtr = SharedPointer<Mutex>;

class Stream;

/**
 * \brief An alias for a shared pointer to a Stream.
 */
using StreamPtr = SharedPointer<Stream>;

class TransportLayer;

/**
 * \brief An alias for a shared pointer to a TransportLayer.
 */
using TransportLayerPtr = SharedPointer<TransportLayer>;

}  // namespace VmbCPP

#endif // VMBCPP_SHAREDPOINTERDEFINES_H
