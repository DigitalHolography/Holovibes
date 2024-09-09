/*=============================================================================
  Copyright (C) 2012-2022 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

-------------------------------------------------------------------------------

  File:        Feature.h

  Description:  Definition of base class VmbCPP::Feature.
                This class wraps every call to BaseFeature resp. its concrete
                subclass. That way  polymorphism is hidden away from the user.
                

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

#ifndef VMBCPP_FEATURE_H
#define VMBCPP_FEATURE_H

/**
* \file   Feature.h
*
* \brief  Definition of base class VmbCPP::Feature
*         This class wraps every call to BaseFeature resp. its concrete
*         subclass. That way  polymorphism is hidden away from the user.
*/

#include <cstddef>
#include <cstring>
#include <map>
#include <type_traits>
#include <vector>

#include <VmbC/VmbCTypeDefinitions.h>

#include "EnumEntry.h"
#include "IFeatureObserver.h"
#include "SharedPointerDefines.h"
#include "VmbCPPCommon.h"

struct VmbFeatureInfo;

namespace VmbCPP {

class BaseFeature;
class FeatureContainer;

/**
 * \brief a type alias for a vector of shared pointers to features. 
 */
using FeaturePtrVector = std::vector<FeaturePtr>;

/**
 * \brief Namespace containing helper functionality for inline functions
 *
 * \warning The members of this namespace are intended exclusively for use
 *          in the implementation of VmbCPP.
 */
namespace impl
{
    /**
     * \brief A helper class for getting the underlying type of a enum type 
     */
    template<class T, bool isEnum>
    struct UnderlyingTypeHelperImpl
    {
    };

    /**
     * \brief A helper class for getting the underlying type of a enum type
     */
    template<class T>
    struct UnderlyingTypeHelperImpl<T, true>
    {
        /**
         * \brief the underlying type of enum type `T` 
         */
        using type = typename std::underlying_type<T>::type;
    };

    /**
     * \brief Helper class for safely detemining the underlying type of an
     *        enum for use with SFINAE
     * 
     * The type provides a type alias `type` containing the underlying type, if and only if
     * the `T` is an enum type.
     * 
     * \tparam T the type that is possibly an enum
     */
    template<class T>
    struct UnderlyingTypeHelper : public UnderlyingTypeHelperImpl<T, std::is_enum<T>::value>
    {
    };
}

/**
 * \brief Class providing access to one feature of one module. 
 */
class Feature final
{
public:

    /**
     * \brief Object is not default constructible
     */
    Feature() = delete;

    /**
     * \brief Object is not copy constructible
     */
    Feature( const Feature& ) = delete;

    /**
     * \brief Object is not copy constructible
     */
    Feature& operator=( const Feature& ) = delete;

    /**
    *  \brief     Queries the value of a feature of type Integer or Enumeration
    *  
    *  \param[out]    value       The feature's value
    * 
    *  \returns ::VmbErrorType
    */ 
    IMEXPORT    VmbErrorType GetValue( VmbInt64_t &value ) const noexcept;
    
    /**
    *  \brief     Queries the value of a feature of type Float
    *  
    *  \param[out]    value       The feature's value
    * 
    *  \returns ::VmbErrorType
    */  
    IMEXPORT    VmbErrorType GetValue( double &value ) const noexcept;

    /**
    * \brief     Queries the value of a feature of type String or Enumeration
    * 
    * \param[out]    value       The feature's value
    * 
    * \returns ::VmbErrorType
    * 
    */
    VmbErrorType GetValue( std::string &value ) const noexcept;

    /**
    *  \brief     Queries the value of a feature of type Bool
    *  
    *  
    *  \param[out]    value       The feature's value
    * 
    *  \returns ::VmbErrorType
    */ 
    IMEXPORT    VmbErrorType GetValue( bool &value ) const noexcept;

    /**
    *  \brief     Queries the value of a feature of type Register
    *  
    *  \param[out]    value       The feature's value
    * 
    *  \returns ::VmbErrorType
    */ 
    VmbErrorType GetValue( UcharVector &value ) const noexcept;

    /**
    *  \brief     Queries the value of a feature of type const Register
    *  
    *  \param[out]    value       The feature's value
    *  \param[out]    sizeFilled  The number of actually received values
    * 
    *  \returns ::VmbErrorType
    */ 
    VmbErrorType GetValue( UcharVector &value, VmbUint32_t &sizeFilled ) const noexcept;

    /**
    *  \brief     Queries the possible integer values of a feature of type Enumeration
    *  
    *  \param[out]    values       The feature's values
    * 
    *  \returns ::VmbErrorType
    */ 
    VmbErrorType GetValues( Int64Vector &values ) noexcept;

    /**
    *  \brief     Queries the string values of a feature of type Enumeration
    *  
    *  \param[out]    values       The feature's values
    * 
    *  \returns ::VmbErrorType
    */ 
    VmbErrorType GetValues( StringVector &values ) noexcept;

    /**
    *  \brief     Queries a single enum entry of a feature of type Enumeration
    *  
    *  \param[out]    entry       An enum feature's enum entry
    *  \param[in ]    pEntryName  The name of the enum entry
    * 
    *  \returns ::VmbErrorType
    */ 
    IMEXPORT    VmbErrorType GetEntry( EnumEntry &entry, const char *pEntryName ) const noexcept;

    /**
    *  \brief     Queries all enum entries of a feature of type Enumeration
    *  
    *  \param[out]    entries       An enum feature's enum entries
    * 
    *  \returns ::VmbErrorType
    */ 
    VmbErrorType GetEntries( EnumEntryVector &entries ) noexcept;

    /**
    *  \brief     Queries the range of a feature of type Float
    *  
    *  \param[out]    minimum   The feature's min value
    *  \param[out]    maximum   The feature's max value
    * 
    *  \returns ::VmbErrorType
    */ 
    IMEXPORT    VmbErrorType GetRange( double &minimum, double &maximum ) const noexcept;

    /**
    *  \brief     Queries the range of a feature of type Integer
    *  
    *  \param[out]    minimum   The feature's min value
    *  \param[out]    maximum   The feature's max value
    * 
    *  \returns ::VmbErrorType
    */ 
    IMEXPORT    VmbErrorType GetRange( VmbInt64_t &minimum, VmbInt64_t &maximum ) const noexcept;

    /**
    *  \brief     Sets and integer or enum feature.
    * 
    * If the feature is an enum feature, the value set is the enum entry
    * corresponding to the integer value.
    * 
    * If known, use pass the string value instead, since this is more performant.
    *  
    *  \param[in ]    value       The feature's value
    * 
    *  \returns ::VmbErrorType
    */
    IMEXPORT    VmbErrorType SetValue( VmbInt64_t value ) noexcept;

    /**
     * \brief Convenience function for calling SetValue(VmbInt64_t)
     *        with an integral value without the need to cast the parameter to VmbInt64_t.
     *
     * Calls `SetValue(static_cast<VmbInt64_t>(value))`
     *
     * \tparam IntegralType an integral type other than ::VmbInt64_t
     */
    template<class IntegralType>
    typename std::enable_if<std::is_integral<IntegralType>::value && !std::is_same<IntegralType, VmbInt64_t>::value, VmbErrorType>::type
        SetValue(IntegralType value) noexcept;

    /**
     * \brief Convenience function for calling SetValue(VmbInt64_t)
     *        with an enum value without the need to cast the parameter to VmbInt64_t.
     *
     * Calls `SetValue(static_cast<VmbInt64_t>(value))`
     *
     * \tparam EnumType an enum type that with an underlying type other than `bool`.
     */
    template<class EnumType>
    typename std::enable_if<!std::is_same<bool, typename impl::UnderlyingTypeHelper<EnumType>::type>::value, VmbErrorType>::type
        SetValue(EnumType value) noexcept;

    /**
    *  \brief     Sets the value of a feature float feature
    *  
    *  \param[in ]    value       The feature's value
    * 
    *  \returns ::VmbErrorType
    */ 
    IMEXPORT    VmbErrorType SetValue( double value ) noexcept;

    /**
    *  \brief     Sets the value of a string feature or an enumeration feature.
    *  
    *  \param[in ]    pValue       The feature's value
    * 
    *  \returns ::VmbErrorType
    */ 
    IMEXPORT    VmbErrorType SetValue( const char *pValue ) noexcept;

    /**
     * \brief null is not allowed as string value 
     */
    VmbErrorType SetValue(std::nullptr_t) noexcept = delete;

    /**
    *  \brief     Sets the value of a feature of type Bool
    *  
    *  \param[in ]    value       The feature's value
    * 
    *  \returns ::VmbErrorType
    */ 
    IMEXPORT    VmbErrorType SetValue( bool value ) noexcept;

    /**
    *  \brief     Sets the value of a feature of type Register
    *  
    *  \param[in ]    value       The feature's value
    * 
    *  \returns ::VmbErrorType
    */ 
    VmbErrorType SetValue( const UcharVector &value ) noexcept;

    /**
    *  \brief     Checks, if a Float or Integer feature provides an increment.
    * 
    * Integer features are always assumed to provide an incement, even if it defaults to 0.
    *  
    *  \param[out]    incrementSupported       The feature's increment support state
    * 
    *  \returns ::VmbErrorType
    */ 
    IMEXPORT    VmbErrorType HasIncrement( VmbBool_t &incrementSupported ) const noexcept;


    /**
    *  \brief     Gets the increment of a feature of type Integer
    *  
    *  \param[out]    increment       The feature's increment
    * 
    *  \returns ::VmbErrorType
    */ 
    IMEXPORT    VmbErrorType GetIncrement( VmbInt64_t &increment ) const noexcept;

    /**
    *  \brief     Gets the increment of a feature of type Float
    *  
    *  \param[out]    increment       The feature's increment
    * 
    *  \returns ::VmbErrorType
    */ 
    IMEXPORT    VmbErrorType GetIncrement( double &increment ) const noexcept;

    /**
    *  \brief     Indicates whether an existing enumeration value is currently available.
    * 
    * An enumeration value might not be available due to the module's
    * current configuration.
    *  
    *  \param[in ]        pValue      The enumeration value as string
    *  \param[out]        available   True when the given value is available
    *  
    *  \returns ::VmbErrorType
    *  
    *  \retval ::VmbErrorSuccess        If no error
    *  \retval ::VmbErrorInvalidValue   If the given value is not a valid enumeration value for this enum
    *  \retval ::VmbErrorApiNotStarted  VmbStartup() was not called before the current command
    *  \retval ::VmbErrorInvalidAccess  Operation is invalid with the current access mode
    *  \retval ::VmbErrorWrongType      The feature is not an enumeration
    */  
    IMEXPORT    VmbErrorType IsValueAvailable( const char *pValue, bool &available ) const noexcept;

    /**
     * \brief Searching for an enum entry given the null string is not allowed 
     */
    VmbErrorType IsValueAvailable( std::nullptr_t, bool&) const noexcept = delete;

    /**
    *  \brief     Indicates whether an existing enumeration value is currently available.
    * 
    * An enumeration value might not be selectable due to the module's
    * current configuration.
    *  
    *  \param[in ]        value       The enumeration value as int
    *  \param[out]        available   True when the given value is available
    *  
    *  \returns ::VmbErrorType
    *  
    *  \retval ::VmbErrorSuccess        If no error
    *  \retval ::VmbErrorInvalidValue   If the given value is not a valid enumeration value for this enum
    *  \retval ::VmbErrorApiNotStarted  VmbStartup() was not called before the current command
    *  \retval ::VmbErrorInvalidAccess  Operation is invalid with the current access mode
    *  \retval ::VmbErrorWrongType      The feature is not an enumeration
    */  
    IMEXPORT    VmbErrorType IsValueAvailable( VmbInt64_t value, bool &available ) const noexcept;

    /**
    *  
    *  \brief     Executes a feature of type Command
    * 
    *  \returns ::VmbErrorType
    */ 
    IMEXPORT    VmbErrorType RunCommand() noexcept;

    /**
    *  
    *  \brief     Checks if the execution of a feature of type Command has finished
    *  
    *  \param[out]    isDone     True when execution has finished
    * 
    *  \returns ::VmbErrorType
    */ 
    IMEXPORT    VmbErrorType IsCommandDone( bool &isDone ) const noexcept;

    /**
    *  \brief     Queries a feature's name
    * 
    * \note The feature name does not change during the lifecycle of the object.
    *  
    * \param[out]    name    The feature's name
    * 
    * \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess
    * \retval ::VmbErrorResources
    */ 
    VmbErrorType GetName( std::string &name ) const noexcept;

    /**
    *  \brief     Queries a feature's display name
    * 
    * \note The display name does not change during the lifecycle of the object.
    * 
    *  \param[out]    displayName    The feature's display name
    * 
    *  \returns ::VmbErrorType
    * 
    * \retval ::VmbErrorSuccess
    * \retval ::VmbErrorResources
    */ 
    VmbErrorType GetDisplayName( std::string &displayName ) const noexcept;

    /**
    * \brief     Queries a feature's type
    *  
    * \note The feature type does not change during the lifecycle of the object.
    * 
    * \param[out]    dataType    The feature's type
    * 
    * \returns ::VmbErrorType
    */ 
    IMEXPORT    VmbErrorType GetDataType( VmbFeatureDataType &dataType ) const noexcept;

    /** 
    * \brief     Queries a feature's access status
    * 
    * The access to the feature may change depending on the state of the module.
    * 
    *  \param[out]    flags    The feature's access status
    * 
    *  \returns ::VmbErrorType
    */ 
    IMEXPORT    VmbErrorType GetFlags( VmbFeatureFlagsType &flags ) const noexcept;

    /**
    *  \brief     Queries a feature's category in the feature tree
    * 
    * \note The category does not change during the lifecycle of the object.
    *  
    *  \param[out]    category    The feature's position in the feature tree
    * 
    *  \returns ::VmbErrorType
    */ 
    VmbErrorType GetCategory( std::string &category ) const noexcept;

    /**
    *  \brief     Queries a feature's polling time
    *  
    *  \param[out]    pollingTime    The interval to poll the feature
    * 
    *  \returns ::VmbErrorType
    */ 
    IMEXPORT    VmbErrorType GetPollingTime( VmbUint32_t &pollingTime ) const noexcept;

    /**
    *  \brief     Queries a feature's unit
    *
    * \note The display name does not change during the lifecycle of the object.
    * 
    * Information about the unit used is only available for features of type Integer or Float.
    * For other feature types the empty string is returned.
    * 
    *  \param[out]    unit    The feature's unit
    * 
    *  \returns ::VmbErrorType
    */ 
    VmbErrorType GetUnit( std::string &unit ) const noexcept;

    /**
    *  \brief     Queries a feature's representation
    *
    * \note The representation does not change during the lifecycle of the object.
    * 
    * Information about the representation used is only available for features of type Integer or Float.
    * For other feature types the empty string is returned.
    * 
    *  \param[out]    representation    The feature's representation
    * 
    *  \returns ::VmbErrorType
    */ 
    VmbErrorType GetRepresentation( std::string &representation ) const noexcept;

    /**
    *  \brief     Queries a feature's visibility
    * 
    * \note The visibiliry does not change during the lifecycle of the object.
    * 
    *  \param[out]    visibility    The feature's visibility
    * 
    *  \returns ::VmbErrorType
    */ 
    IMEXPORT    VmbErrorType GetVisibility( VmbFeatureVisibilityType &visibility ) const noexcept;

    /**
    *  \brief     Queries a feature's tooltip to display in the GUI
    * 
    * \note The tooltip does not change during the lifecycle of the object.
    *  
    *  \param[out]    toolTip    The feature's tool tip
    * 
    *  \returns ::VmbErrorType
    */ 
    VmbErrorType GetToolTip( std::string &toolTip ) const noexcept;

    /**
    *  \brief     Queries a feature's description
    *
    * \note The description does not change during the lifecycle of the object.
    * 
    *  \param[out]    description    The feature's description
    * 
    *  \returns ::VmbErrorType
    */ 
    VmbErrorType GetDescription( std::string &description ) const noexcept;

    /**
    *  \brief     Queries a feature's Standard Feature Naming Convention namespace
    * 
    * \note The namespace does not change during the lifecycle of the object.
    * 
    *  \param[out]    sFNCNamespace    The feature's SFNC namespace
    * 
    *  \returns ::VmbErrorType
    */ 
    VmbErrorType GetSFNCNamespace( std::string &sFNCNamespace ) const noexcept;

    /**
    *  \brief     Gets the features that get selected by the current feature
    *
    * \note The selected features do not change during the lifecycle of the object.
    * 
    * \param[out]    selectedFeatures    The selected features
    * 
    *  \returns ::VmbErrorType
    */ 
    VmbErrorType GetSelectedFeatures( FeaturePtrVector &selectedFeatures ) noexcept;


    /**
    * \brief Retrieves info about the valid value set of an integer feature. Features of other types will retrieve an error ::VmbErrorWrongType.
    * 
    * \note Only some specific integer features support valid value sets.
    * 
    * \param[out]   validValues                  Vector of int64, after the call it contains the valid value set.
    * 
    *
    * \return An error code indicating success or the type of error that occured.
    *
    * \retval ::VmbErrorSuccess                        The call was successful 
    *
    * \retval ::VmbErrorApiNotStarted                  ::VmbStartup() was not called before the current command
    *
    * \retval ::VmbErrorBadHandle                      The current feature handle is not valid
    *
    * \retval ::VmbErrorWrongType                      The type of the feature is not Integer
    *
    * \retval ::VmbErrorValidValueSetNotPresent        The feature does not provide a valid value set
    * 
    * \retval ::VmbErrorResources                      Resources not available (e.g. memory)
    *
    * \retval ::VmbErrorOther                          Some other issue occured
    */
    VmbErrorType GetValidValueSet(Int64Vector& validValues) const noexcept;

    /**
    *  \brief     Queries the read access status of a feature
    *  
    *  \param[out]    isReadable    True when feature can be read
    * 
    *  \returns ::VmbErrorType
    */ 
    IMEXPORT    VmbErrorType IsReadable( bool &isReadable ) noexcept;

    /**
    *  
    *  \brief     Queries the write access status of a feature
    *  
    *  \param[out]    isWritable    True when feature can be written
    * 
    *  \returns ::VmbErrorType
    */ 
    IMEXPORT    VmbErrorType IsWritable( bool &isWritable ) noexcept;

    /**
    *  \brief     Queries whether a feature's should be persisted to store the state of a module.
    * 
    * \note The information does not change during the lifecycle of the object.
    *  
    *  \param[out]    isStreamable    True when streamable
    * 
    *  \returns ::VmbErrorType
    */ 
    IMEXPORT    VmbErrorType IsStreamable( bool &isStreamable ) const noexcept;

    /**
    *  \brief     Registers an observer that notifies the application whenever a feature is invalidated
    * 
    * \note A feature may be invalidated even though the value hasn't changed. A notification of the observer
    *       just provides a notification that the value needs to be reread, to be sure the current module value
    *       is known.
    *  
    *  \param[out]    pObserver    The observer to be registered
    *  
    *  \returns ::VmbErrorType
    *  
    *  \retval ::VmbErrorSuccess        If no error
    *  \retval ::VmbErrorBadParameter   \p pObserver is null.
    *  \retval ::VmbErrorAlready        \p pObserver is already registered
    *  \retval ::VmbErrorDeviceNotOpen  Device is not open (FeatureContainer is null)
    *  \retval ::VmbErrorInvalidCall    If called from a chunk access callback
    *  \retval ::VmbErrorInvalidAccess  Operation is invalid with the current access mode
    */  
    IMEXPORT    VmbErrorType RegisterObserver( const IFeatureObserverPtr &pObserver );

    /**
    *  \brief     Unregisters an observer
    *  
    *  \param[out]    pObserver    The observer to be unregistered
    *  
    *  \returns ::VmbErrorType
    *  
    *  \retval ::VmbErrorSuccess        If no error
    *  \retval ::VmbErrorBadParameter   \p pObserver is null.
    *  \retval ::VmbErrorUnknown        \p pObserver is not registered
    *  \retval ::VmbErrorDeviceNotOpen  Device is not open (FeatureContainer is null)
    *  \retval ::VmbErrorInvalidCall    If called from a chunk access callback
    *  \retval ::VmbErrorInvalidAccess  Operation is invalid with the current access mode
    *  \retval ::VmbErrorInternalFault  Could not lock feature observer list for writing.
    */  
    IMEXPORT    VmbErrorType UnregisterObserver( const IFeatureObserverPtr &pObserver );


private:
    Feature(const VmbFeatureInfo& pFeatureInfo, FeatureContainer& pFeatureContainer);
    ~Feature();

    friend class FeatureContainer;
    friend SharedPointer<Feature>;

    void ResetFeatureContainer();

    BaseFeature *m_pImpl;

    //   Array functions to pass data across DLL boundaries
    IMEXPORT    VmbErrorType GetValue( char * const pValue, VmbUint32_t &length ) const noexcept;
    IMEXPORT    VmbErrorType GetValue( VmbUchar_t *pValue, VmbUint32_t &size, VmbUint32_t &sizeFilled ) const noexcept;
    IMEXPORT    VmbErrorType GetValues( const char **pValues, VmbUint32_t &size ) noexcept;
    IMEXPORT    VmbErrorType GetValues( VmbInt64_t *pValues, VmbUint32_t &Size ) noexcept;

    IMEXPORT    VmbErrorType GetEntries( EnumEntry *pEnumEntries, VmbUint32_t &size ) noexcept;

    IMEXPORT    VmbErrorType SetValue( const VmbUchar_t *pValue, VmbUint32_t size ) noexcept;

    IMEXPORT    VmbErrorType GetName( char * const pName, VmbUint32_t &length ) const noexcept;
    IMEXPORT    VmbErrorType GetDisplayName( char * const pDisplayName, VmbUint32_t &length ) const noexcept;
    IMEXPORT    VmbErrorType GetCategory( char * const pCategory, VmbUint32_t &length ) const noexcept;
    IMEXPORT    VmbErrorType GetUnit( char * const pUnit, VmbUint32_t &length ) const noexcept;
    IMEXPORT    VmbErrorType GetRepresentation( char * const pRepresentation, VmbUint32_t &length ) const noexcept;
    IMEXPORT    VmbErrorType GetToolTip( char * const pToolTip, VmbUint32_t &length ) const noexcept;
    IMEXPORT    VmbErrorType GetDescription( char * const pDescription, VmbUint32_t &length ) const noexcept;
    IMEXPORT    VmbErrorType GetSFNCNamespace( char * const pSFNCNamespace, VmbUint32_t &length ) const noexcept;
    IMEXPORT    VmbErrorType GetSelectedFeatures( FeaturePtr *pSelectedFeatures, VmbUint32_t &nSize ) noexcept;
    IMEXPORT    VmbErrorType GetValidValueSet(VmbInt64_t* buffer, VmbUint32_t bufferSize, VmbUint32_t* setSize) const noexcept;
};


} // namespace VmbCPP

#include "Feature.hpp"

#endif
