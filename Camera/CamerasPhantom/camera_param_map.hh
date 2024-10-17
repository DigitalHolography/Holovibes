/*! \file camera_param_map.hh
 *
 * \brief data structure for parameters camera handling
 */
#pragma once

#include <string>
#include <map>
#include "spdlog/spdlog.h"
#include "camera.hh"

namespace camera
{
/*! \class CameraParamInt
 *
 * \brief Interface of CameraParam<\T> for manipulate each CameraParam regardless of the specific type \T
 */
class CameraParamInt
{
  public:
    /*! \brief Update the value from the ini file
     *
     * Must be pure in the interface
     *
     * \param pt property tree (ini file)
     */
    virtual void set_from_ini(const boost::property_tree::ptree& pt) = 0;
};

/*! \class CameraParam
 *
 * \brief Container of a value with its name and ini file information
 *
 * \tparam T type of the value stored
 */
template <typename T>
class CameraParam : public CameraParamInt
{
  public:
    /*! \brief Constructor of CameraParam
     *
     * \param value value
     * \param name name of the parameter must be the same on the ini file (if this parameter is inside of the ini file)
     * \param prefix prefix of the camera inside of the ini file
     * \param is_inside_ini if false set_from_ini do nothing
     */
    CameraParam(T value, std::string name, std::string prefix, bool is_inside_ini = true);

    /*! \brief Getter of value_
     *
     * \return value_
     */
    T get_value();

    /*! \brief Setter of value_
     *
     * \param value new value
     */
    void set_value(T value);

    /*! \brief Update the value from the ini file
     *
     * \param pt property tree (ini file)
     */
    void set_from_ini(const boost::property_tree::ptree& pt) override;

  private:
    /*! \brief value stored */
    T value_;
    /*! \brief name of the parameter */
    std::string name_;
    /*! \brief prefix of the camera inside of the ini file */
    std::string prefix_;
    /*! \brief false if the parameter is not inside of the ini file */
    bool is_inside_ini_;
};

/*! \class CameraParam
 *
 * \brief Map of CameraParam
 */
class CameraParamMap
{
  public:
    /*! \brief Constructor of CameraParamMap
     *
     * \param prefix prefix of the camera inside of the ini file
     */
    CameraParamMap(std::string prefix);

    /*! \brief Destructor of CameraParamMap, delete each CameraParam */
    ~CameraParamMap();

    /*! \brief check if the map contain the key
     *
     * \param key name of the parameter
     * \return true if map_ contains key
     */
    bool has(const std::string key) const;

    /*! \brief return the value of the parameter \key if its present
     *
     * \tparam T type of return value
     * \param key name of the parameter
     * \return optional containing the value of the parameter \key if its present
     */
    template <typename T>
    std::optional<T> get(const std::string key) const;

    /*! \brief return the value of the parameter \key
     *
     * /!\ Pay attention this function will abort the program if the type is wrong or the key not present!!!
     *
     * \tparam T type of return value
     * \param key name of the parameter
     * \return the value of the parameter \key
     */
    template <typename T>
    T at(const std::string key) const;

    /*! \brief Set the value of the parameter \key of insert it on the map
     *
     * /!\ Pay attention this function will abort the program if the key present and type is wrong !!!
     *
     * \tparam T type of the value
     * \param key name of the parameter
     * \param value new value
     * \param is_inside_ini false if the parameter is not inside of the ini file (if the key is already present this
     * argument is useless)
     */
    template <typename T>
    void set(const std::string key, T value, bool is_inside_ini = true);

    /*! \brief Update all parameter with value from the ini file
     *
     * \param pt property tree (ini file)
     */
    void set_from_ini(const boost::property_tree::ptree& pt);

  private:
    /*! \brief prefix of the camera inside of the ini file */
    std::string prefix_;
    /*! \brief map of CameraParam */
    std::map<const std::string, CameraParamInt*> map_;
};

} // namespace camera

#include "camera_param_map.hxx"
