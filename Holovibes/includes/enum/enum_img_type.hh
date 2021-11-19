
/*! \file
 *
 * \brief Enum for the different type of displaying images
 */
#pragma once

namespace holovibes
{
/*! \enum ImgType
 *
 * \brief Displaying type of the image
 */
enum class ImgType
{
    Modulus = 0,    /*!< Modulus of the complex data */
    SquaredModulus, /*!<  Modulus taken to its square value */
    Argument,       /*!<  Phase (angle) value of the complex pixel c, computed with atan(Im(c)/Re(c)) */
    PhaseIncrease,  /*!<  Phase value, the conjugate between the phase of the last image and the previous one */
    Composite       /*!<  Displays different frequency intervals on color RBG or HSV chanels*/
};

namespace _internal
{

const std::map<ImgType, std::string> img_type_to_string{{ImgType::Modulus, "Magnitude"},
                                                        {ImgType::SquaredModulus, "Squared magnitude"},
                                                        {ImgType::Argument, "Argument"},
                                                        {ImgType::PhaseIncrease, "Phase increase"},
                                                        {ImgType::Composite, "Composite image"}};

const std::map<std::string, ImgType> string_to_img_type{{"Magnitude", ImgType::Modulus},
                                                        {"Squared magnitude", ImgType::SquaredModulus},
                                                        {"Argument", ImgType::Argument},
                                                        {"Phase increase", ImgType::PhaseIncrease},
                                                        {"Composite image", ImgType::Composite}};

} // namespace _internal

inline std::string img_type_to_string(ImgType value) { return _internal::img_type_to_string.at(value); }

inline ImgType img_type_from_string(const std::string& in) { return _internal::string_to_img_type.at(in); }

inline std::ostream& operator<<(std::ostream& os, holovibes::ImgType value) { return os << img_type_to_string(value); }

} // namespace holovibes