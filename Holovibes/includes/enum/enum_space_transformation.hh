/*! \file
 *
 * \brief Enum for the different space transformations
 */
#pragma once
#include <map>

namespace holovibes
{
/*! \enum SpaceTransformation
 *
 * \brief Rendering mode for Hologram (Space transformation)
 */
enum class SpaceTransformation
{
    NONE = 0, /*!< Nothing Applied */
    FFT1,     /*!< Fresnel Transform */
    FFT2      /*!< Angular spectrum propagation */
};

namespace _internal
{

const std::map<std::string, SpaceTransformation> string_to_space_transform{
    {"None", SpaceTransformation::NONE}, {"1FFT", SpaceTransformation::FFT1}, {"2FFT", SpaceTransformation::FFT2}};

const std::map<SpaceTransformation, std::string> space_transform_to_string{
    {SpaceTransformation::NONE, "None"}, {SpaceTransformation::FFT1, "1FFT"}, {SpaceTransformation::FFT2, "2FFT"}};

} // namespace _internal

inline std::string space_transformation_to_string(SpaceTransformation value)
{
    return _internal::space_transform_to_string.at(value);
}

inline SpaceTransformation space_transformation_from_string(const std::string& in)
{
    return _internal::string_to_space_transform.at(in);
}

inline std::ostream& operator<<(std::ostream& os, holovibes::SpaceTransformation value)
{
    return os << space_transformation_to_string(value);
}

} // namespace holovibes
