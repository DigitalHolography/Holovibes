/*! \file
 *
 * \brief Enum for the different return code of the API
 */
#pragma once

namespace holovibes
{
/*! \enum ApiCode
 *
 * \brief Describe the different return code of the API
 */
enum class ApiCode
{
    OK = 0,          /*!< Everything went well */
    NO_CHANGE,       /*!< No change was made (set with same value) */
    NO_IN_DATA,      /*!< Can't start because their is no source (file/camera) */
    NOT_STARTED,     /*!< The operation could not be performed because the computation has not started */
    WRONG_COMP_MODE, /*!< The operation could not be performed on this computation mode */
    WRONG_IMG_TYPE,  /*!< The operation could not be performed on this computation mode */
    FAILURE          /*!< An error occurred */
};
} // namespace holovibes