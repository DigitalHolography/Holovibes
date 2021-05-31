#pragma once

namespace holovibes::io_files
{
class AviFile
{
  protected:
    /*!
     *  \brief    Default constructor
     */
    AviFile() = default;

    /*!
     *  \brief    Abstract destructor to make class abstract
     */
    virtual ~AviFile() = 0;

    /*!
     *  \brief    Default copy constructor
     */
    AviFile(const AviFile&) = default;

    /*!
     *  \brief    Default copy operator
     */
    AviFile& operator=(const AviFile&) = default;
};
} // namespace holovibes::io_files

#include "avi_file.hxx"
