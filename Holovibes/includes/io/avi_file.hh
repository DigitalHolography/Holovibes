/*! \file avi_file.hh
 *
 * \brief This file contains the AviFile base class definition (inherited for writing avi files).
 */
#pragma once

namespace holovibes::io_files
{
/*! \class AviFile
 *
 * \brief A base class for the AviFile class.
 */
class AviFile
{
  protected:
    /*! \brief Default constructor */
    AviFile() = default;

    /*! \brief Abstract destructor to make class abstract */
    virtual ~AviFile(){};

    /*! \brief Default copy constructor */
    AviFile(const AviFile&) = default;

    /*! \brief Default copy operator */
    AviFile& operator=(const AviFile&) = default;
};
} // namespace holovibes::io_files
