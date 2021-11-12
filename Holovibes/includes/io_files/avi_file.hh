/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

namespace holovibes::io_files
{
/*! \class AviFile
 *
 * \brief #TODO Add a description for this class
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
