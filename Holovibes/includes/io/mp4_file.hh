/*! \file mp4_file.hh
 *
 * \brief This file contains the Mp4File base class (inherited for writing mp4 files).
 */
#pragma once

namespace holovibes::io_files
{
/*! \class Mp4File
 *
 * \brief Base class of mp4 files.
 */
class Mp4File
{
  protected:
    /*! \brief Default constructor */
    Mp4File() = default;

    /*! \brief Abstract destructor to make class abstract */
    virtual ~Mp4File() {};

    /*! \brief Default copy constructor */
    Mp4File(const Mp4File&) = default;

    /*! \brief Default copy operator */
    Mp4File& operator=(const Mp4File&) = default;
};
} // namespace holovibes::io_files
