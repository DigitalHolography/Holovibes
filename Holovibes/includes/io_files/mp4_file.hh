/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

namespace holovibes::io_files
{
class Mp4File
{
  protected:
    /*!
     *  \brief    Default constructor
     */
    Mp4File() = default;

    /*!
     *  \brief    Abstract destructor to make class abstract
     */
    virtual ~Mp4File() = 0;

    /*!
     *  \brief    Default copy constructor
     */
    Mp4File(const Mp4File&) = default;

    /*!
     *  \brief    Default copy operator
     */
    Mp4File& operator=(const Mp4File&) = default;
};
} // namespace holovibes::io_files

#include "mp4_file.hxx"
