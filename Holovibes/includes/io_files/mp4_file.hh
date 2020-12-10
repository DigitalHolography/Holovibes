/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

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
