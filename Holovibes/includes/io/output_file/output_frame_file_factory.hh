/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "output_frame_file.hh"
#include "file_exception.hh"
#include "enum_recorded_data_type.hh"

namespace holovibes::io_files
{
class OutputFrameFile;

/*! \class OutputFrameFileFactory
 *
 * \brief Used to create an output file
 *
 * \details This class is a factory,
 *          the created input file depends on the file path extension
 */
class OutputFrameFileFactory
{
  public:
    /*! \brief Deleted default constructor */
    OutputFrameFileFactory() = delete;

    /*! \brief Deleted default destructor */
    ~OutputFrameFileFactory() = delete;

    /*! \brief Deleted default copy constructor */
    OutputFrameFileFactory(const OutputFrameFileFactory&) = delete;

    /*! \brief Deleted default copy operator */
    OutputFrameFileFactory& operator=(const OutputFrameFileFactory&) = delete;

    /*! \brief    Create an output file
     *
     * This methods allocates the output file attribute.
     * Thus, it must be called before the other methods
     *
     * \param file_path The path of the file to create, the extension must be supported
     * \throw FileException if the OutputFrameFile is not created or if the file extension is not supported
     */
    static OutputFrameFile* create(const std::string& file_path,
                                   const camera::FrameDescriptor& fd,
                                   uint64_t img_nb,
                                   RecordedDataType data_type);
};
} // namespace holovibes::io_files
