/*! \file output_hdf5_file.hh
 *
 * \brief This file contains the OutputHdf5File class, responsible for exporting data to an HDF5 file.
 */
#pragma once

#include "output_frame_file.hh"
#include "H5Cpp.h"
#include <complex>

namespace holovibes::io_files
{
/*! \class OutputHdf5File
 *
 * \brief Class responsible for exporting data to an HDF5 file.
 */
class OutputHdf5File : public OutputFrameFile
{
  public:
    size_t get_total_nb_frames() const override { return img_nb_; }

    void export_compute_settings(int input_fps, size_t contiguous) override {}

    void write_header() override;

    size_t write_frame(const char* frame, size_t frame_size) override;

    void write_footer() override;

    void correct_number_of_frames(size_t nb_frames_written) override { img_nb_ = nb_frames_written; }

  private:
    friend class OutputFrameFileFactory;
    OutputHdf5File(const std::string& file_path, const camera::FrameDescriptor& fd, uint64_t img_nb);

    size_t img_nb_;
    H5::H5File h5_file_;
    H5::DataSet dataset_;
    hsize_t current_frame_;
    hsize_t current_cube_;
    hsize_t current_depth_;
};
} // namespace holovibes::io_files
