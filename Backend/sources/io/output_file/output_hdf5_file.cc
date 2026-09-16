#include "output_hdf5_file.hh"
#include "file_exception.hh"
#include "API.hh"
#include <stdexcept>

namespace holovibes::io_files
{

OutputHdf5File::OutputHdf5File(const std::string& file_path, const camera::FrameDescriptor& fd, uint64_t img_nb)
    : OutputFrameFile(file_path)
    , img_nb_(img_nb)
    , current_frame_(0)
{
    // Create HDF5 file
    try
    {
        h5_file_ = H5::H5File(file_path, H5F_ACC_TRUNC);
        fd_ = fd;
    }
    catch (const H5::Exception&)
    {
        throw FileException("Failed to open/create HDF5 file", false);
    }
}

void OutputHdf5File::write_header()
{
    try
    {
        hsize_t depth =
            (API.record.get_record_mode() == RecordMode::MOMENTS) ? 3 : API.transform.get_time_transformation_size();

        hsize_t img_count = (API.record.get_record_mode() == RecordMode::MOMENTS)
                                ? img_nb_
                                : img_nb_ / API.transform.get_time_transformation_size();

        // [#images, height, width, depth]
        hsize_t dims[4] = {img_count, fd_.height, fd_.width, depth};

        H5::DataSpace dataspace(4, dims);

        hsize_t chunk_dims[4] = {1, fd_.height, fd_.width, depth};
        H5::DSetCreatPropList plist;
        plist.setChunk(4, chunk_dims);

        if (API.record.get_record_mode() == RecordMode::OCT_CUBE)
        {
            H5::CompType complex_type(sizeof(std::complex<float>));
            complex_type.insertMember("r", 0, H5::PredType::NATIVE_FLOAT);
            complex_type.insertMember("i", sizeof(float), H5::PredType::NATIVE_FLOAT);
            dataset_ = h5_file_.createDataSet("frames", complex_type, dataspace, plist);
        }
        else if (API.record.get_record_mode() == RecordMode::OCT_CUBE_FLOAT)
        {
            dataset_ = h5_file_.createDataSet("frames", H5::PredType::NATIVE_FLOAT, dataspace, plist);
        }
        else if (API.record.get_record_mode() == RecordMode::MOMENTS)
        {
            // [#images, height, width, 3]
            dataset_ = h5_file_.createDataSet("moments", H5::PredType::NATIVE_FLOAT, dataspace, plist);
        }
        else
        {
            throw FileException("Unsupported record mode for HDF5 output", false);
        }

        current_frame_ = 0;
    }
    catch (const H5::Exception&)
    {
        throw FileException("Failed to write HDF5 header", false);
    }
}

size_t OutputHdf5File::write_frame(const char* frame, size_t frame_size)
{
    try
    {
        H5::DataSpace filespace = dataset_.getSpace();
        hsize_t dims[4];
        filespace.getSimpleExtentDims(dims); // dims[3]

        hsize_t offset[4] = {current_frame_, 0, 0, 0};
        hsize_t count[4] = {1, fd_.height, fd_.width, dims[3]};

        filespace.selectHyperslab(H5S_SELECT_SET, count, offset);

        H5::DataSpace memspace(4, count);

        if (API.record.get_record_mode() == RecordMode::OCT_CUBE)
        {
            dataset_.write(frame, dataset_.getDataType(), memspace, filespace);
        }
        else // OCT_CUBE_FLOAT or MOMENTS
        {
            dataset_.write(frame, H5::PredType::NATIVE_FLOAT, memspace, filespace);
        }

        ++current_frame_;
    }
    catch (const H5::Exception&)
    {
        throw FileException("Unable to write frame to HDF5 file", false);
    }

    return frame_size;
}

void OutputHdf5File::write_footer()
{
    // Flush and add metadata (Used in OCT to annotate dataset).
    h5_file_.flush(H5F_SCOPE_GLOBAL);
}

} // namespace holovibes::io_files
