/*!
    \file circular_video_buffer.hh
*/
#pragma once

#include "unique_ptr.hh"

namespace holovibes
{
/*!
 * \brief A circular buffer that stores a fixed number of video frames.
 *
 * This class implements a circular buffer that stores a fixed number of
 * video frames, where the oldest frame is overwritten by the newest frame
 * when the buffer is full.
 *
 * The class provides methods for adding new frames to the buffer,
 * accessing the oldest and newest frames, and checking the size.
 */
class CircularVideoBuffer
{
  public:
    /*!
     * \brief Constructs a CircularVideoBuffer object.
     *
     * This constructor initializes a CircularVideoBuffer object with the specified frame resolution, buffer capacity,
     * and CUDA stream. It allocates the necessary GPU memory buffers and initializes member variables.
     *
     * \param [in] frame_res The resolution of each frame (number of elements per frame).
     * \param [in] buffer_capacity The capacity of the buffer (number of frames it can hold).
     * \param [in] stream The CUDA stream to use for memory operations and kernel launches.
     *
     * \note The constructor allocates GPU memory for the internal buffer, sum image, mean image, and a buffer for
     *       computing the mean. It also initializes the sum image to zero.
     */
    CircularVideoBuffer(const size_t frame_res, const size_t frame_capacity, cudaStream_t stream);

    /*!
     * \brief Destructs a CircularVideoBuffer object.
     *
     * This destructor releases the GPU memory buffers allocated by the CircularVideoBuffer object.
     * It ensures that all dynamically allocated resources are properly freed to avoid memory leaks.
     *
     * \note The destructor calls the `reset` method on the GPU memory buffers to release the allocated memory.
     */
    ~CircularVideoBuffer();

    /*!
     * \brief Copy constructor for CircularVideoBuffer.
     *
     * This copy constructor initializes a new CircularVideoBuffer object by copying the state and data from an existing
     * CircularVideoBuffer object. It allocates the necessary GPU memory buffers and copies the data from the reference
     * object.
     *
     * \param [in] ref The reference CircularVideoBuffer object to copy from.
     *
     * \note The constructor allocates GPU memory for the internal buffer, sum image, mean image, and a buffer for
     *       computing the mean. It also initializes the sum image to zero and copies the data from the reference object
     *       using an asynchronous memory copy operation.
     */
    CircularVideoBuffer(CircularVideoBuffer& ref);

    /*!
     * \brief Assignment operator for CircularVideoBuffer.
     *
     * This assignment operator copies the state and data from an existing CircularVideoBuffer object to the current
     * object. It performs an asynchronous memory copy operation to transfer the data from the reference object to the
     * current object.
     *
     * \param [in] ref The reference CircularVideoBuffer object to copy from.
     * \return A reference to the current CircularVideoBuffer object.
     *
     * \note The operator copies the internal state variables and performs asynchronous memory copy operations for the
     * data, sum image, and mean image buffers using the provided CUDA stream.
     */
    CircularVideoBuffer& operator=(CircularVideoBuffer& ref);

    /*!
     * \brief Retrieves a pointer to the first frame in the buffer.
     *
     * This function returns a pointer to the first frame in the circular video buffer.
     * If the buffer is empty (i.e., no frames have been added), it returns nullptr.
     *
     * \return A pointer to the first frame in the buffer, or nullptr if the buffer is empty.
     *
     * \note The function calculates the pointer to the first frame based on the start index and frame resolution.
     */
    float* get_first_frame();

    /*!
     * \brief Retrieves a pointer to the last frame in the buffer.
     *
     * This function returns a pointer to the last frame in the circular video buffer.
     * If the buffer is empty (i.e., no frames have been added), it returns nullptr.
     *
     * \return A pointer to the last frame in the buffer, or nullptr if the buffer is empty.
     *
     * \note The function calculates the pointer to the last frame based on the end index and buffer capacity.
     *       If the end index is 0, it returns the pointer to the last frame in the buffer.
     *       Otherwise, it returns the pointer to the frame just before the end index.
     */
    float* get_last_frame();

    /*!
     * \brief Computes the mean (temporal mean) image from the sum image and the number of frames.
     *
     * This function computes the mean image by dividing the sum image by the number of frames.
     * It uses a helper function `compute_mean` to perform the computation.
     *
     * \note The function assumes that the sum image has already been computed and stored in `sum_image_`.
     *       It also assumes that `nb_frames_` contains the number of frames used to compute the sum image.
     */
    void compute_mean_image();

    /*!
     * \brief Retrieves a pointer to the mean image.
     *
     * This function returns a pointer to the mean image stored in the CircularVideoBuffer.
     * The mean image is computed from the sum image and the number of frames.
     *
     * \return A pointer to the mean image.
     *
     * \note The function assumes that the mean image has already been computed and stored in `mean_image_`.
     */
    float* get_mean_image();

    /*!
     * \brief Adds a new frame to the circular video buffer.
     *
     * This function adds a new frame to the circular video buffer. If the buffer is full, it removes the oldest frame
     * to make room for the new frame. The function updates the sum image by subtracting the oldest frame (if necessary)
     * and adding the new frame. It also updates the start and end indices accordingly.
     *
     * \param [in] new_frame Pointer to the new frame to be added to the buffer.
     *
     * \note The function performs the following steps:
     *       1. Calculates the position for the new frame in the buffer.
     *       2. Updates the end index to point to the next position in the buffer.
     *       3. If the buffer is full, removes the oldest frame by subtracting it from the sum image and updating the
     * start index.
     *       4. Adds the new frame to the sum image.
     *       5. Copies the new frame to the calculated position in the buffer.
     *       6. Increments the number of frames if the buffer is not full.
     */
    void add_new_frame(const float* const new_frame);

    /*!
     * \brief Checks if the circular video buffer is full.
     *
     * This function checks whether the circular video buffer is full by comparing the number of frames currently in the
     * buffer with the buffer's capacity.
     *
     * \return True if the buffer is full, false otherwise.
     */
    bool is_full();

    /*!
     * \brief Retrieves the number of frames currently in the buffer.
     *
     * This function returns the number of frames currently stored in the circular video buffer.
     *
     * \return The number of frames in the buffer.
     */
    size_t get_frame_count();

    /*!
     * \brief Retrieves a pointer to the data buffer.
     *
     * This function returns a pointer to the internal data buffer of the CircularVideoBuffer.
     * The data buffer contains the frames stored in the circular video buffer.
     *
     * \return A pointer to the data buffer.
     */
    float* get_data_ptr();

    /*!
     * \brief Computes the mean of the element-wise multiplication of the buffer frames with a given frame.
     *
     * This function computes the mean of the element-wise multiplication of the frames stored in the circular video
     * buffer with a given frame. It initializes a buffer to store the intermediate results and then calls the
     * `compute_multiplication_mean` function to perform the computation.
     *
     * \param [in] frame Pointer to the frame to be multiplied with the buffer frames.
     *
     * \note The function initializes the `compute_mean_1_2_buffer_` to zero and then calls
     * `compute_multiplication_mean` to perform the multiplication and mean computation. The results are stored in
     * `compute_mean_1_2_buffer_`.
     */
    void compute_mean_1_2(float* frame);

    /*!
     * \brief Retrieves a pointer to the buffer containing the mean of the element-wise multiplication.
     *
     * This function returns a pointer to the buffer that stores the mean of the element-wise multiplication
     * of the frames in the circular video buffer with a given frame. This buffer is computed by the `compute_mean_1_2`
     * function.
     *
     * \return A pointer to the buffer containing the mean of the element-wise multiplication.
     *
     * \note The function assumes that the `compute_mean_1_2` function has already been called to populate the buffer.
     */
    float* get_mean_1_2_();

  private:
    /*! \brief Video of the last 'time_window_' frames */
    cuda_tools::UniquePtr<float> data_{};

    /*! \brief Index of the first image of the buffer */
    size_t start_index_;

    /*! \brief Index of the index AFTER the last image of the buffer */
    size_t end_index_;

    /*! \brief Number of frames currently stored */
    size_t nb_frames_;

    /*! \brief Max number of frames that the buffer can store */
    size_t buffer_capacity_;

    /*! \brief Resolution of one frame in pixels */
    size_t frame_res_;

    /*! \brief Size of one frame in bytes */
    size_t frame_size_;

    /*! \brief Image with each pixel value equal to the sum of each value at the same pixel in the buffer */
    cuda_tools::UniquePtr<float> sum_image_{};

    /*! \brief Image with each pixel value equal to the mean of each value at the same pixel in the buffer */
    cuda_tools::UniquePtr<float> mean_image_{};

    /*! \brief Video with each frame being a 1x1 pixel with value equal to the mean of pixel in its corresponding frame
     */
    cuda_tools::UniquePtr<float> compute_mean_1_2_buffer_{};

    /*! \brief Cuda stream used for async computations */
    cudaStream_t stream_;
};
} // namespace holovibes
