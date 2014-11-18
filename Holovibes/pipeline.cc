#include "stdafx.h"
#include "pipeline.hh"

#include <cassert>
#include "fft1.cuh"
#include "fft2.cuh"
#include "tools.cuh"

namespace holovibes
{
  Pipeline::Pipeline(
    Queue& input,
    Queue& output,
    ComputeDescriptor& desc)
    : fn_vect_()
    , compute_desc_(desc)
    , res_(input, output, desc.nsamples)
    , autofocus_requested_(false)
    , autocontrast_requested_(false)
  {
    refresh();
  }

  Pipeline::~Pipeline()
  {}

  void Pipeline::refresh()
  {
    const camera::FrameDescriptor& input_fd =
      res_.get_input_queue().get_frame_desc();
    const camera::FrameDescriptor& output_fd =
      res_.get_output_queue().get_frame_desc();

    /* Clean current vector. */
    fn_vect_.clear();

    if (update_n_requested_)
    {
      update_n_requested_ = false;
      res_.update_n_parameter(compute_desc_.nsamples);
    }

    if (compute_desc_.algorithm == ComputeDescriptor::FFT1)
    {
      // Initialize FFT1 lens.
      fft1_lens(
        res_.get_lens(),
        input_fd,
        compute_desc_.lambda,
        compute_desc_.zdistance);

      // Add FFT1.
      fn_vect_.push_back(std::bind(
        fft_1,
        res_.get_pbuffer(),
        std::ref(res_.get_input_queue()),
        res_.get_lens(),
        res_.get_sqrt_vector(),
        res_.get_plan3d(),
        compute_desc_.nsamples.load()));

      res_.set_output_frame_ptr(
        res_.get_pbuffer() + compute_desc_.pindex * output_fd.frame_res());
    }
    else if (compute_desc_.algorithm == ComputeDescriptor::FFT2)
    {
      fft2_lens(
        res_.get_lens(),
        input_fd,
        compute_desc_.lambda,
        compute_desc_.zdistance);
    }
    else
      assert(!"Impossible case.");

    /* [POSTPROCESSING] Everything behind this line uses output_frame_ptr */

    if (compute_desc_.shift_corners_enabled)
    {
      fn_vect_.push_back(std::bind(
        shift_corners,
        res_.get_output_frame_ptr(),
        output_fd.width,
        output_fd.height));
    }

    if (autofocus_requested_)
    {
      autofocus_requested_ = false;
      // push autofocus();
    }

    if (autocontrast_requested_)
    {
      autocontrast_requested_ = false;
      // push autocontrast();
    }

    if (autocontrast_requested_ ||
      autofocus_requested_)
    {
      request_refresh();
    }
  }

  void Pipeline::exec()
  {
    if (res_.get_input_queue().get_current_elts() >= compute_desc_.nsamples)
    {
      for (FnVector::const_iterator cit = fn_vect_.cbegin();
        cit != fn_vect_.cend();
        ++cit)
        (*cit)();

      res_.get_output_queue().enqueue(
        res_.get_output_frame_ptr(),
        cudaMemcpyDeviceToDevice);
      res_.get_input_queue().dequeue();
    }
  }

  void Pipeline::request_refresh()
  {
    fn_vect_.push_back(std::bind(&Pipeline::refresh, this));
  }

  void Pipeline::request_autocontrast()
  {
    autocontrast_requested_ = true;
    request_refresh();
  }

  void Pipeline::request_autofocus()
  {
    autofocus_requested_ = true;
    request_refresh();
  }

  void Pipeline::request_update_n(unsigned short n)
  {
    compute_desc_.nsamples = n;
    update_n_requested_ = true;
    request_refresh();
  }
}