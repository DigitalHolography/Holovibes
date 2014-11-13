#include "stdafx.h"
#include "pipeline.hh"

#include <cassert>
#include "fft1.cuh"
#include "fft2.cuh"

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
    /* Clean current vector. */
    fn_vect_.clear();

    if (compute_desc_.algorithm == ComputeDescriptor::FFT1)
    {
      fft1_lens(
        res_.get_lens(),
        res_.get_input_queue().get_frame_desc(),
        compute_desc_.lambda,
        compute_desc_.zdistance);

      fn_vect_.push_back(std::bind(
        fft_1,
        res_.get_pbuffer(),
        std::ref(res_.get_input_queue()),
        res_.get_lens(),
        res_.get_sqrt_vector(),
        res_.get_plan3d(),
        compute_desc_.nsamples));
    }
    else if (compute_desc_.algorithm == ComputeDescriptor::FFT2)
    {
      fft2_lens(
        res_.get_lens(),
        res_.get_input_queue().get_frame_desc(),
        compute_desc_.lambda,
        compute_desc_.zdistance);
    }
    else
      assert(!"Impossible case.");

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

      unsigned short* output_frame = res_.get_pbuffer() +
        compute_desc_.pindex * res_.get_input_queue().get_pixels();
      res_.get_output_queue().enqueue(output_frame, cudaMemcpyDeviceToDevice);
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
}