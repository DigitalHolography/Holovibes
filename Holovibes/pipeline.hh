#ifndef PIPELINE_HH
# define PIPELINE_HH

# include <vector>
# include <functional>

# include "queue.hh"
# include "compute_descriptor.hh"
# include "pipeline_resources.hh"

namespace holovibes
{
  class Pipeline
  {
    using FnVector = std::vector<std::function<void()>>;
  public:
    Pipeline(
      Queue& input,
      Queue& output,
      ComputeDescriptor& desc);
    virtual ~Pipeline();

    void request_refresh();
    void request_autofocus();
    void request_autocontrast();
    void request_update_n(unsigned short n);
    void exec();
  private:
    void refresh();

    Pipeline& operator=(const Pipeline&) = delete;
    Pipeline(const Pipeline&) = delete;
  private:
    FnVector fn_vect_;
    ComputeDescriptor& compute_desc_;
    PipelineResources res_;

    bool autofocus_requested_;
    bool autocontrast_requested_;
    bool update_n_requested_;
  };
}

#endif /* !PIPELINE_HH */
