#pragma once

#include <unordered_map>
#include <vector>
#include <functional>
#include <memory>
#include "parameter.hh"
#include "static_container.hh"
#include "logger.hh"

#include "batch_size.hh"

namespace holovibes
{
class ParametersHandler
{
  protected:
    MapKeyParams key_params_;
    StaticContainer<BatchSize> params_;

  public:
    ParametersHandler()
        : key_params_()
        , params_(key_params_)
    {
    }

  public:
    void force_sync_with(ParametersHandler& handler) { params_.force_sync_with(handler.params_); }

    template <typename FunctionClass, typename... Args>
    void call(Args&&... args)
    {
        FunctionClass functions_class;
        params_.call(functions_class, std::forward<Args>(args)...);
    }

  public:
    template <typename T>
    const T& get_type() const
    {
        return params_.get<T>();
    }

    template <typename T>
    T& get_type()
    {
        return params_.get<T>();
    }

    template <typename T>
    typename T::TransfertType get_value() const
    {
        return params_.get<T>().get_value();
    }

    template <typename T>
    typename T::ValueType& get_value()
    {
        return params_.get<T>().get_value();
    }

    template <typename T>
    void set_value(T&& type)
    {
        params_.get<T>().set_value(type.get_value());
    }
};

struct ParamsChange
{
  public:
    IParameter* ref = nullptr;
    IParameter* param_to_change = nullptr;
};

class SetSynchronize
{
  public:
    template <typename T>
    void call(T& value)
    {
        value.set_has_been_synchronized(false);
    }
};

class ParametersHandlerCache : public ParametersHandler
{
  public:
    ParametersHandlerCache()
        : ParametersHandler()
        , change_pool{}
    {
    }

    template <typename T>
    void add_params_change(IParameter* ref)
    {
        change_pool.push_back(ParamsChange{ref, &get_type<T>()});
    }

  public:
    void synchronize()
    {
        for (auto change : change_pool)
        {
            change.param_to_change->sync_with(change.ref);
            change.param_to_change->set_has_been_synchronized(true);
        }
        change_pool.clear();
    }

    template <typename FunctionClass, typename... Args>
    void call_synchronize(Args&&... args)
    {
        SetSynchronize set_synchronize;
        params_.template call(set_synchronize);

        synchronize();

        FunctionClass functions_class;
        params_.template call(functions_class, std::forward<Args>(args)...);
    }

    std::vector<ParamsChange>& get_change_pool() { return change_pool; }
    const std::vector<ParamsChange>& get_change_pool() const { return change_pool; }

  private:
    std::vector<ParamsChange> change_pool;
};

class ParametersHandlerRef : public ParametersHandler
{
  public:
    ParametersHandlerRef()
        : ParametersHandler()
        , caches_to_sync_{}
    {
    }

  public:
    void add_cache_to_synchronize(ParametersHandlerCache& cache)
    {
        caches_to_sync_.push_back(&cache);
        cache.force_sync_with(*this);
    }

    template <typename T>
    void add_params_change()
    {
        IParameter* ref = &get_type<T>();
        for (auto cache : caches_to_sync_)
            cache->add_params_change<T>(ref);
    }

  public:
    template <typename T>
    void set(T&& value)
    {
        LOG_DEBUG(main, "++++++ SET VALUE ON REF");
        add_params_change<T>();
        ParametersHandler::set_value<T>(std::forward<T>(value));
    }

  private:
    std::vector<ParametersHandlerCache*> caches_to_sync_;
};

} // namespace holovibes
