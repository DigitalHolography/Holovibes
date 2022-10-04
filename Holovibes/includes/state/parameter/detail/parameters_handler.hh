#pragma once

#include <unordered_map>
#include <vector>
#include <deque>
#include <map>
#include <functional>
#include <memory>
#include <utility>
#include <mutex>
#include <thread>
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
    virtual void synchronize(){};
    void force_sync_with(ParametersHandler& handler) { params_.force_sync_with(handler.params_); }

  public:
    template <typename FunctionClass, typename... Args>
    void call(Args&&... args)
    {
        FunctionClass functions_class;

        if constexpr (requires { typename FunctionClass::BeforeMethods; })
        {
            typename FunctionClass::BeforeMethods before;
            params_.call(before);
            before.template call(*this);
        }

        params_.call(functions_class, std::forward<Args>(args)...);

        if constexpr (requires { typename FunctionClass::AfterMethods; })
        {
            typename FunctionClass::AfterMethods after;
            params_.call(after);
            after.template call(*this);
        }
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

  public:
    template <typename T>
    void setter(T& old_value, T&& new_value)
    {
        old_value = std::forward<T>(new_value);
    }

    template <typename T>
    void set_value(T&& value)
    {
        setter<T>(params_.get<T>(), std::forward<T>(value));
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
    void trigger_param(IParameter* ref)
    {
        std::lock_guard<std::mutex> guard(change_pool_mutex);
        IParameter* param_to_change = static_cast<IParameter*>(&ParametersHandler::get_type<T>());
        change_pool[param_to_change] = ref;
    }

  public:
    void synchronize() override
    {
        std::lock_guard<std::mutex> guard(change_pool_mutex);
        for (auto change : change_pool)
        {
            change.first->sync_with(change.second);
            change.second->set_has_been_synchronized(true);
        }
        change_pool.clear();
    }

    bool has_change_requested()
    {
        std::lock_guard<std::mutex> guard(change_pool_mutex);
        return change_pool.size() > 0;
    }

  private:
    // first is param_to_change ; second is ref
    std::map<IParameter*, IParameter*> change_pool;
    std::mutex change_pool_mutex;
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
        caches_to_sync_.insert(&cache);
        cache.force_sync_with(*this);
    }

    void remove_cache_to_synchronize(ParametersHandlerCache& cache) { caches_to_sync_.erase(&cache); }

    template <typename T>
    void trigger_params()
    {
        IParameter* ref = &get_type<T>();
        for (auto cache : caches_to_sync_)
            cache->trigger_param<T>(ref);
    }

  public:
    template <typename T>
    void set_value(T&& value)
    {
        setter<T>(params_.get<T>(), std::forward<T>(value));
        trigger_params<T>();
    }

  private:
    std::set<ParametersHandlerCache*> caches_to_sync_;
};

} // namespace holovibes
