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

namespace holovibes
{

template <typename... Params>
class ParametersHandler
{
  protected:
    MapKeyParams key_container_;
    StaticContainer<Params...> container_;

  public:
    ParametersHandler()
        : key_container_()
        , container_(key_container_)
    {
    }

  public:
    virtual void synchronize(){};

    template <typename ParametersHandlerRef>
    void force_sync_with(ParametersHandlerRef& ref)
    {
        container_.force_sync_with(ref.get_container());
    }

  public:
    template <typename FunctionClass, typename... Args>
    void call(Args&&... args)
    {
        FunctionClass functions_class;

        if constexpr (requires { typename FunctionClass::BeforeMethods; })
        {
            typename FunctionClass::BeforeMethods before;
            container_.call(before);
            before.template call_handler(*this);
        }

        container_.call(functions_class, std::forward<Args>(args)...);

        if constexpr (requires { typename FunctionClass::AfterMethods; })
        {
            typename FunctionClass::AfterMethods after;
            container_.call(after);
            after.template call_handler(*this);
        }
    }

  public:
    const MapKeyParams& get_map_key() const { return key_container_; }
    MapKeyParams& get_map_key() { return key_container_; }

    StaticContainer<Params...>& get_container() { return container_; }

  public:
    template <typename T>
    const T& get_type() const
    {
        return container_.template get<T>();
    }

    template <typename T>
    T& get_type()
    {
        return container_.template get<T>();
    }

    template <typename T>
    typename T::TransfertType get_value() const
    {
        return container_.template get<T>().get_value();
    }

    template <typename T>
    typename T::ValueType& get_value()
    {
        return container_.template get<T>().get_value();
    }
};

template <typename... Params>
class ParametersHandlerCache : public ParametersHandler<Params...>
{
  public:
    ParametersHandlerCache()
        : ParametersHandler<Params...>()
        , change_pool{}
    {
    }

    template <typename T>
    void trigger_param(IParameter* ref)
    {
        std::lock_guard<std::mutex> guard(change_pool_mutex);
        IParameter* param_to_change = static_cast<IParameter*>(&ParametersHandler<Params...>::template get_type<T>());
        change_pool[param_to_change] = ref;
    }

  public:
    void synchronize() override
    {
        std::lock_guard<std::mutex> guard(change_pool_mutex);
        for (auto change : change_pool)
            change.first->sync_with(change.second);
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

template <typename... T>
class CachesToSync
{
};

template <typename T>
class CachesToSync<T>
{
  public:
    using CacheType = T;
    using Next = void;
};

template <typename T, typename... R>
class CachesToSync<T, R...> : CachesToSync<R...>
{
  public:
    using CacheType = T;
    using Next = CachesToSync<R...>;
};

template <typename Setters, typename CachesToSync, typename... Params>
class BasicParametersHandlerRef : public BasicParametersHandlerRef<Setters, typename CachesToSync::Next, Params...>
{
  public:
    using Base = BasicParametersHandlerRef<Setters, typename CachesToSync::Next, Params...>;
    using CacheType = typename CachesToSync::CacheType;

    using Base::trigger_params_all;

  public:
    BasicParametersHandlerRef()
        : Base()
        , caches_to_sync_{}
    {
    }

  private:
    template <typename T>
    void trigger_params()
    {
        if (caches_to_sync_.size() == 0)
            return;
        IParameter* ref = &this->template get_type<T>();
        for (auto cache : caches_to_sync_)
            cache->template trigger_param<T>(ref);
    }

  protected:
    template <typename T>
    void trigger_params_all()
    {
        trigger_params<T>();
        Base::template trigger_params_all<T>();
    }

  public:
    void add_cache_to_synchronize(CacheType& cache)
    {
        caches_to_sync_.insert(&cache);
        cache.force_sync_with(*this);
    }

    void remove_cache_to_synchronize(CacheType& cache)
    {
        if (caches_to_sync_.erase(&cache))
            LOG_ERROR(main, "Maybe a problem here...");
    }

  private:
    std::set<CacheType*> caches_to_sync_;
};
template <typename Setters, typename... Params>
class BasicParametersHandlerRef<Setters, void, Params...> : public ParametersHandler<Params...>
{
  public:
    using Base = ParametersHandler<Params...>;

  public:
    BasicParametersHandlerRef()
        : ParametersHandler<Params...>()
    {
    }

  protected:
    template <typename T>
    void trigger_params_all()
    {
    }
};

template <typename Master, typename Setters, typename CachesToSync, typename... Params>
class ParametersHandlerRef : public BasicParametersHandlerRef<Setters, CachesToSync, Params...>
{
  public:
    using Base = BasicParametersHandlerRef<Setters, CachesToSync, Params...>;

  public:
    template <typename T>
    void default_setter(T& old_value, T&& new_value)
    {
        old_value = std::forward<T>(new_value);
    }

    template <typename T>
    void set_value(T&& value)
    {
        constexpr bool has_member_setter = requires(Setters setters)
        {
            setters.template setter<T>(*static_cast<Master*>(this),
                                       this->container_.template get<T>(),
                                       std::forward<T>(value));
        };
        if constexpr (has_member_setter)
        {
            Setters setters;
            setters.template setter<T>(*static_cast<Master*>(this),
                                       this->container_.template get<T>(),
                                       std::forward<T>(value));
        }
        else { default_setter<T>(this->container_.template get<T>(), std::forward<T>(value)); }

        Base::template trigger_params_all<T>();
    } // namespace holovibes
};
} // namespace holovibes
