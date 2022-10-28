#pragma once

#include <unordered_map>
#include <vector>
#include <deque>
#include <map>
#include <unordered_map>
#include <set>
#include <functional>
#include <memory>
#include <utility>
#include <mutex>
#include <thread>
#include "parameter.hh"
#include "logger.hh"

#include "static_container.hh"
#include "micro_cache_trigger.hh"

namespace holovibes
{
template <bool has_been_sync>
class SetHasBeenSynchronized
{
  public:
    template <typename T>
    void operator()(T& value)
    {
        value.set_has_been_synchronized(has_been_sync);
    }
};

using MapKeyParams = std::unordered_map<std::string_view, IParameter*>;
class FillMapKeyParams
{
  public:
    template <typename T>
    void operator()(T& value, MapKeyParams& map)
    {
        map[value.get_key()] = &value;
    }
};

//! <param_to_change, <ref_new_value, ref_old_value>>
using ChangePool = std::map<IParameter*, std::pair<IParameter*, IDuplicatedParameter*>>;
template <typename FunctionClass, typename... Args>
class OnSync
{
  public:
    FunctionClass functions_to_call;

  public:
    template <typename T>
    void operator()(T& value, ChangePool& pool, Args&&... args)
    {
        if (value.get_has_been_synchronized() == true)
        {
            IDuplicatedParameter* Iold_value = pool[&value].second;
            DuplicatedParameter<T>* old_value = dynamic_cast<DuplicatedParameter<T>*>(Iold_value);
            if (old_value == nullptr)
            {
                LOG_ERROR(main, "Not supposed to end here : fail to cast DuplicatedParameter<T>");
                return;
            }
            functions_to_call.template operator()<T>(value, old_value->get_value(), std::forward<Args>(args)...);
        }
    }
};

template <typename... Params>
class MicroCache
{
  protected:
    class BasicMicroCache
    {
      public:
        BasicMicroCache()
            : key_container_{}
            , container_{}
        {
            container_.template call<FillMapKeyParams>(key_container_);
        }

      public:
        template <typename FunctionClass, typename... Args>
        void call(Args&&... args)
        {
            container_.call<FunctionClass>(std::forward<Args>(args)...);
        }

      public:
        const MapKeyParams& get_map_key() const { return key_container_; }
        MapKeyParams& get_map_key() { return key_container_; }

        StaticContainer<Params...>& get_container() { return container_; }

      public:
        template <typename T>
        typename T::ConstRefType get_value() const
        {
            return container_.template get<T>().get_value();
        }

        template <typename T>
        static constexpr bool has()
        {
            return StaticContainer<Params...>::template has<T>();
        }

      protected:
        template <typename T>
        T& get_type()
        {
            return container_.template get<T>();
        }

      protected:
        MapKeyParams key_container_;
        StaticContainer<Params...> container_;
    };

  public:
    class Ref;

  public:
    class Cache : public BasicMicroCache
    {
      public:
        friend Ref;

      public:
        Cache();
        ~Cache();

      protected:
        template <typename MicroCacheToSync>
        void sync_with(MicroCacheToSync& ref)
        {
            this->container_.sync_with(ref.get_container());
        }

        template <typename T>
        void trigger_param(IParameter* ref)
        {
            IParameter* param_to_change = static_cast<IParameter*>(&BasicMicroCache::template get_type<T>());
            IDuplicatedParameter* old_value = &duplicate_container_.template get<DuplicatedParameter<T>>();

            std::lock_guard<std::mutex> guard(lock_);
            change_pool_[param_to_change] = std::pair{ref, old_value};
        }

      public:
        template <typename FunctionClass, typename... Args>
        void synchronize(Args&&... args)
        {
            if (change_pool_.size() == 0)
                return;

            LOG_TRACE(main, "Cache sync {} elements", change_pool_.size());
            std::lock_guard<std::mutex> guard(lock_);

            this->container_.template call<SetHasBeenSynchronized<false>>();

            for (auto change : change_pool_)
            {
                IParameter* param_to_change = change.first;
                IParameter* ref_param = change.second.first;
                param_to_change->sync_with(ref_param);
            }

            this->container_.template call<OnSync<FunctionClass, Args...>>(change_pool_, std::forward<Args>(args)...);

            change_pool_.clear();
        }

        bool has_change_requested() { return change_pool_.size() > 0; }

      private:
        ChangePool change_pool_;
        StaticContainer<DuplicatedParameter<Params>...> duplicate_container_;
        std::mutex lock_;
    };

  public:
    class Ref : public BasicMicroCache
    {
      public:
        friend Cache;

      public:
        Ref()
            : BasicMicroCache()
            , caches_to_sync_{}
        {
        }

      private:
        template <typename T>
        void trigger_param()
        {
            IParameter* ref = &this->BasicMicroCache::template get_type<T>();
            for (auto cache : caches_to_sync_)
                cache->template trigger_param<T>(ref);
        }

      public:
        template <typename T>
        void set_value(typename T::ConstRefType value)
        {
            this->BasicMicroCache::template get_type<T>().set_value(value);
            trigger_param<T>();
        }

        template <typename T>
        void callback_trigger_change_value()
        {
            trigger_param<T>();
        }

        template <typename T>
        TriggerChangeValue<typename T::ValueType> change_value()
        {
            return TriggerChangeValue<typename T::ValueType>([this]() { this->callback_trigger_change_value<T>(); },
                                                             &BasicMicroCache::template get_type<T>().get_value());
        }

      public:
        //! this function must be handle with care (hence the W, may_be we can change this...)
        template <typename T>
        typename T::ValueType& get_value_ref_W()
        {
            // Only way to get the value as reference, for safe reason, the get_value non-const ref has not been
            // declared.
            return BasicMicroCache::template get_type<T>().get_value();
        }

        //! this function must be handle with care (hence the W, may_be we can change this...)
        template <typename T>
        void force_trigger_param_W()
        {
            trigger_param<T>();
        }

      protected:
        void add_cache_to_synchronize(Cache& cache)
        {
            caches_to_sync_.insert(&cache);
            cache.sync_with(*this);
        }

        void remove_cache_to_synchronize(Cache& cache)
        {
            if (caches_to_sync_.erase(&cache) != 1)
                LOG_ERROR(main, "Maybe a problem here...");
        }

      private:
        std::set<Cache*> caches_to_sync_;
    };

    class RefSingleton
    {
      public:
        static Ref& get()
        {
            if (instance == nullptr)
                throw std::runtime_error("No Ref has been set for this cache");
            return *instance;
        }

      public:
        static void set_main_ref(Ref& ref) { instance = &ref; }
        static void remove_main_ref(Ref& ref)
        {
            if (instance == &ref)
                instance = nullptr;
        }

      private:
        static inline Ref* instance;
    };

  public:
    template <typename T>
    static constexpr bool has()
    {
        return BasicMicroCache::template has<T>();
    }
};

template <typename... Params>
MicroCache<Params...>::Cache::Cache()
    : BasicMicroCache()
    , change_pool_{}
    , duplicate_container_{}
    , lock_{}
{
    MicroCache<Params...>::RefSingleton::get().add_cache_to_synchronize(*this);
}

template <typename... Params>
MicroCache<Params...>::Cache::~Cache()
{
    MicroCache<Params...>::RefSingleton::get().remove_cache_to_synchronize(*this);
}

} // namespace holovibes
