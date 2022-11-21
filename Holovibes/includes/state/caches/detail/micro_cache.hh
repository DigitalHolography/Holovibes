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
#include "Iparameter.hh"
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
                LOG_ERROR(main,
                          "Not supposed to end here : fail to cast DuplicatedParameter<T> ; T = {}",
                          typeid(T).name());
                return;
            }
            functions_to_call.template on_sync<T>(value, old_value->get_value(), std::forward<Args>(args)...);
            old_value->save_current_value(&value);
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
            container_.template call<FunctionClass>(std::forward<Args>(args)...);
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

  protected:
    class BasicRef;

  public:
    class Cache : public BasicMicroCache
    {
      public:
        friend class BasicRef;

      public:
        Cache();
        ~Cache();

      protected:
        template <typename MicroCacheToSync>
        void set_all_values(MicroCacheToSync& ref)
        {
            this->container_.set_all_values(ref.get_container());
        }

        template <typename T>
        void trigger_param(IParameter* Iref)
        {
            auto& param_to_change = BasicMicroCache::template get_type<T>();
            if (param_to_change.value_has_changed(Iref) == false)
                return;

            IParameter* Iparam_to_change = static_cast<IParameter*>(&param_to_change);
            IDuplicatedParameter* Iold_value = &duplicate_container_.template get<DuplicatedParameter<T>>();

#ifndef DISABLE_LOG_TRIGGER_MICROCACHE
            T* ref = dynamic_cast<T*>(Iref);
            if (ref)
                LOG_TRACE(main, "MicroCache : TRIGGER {} = {}", Iref->get_key(), ref->get_value());
            else
                LOG_TRACE(main, "MicroCache : TRIGGER {} = ? (unable to cast ref)", Iref->get_key());
#endif

            std::lock_guard<std::mutex> guard(lock_);
            change_pool_[Iparam_to_change] = std::pair{Iref, Iold_value};
        }

      public:
        class DefaultFunctionsOnSync
        {
          public:
            template <typename T>
            void operator()()
            {
            }
        };

        template <typename FunctionClass = DefaultFunctionsOnSync, typename... Args>
        void synchronize(Args&&... args)
        {
            if (change_pool_.size() == 0)
                return;

#ifndef DISABLE_LOG_SYNC_MICROCACHE
            LOG_TRACE(main, "Cache sync {} elements", change_pool_.size());
#endif

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

        template <typename FunctionClass, typename... Args>
        void synchronize_force(Args&&... args);

        bool has_change_requested()
        {
            std::lock_guard<std::mutex> guard(lock_);
            return change_pool_.size() > 0;
        }

      public:
        // for debugging purpose ONLY
        template <typename T, typename FunctionsClass, typename... Args>
        void virtual_synchronize_W(Args&&... args)
        {
            auto param_to_change = BasicMicroCache::template get_type<T>();
            IDuplicatedParameter* Iold_value = &duplicate_container_.template get<DuplicatedParameter<T>>();
            DuplicatedParameter<T>* old_value = dynamic_cast<DuplicatedParameter<T>*>(Iold_value);

            if (old_value == nullptr)
            {
                LOG_ERROR(main,
                          "Not supposed to end here : fail to cast DuplicatedParameter<T> ; T = {}",
                          typeid(T).name());
                return;
            }

            FunctionsClass functions;
            functions.template on_sync<T>(param_to_change.get_value(),
                                          old_value->get_value(),
                                          std::forward<Args>(args)...);
        }

      private:
        ChangePool change_pool_;
        StaticContainer<DuplicatedParameter<Params>...> duplicate_container_;
        std::mutex lock_;
    };

  public:
    class RefSingleton
    {
      public:
        friend BasicRef;

      public:
        static BasicRef& get()
        {
            if (instance == nullptr)
                throw std::runtime_error("No Ref has been set for this cache");
            return *instance;
        }

      protected:
        static void set_main_ref(BasicRef& ref) { instance = &ref; }
        static void remove_main_ref(BasicRef& ref)
        {
            if (instance == &ref)
                instance = nullptr;
        }

      private:
        static inline BasicRef* instance;
    };

  protected:
    class BasicRef : public BasicMicroCache
    {
      public:
        friend Cache;

      public:
        BasicRef()
            : BasicMicroCache()
            , caches_to_sync_{}
        {
            RefSingleton::set_main_ref(*this);
        }

        ~BasicRef() { RefSingleton::remove_main_ref(*this); }

      protected:
        template <typename T>
        void trigger_param()
        {
            IParameter* ref = &this->BasicMicroCache::template get_type<T>();

            // This is only for current holovibes version because each caches only exists once. If you want
            // more caches at the same time remove this to don't get this warning
            size_t nb_caches = caches_to_sync_.size();
            if (nb_caches > 1)
                LOG_WARN(main, "Number to cache to sync > 1; current value {}", nb_caches);

            for (auto cache : caches_to_sync_)
                cache->template trigger_param<T>(ref);
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
            cache.set_all_values(*this);
        }

        void remove_cache_to_synchronize(Cache& cache)
        {
            if (caches_to_sync_.erase(&cache) != 1)
                LOG_ERROR(main, "Maybe a problem here...");
        }

      private:
        std::set<Cache*> caches_to_sync_;
    };

  public:
    class DefaultFunctionsOnChange
    {
      public:
        template <typename T>
        void operator()(typename T::ValueType&)
        {
        }
    };

    template <typename FunctionsOnChange = DefaultFunctionsOnChange>
    class Ref : public BasicRef
    {
      public:
        friend Cache;

      public:
        template <typename T>
        static constexpr bool has()
        {
            return BasicMicroCache::template has<T>();
        }

      public:
        Ref() {}
        ~Ref() {}

      public:
        template <typename T>
        void callback_trigger_change_value()
        {
            FunctionsOnChange functions;
            functions.template operator()<T>(this->BasicMicroCache::template get_type<T>().get_value());

            this->BasicRef::template trigger_param<T>();
        }

        template <typename T>
        void set_value(typename T::ConstRefType value)
        {
            this->BasicMicroCache::template get_type<T>().set_value(value);
            callback_trigger_change_value<T>();
        }

        template <typename T>
        TriggerChangeValue<typename T::ValueType> change_value()
        {
            return TriggerChangeValue<typename T::ValueType>([this]() { this->callback_trigger_change_value<T>(); },
                                                             &BasicMicroCache::template get_type<T>().get_value());
        }
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

template <typename... Params>
template <typename FunctionClass, typename... Args>
void MicroCache<Params...>::Cache::synchronize_force(Args&&... args)
{
    set_all_values(MicroCache<Params...>::RefSingleton::get());
    this->template call<FunctionClass>(std::forward<Args>(args)...);

    std::lock_guard<std::mutex> guard(lock_);
    change_pool_.clear();
}

} // namespace holovibes
