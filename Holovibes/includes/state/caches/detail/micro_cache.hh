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

// FIXME methods of Callback FunctionsClass could be static

namespace holovibes
{

class DefaultFunctionsOnSync
{
  public:
    template <typename T>
    void on_sync()
    {
    }

    template <typename T>
    void operator()()
    {
    }
};

class DefaultFunctionsOnChange
{
  public:
    template <typename T>
    void operator()(typename T::ValueType&)
    {
    }

    template <typename T>
    bool change_accepted(typename T::ConstRefType)
    {
        return true;
    }
};

template <typename... T>
class MultiFunctions;

template <>
class MultiFunctions<>
{
  public:
    template <typename T, typename... Args>
    void operator()(Args&&... args)
    {
    }
};

template <typename Functions, typename... Rest>
class MultiFunctions<Functions, Rest...> : public MultiFunctions<Rest...>
{
  public:
    template <typename T, typename... Args>
    void operator()(Args&&... args)
    {
        Functions functions;
        functions.template operator()<T>(std::forward<Args>(args)...);
        MultiFunctions<Rest...>::template operator()<T>(std::forward<Args>(args)...);
    }
};

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

//! <param_to_change, <ref_new_value, cache_old_value>>
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
                LOG_ERROR("Not supposed to end here : fail to cast DuplicatedParameter<T> ; T = {}", typeid(T).name());
                return;
            }

            try
            {
                // LOG_TRACE("Call OnSync with new value : {}; old value : {}", value, old_value->get_value());
                functions_to_call.template on_sync<T>(value, old_value->get_value(), std::forward<Args>(args)...);
            }
            catch (const std::exception& e)
            {
                value.set_value(old_value->get_value());

                LOG_ERROR("Got an exception with the OnSync functions on a Setter : {} ; Skip the change", e.what());
                throw;
            }
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

        static constexpr uint size() { return StaticContainer<Params...>::size(); }

        template <typename T>
        static constexpr uint get_index_of()
        {
            return StaticContainer<Params...>::template get_index_of<T>();
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

  private:
    class BasicCache : public BasicMicroCache
    {
      public:
        friend class BasicRef;

      public:
        BasicCache()
            : BasicMicroCache()
            , change_pool_{}
            , duplicate_container_{}
        {
            MicroCache<Params...>::RefSingleton::get().add_cache_to_synchronize(*this);
        }

        ~BasicCache() { MicroCache<Params...>::RefSingleton::get().remove_cache_to_synchronize(*this); }

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
            if (param_to_change.has_parameter_change(Iref) == false)
                return;

            IParameter* Iparam_to_change = &param_to_change;
            IDuplicatedParameter* Iold_value = &duplicate_container_.template get<DuplicatedParameter<T>>();

#ifndef DISABLE_LOG_TRIGGER_CACHE
            T* ref = dynamic_cast<T*>(Iref);
            if (ref)
                LOG_TRACE("TRIGGER On Cache {} = {}", Iref->get_key(), ref->get_value());
            else
                LOG_TRACE("TRIGGER On Cache {} = ? (unable to cast ref)", Iref->get_key());
#endif
            change_pool_[Iparam_to_change] = std::pair{Iref, Iold_value};
        }

      public:
        bool has_change_requested() { return change_pool_.size() > 0; }

      protected:
        ChangePool change_pool_;
        StaticContainer<DuplicatedParameter<Params>...> duplicate_container_;
    };

  public:
    template <typename FunctionsClass = DefaultFunctionsOnSync>
    class Cache : public BasicCache
    {
      public:
        Cache() = default;

        template <typename... Args>
        Cache(Args&&... args)
        {
            synchronize_force(std::forward<Args>(args)...);
        }

      public:
        // Synchronize this cache with the cache ref using the pool change
        template <typename... Args>
        void synchronize(Args&&... args)
        {
            if (this->change_pool_.size() == 0)
                return;

#ifndef DISABLE_LOG_SYNC_MICROCACHE
            LOG_TRACE("Cache sync {} elements", this->change_pool_.size());
#endif

            this->container_.template call<SetHasBeenSynchronized<false>>();

            std::lock_guard<std::recursive_mutex> guard(RefSingleton::get().get_lock());
            for (auto change : this->change_pool_)
            {
                IParameter* param_to_change = change.first;
                IParameter* ref_param = change.second.first;
                param_to_change->sync_with(ref_param);
            }

            this->container_.template call<OnSync<FunctionsClass, Args...>>(this->change_pool_,
                                                                            std::forward<Args>(args)...);
            this->change_pool_.clear();
        }

        template <typename... Args>
        void synchronize_force(Args&&... args)
        {
            this->set_all_values(MicroCache<Params...>::RefSingleton::get());
            this->template call<FunctionsClass>(std::forward<Args>(args)...);

            std::lock_guard<std::recursive_mutex> guard(RefSingleton::get().get_lock());
            this->change_pool_.clear();
        }

      public:
        // for debugging purpose ONLY
        template <typename T, typename... Args>
        void virtual_synchronize_W(Args&&... args)
        {

            auto& param_to_change = BasicMicroCache::template get_type<T>();
            IParameter* Iparam_to_change = &param_to_change;
            if (this->change_pool_.contains(Iparam_to_change) == false)
            {
                FunctionsClass functions;
                functions.template operator()<T>(param_to_change.get_value(), std::forward<Args>(args)...);
            }
            else
            {
                auto& old_value = this->duplicate_container_.template get<DuplicatedParameter<T>>();
                old_value.set_value(param_to_change);
                IParameter* ref_param = this->change_pool_[Iparam_to_change].first;
                param_to_change.sync_with(ref_param);

#ifndef DISABLE_LOG_TRIGGER_CACHE
                LOG_TRACE("Call On Sync (Forced) {} = {}", param_to_change.get_key(), param_to_change.get_value());
#endif

                FunctionsClass functions;
                functions.template on_sync<T>(param_to_change.get_value(),
                                              old_value.get_value(),
                                              std::forward<Args>(args)...);
            }
        }
    };

  protected:
    class BasicRef : public BasicMicroCache
    {
      public:
        friend BasicCache;

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

#ifndef DISABLE_LOG_TRIGGER_REF
            LOG_TRACE("TRIGGER On Ref {} = {}",
                      this->BasicMicroCache::template get_type<T>().get_key(),
                      this->BasicMicroCache::template get_type<T>().get_value());
#endif

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
        void add_cache_to_synchronize(BasicCache& cache)
        {
            caches_to_sync_.insert(&cache);
            cache.set_all_values(*this);
        }

        void remove_cache_to_synchronize(BasicCache& cache)
        {
            if (caches_to_sync_.erase(&cache) != 1)
                LOG_ERROR("Maybe a problem here...");
        }

      public:
        std::recursive_mutex& get_lock() { return lock_; }
        void lock() { lock_.lock(); }
        void unlock() { lock_.unlock(); }

      protected:
        std::set<BasicCache*> caches_to_sync_;
        std::recursive_mutex lock_;
    };

  public:
    template <typename FunctionsOnChange = DefaultFunctionsOnChange>
    class Ref : public BasicRef
    {
      public:
        friend BasicCache;

      public:
        template <typename T>
        static constexpr bool has()
        {
            return BasicMicroCache::template has<T>();
        }

      public:
        Ref() {}
        ~Ref() {}

      private:
        template <typename T>
        bool is_change_accepted(typename T::ConstRefType new_value)
        {
            FunctionsOnChange functions;
            if (functions.template change_accepted<T>(new_value) == false)
            {
                LOG_WARN("Refused the change on {} with old_value : {} ; new_value : {} ; Skip the change",
                         typeid(T).name(),
                         this->BasicMicroCache::template get_type<T>().get_value(),
                         new_value);
                return false;
            }
            return true;
        }

      public:
        template <typename T>
        void callback_trigger_change_value(typename T::ConstRefType value_for_restore)
        {
            std::vector<BasicCache*> start_caches;
            try
            {
                FunctionsOnChange functions;
                functions.template operator()<T>(this->BasicMicroCache::template get_type<T>().get_value());
            }
            catch (const std::exception& e)
            {
                LOG_ERROR("Got an exception with the OnChange functions of {} with old_value : {} ; new_value : {} ; "
                          "error : {} ; Skip the change",
                          typeid(T).name(),
                          value_for_restore,
                          this->BasicMicroCache::template get_type<T>().get_value(),
                          e.what());

                this->BasicMicroCache::template get_type<T>().set_value(value_for_restore);
                throw;
            }

            this->BasicRef::template trigger_param<T>();
        }

        template <typename T>
        void set_value(typename T::ConstRefType value)
        {
            this->lock_.lock();
            const T& old_value = this->BasicMicroCache::template get_type<T>();
            if (old_value.has_parameter_change_valuetype(value) && is_change_accepted<T>(value))
            {
                typename T::ValueType save_for_restore = this->BasicMicroCache::template get_type<T>().get_value();
                this->BasicMicroCache::template get_type<T>().set_value(value);
                callback_trigger_change_value<T>(save_for_restore);
            }
            this->lock_.unlock();
        }

        template <typename T>
        TriggerChangeValue<typename T::ValueType> change_value()
        {
            this->lock_.lock();
            return TriggerChangeValue<typename T::ValueType>(
                [this, old_value = this->BasicMicroCache::template get_type<T>()]()
                {
                    if (is_change_accepted<T>(BasicMicroCache::template get_type<T>().get_value()) == false)
                    {
                        this->BasicMicroCache::template get_type<T>().set_value(old_value);
                    }
                    if (old_value.has_parameter_change_valuetype(BasicMicroCache::template get_type<T>().get_value()))
                    {
                        this->callback_trigger_change_value<T>(old_value.get_value());
                    }

                    this->lock_.unlock();
                },
                &BasicMicroCache::template get_type<T>().get_value());
        }
    };

  public:
    // Allows use of AdvancedCache::size
    template <typename T>
    static constexpr bool has()
    {
        return BasicMicroCache::template has<T>();
    }

    static constexpr uint size() { return BasicMicroCache::size(); }

    template <typename T>
    static constexpr uint get_index_of()
    {
        return BasicMicroCache::template get_index_of<T>();
    }
};

} // namespace holovibes
