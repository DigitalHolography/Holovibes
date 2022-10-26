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
template <bool value>
class SetHasBeenSynchronized
{
  public:
    void operator()() { value.set_has_been_synchronized(value); }
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

using MapDuplicatedParameter = std::unordered_map<IParameter*, IDuplicatedParameter*>;
template <typename DuplicatedContainer>
class FillDuplicatedContainer
{
  public:
    template <typename T>
    void operator()(T& value, DuplicatedContainer& dup_container, MapDuplicatedParameter& map)
    {
        DuplicatedParameter<T>* dup_ptr = &dup_container.template get<DuplicatedParameter<T>>();
        dup_ptr->set_value(value.get_value());
        map[&value] = dup_ptr;
    }
};

//! <param_to_change, <ref_new_value, ref_old_value>>
using ChangePool = std::map<IParameter*, std::pair<IParameter*, IDuplicatedParameter*>>;
template <typename FunctionClass, typename... Args>
class OnSync
{
  public:
    template <typename T>
    void operator()(T& value, FunctionClass& functions_to_call, ChangePool& pool, Args&&... args)
    {
        if (value.get_has_been_synchronized() == true)
        {
            DuplicatedParameter<T>& old_value = pool[&value].second;
            functions_to_call.operator<T>()(value, old_value.get_value(), std::forward<Args>(args)...);
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
            FunctionClass functions_class;
            container_.call(functions_class, std::forward<Args>(args)...);
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
        void trigger_param(IParameter* ref, IDuplicatedParameter* old_value)
        {
            IParameter* param_to_change = static_cast<IParameter*>(&BasicMicroCache::template get_type<T>());

            std::lock_guard<std::mutex> guard(lock_);
            change_pool_[param_to_change] = std::pair{ref, old_value};
        }

      public:
        template <typename FunctionClass, typename... Args>
        void synchronize(Args&&... args)
        {
            call<SetHasBeenSynchronized<false>>();

            std::lock_guard<std::mutex>(lock_);
            for (auto change : change_pool_)
            {
                IParameter* param_to_change = change.first;
                IParameter* ref_param = change.second.first;
                param_to_change->sync_with(ref_param);
            }

            FunctionClass function_class;
            call<OnSync<FunctionClass>>(function_class, change_pool_, std::forward<Args>(args)...);

            change_pool_.clear();
        }

        bool has_change_requested() { return change_pool_.size() > 0; }

      private:
        ChangePool change_pool_;
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
            , lock_{}
            , duplicate_container_{}
            , duplicate_container_lut_{}
            , change_pool_{}
        {
            this->container_.template call<FillDuplicatedContainer<StaticContainer<DuplicatedParameter<Params>...>>>(
                duplicate_container_,
                duplicate_container_lut_);
        }

      public:
        bool has_change() { return change_pool_.size() > 0; }

      private:
        template <typename T>
        void trigger_param()
        {
            std::lock_guard<std::mutex> guard(lock_);

            IParameter* ref = &this->BasicMicroCache::template get_type<T>();
            IDuplicatedParameter* old_value = duplicate_container_lut_[ref];

            for (auto cache : caches_to_sync_)
                cache->template trigger_param<T>(ref, old_value);

            change_pool_.insert(ref);
        }

      public:
        template <typename T>
        void set_value(typename T::ConstRefType value)
        {
            std::lock_guard<std::mutex> guard(lock_);

            this->BasicMicroCache::template get_type<T>().set_value(value);
            trigger_param<T>();
        }

        template <typename T>
        void callback_trigger_change_value()
        {
            std::lock_guard<std::mutex> guard(lock_);

            trigger_param<T>();
        }

        template <typename T>
        TriggerChangeValue<typename T::ValueType> change_value()
        {
            std::lock_guard<std::mutex> guard(lock_);

            return TriggerChangeValue<typename T::ValueType>([this]() { this->callback_trigger_change_value<T>(); },
                                                             &BasicMicroCache::template get_type<T>().get_value());
        }

      public:
        //! this function must be handle with care (hence the W, may_be we can change this...)
        template <typename T>
        typename T::ValueType& get_value_ref_W()
        {
            std::lock_guard<std::mutex> guard(lock_);

            // Only way to get the value as reference, for safe reason, the get_value non-const ref has not been
            // declared.
            return BasicMicroCache::template get_type<T>().get_value();
        }

        //! this function must be handle with care (hence the W, may_be we can change this...)
        template <typename T>
        void force_trigger_param_W()
        {
            std::lock_guard<std::mutex> guard(lock_);

            trigger_param<T>();
        }

      public:
        void save_ref_for_next_sync()
        {
            for (IParameter* param_to_save : change_pool_)
                duplicate_container_lut_[param_to_save].save_current_value(param_to_save);
            change_pool_.clear();
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

      public:
        std::mutex& get_lock() { return mutex_; }

      private:
        std::set<Cache*> caches_to_sync_;
        std::mutex lock_;
        StaticContainer<DuplicatedParameter<Params>...> duplicate_container_;
        MapDuplicatedParameter duplicate_container_lut_;
        std::set<IParameter*> change_pool_;
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

      public:
        static bool has_change() { return get().has_change(); }
        static void lock() { get().get_lock().lock(); }
        static void unlock() { get().get_lock().unlock(); }
        static void end_synchronize() { get().save_ref_for_next_sync(); }

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
    , change_pool{}
{
    MicroCache<Params...>::RefSingleton::get().add_cache_to_synchronize(*this);
}

template <typename... Params>
MicroCache<Params...>::Cache::~Cache()
{
    MicroCache<Params...>::RefSingleton::get().remove_cache_to_synchronize(*this);
}

} // namespace holovibes
