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
#include "logger.hh"

#include "static_container.hh"
#include "micro_cache_trigger.hh"

namespace holovibes
{

template <typename... Params>
class MicroCache
{
  protected:
    class BasicMicroCache
    {
      protected:
        MapKeyParams key_container_;
        StaticContainer<Params...> container_;

      public:
        BasicMicroCache()
            : key_container_()
            , container_(key_container_)
        {
        }

      public:
        template <typename FunctionClass, typename... Args>
        void call(Args&&... args)
        {
            FunctionClass functions_class;

            if constexpr (requires { typename FunctionClass::BeforeMethods; })
                call<typename FunctionClass::BeforeMethods>();

            container_.call(functions_class, std::forward<Args>(args)...);

            constexpr bool has_member_call_on_cache = requires(FunctionClass functions_class)
            {
                functions_class.template call_on_cache(*this);
            };
            if constexpr (has_member_call_on_cache) functions_class.template call_on_cache(*this);

            if constexpr (requires { typename FunctionClass::AfterMethods; })
                call<typename FunctionClass::AfterMethods>();
        }

      public:
        virtual void synchronize() {}

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
            std::lock_guard<std::mutex> guard(change_pool_mutex);
            IParameter* param_to_change = static_cast<IParameter*>(&BasicMicroCache::template get_type<T>());
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
        std::mutex get_change_pool_mutex() { return change_pool_mutex; }

      private:
        // first is param_to_change ; second is ref
        std::map<IParameter*, IParameter*> change_pool;
        std::mutex change_pool_mutex;
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

        // Singleton
      private:
        static inline Ref* instance;

      public:
        void set_as_ref() { instance = this; }
        static Ref& get_ref()
        {
            if (instance == nullptr)
                throw std::runtime_error("No Ref has been set for this cache");
            return *instance;
        }
        static void remove_ref(Ref& ref)
        {
            if (instance == &ref)
                instance = nullptr;
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
    , change_pool{}
{
    MicroCache<Params...>::Ref::get_ref().add_cache_to_synchronize(*this);
}

template <typename... Params>
MicroCache<Params...>::Cache::~Cache()
{
    MicroCache<Params...>::Ref::get_ref().remove_cache_to_synchronize(*this);
}

} // namespace holovibes
