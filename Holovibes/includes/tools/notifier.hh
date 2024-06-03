#pragma once

#ifndef NOTIFIER_HH
#define NOTIFIER_HH

#include <functional>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <atomic>
#include <memory>
#include <iostream>

// Forward declaration
template <typename T>
class Notifier;

class NotifierManager
{
public:
    static NotifierManager& get_instance() {
        static NotifierManager instance;
        return instance;
    }

    template <typename T>
    std::shared_ptr<Notifier<T>> get_notifier(const std::string& name);

private:
    NotifierManager() = default;
    ~NotifierManager() = default;
    NotifierManager(const NotifierManager&) = delete;
    NotifierManager& operator=(const NotifierManager&) = delete;

    std::unordered_map<std::string, std::shared_ptr<void>> notifiers_;
    std::mutex mutex_;
};

template <typename T>
class Notifier
{
public:
    using SubscriptionId = std::size_t;

    Notifier();

    void notify(const T& data);

    SubscriptionId subscribe(const std::function<void(const T&)>& subscriber);

    void unsubscribe(SubscriptionId id);

private:
    std::unordered_map<SubscriptionId, std::function<void(const T&)>> subscribers_;
    std::mutex mutex_;
    std::atomic<SubscriptionId> nextId_;
};

template <typename T>
class Subscriber
{
public:
    template <typename Func>
    Subscriber(const std::string& name, Func&& callback);

    ~Subscriber();

private:
    std::shared_ptr<Notifier<T>> notifier_;
    typename Notifier<T>::SubscriptionId subscriptionId_;
    std::function<void(const T&)> callback_;
};

// Include the implementation file for template functions
#include "notifier.hxx"
 
#endif // NOTIFIER_HH