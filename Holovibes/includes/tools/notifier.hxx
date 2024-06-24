#pragma once

#include "tools.hh"

#include "notifier.hh"

template <typename T>
std::shared_ptr<Notifier<T>> NotifierManager::get_notifier(const std::string& name)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = notifiers_.find(name);
    if (it == notifiers_.end())
    {
        auto notifier = std::make_shared<Notifier<T>>();
        notifiers_.emplace(name, notifier);
        return notifier;
    }
    return std::static_pointer_cast<Notifier<T>>(it->second);
}

template <typename T>
Notifier<T>::Notifier()
    : nextId_(0)
{
}

template <typename T>
void Notifier<T>::notify(const T& data)
{
    std::unordered_map<SubscriptionId, std::function<void(const T&)>> subscribersCopy;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        subscribersCopy = subscribers_;
    }

    for (const auto& [id, subscriber] : subscribersCopy)
    {
        try
        {
            subscriber(data);
        }
        catch (const std::exception& e)
        {
            LOG_ERROR("Exception caught in notifier: {}", e.what());
        }
    }
}

template <typename T>
typename Notifier<T>::SubscriptionId Notifier<T>::subscribe(const std::function<void(const T&)>& subscriber)
{
    std::lock_guard<std::mutex> lock(mutex_);
    SubscriptionId id = nextId_++;
    subscribers_.emplace(id, subscriber);
    return id;
}

template <typename T>
void Notifier<T>::unsubscribe(SubscriptionId id)
{
    std::lock_guard<std::mutex> lock(mutex_);
    subscribers_.erase(id);
}

template <typename T>
template <typename Func>
Subscriber<T>::Subscriber(const std::string& name, Func&& callback)
    : notifier_(NotifierManager::get_instance().get_notifier<T>(name))
    , callback_(std::forward<Func>(callback))
{
    subscriptionId_ = notifier_->subscribe(
        [this](const T& value)
        {
            this->callback_(value);
        });
}

template <typename T>
Subscriber<T>::~Subscriber()
{
    notifier_->unsubscribe(subscriptionId_);
}
