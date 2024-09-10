#pragma once

#include "tools.hh"

#include "notifier.hh"

template <typename T, typename D>
std::shared_ptr<Notifier<T, D>> NotifierManager::get_notifier(const std::string& name)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = notifiers_.find(name);
    if (it == notifiers_.end())
    {
        auto notifier = std::make_shared<Notifier<T, D>>();
        notifiers_.emplace(name, notifier);
        return notifier;
    }
    return std::static_pointer_cast<Notifier<T, D>>(it->second);
}

template <typename T, typename D>
inline D NotifierManager::notify(const std::string& name, const T& data)
{
    return NotifierManager::get_instance().get_notifier<T, D>(name)->notify(data);
}

template <typename T, typename D>
Notifier<T, D>::Notifier()
    : nextId_(0)
{
}

template <typename T, typename D>
D Notifier<T, D>::notify(const T& data)
{
    std::unordered_map<SubscriptionId, std::function<D(const T&)>> subscribersCopy;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        subscribersCopy = subscribers_;
    }

    for (const auto& [id, subscriber] : subscribersCopy)
    {
        try
        {
            if (std::is_same<D, void>::value)
                subscriber(data); // The notifier doesn't return anything.
            else
            {
                // The notifier has a return value.
                // In that neesh case reserved only to a few usages, it is assumed that
                // there is only one subscriber for the notifier in question.

                if (subscribers_.size() > 1)
                    LOG_WARN("Multiple subscribers for non-void notifier: {}", subscribers_.size());
                return subscriber(data);
            }
        }
        catch (const std::exception& e)
        {
            LOG_ERROR("Exception caught in notifier: {}", e.what());
        }
    }
}

template <typename T, typename D>
typename Notifier<T, D>::SubscriptionId Notifier<T, D>::subscribe(const std::function<D(const T&)>& subscriber)
{
    std::lock_guard<std::mutex> lock(mutex_);
    SubscriptionId id = nextId_++;
    subscribers_.emplace(id, subscriber);
    return id;
}

template <typename T, typename D>
void Notifier<T, D>::unsubscribe(SubscriptionId id)
{
    std::lock_guard<std::mutex> lock(mutex_);
    subscribers_.erase(id);
}

template <typename T, typename D>
template <typename Func>
Subscriber<T, D>::Subscriber(const std::string& name, Func&& callback)
    : notifier_(NotifierManager::get_instance().get_notifier<T, D>(name))
    , callback_(std::forward<Func>(callback))
{
    subscriptionId_ = notifier_->subscribe(
        [this](const T& value)
        {
            return this->callback_(value);
        });
}

template <typename T, typename D>
Subscriber<T, D>::~Subscriber()
{
    notifier_->unsubscribe(subscriptionId_);
}
