/**
 * @file notifier.hh
 * @brief Defines the Notifier, NotifierManager, and Subscriber classes.
 */

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
template <typename T, typename D>
class Notifier;

/**
 * @brief Manages instances of Notifier objects.
 *
 * @note This class is a singleton and should be accessed using the get_instance() method.
 *
 * @see get_instance()
 * @see Notifier
 */
class NotifierManager
{
  public:
    /**
     * @brief Gets the singleton instance of NotifierManager.
     * @return The singleton instance of NotifierManager.
     */
    static NotifierManager& get_instance()
    {
        static NotifierManager instance;
        return instance;
    }

    /**
     * @brief Gets a notifier for a specific type and name.
     * @tparam T The type of data the notifier will handle.
     * @tparam D The return type of the notifier callback (defaults to void).
     * @param name A unique name for the notifier.
     * @return A shared pointer to the notifier.
     */
    template <typename T, typename D = void>
    std::shared_ptr<Notifier<T, D>> get_notifier(const std::string& name);

    /**
     * @brief Generic function to compact a Notify call.
     * Basically the same as getting the instance of NotifierManager,
     * getting the notifier and calling notify.
     *
     * @tparam T The type of data the notifier will handle.
     * @tparam D The return type of the notifier callback (defaults to void).
     * @param name A unique name for the notifier.
     * @return D The return value of the notifier callback, if any.
     *
     * @note In the case of a non-void return type, it is intended that only one subscriber is present.
     * Indeed, in such case, only the return value of the first subscriber will be returned.
     */
    template <typename T, typename D = void>
    static D notify(const std::string& name, const T& data);

  private:
    NotifierManager() = default;
    ~NotifierManager() = default;
    NotifierManager(const NotifierManager&) = delete;
    NotifierManager& operator=(const NotifierManager&) = delete;

    std::unordered_map<std::string, std::shared_ptr<void>> notifiers_; ///< Stores the notifiers
    std::mutex mutex_;                                                 ///< Mutex for thread-safe access
};

/**
 * @brief Handles notification subscriptions and notifications.
 * @tparam T The type of data the notifier will handle.
 * @tparam D The return type of the subscriber function (defaults to void).
 *
 * @see NotifierManager
 * @see Subscriber
 */
template <typename T, typename D = void>
class Notifier
{
  public:
    using SubscriptionId = std::size_t; ///< Type alias for subscription ID

    /**
     * @brief Constructs a new Notifier object.
     *
     * @note This constructor is private. Use the NotifierManager to get a notifier.
     * @see NotifierManager::get_notifier()
     */
    Notifier();

    /**
     * @brief Notifies all subscribers with the provided data.
     * @param data The data to notify subscribers with.
     */
    D notify(const T& data);

    /**
     * @brief Subscribes a new subscriber.
     * @param subscriber The subscriber function to be called upon notification.
     * @return The subscription ID of the subscriber.
     */
    SubscriptionId subscribe(const std::function<D(const T&)>& subscriber);

    /**
     * @brief Unsubscribes a subscriber using its subscription ID.
     * @param id The subscription ID of the subscriber to be removed.
     */
    void unsubscribe(SubscriptionId id);

  private:
    std::unordered_map<SubscriptionId, std::function<D(const T&)>> subscribers_; ///< Stores subscribers
    std::mutex mutex_;                                                           ///< Mutex for thread-safe access
    std::atomic<SubscriptionId> nextId_; ///< Atomic counter for subscription IDs
};

/**
 * @brief Manages a subscription to a notifier.
 * @tparam T The type of data the notifier will handle.
 * @tparam D The return type of the notifier callback (defaults to void).
 */
template <typename T, typename D = void>
class Subscriber
{
  public:
    /**
     * @brief Constructs a new Subscriber object and subscribes to the notifier.
     * @tparam Func The type of the callback function.
     * @param name The name of the notifier to subscribe to.
     * @param callback The callback function to be called upon notification.
     */
    template <typename Func>
    Subscriber(const std::string& name, Func&& callback);

    /**
     * @brief Destroys the Subscriber object and unsubscribes from the notifier.
     */
    ~Subscriber();

  private:
    std::shared_ptr<Notifier<T, D>> notifier_;               ///< The notifier this subscriber is subscribed to
    typename Notifier<T, D>::SubscriptionId subscriptionId_; ///< The subscription ID of this subscriber
    std::function<D(const T&)> callback_;                    ///< The callback function to be called upon notification
};

// Include the implementation file for template functions
#include "notifier.hxx"

#endif // NOTIFIER_HH
