/*! \file
 *
 * \brief A thread safe wrapper on std::deque.
 */
#pragma once

#include <algorithm>
#include <deque>
#include <mutex>
#include <tuple>

#undef max
#undef min

namespace holovibes
{
using LockGuard = std::lock_guard<std::mutex>;

/*! \class ConcurrentDeque
 *
 * \brief This class is a thread safe wrapper on std::deque.
 *
 * It is used mainly to store Chart values.
 * Every method locks a mutex, do the action and delocks the mutex.
 */
template <class T>
class ConcurrentDeque
{
  public:
    using iterator = typename std::deque<T>::iterator;

    using reverse_iterator = typename std::deque<T>::reverse_iterator;

  public:
    /*! \brief Constructor. */
    ConcurrentDeque() {}

    /*! \brief Destructor. */
    ~ConcurrentDeque() {}

    /*! \brief Returns the begin iterator. */
    iterator begin()
    {
        LockGuard guard(mutex_);
        return deque_.begin();
    }

    /*! \brief Returns the end iterator. */
    iterator end()
    {
        LockGuard guard(mutex_);
        return deque_.end();
    }

    /*! \brief Returns the rbegin iterator. */
    reverse_iterator rbegin()
    {
        LockGuard guard(mutex_);
        return deque_.rbegin();
    }

    /*! \brief Returns the rend iterator. */
    reverse_iterator rend()
    {
        LockGuard guard(mutex_);
        return deque_.rend();
    }

    /*! \brief Returns size of the queue. */
    size_t size() const
    {
        LockGuard guard(mutex_);
        return deque_.size();
    }

    /*! \brief Resize the queue. */
    void resize(unsigned int new_size)
    {
        LockGuard guard(mutex_);
        deque_.resize(new_size);
    }

    /*! \brief Checks if queue is empty. */
    bool empty() const
    {
        LockGuard guard(mutex_);
        return deque_.empty();
    }

    /*! \brief [] operator. */
    T& operator[](unsigned int index) { return deque_[index]; }

    /*! \brief Insert element to the back. */
    void push_back(const T& elt)
    {
        LockGuard guard(mutex_);
        deque_.push_back(elt);
    }

    /*! \brief Insert element to the front. */
    void push_front(const T& elt)
    {
        LockGuard guard(mutex_);
        deque_.push_front(elt);
    }

    /*! \brief Retrieve element from the back. */
    void pop_back()
    {
        LockGuard guard(mutex_);
        deque_.pop_back();
    }

    /*! \brief Retrieve element from the front. */
    void pop_front()
    {
        LockGuard guard(mutex_);
        deque_.pop_front();
    }

    /*! \brief Clear the queue. */
    void clear()
    {
        LockGuard guard(mutex_);
        deque_.clear();
    }

    /*! \brief Fill a given vector with deque values
     *
     * \param vect Vector to fill
     * \param nb_elts Number of elements to copy
     */
    size_t fill_array(std::vector<T>& vect, size_t nb_elts)
    {
        LockGuard guard(mutex_);

        reverse_iterator q_end = deque_.rbegin();
        size_t limit = std::min(nb_elts, deque_.size());
        std::advance(q_end, limit);

        std::transform(deque_.rbegin(), q_end, vect.begin(), [](T& elt) { return elt; });

        return limit;
    }

  private:
    std::deque<T> deque_;
    mutable std::mutex mutex_;
};
} // namespace holovibes
