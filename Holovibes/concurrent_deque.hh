#pragma once

# include <deque>
# include <mutex>
# include <array>

namespace holovibes
{
  using guard = std::lock_guard<std::mutex>;

  /*! \brief This class is a thread safe wrapper on std::deque.
   *
   * It is used mainly to store average/ROI values.
   * Every method locks a mutex, do the action and delocks the mutex.
   */
  template <class T> class ConcurrentDeque
  {
  public:
    using iterator = typename
      std::deque<T>::iterator;

    using reverse_iterator = typename
      std::deque<T>::reverse_iterator;

  public:
    ConcurrentDeque()
    {
    }

    ~ConcurrentDeque()
    {
    }

    iterator begin()
    {
      guard guard(mutex_);
      return deque_.begin();
    }

    iterator end()
    {
      guard guard(mutex_);
      return deque_.end();
    }

    reverse_iterator rbegin()
    {
      guard guard(mutex_);
      return deque_.rbegin();
    }

    reverse_iterator rend()
    {
      guard guard(mutex_);
      return deque_.rend();
    }

    size_t size() const
    {
      guard guard(mutex_);
      return deque_.size();
    }

    void resize(unsigned int new_size)
    {
      guard guard(mutex_);
      deque_.resize(new_size);
    }

    bool empty() const
    {
      guard guard(mutex_);
      return deque_.empty();
    }

    T& operator[](unsigned int index)
    {
      return deque_[index];
    }

    void push_back(const T& elt)
    {
      guard guard(mutex_);
      deque_.push_back(elt);
    }

    void push_front(const T& elt)
    {
      guard guard(mutex_);
      deque_.push_front(elt);
    }

    void pop_back()
    {
      guard guard(mutex_);
      deque_.pop_back();
    }

    void pop_front()
    {
      guard guard(mutex_);
      deque_.pop_front();
    }

    void clear()
    {
      guard guard(mutex_);
      deque_.clear();
    }

    /*! \brief Fill a given vector with deque values
    **
    ** \param vect Vector to fill
    ** \param nb_elts Number of elements to copy
    */
    size_t fill_array(std::vector<T>& vect, size_t nb_elts)
    {
      guard guard(mutex_);

      reverse_iterator q_end = deque_.rbegin();
      unsigned limit = std::min(nb_elts, deque_.size());
      std::advance(q_end, limit);

      std::transform(deque_.rbegin(),
        q_end,
        vect.begin(),
        [](T& elt) { return elt; });

      return limit;
    }

  private:
    std::deque<T> deque_;
    mutable std::mutex mutex_;
  };
}