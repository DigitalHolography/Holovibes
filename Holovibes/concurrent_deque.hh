#ifndef CONCURRENT_DEQUE_HH
# define CONCURRENT_DEQUE_HH

# include <deque>
# include <mutex>
# include <array>

namespace holovibes
{
  /*! \class Concurrent Deque
  **
  ** This class is a thread safe wrapper on std::deque.
  ** It is used mainly to store average/ROI values.
  ** Every method locks a mutex, do the action and delocks the mutex.
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
      std::deque<T>::iterator it;

      mutex_.lock();
      it = deque_.begin();
      mutex_.unlock();

      return it;
    }

    iterator end()
    {
      std::deque<T>::iterator it;

      mutex_.lock();
      it = deque_.end();
      mutex_.unlock();

      return it;
    }

    reverse_iterator rbegin()
    {
      std::deque<T>::reverse_iterator rit;

      mutex_.lock();
      rit = deque_.rbegin();
      mutex_.unlock();

      return rit;
    }

    reverse_iterator rend()
    {
      std::deque<T>::reverse_iterator rit;

      mutex_.lock();
      rit = deque_.rend();
      mutex_.unlock();

      return rit;
    }
    
    size_t size()
    {
      size_t s;

      mutex_.lock();
      s = deque_.size();
      mutex_.unlock();

      return s;
    }

    void resize(unsigned int new_size)
    {
      mutex_.lock();
      deque_.resize(new_size);
      mutex_.unlock();
    }

    bool empty()
    {
      bool is_empty;

      mutex_.lock();
      is_empty = deque_.empty();
      mutex_.unlock();

      return is_empty;
    }

    T& operator[](unsigned int index)
    {
      return deque_[index];
    }

    void push_back(const T& elt)
    {
      mutex_.lock();
      deque_.push_back(elt);
      mutex_.unlock();
    }

    void push_front(const T& elt)
    {
      mutex_.lock();
      deque_.push_front(elt);
      mutex_.unlock();
    }

    void pop_back()
    {
      mutex_.lock();
      deque_.pop_back();
      mutex_.unlock();
    }

    void pop_front()
    {
      mutex_.lock();
      deque_.pop_front();
      mutex_.unlock();
    }

    void clear()
    {
      mutex_.lock();
      deque_.clear();
      mutex_.unlock();
    }

    /*! \brief Fill a given vector with deque values
    **
    ** \param vect vecto to fill
    ** \param nb_elts number of elements to copy
    */
    size_t fill_array(std::vector<T>& vect, size_t nb_elts)
    {
      mutex_.lock();
      unsigned int i = 0;

      for (auto it = deque_.rbegin(); it != deque_.rend() && i < nb_elts; ++it)
      {
        vect[i] = *it;
        ++i;
      }
      mutex_.unlock();

      return i;
    }

  private:
    std::deque<T> deque_;
    std::mutex mutex_;
  };
}

#endif /* !CONCURRENT_DEQUE_HH */