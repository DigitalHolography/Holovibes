
#pragma once

template<typename Container, typename Functor>
void delete_them(Container& c, const Functor& f) {
  std::for_each(c.begin(),
    c.end(),
    f);
  c.clear();
}