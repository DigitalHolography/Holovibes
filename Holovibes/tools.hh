
#pragma once

template<typename Container>
void delete_them(Container& c) {
  std::for_each(c.begin(),
    c.end(),
    [](Container::value_type module) { delete module; });
  c.clear();
}