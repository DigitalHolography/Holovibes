#ifndef CAMERA_HH
# define CAMERA_HH

//Struct of a 2D coordinate
template <typename T>
struct coord2D
{
  T x;
  T y;
};

class Camera
{
 public:
  Camera(char* name);
  ~Camera();

  //Getters, setters
  char* getName();

  int get_support_external_buffer();
  int get_support_non_paged_buffer();
  int get_has_been_externally_allocated();
  void set_support_external_buffer();
  void set_support_non_paged_buffer();
  void set_has_been_externally_allocated();

 private:
  char* _name;

  //Buffer flags
  int _support_external_buffer;
  int _support_non_paged_buffer;
  int _has_been_externally_allocated;
};

#endif /* !CAMERA_HH */
