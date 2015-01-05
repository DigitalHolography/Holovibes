#ifndef DEVICE_INIT_HH
# define DEVICE_INIT_HH

#include "gpib_controller.h"

int initialize_board(int board_index);
Addr4882_t *find_listeners(int board_index);
int clear_board(int device_handler);

#endif /* !DEVICE_INIT_HH */