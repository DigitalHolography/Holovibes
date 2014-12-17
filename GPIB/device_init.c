#include "device_init.h"

global_s global;

int initialize_board(int board_index) // return 1 on success or 0 fail
{
  int device_handle = -1;
  SendIFC(0);
  if (ibsta & ERR)
    printf("Unable to open the board \n");
  else
    printf("Board found\n");
  Addr4882_t *listeners_list = find_listeners(board_index);
  if (listeners_list)
  {
    int i = 0;
    global.device_tab = (device_s*)calloc(global.nb_devices, sizeof(device_s));
    while (i < global.nb_devices)
    {
      global.device_tab[i].device_handler = ibdev(board_index, GetPAD(listeners_list[i]), GetSAD(listeners_list[i]), TIMEOUT, EOTMODE, EOSMODE);
      global.device_tab[i].adress = listeners_list[i];
      if (global.device_tab[i].device_handler == -1)
        printf("Unable to open the device at address %i\n", i);
      else
        printf("Device at address %i successfully opened\n", listeners_list[i]);
      i++;
    }
  }
  else
    return 0;
  if (!clear_board(device_handle))
    return 0;
  for (int i = 0; i < global.nb_devices; i++)
  {
    ibrsc(global.device_tab[i].device_handler, 0);
    ibsic(global.device_tab[i].device_handler);
  }
  return 1;
}

Addr4882_t *find_listeners(int board_index)
{
  int nbadress = 31;
  Addr4882_t *adress_to_test = (Addr4882_t*)malloc(sizeof (Addr4882_t)* nbadress);
  if (adress_to_test)
  {
    for (int i = 0; i < 30; i++)
      adress_to_test[i] = (Addr4882_t)(i + 1);
    adress_to_test[30] = NOADDR;

    Addr4882_t *listeners = (Addr4882_t*)calloc(1, sizeof (Addr4882_t)* nbadress);
    FindLstn(board_index, adress_to_test, listeners, nbadress);
    int device_found = ibcntl;
    printf("%i device(s) found on GPIB\n", device_found);
    if (iberr & EBUS)
    {
      printf("No device connected to the GPIB interface- \n");
      free(adress_to_test);
      free(listeners);
      return NULL;
    }
    free(adress_to_test);
    global.nb_devices = device_found;
    return listeners;
  }
  return NULL;
}

int clear_board(int device_handler)
{
  ibclr(device_handler);
  if (ibsta & ERR)
  {
    printf("Device can't be cleared \n");
    return 0;
  }
  else
    printf("Device successfully cleared \n");
  return 1;
} 
