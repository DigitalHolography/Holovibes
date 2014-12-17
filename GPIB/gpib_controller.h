#ifndef GPIB_CONTROLLER_HH
#define GPIB_CONTROLLER_HH
#define PRIMARY_ADDR_OF_DMM   1     // Primary address of device
#define NO_SECONDARY_ADDR     0     // Secondary address of device
#define TIMEOUT               T10s  // Timeout value = 10 seconds
#define EOTMODE               1     // Enable the END message
#define EOSMODE               0     // Disable the EOS mode
#define TABSIZE               1000000

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <conio.h>
#include <ni4882.h>

#include "parser.h"
#include "device_init.h"

typedef struct device
{
  int adress;
  int device_handler;
}device_s;

typedef struct global_s
{
  int next_to_execute;
  char **commands;
  int nb_devices;
  int device_handler;
  int actual_adress;
  device_s *device_tab;
  char *file;
  char *bloc_number;
}global_s;

void send_data(int device_handler, char *cmd);
#endif