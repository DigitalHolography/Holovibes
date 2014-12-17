// gpib_controller.cpp : Defines the entry point for the console application.
//
#include "gpib_controller.h"
#include "include\gpib.h"

global_s global;

void send_data(int device_handler,char *cmd) 
{
  ibsta = ibwrt(device_handler, cmd, strlen(cmd));
  if (ibsta & ERR)
    printf("Failed to send a command to the device \n");
  else
    printf("Command successfully sent to the device \n");

  if (cmd[strlen(cmd) - 2] == '?')
  {
    char *result = (char*)calloc(256, sizeof (char));
    ibrd(device_handler, result, 255);
    if (ibsta & ERR)
      printf("Failed receive a response from the device \n");
    else
      printf("Response successfully received from the device: %s", result);
    free(result);
  }
}


void free_cmd_tab()
{
  if (global.commands)
  {
    for (size_t i = 0; global.commands[i]; ++i)
      free(global.commands[i]);
    free(global.commands);
    global.commands = NULL;
  }
}

int load_batch_file(const char* filepath)
{
  initialize_board(0);
  global.next_to_execute = 1;
  free_cmd_tab();
  FILE* f = NULL;
  if (fopen_s(&f, filepath, "r") != 0)
    return -1;
  global.commands = parsefile(f);
  fclose(f);
  return 0;
}

int execute_next_block(void)
{
  if (global.commands)
    return(get_all_blocks(global.commands));

  return 0;
}
