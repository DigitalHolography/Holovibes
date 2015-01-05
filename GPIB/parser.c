#include "parser.h"
#include <Windows.h>

extern global_s global;
int parse_options(int argc, char *argv[])
{
  global.file = NULL;
  global.bloc_number = NULL;
  if (argc == 5)
  {
    if (strcmp(argv[1], "-f") == 0)
      global.file = argv[2];
    if (strcmp(argv[1], "-b") == 0)
      global.bloc_number = argv[2];
    if (strcmp(argv[3], "-f") == 0)
      global.file = argv[4];
    if (strcmp(argv[3], "-b") == 0)
      global.bloc_number = argv[4];
    if (global.file && global.bloc_number)
      return 1;
  }
  printf("Not enough arguments passed to the program.\n");
  return 0;
}

char **parsefile(FILE *cmd_file)
{
  char **cmd_tab = (char **)calloc(TABSIZE, sizeof(char *));
  char *buffer = (char *)malloc(25000 * sizeof (char));
  for (int i = 0; i < TABSIZE; i++)
  {
    fgets(buffer, 25000, cmd_file);
    size_t line_length = strlen(buffer);
    char *to_store = (char*)calloc(line_length + 1, sizeof(char));
    memcpy(to_store, buffer, line_length + 1);
    if (feof(cmd_file))
      break;
    else
      cmd_tab[i] = to_store;
  }
  free(buffer);
  return cmd_tab;
}

void get_bloc(char *bloc_number, char **cmd_tab)
{
  char *bloc = (char *)malloc(sizeof(char)* (9 + strlen(bloc_number)));
  memcpy(bloc, "#Block ", 7 * sizeof(char));
  memcpy(bloc + 7, bloc_number, strlen(bloc_number) + 1);
  bloc[9 + strlen(bloc_number) - 2] = '\n';
  bloc[9 + strlen(bloc_number) - 1] = '\0';
  printf("Requested block: %s", bloc);
  for (int i = 0; i < TABSIZE; i++)
  {
    if (cmd_tab[i])
    {
      if (strcmp(bloc, cmd_tab[i]) == 0)
      {
        i++;
        printf("Requested block found\n");
        while (cmd_tab[i] && i < TABSIZE && strncmp("#Block", cmd_tab[i], 6) != 0)
        {
          manage_cmd(cmd_tab[i]);
          i++;
        }
        printf("End of block reached\n");
        break;
      }
    }
    else
    {
      printf("%s not found\n", bloc);
      break;
    }
  }
}

int get_all_blocks(char **cmd_tab) // return 1 if another block exist else return 0
{
  while (cmd_tab[global.next_to_execute] &&
    global.next_to_execute < TABSIZE - 1 &&
    strncmp("#Block", cmd_tab[global.next_to_execute], 6) != 0)
  {
    manage_cmd(cmd_tab[global.next_to_execute]);
    global.next_to_execute++;
  }
  printf("End of block reached\n");
  global.next_to_execute++;
  if (cmd_tab[global.next_to_execute])
    return 1;
  return 0;
}

void manage_cmd(char *cmd)
{
  if (strncmp(cmd, "#Wait ", 6) == 0)
  {
    char *time = (char*)calloc(256, sizeof (char));
    strncpy_s(time, 256, cmd + 6, strlen(cmd + 6) - 1);
    int wait_time = atoi(time);
    printf("Wait %i ms\n", wait_time);
    Sleep(wait_time);
    free(time);
  }
  else if (strncmp(cmd, "#InstrumentAddress", strlen("#InstrumentAddress")) == 0)
  {
    char *address = (char*)calloc(256, sizeof(char));
    strncpy_s(address, 256, cmd + strlen("InstrumentAddress "), strlen("InstrumentAddress ") - 1);
    int new_address = atoi(address);
    free(address);
    printf("Current address: %i\n", new_address);
    global.actual_adress = new_address;
    for (int i = 0; i < global.nb_devices; i++)
    {
      if (global.device_tab[i].adress == new_address)
      {
        global.device_handler = global.device_tab[i].device_handler;
        printf("Device at address %i will receive commands now\n", new_address);
      }
    }
  }
  else if (cmd[0] != '#' && cmd[0] != '\n' && cmd[0] != ' ')
  {
    printf("Sent: %s", cmd);
    send_data(global.device_handler, cmd);
  }
}