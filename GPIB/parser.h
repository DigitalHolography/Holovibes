#ifndef PARSER_H
# define PARSER_H

#include "gpib_controller.h"

char **parsefile(FILE *cmd_file);
void get_block(char *bloc_number, char **cmd_tab);
int get_all_blocks(char **cmd_tab);
void manage_cmd(char *cmd);

#endif