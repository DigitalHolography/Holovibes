#ifndef GPIB_H
# define GPIB_H

int load_batch_file(const char* filepath);
int execute_next_block(void);

#endif /* !GPIB_H */