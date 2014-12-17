#ifndef GPIB_H
# define GPIB_H

#ifdef __cplusplus
extern "C" {
#endif
  int load_batch_file(const char* filepath);
  int execute_next_block(void);
#ifdef __cplusplus
};
#endif

#endif /* !GPIB_H */