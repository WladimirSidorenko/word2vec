#ifndef __WORD2VEC_COMMON_H__
# define __WORD2VEC_COMMON_H__

//////////////
// Includes //
//////////////
#include <stdlib.h>

////////////
// Macros //
////////////
# define MAX_STRING 100
# define MAX_TASKS 1024

//////////////
// typedefs //
//////////////
typedef struct opt opt_t;
typedef float real;                    // Precision of float numbers

/////////////
// Structs //
/////////////

/**
 * Statistics about multiple task classes.
 */
typedef struct {
  //< number of tasks in a multitask setting
  size_t m_n_tasks;
  //< maximum number of classes for each task
  size_t m_max_classes[MAX_TASKS];
} multiclass_t;


struct opt {
  char m_train_file[MAX_STRING];
  char m_output_file[MAX_STRING];

  long long m_layer1_size;
  long long m_iter;

  real m_alpha;
  real m_sample;

  int m_binary;
  int m_cbow;
  int m_debug_mode;
  int m_hs;
  int m_least_sq;
  int m_min_count;
  int m_min_reduce;
  int m_negative;
  int m_num_threads;
  int m_task_specific;
  int m_window;
  int m_w2v;
};

/////////////
// Methods //
/////////////

void reset_opt(opt_t *opt);

opt_t *create_opt(void);

#endif  /* ifndef __WORD2VEC_COMMON_H__ */
