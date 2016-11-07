#ifndef WORD2VEC_COMMON_H_
# define WORD2VEC_COMMON_H_

//////////////
// Includes //
//////////////
#include <stdlib.h>

////////////
// Macros //
////////////
# define MAX_STRING 100
# define MAX_TASKS 1024
# define UNUSED(x) (void)(x)

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
  int m_min_count;
  int m_min_reduce;
  int m_negative;
  int m_num_threads;
  /*< train task-specific embeddings only */
  int m_ts;
  int m_window;
  /*< train task-specific embeddings along with the normal word2vec
   *  objective */
  int m_ts_w2v;
  /*< train word2vec and task-specific embeddings, applying the
   *  least-squares method in the end to map the former vectors to the
   *  latter representation */
  int m_ts_least_sq;
};

/////////////
// Methods //
/////////////

void reset_opt(opt_t *opt);

opt_t *create_opt(void);

#endif  /* ifndef WORD2VEC_COMMON_H_ */
