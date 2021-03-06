/**
 * @file common.h
 * @brief Declaration of common data structs and utils.
 */

#ifndef WORD2VEC_COMMON_H_
# define WORD2VEC_COMMON_H_

//////////////
// Includes //
//////////////
#include <stdlib.h>

////////////
// Macros //
////////////

/** @brief Maximum length of stored string. */
# define MAX_STRING 100
/** @brief Maximum number of user-defined tasks to train. */
# define MAX_TASKS 1024
/** @brief Custom macro to prevent warning about unused variables. */
# define UNUSED(x) (void)(x)

//////////////
// typedefs //
//////////////

/**
 * @typedef opt_t
 * @brief command line options
 */
typedef struct opt opt_t;
/**
 * @typedef real
 * @brief default floating point type
 */
typedef float real;                    // Precision of float numbers

/////////////
// Structs //
/////////////

/**
 * @struct multiclass_t
 * @brief Statistics about specific tasks.
 *
 * This struct holds statistics about user-defined tasks such as the
 * total number of tasks and the maximum number of classes
 * distinguished by each task.
 */
typedef struct {
  /**
   * @brief Total number of user-defined tasks.
   */
  size_t m_n_tasks;
  /**
   * @brief Maximum number or specific labels for each particular task.
   */
  int m_classes[MAX_TASKS];
} multiclass_t;


/**
 * @struct opt
 * @brief command line options.
 */
struct opt {
  char m_train_file[MAX_STRING]; /**< name of the input file  */
  char m_output_file[MAX_STRING]; /**< name of the output file  */

  long long m_layer1_size;	/**< dimensionality of the embeddings */
  long long m_iter;		/**< number of iterations to run */

  real m_alpha;			/**< Update rate for gradient descent.  */
  real m_sample;		/**< randomly discard frequent words
				   while keeping the ranking same */
  int m_binary;			/**< Store resulting embeddings in the binary format. */
  int m_cbow;			/**< Use continuous bag of words if > 0. */
  int m_debug_mode;		/**< Turn on debug messages. */
  int m_hs;			/**< Use hierarchical softmax if > 0.  */
  int m_min_count;		/**< Minimum number of occurrences for a word to be analyzed. */
  int m_min_reduce;		/**< Reduce vocabulary. */
  int m_negative;    /**< Use negative sampling for word2vec
			embeddings */
  int m_num_threads;		/**< Maximum number of threads to use. */
  int m_ts;			/**< Train task-specific embeddings only. */
  int m_window;			/**< Size of context window. */
  int m_ts_w2v;			/**< Simultaneously train
				   task-specific and word2vec
				   embeddings.  */
  /**
   * @brief mapword2vec to task-specific vectors using least-squares
   *
   *  Train word2vec and task-specific embeddings, applying the
   *  least-squares method in the end to map the former vectors to the
   *  latter representation.
   */
  int m_ts_least_sq;
};

/**
 * @struct nnet_t
 * @brief Neural network data.
 */
typedef struct {
  /**
   * @brief Actual word embedding layer.
   */
  real *m_syn0;
  /**
   * @brief Isolated task-specific word embeddings used in
   * least-squares method.
   */
  real *m_ts_syn0;
  /**
   * @brief Flags of tokens which were trained during task-specific
   * training.
   */
  short *m_ts_syn0_active;
  /**
   * @brief Word embedding 2 output layer for hierarchical softmax.
   */
  real *m_syn1;
  /**
   * @brief Word embedding 2 output layer for negative sampling.
   */
  real *m_syn1neg;
  /**
   * @brief Task-specific embeddings (used in the least squares
   * method)
   */
  real *m_syn0_ts;
  /**
   * @brief Number of user-defined tasks.
   */
  size_t m_n_tasks;
  /**
   * @brief Word embedding 2 output layers for specific tasks.
   */
  real **m_vec2task;
} nnet_t;

/////////////
// Methods //
/////////////

/**
 * Set options to their default values.
 *
 * @param opt - pointer to an #opt struct whose values should be set
 * to default
 *
 * @return \c void
 */
void reset_opt(opt_t *opt);

#endif  /* ifndef WORD2VEC_COMMON_H_ */
