//////////////
// Includes //
//////////////
#include "common.h"
#include "train.h"
#include "vocab.h"
#include "w2vio.h"

#include <math.h>
#include <pthread.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>  /* memset */
#include <time.h>

/////////////
// Structs //
/////////////

/**
 * \struct thread_opts_t
 * \brief options required by each thread
 *
 */
typedef struct {
  clock_t m_start;
  size_t m_file_size;
  real m_alpha;
  real m_starting_alpha;
  long m_thread_id;
  const opt_t *m_w2v_opts;
  const vocab_t *m_vocab;
  nnet_t *m_nnet;
  const real *m_exp_table;
  const int *m_ugram_table;
  size_t m_n_tasks;
} thread_opts_t;

///////////////
// Constants //
///////////////
pthread_mutex_t tlock;

/////////////
// Methods //
/////////////
static void init_mtx(void **a_mtx, long long a_size) {
  long long ret = posix_memalign(a_mtx, 128, a_size);
  if (ret) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(EXIT_FAILURE);
  }
}

static void copy_thread_opts(const thread_opts_t *src_opts,
                             thread_opts_t *trg_opts, long a_thread_id) {
  trg_opts->m_start = clock();
  trg_opts->m_file_size = src_opts->m_file_size;
  trg_opts->m_alpha = src_opts->m_alpha;
  trg_opts->m_starting_alpha = src_opts->m_starting_alpha;
  trg_opts->m_thread_id = a_thread_id;
  trg_opts->m_w2v_opts = src_opts->m_w2v_opts;
  trg_opts->m_vocab = src_opts->m_vocab;
  trg_opts->m_nnet = src_opts->m_nnet;
  trg_opts->m_exp_table = src_opts->m_exp_table;
  trg_opts->m_ugram_table = src_opts->m_ugram_table;
  trg_opts->m_n_tasks = src_opts->m_n_tasks;
}

static void reset_multiclass(multiclass_t *a_multiclass) {
  a_multiclass->m_n_tasks = 0;
  memset(a_multiclass->m_classes, -1, sizeof(int) * MAX_TASKS);
}

static real *init_exp_table(void) {
  real *exp_table = (real *) malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  int i;
  for (i = 0; i < EXP_TABLE_SIZE; ++i) {
    exp_table[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    exp_table[i] = exp_table[i] / (exp_table[i] + 1); // Precompute f(x) = x / (x + 1)
  }
  return exp_table;
}

static void reset_nnet(nnet_t *a_nnet) {
  /*@null@*/
  a_nnet->m_syn0 = NULL;
  /*@null@*/
  a_nnet->m_ts_syn0 = NULL;
  /*@null@*/
  a_nnet->m_syn1 = NULL;
  /*@null@*/
  a_nnet->m_syn1neg = NULL;

  a_nnet->m_n_tasks = 0;
  a_nnet->m_vec2task = NULL;
}

static void free_nnet(nnet_t *a_nnet) {
  free(a_nnet->m_syn0);
  free(a_nnet->m_ts_syn0);
  free(a_nnet->m_syn1);
  free(a_nnet->m_syn1neg);

  size_t i;
  for (i = 0; i < a_nnet->m_n_tasks; ++i) {
    free(a_nnet->m_vec2task[i]);
  }
  free(a_nnet->m_vec2task);
  reset_nnet(a_nnet);
}

static void init_ts_nnet(nnet_t *a_nnet, const vocab_t *a_vocab,
                         const opt_t *a_opts, const multiclass_t *a_multiclass) {
  unsigned long long next_random = 1;
  long long layer1_size = a_opts->m_layer1_size;
  long long vocab_size = a_vocab->m_vocab_size;
  a_nnet->m_n_tasks = a_multiclass->m_n_tasks;
  /* allocate memory for task-specific weight matrices */
  a_nnet->m_vec2task = calloc(a_nnet->m_n_tasks, sizeof(real *));
  if (a_nnet->m_vec2task == NULL) {
    fprintf(stderr, "Could not allocate memory for m_vec2task.\n");
    exit(8);
  }
  long long a, n;
  size_t i;
  real *v2t_layer;
  for (i = 0; i < a_nnet->m_n_tasks; ++i) {
    n = a_multiclass->m_classes[i] * layer1_size;
    init_mtx((void **) &a_nnet->m_vec2task[i], n * sizeof(real));
    v2t_layer = a_nnet->m_vec2task[i];
    for (a = 0; a < n; ++a) {
      next_random = next_random * (unsigned long long)25214903917 + 11;
      v2t_layer[a] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
    }
  }

  /* initialize array for storing task-specific embeddings */
  void **layer_address = NULL;

  if (a_opts->m_ts_w2v > 0) {
    return;
  } else if (a_opts->m_ts > 0) {
    layer_address = (void **) &a_nnet->m_syn0;
  else if (a_opts->m_ts_least_sq > 0)
    layer_address = (void **) &a_nnet->m_ts_syn0;

  if (layer_address) {
    init_mtx(layer_address,
	     vocab_size * layer1_size * sizeof(real));
    real *layer = (real *) (*layer_address);
    long long b;
    for (a = 0; a < vocab_size; ++a) {
      for (b = 0; b < layer1_size; ++b) {
        next_random = next_random * (unsigned long long)25214903917 + 11;
        layer[a * layer1_size + b] = (((next_random & 0xFFFF)
                                       / (real)65536) - 0.5) / layer1_size;
      }
    }
  }
}

static void init_w2v_nnet(nnet_t *a_nnet, const vocab_t *a_vocab, const opt_t *a_opts) {
  long long a, b;
  unsigned long long next_random = 1;
  long long vocab_size = a_vocab->m_vocab_size;
  long long layer1_size = a_opts->m_layer1_size;
  init_mtx((void **) &a_nnet->m_syn0, vocab_size * layer1_size * sizeof(real));

  if (a_opts->m_hs) {
    init_mtx((void **) &a_nnet->m_syn1,
	     vocab_size * layer1_size * sizeof(real));
    for (a = 0; a < vocab_size; ++a)
      for (b = 0; b < layer1_size; ++b)
        a_nnet->m_syn1[a * layer1_size + b] = 0;
  }
  if (a_opts->m_negative > 0) {
    init_mtx((void **) &a_nnet->m_syn1neg,
	     vocab_size * layer1_size * sizeof(real));
    for (a = 0; a < vocab_size; ++a)
      for (b = 0; b < layer1_size; ++b)
        a_nnet->m_syn1neg[a * layer1_size + b] = 0;
  }

  for (a = 0; a < vocab_size; ++a) {
    for (b = 0; b < layer1_size; ++b) {
      next_random = next_random * (unsigned long long)25214903917 + 11;
      a_nnet->m_syn0[a * layer1_size + b] = (((next_random & 0xFFFF)
                                              / (real)65536) - 0.5) / layer1_size;
    }
  }
}

static void init_nnet(nnet_t *a_nnet, vocab_t *a_vocab,
                     const opt_t *a_opts, const multiclass_t *a_multiclass) {
  reset_nnet(a_nnet);
  if (a_opts->m_ts > 0 || a_opts->m_ts_w2v > 0 || a_opts->m_ts_least_sq > 0)
    init_ts_nnet(a_nnet, a_vocab, a_opts, a_multiclass);

  if (a_opts->m_ts <= 0)
    init_w2v_nnet(a_nnet, a_vocab, a_opts);
}

static real train_w2v(const opt_t *w2v_opts,
                      const vw_t *vocab, const long long vocab_size,
                      const real *exp_table, const int *table,
                      const int window, const long long layer1_size,
                      nnet_t *nnet, long long sen[], long long word,
                      real *neu1, real *neu1e,
                      long long sentence_length,
                      long long sentence_position,
                      unsigned long long *next_random) {
  real f, g, total_cost = 0;
  long long a, b, c, cw, d, last_word, label, l1, l2, target;

  for (c = 0; c < layer1_size; ++c)
    neu1[c] = 0;

  for (c = 0; c < layer1_size; ++c)
    neu1e[c] = 0;

  *next_random = (*next_random) * (unsigned long long)25214903917 + 11;
  b = (*next_random) % window;
  if (w2v_opts->m_cbow) {  //train the cbow architecture
    // in -> hidden
    cw = 0;
    for (a = b; a < window * 2 + 1 - b; ++a) {

      if (a != window) {
        c = sentence_position - window + a;
        if (c < 0)
          continue;

        if (c >= sentence_length)
          continue;

        last_word = sen[c];
        if (last_word == -1)
          continue;

        for (c = 0; c < layer1_size; ++c) {
          neu1[c] += nnet->m_syn0[c + last_word * layer1_size];
        }
        ++cw;
      }
    }
    if (cw) {
      for (c = 0; c < layer1_size; ++c)
        neu1[c] /= cw;

      if (w2v_opts->m_hs) {
        for (d = 0; d < vocab[word].codelen; ++d) {
          f = 0;
          l2 = vocab[word].point[d] * layer1_size;
          // Propagate hidden -> output
          pthread_mutex_lock(&tlock);
          for (c = 0; c < layer1_size; ++c)
            f += neu1[c] * nnet->m_syn1[c + l2];

          if (f <= -MAX_EXP || f >= MAX_EXP)
            continue;
          else
            f = exp_table[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];

          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f);
          total_cost += g;
          g *= w2v_opts->m_alpha;
          // Propagate errors output -> hidden
          for (c = 0; c < layer1_size; ++c) {
            neu1e[c] += g * nnet->m_syn1[c + l2];
          }
          // Learn weights hidden -> output
          for (c = 0; c < layer1_size; ++c) {
            nnet->m_syn1[c + l2] += g * neu1[c];
          }
          pthread_mutex_unlock(&tlock);
        }
      }
      // NEGATIVE SAMPLING
      if (w2v_opts->m_negative > 0)
        for (d = 0; d < w2v_opts->m_negative + 1; ++d) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            *next_random = (*next_random) * (unsigned long long)25214903917 + 11;
            target = table[((*next_random) >> 16) % TABLE_SIZE];
            if (target == 0) target = (*next_random) % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0;
          pthread_mutex_lock(&tlock);
          for (c = 0; c < layer1_size; ++c) {
            f += neu1[c] * nnet->m_syn1neg[c + l2];
          }
          if (f > MAX_EXP) {
            g = (label - 1) * w2v_opts->m_alpha;
          } else if (f < -MAX_EXP) {
            g = (label - 0) * w2v_opts->m_alpha;
          } else {
            g = (label - exp_table[(int)((f + MAX_EXP)
                                         * (EXP_TABLE_SIZE / MAX_EXP / 2))]);
            total_cost += g;
            g *= w2v_opts->m_alpha;
          }
          for (c = 0; c < layer1_size; ++c) {
            neu1e[c] += g * nnet->m_syn1neg[c + l2];
          }
          for (c = 0; c < layer1_size; ++c) {
            nnet->m_syn1neg[c + l2] += g * neu1[c];
          }
          pthread_mutex_unlock(&tlock);
        }
      // hidden -> in
      for (a = b; a < window * 2 + 1 - b; ++a) {
        if (a != window) {
          c = sentence_position - window + a;
          if (c < 0)
            continue;

          if (c >= sentence_length)
            continue;

          last_word = sen[c];
          if (last_word == -1)
            continue;

          pthread_mutex_lock(&tlock);
          for (c = 0; c < layer1_size; ++c) {
            nnet->m_syn0[c + last_word * layer1_size] += neu1e[c];
          }
          pthread_mutex_unlock(&tlock);
        }
      }
    }
  } else {  //train skip-gram
    for (a = b; a < window * 2 + 1 - b; ++a) {
      if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        l1 = last_word * layer1_size;
        for (c = 0; c < layer1_size; ++c)
          neu1e[c] = 0;
        // HIERARCHICAL SOFTMAX
        if (w2v_opts->m_hs) for (d = 0; d < vocab[word].codelen; ++d) {
            f = 0;
            l2 = vocab[word].point[d] * layer1_size;
            // Propagate hidden -> output
            pthread_mutex_lock(&tlock);
            for (c = 0; c < layer1_size; c++)
              f += nnet->m_syn0[c + l1] * nnet->m_syn1[c + l2];

            if (f <= -MAX_EXP || f >= MAX_EXP)
              continue;
            else
              f = exp_table[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];

            // 'g' is the gradient multiplied by the learning rate
            g = (1 - vocab[word].code[d] - f);
            total_cost += g;
            g *= w2v_opts->m_alpha;
            // Propagate errors output -> hidden
            for (c = 0; c < layer1_size; ++c) neu1e[c] += g * nnet->m_syn1[c + l2];
            // Learn weights hidden -> output
            for (c = 0; c < layer1_size; ++c) {
              nnet->m_syn1[c + l2] += g * nnet->m_syn0[c + l1];
            }
            pthread_mutex_unlock(&tlock);
          }
        // NEGATIVE SAMPLING
        if (w2v_opts->m_negative > 0)
          for (d = 0; d < w2v_opts->m_negative + 1; ++d) {
            if (d == 0) {
              target = word;
              label = 1;
            } else {
              (*next_random) = (*next_random) * (unsigned long long)25214903917 + 11;
              target = table[(*next_random >> 16) % TABLE_SIZE];
              if (target == 0) target = (*next_random) % (vocab_size - 1) + 1;
              if (target == word) continue;
              label = 0;
            }
            l2 = target * layer1_size;
            f = 0;
            pthread_mutex_lock(&tlock);
            for (c = 0; c < layer1_size; ++c)
              f += nnet->m_syn0[c + l1] * nnet->m_syn1neg[c + l2];

            total_cost += f;
            if (f > MAX_EXP)
              g = (label - 1) * w2v_opts->m_alpha;
            else if (f < -MAX_EXP)
              g = (label - 0) * w2v_opts->m_alpha;
            else
              g = (label - exp_table[(int)((f + MAX_EXP)
                                           * (EXP_TABLE_SIZE / MAX_EXP / 2))])
                  * w2v_opts->m_alpha;

            for (c = 0; c < layer1_size; ++c)
              neu1e[c] += g * nnet->m_syn1neg[c + l2];

            for (c = 0; c < layer1_size; ++c) {
              nnet->m_syn1neg[c + l2] += g * nnet->m_syn0[c + l1];
            }
            pthread_mutex_unlock(&tlock);
          }
        // Learn weights input -> hidden
        pthread_mutex_lock(&tlock);
        for (c = 0; c < layer1_size; ++c) {
          nnet->m_syn0[c + l1] += neu1e[c];
        }
        pthread_mutex_unlock(&tlock);
      }
    }
  }
  return total_cost;
}

static real train_ts(const multiclass_t  *multiclass, long long word,
                     const real alpha, const int active_tasks,
                     const long long layer1_size, const real *exp_table,
                     nnet_t *nnet, real *embeddings) {
  real f, g, emb_orig, total_cost = 0;
  long long l2, w_idx = word * layer1_size;
  real *label_weights = NULL;
  size_t i;
  int label, j = 0, c = 0;
  for (i = 0; i < multiclass->m_n_tasks && j < active_tasks; ++i) {
    label = multiclass->m_classes[i];
    if (label >= 0) {
      ++j;
      f = 0;

      label_weights = nnet->m_vec2task[i];
      l2 = label * layer1_size;
      pthread_mutex_lock(&tlock);
      /* compute decision */
      for (c = 0; c < layer1_size; ++c)
        f += embeddings[c + w_idx]  * label_weights[l2 + c];

      /* compute gradient */
      if (f > MAX_EXP) {
        /* for maximum scores, do nothing */
        pthread_mutex_unlock(&tlock);
        continue;
      } else {
        if (f < -MAX_EXP)
          f = -MAX_EXP;

        g = exp_table[(int)((f + MAX_EXP)
                            * (EXP_TABLE_SIZE / MAX_EXP / 2))];
        total_cost += 1 - g;
        g *= alpha;
        /* propagate gradient to word embeddings and task-specific coefficients */
        for (c = 0; c < layer1_size; ++c) {
          emb_orig = embeddings[c + w_idx];
          embeddings[c + w_idx] += g * label_weights[l2 + c];
          label_weights[l2 + c] += g * emb_orig;
        }
        pthread_mutex_unlock(&tlock);
      }
    }
  }
  return total_cost;
}

static void *train_model_thread(void *a_opts) {
  real total_cost = 0;
  thread_opts_t *thread_opts = (thread_opts_t *) a_opts;
  const long long file_size = thread_opts->m_file_size;
  nnet_t *nnet = thread_opts->m_nnet;
  const real *exp_table = thread_opts->m_exp_table;
  const int *table = thread_opts->m_ugram_table;
  const vw_t *vocab = thread_opts->m_vocab->m_vocab;
  const int *vocab_hash = thread_opts->m_vocab->m_vocab_hash;
  const long long vocab_size = thread_opts->m_vocab->m_vocab_size;
  const long long train_words = thread_opts->m_vocab->m_train_words;

  const opt_t *w2v_opts = thread_opts->m_w2v_opts;
  const long long thread_id = (long long) thread_opts->m_thread_id;
  const int window = w2v_opts->m_window;;
  const long long layer1_size = w2v_opts->m_layer1_size;
  const long long num_threads = w2v_opts->m_num_threads;
  const real sample = w2v_opts->m_sample;
  const int consume_tab = w2v_opts->m_ts <= 0 && w2v_opts->m_ts_w2v <= 0 && \
                          w2v_opts->m_ts_least_sq <= 0;

  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));

  int active_tasks = 0;
  multiclass_t multiclass;
  reset_multiclass(&multiclass);
  multiclass.m_n_tasks = thread_opts->m_n_tasks;
  long long word, sentence_length = 0, sentence_position = 0;
  long long word_count_actual = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long local_iter = w2v_opts->m_iter;
  unsigned long long next_random = thread_id;
  clock_t now;

  FILE *fi = fopen(w2v_opts->m_train_file, "rb");
  long offset = 1 + (long long) next_random * file_size / (long long) num_threads;
  /* proceed to the beginning of the line */
  do {
    fseek(fi, --offset, SEEK_SET);
  } while (offset > 0 && fgetc(fi) != '\n');

  if (offset > 0)
    ++offset;

  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((w2v_opts->m_debug_mode > 1)) {
        now = clock();
        fprintf(stderr,
                "%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13,
                thread_opts->m_alpha,
                word_count_actual / (real)(w2v_opts->m_iter * train_words + 1) * 100,
                word_count_actual / ((real)(now - thread_opts->m_start + 1)
                                     / (real) CLOCKS_PER_SEC * 1000));
        fflush(stderr);
      }
      thread_opts->m_alpha = thread_opts->m_starting_alpha              \
                             * (1 - word_count_actual /                 \
                                (real)(w2v_opts->m_iter * train_words + 1));
      if (thread_opts->m_alpha < thread_opts->m_starting_alpha * 0.0001)
        thread_opts->m_alpha = thread_opts->m_starting_alpha * 0.0001;
    }

    if (sentence_length == 0) {
      while (1) {
        word = read_word_index(fi, vocab, vocab_hash, consume_tab);
        if (feof(fi))
          break;

        if (word == -1)
          continue;

        ++word_count;
        if (word == 0)
          break;

        // The subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn /
                           (sample * train_words)) + 1)
                     * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long) 25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real) 65536)
            continue;
        }
        sen[sentence_length] = word;
        ++sentence_length;
        if (sentence_length >= MAX_SENTENCE_LENGTH)
          break;
      }
      if (!consume_tab) {
        active_tasks = read_tags(fi, &multiclass);
        if (active_tasks < 0) {
	  fprintf(stderr, "No active tasks found.\n");
          exit(EXIT_FAILURE);
        /* skip lines for which no active tasks are defined */
        } else if (active_tasks == 0 && w2v_opts->m_ts > 0) {
          sentence_length = 0;
        }
      }
      sentence_position = 0;
    }
    if (feof(fi) || (word_count > train_words / num_threads)) {
      /* fprintf(stderr, "thread %lld [iteration %lld]: total_cost = %f\n", */
      /*         thread_id, local_iter, total_cost); */
      word_count_actual += word_count - last_word_count;
      --local_iter;

      if (local_iter == 0)
        break;

      total_cost = 0;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, offset, SEEK_SET);
      continue;
    }
    word = sen[sentence_position];
    if (word == -1)
      continue;

    /* train task-specific embeddings */
    if (w2v_opts->m_ts > 0 || w2v_opts->m_ts_w2v > 0) {
      total_cost += train_ts(&multiclass, word, thread_opts->m_alpha,
                             active_tasks, layer1_size, exp_table,
                             nnet, nnet->m_syn0);
    } else if (w2v_opts->m_ts_least_sq > 0) {
      total_cost += train_ts(&multiclass, word, thread_opts->m_alpha,
                             active_tasks, layer1_size, exp_table,
                             nnet, nnet->m_ts_syn0);
    }
    /* train plain word2vec embeddings */
    if (w2v_opts->m_ts <= 0) {
      total_cost += train_w2v(w2v_opts, vocab, vocab_size, exp_table, table, window,
                              layer1_size, nnet, sen, word, neu1, neu1e, sentence_length,
                              sentence_position, &next_random);
    }

    ++sentence_position;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}

void train_model(opt_t *a_opts) {
  fprintf(stderr, "Starting training using file '%s'\n",
          a_opts->m_train_file);

  /* initialize vocabulary and exp table */
  vocab_t vocab;
  init_vocab(&vocab);
  multiclass_t multiclass;
  reset_multiclass(&multiclass);
  size_t file_size = learn_vocab_from_trainfile(&vocab, &multiclass, a_opts);
  real *exp_table = init_exp_table();
  nnet_t nnet;
  init_nnet(&nnet, &vocab, a_opts, &multiclass);
  int *ugram_table = NULL;
  if (a_opts->m_negative > 0)
    ugram_table = init_unigram_table(&vocab);

  thread_opts_t thread_opts = {clock(), file_size,
                               a_opts->m_alpha, a_opts->m_alpha,
                               0, a_opts, &vocab, &nnet,
                               exp_table, ugram_table, multiclass.m_n_tasks};
  thread_opts_t *ptopts = (thread_opts_t *) malloc(a_opts->m_num_threads
                                                   * sizeof(thread_opts_t));
  pthread_t *pt = (pthread_t *) malloc(a_opts->m_num_threads
                                       * sizeof(pthread_t));
  if (pthread_mutex_init(&tlock, NULL)) {
    fprintf(stderr, "\nmutex init failed\n");
    exit(5);
  }

  long a;
  for (a = 0; a < a_opts->m_num_threads; ++a) {
    copy_thread_opts(&thread_opts, &ptopts[a], a);
    pthread_create(&pt[a], NULL, train_model_thread,
                   (void *) &ptopts[a]);
  }
  for (a = 0; a < a_opts->m_num_threads; ++a)
    pthread_join(pt[a], NULL);

  save_embeddings(a_opts, &vocab, &nnet);
  pthread_mutex_destroy(&tlock);

  free(pt);
  free(ptopts);
  free_nnet(&nnet);
  free(ugram_table);
  free(exp_table);
  free_vocab(&vocab);
}
