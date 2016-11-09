//////////////
// Includes //
//////////////
#include "common.h"
#include "w2vio.h"

#include <errno.h>  /* errno */
#include <ctype.h>  /* isspace() */
#include <stdio.h>  /* sscanf() */
#include <string.h> /* strcpy() */

/////////////
// Methods //
/////////////

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void read_word(char *a_word, FILE *a_fin, const int a_consume_tab) {
  int a = 0, ch;
  while (!feof(a_fin)) {
    ch = fgetc(a_fin);
    if (ch == 13)
      continue;

    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n' || (ch == '\t' && !a_consume_tab))
          ungetc(ch, a_fin);

        break;
      }
      if (ch == '\n' || (ch == '\t' && !a_consume_tab)) {
        strcpy(a_word, (char *)"</s>");
        return;
      } else {
        continue;
      }
    }
    a_word[a] = ch;
    ++a;
    if (a >= MAX_STRING - 1)
      --a;   // Truncate too long words
  }
  a_word[a] = 0;
}

// Reads a word and returns its index in the vocabulary
int read_word_index(FILE *a_fin, const vw_t *a_vocab,
                    const int *a_vocab_hash, const int a_consume_tab) {
  char word[MAX_STRING];
  read_word(word, a_fin, a_consume_tab);
  if (feof(a_fin))
    return -1;

  return search_vocab(word, a_vocab, a_vocab_hash);
}

int read_tags(FILE *a_fin, multiclass_t *a_multiclass) {
  int active_tasks = 0, ntasks = 0;

  int ch;
  size_t nchars = 0;
  int ret = 0, space_seen = 1;
  char tag[MAX_STRING];

  while (!feof(a_fin)) {
    ch = fgetc(a_fin);
    if (ch == 13)
      continue;

    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (nchars) {
        tag[nchars] = '\0';
        ret = scanf(tag, "%d",
                    &a_multiclass->m_max_classes[ntasks++]);
        if (ret <= 0)
          return -1;

        ++active_tasks;
        nchars = 0;
      }
      if (ch == '\n')
        break;

      space_seen = 1;
    } else if (ch == '_') {
      if (space_seen)
        a_multiclass->m_max_classes[ntasks++] = -1;
    } else {
      tag[nchars++] = ch;
      space_seen = 0;
    }
  }
  return active_tasks;
}

static int process_line_w2v(vocab_t *a_vocab,
                            multiclass_t *a_multiclass,
                            const int a_use_w2v,
                            char *a_word, const char *a_line, ssize_t a_read) {
  int n_words = 0;
  UNUSED(a_multiclass);
  UNUSED(a_use_w2v);

  ssize_t i, n_chars;
  for (i = 0, n_chars = 0; i < a_read; ++i, ++n_chars) {
    if (isspace(a_line[i])) {
      if (n_chars > 0) {
        a_word[n_chars] = 0;
        add_word2vocab(a_vocab, a_word);
        ++n_words;
      }
      n_chars = -1;
    } else if (n_chars >= MAX_STRING) {
      n_chars = MAX_STRING - 1;
      a_word[n_chars] = 0;
    } else {
      a_word[n_chars] = a_line[i];
    }
  }

  if (n_chars > 0) {
    a_word[n_chars] = 0;
    add_word2vocab(a_vocab, a_word);
    ++n_words;
  }
  return n_words;
}

static int process_tag_line(multiclass_t *a_multiclass, const char *a_line,
                            ssize_t a_read) {
  ssize_t i;
  int tag_len = 0;
  /* `m' counts actual tasks, `n' counts active tasks  */
  size_t m = 0, label = 0;
  int n = 0, seen_space = 0;
  int first_run = (a_multiclass->m_n_tasks == 0);

  for (i = 0; i < a_read; ++i) {
    if (isspace(a_line[i])) {
      seen_space = 1;
      continue;
    } else if (a_line[i] == '_') {
      if (seen_space)
        ++m;
      else
        continue;
    } else {
      errno = 0;
      tag_len = sscanf(&a_line[i], "%zu", &label);
      if (tag_len == 0) {
        fprintf(stderr, "Invalid tag specification '%s'\n", &a_line[i]);
        return -1;
      } else if (tag_len == EOF) {
        if (errno) {
          perror("sscanf():");
        } else {
          fprintf(stderr, "EOF reached while looking for tag: '%s'\n",
                  &a_line[i]);
        }
        return -1;
      } else {
        /* increment the label by one for comparison */
        if ((int) (++label) > a_multiclass->m_max_classes[m])
          a_multiclass->m_max_classes[m] = (int) label;

        /* advance the pointer to the end of number */
        i += (size_t) tag_len;
        ++n;
      }
      ++m;
    }
    seen_space = 0;
  }
  if (first_run) {
    if (m == 0) {
      fprintf(stderr,
              "Invalid line format "
              "(no tags specified for task-specific embeddings):"
              " '%s'\n", a_line);
      return -1;
    } else {
      a_multiclass->m_n_tasks = m;
    }
  } else if (m != a_multiclass->m_n_tasks) {
      fprintf(stderr,
              "Invalid line format "
              "(different number of tags specified for task-specific embeddings):"
              " %zu versus %zu\n", m, a_multiclass->m_n_tasks);
      return -1;
  }
  return n;
}

static int process_line_task_specific(vocab_t *a_vocab,
                                      multiclass_t *a_multiclass,
                                      const int a_use_w2v,
                                      char *a_word,
                                      const char *a_line, ssize_t a_read) {
  int active_tasks = 0;
  const char *tag_line = strchr(a_line, '\t');
  if (tag_line == NULL) {
    fprintf(stderr,
            "Invalid line format (missing tags): '%s'\n", a_line);
    exit(5);
  }
  ssize_t line_read = (ssize_t)(tag_line - a_line);

  if ((active_tasks = process_tag_line(a_multiclass,
                                       tag_line, a_read - line_read)) < 0)
    exit(6);

  if (!active_tasks && !a_use_w2v)
    return 0;

  return process_line_w2v(a_vocab, NULL, 0, a_word, a_line, line_read);
}

size_t learn_vocab_from_trainfile(vocab_t *a_vocab, multiclass_t *a_multiclass,
                                  opt_t *a_opts) {
  FILE *fin = fopen(a_opts->m_train_file, "rb");
  if (fin == NULL) {
    fprintf(stderr, "ERROR: training data file not found!\n");
    exit(EXIT_FAILURE);
  }

  int a;
  for (a = 0; a < VOCAB_HASH_SIZE; a++) {
    a_vocab->m_vocab_hash[a] = -1;
  }

  ssize_t read;
  char *line = NULL;
  size_t len = 0;
  char word[MAX_STRING];
  const int use_w2v = !a_opts->m_ts;
  int (*process_line)(vocab_t *a_vocab, multiclass_t *a_multiclass,
                      const int a_use_w2v, char *a_word,
                      const char *a_line, ssize_t a_read) = NULL;

  if (a_opts->m_ts || a_opts->m_ts_least_sq || a_opts->m_ts_w2v)
    process_line = process_line_task_specific;
  else
    process_line = process_line_w2v;

  long long train_words = 1;
  while ((read = getline(&line, &len, fin)) != -1) {
    if (read > 1) {
      add_word2vocab(a_vocab, EOS);
    }
    if ((a_opts->m_debug_mode > 1) && (train_words % 100000 == 0)) {
      fprintf(stderr, "%lldK%c", train_words / 1000, 13);
      fflush(stderr);
    }
    train_words += process_line(a_vocab, a_multiclass, use_w2v,
                                word, line, read);

    if (a_vocab->m_vocab_size > VOCAB_HASH_SIZE * 0.7)
      reduce_vocab(a_vocab, a_opts);
  }
  free(line);
  a_vocab->m_train_words = sort_vocab(a_vocab, a_opts->m_min_count);
  create_binary_tree(a_vocab);

  if (ferror(fin)) {
    fprintf(stderr, "ERROR: reading input file\n");
    exit(EXIT_FAILURE);
  }
  if (a_opts->m_debug_mode > 0) {
    fprintf(stderr, "Vocab size: %lld\n", a_vocab->m_vocab_size);
    fprintf(stderr, "Words in train file: %lld\n", a_vocab->m_train_words);
  }
  size_t file_size = ftell(fin);
  fclose(fin);
  return file_size;
}
