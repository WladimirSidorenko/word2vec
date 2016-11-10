#ifndef __WORD2VEC_IO_H__
# define __WORD2VEC_IO_H__

//////////////
// Includes //
//////////////
#include "common.h"
#include "vocab.h"

#include <stdio.h>   /* fopen, getline, ferror */

/////////////
// Methods //
/////////////

/**
 * Reads a single word from a file.
 *
 * @param word - target word to populate
 * @type char *
 * @param fin - input stream
 * @type FILE *
 * @param a_consume_tab - digest tab as a normal white-space character
 * @type const int
 *
 * @return \c void
 */
void read_word(char *a_word, FILE *a_fin, const int a_consume_tab);

/**
 * Read a word and return its index in the vocabulary.
 *
 * @param a_fin - input stream
 * @type FILE *
 * @param a_vocab - vocabulary to search in
 * @type vw_t *
 * @param a_vocab_hash - hash of word indices
 * @type int *
 * @param a_consume_tab - digest tab as a normal white-space character
 * @type const int
 *
 * @return \c void
 */
int read_word_index(FILE *a_fin, const vw_t *a_vocab, const int *a_vocab_hash,
                    const int a_consume_tab);

/**
 * Read a word and return its index in the vocabulary.
 *
 * @param a_fin - input stream
 * @type FILE *
 * @param a_multiclass - statistics on task-specific classes
 * @type multiclass_t *
 *
 * @return negative \c int on error, otherwise a non-negative number
 *   of active tasks
 */
int read_tags(FILE *a_fin, multiclass_t *a_multiclass);

/**
 * Create vocabulary from words in the training file.
 *
 * @param a_vocab - vocabulary to populate
 * @type vocab_t *
 * @param a_multiclass - statistics about multiple training classes
 * @type multiclass_t *
 * @param a_opts - word to search for
 * @type opt_t *
 *
 * @return \c size_t - size of the input file
 */
size_t learn_vocab_from_trainfile(vocab_t *a_vocab, multiclass_t *a_multiclass,
                                  opt_t *a_opts);

/**
 * Output .
 *
 * @param a_opts - word to search for
 * @type const opt_t *
 * @param a_vocab - vocabulary to populate
 * @type const vocab_t *
 * @param a_nnet - neural net with trained parameters
 * @type const nnet_t *
 *
 * @return \c size_t - size of the input file
 */
void save_embeddings(const opt_t *a_fo, const vocab_t *a_vocab, const nnet_t *a_nnet);
#endif  /* ifndef __WORD2VEC_IO_H__ */
