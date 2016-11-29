/**
 * @file w2vio.h
 * @brief Declaration of input/output utils.
 */
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
 * @param a_word - target word to populate
 * @param a_fin - input stream
 * @param a_consume_tab - digest tab as a normal white-space character
 *
 * @return \c void
 */
void read_word(char *a_word, FILE *a_fin, const int a_consume_tab);

/**
 * Read a word and return its index in the vocabulary.
 *
 * @param a_fin - input stream
 * @param a_vocab - vocabulary to search in
 * @param a_vocab_hash - hash of word indices
 * @param a_consume_tab - digest tab as a normal white-space character
 *
 * @return \c void
 */
int read_word_index(FILE *a_fin, const vw_t *a_vocab, const int *a_vocab_hash,
                    const int a_consume_tab);

/**
 * Read a word and return its index in the vocabulary.
 *
 * @param a_fin - input stream
 * @param a_multiclass - statistics on task-specific classes
 *
 * @return negative \c int on error, otherwise a non-negative number
 *   of active tasks
 */
int read_tags(FILE *a_fin, multiclass_t *a_multiclass);

/**
 * Create vocabulary from words in the training file.
 *
 * @param a_vocab - vocabulary to populate
 * @param a_multiclass - statistics about multiple training classes
 * @param a_opts - word to search for
 *
 * @return \c size_t - size of the input file
 */
size_t learn_vocab_from_trainfile(vocab_t *a_vocab, multiclass_t *a_multiclass,
                                  opt_t *a_opts);

/**
 * Output embeddings to the specified file.
 *
 * @param a_fo - output stream
 * @param a_vocab - vocabulary to populate
 * @param a_nnet - neural net with trained parameters
 *
 * @return \c size_t - size of the input file
 */
void save_embeddings(const opt_t *a_fo, const vocab_t *a_vocab, const nnet_t *a_nnet);
#endif  /* ifndef __WORD2VEC_IO_H__ */
