/**
 * @file vocab.h
 * @brief Declaration of the @link vocab_t vocabulary struct @endlink and its interface.
 */

#ifndef __WORD2VEC_VOCAB_H__
#define __WORD2VEC_VOCAB_H__

//////////////
// Includes //
//////////////
#include "common.h"                   /* real */

#include <stddef.h>                   /* NULL */
#include <stdlib.h>                   /* malloc(), free() */
#include <stdio.h>                    /* FILE * */

////////////
// Macros //
////////////

///////////////
// Constants //
///////////////
extern const int TABLE_SIZE;	  /**< size of the unigram table */
extern const int MAX_CODE_LENGTH; /**< maximum length of a binary
				     code */
extern const int MAX_SENTENCE_LENGTH; /**< maximum allowed sentence
					 length */
extern const int VOCAB_HASH_SIZE; /**< Maximum 30 * 0.7 = 21M words in
				      the vocabulary */
extern const char EOS[];	/**< end-of-sentence mark */

/////////////
// Structs //
/////////////

/**
 * @brief Single word stored in the vocabulary.
 */
typedef struct vocab_word {
  long long cn;		     /**< word count  */
  int *point;		     /**< word's binary code */
  char *word;	    /**< String representation of the word.*/
  char *code;	    /**< Numerical code of the word used for
		       hashing.*/
  char codelen;			/**< Length of word's code.*/
} vw_t;

/**
 * @brief Whole vocabulary.
 */
typedef struct vocab {
  long long m_vocab_size;	/**< number of words in vocabulary  */
  long long m_max_vocab_size;	/**< maximum reserved size for the vocabulary  */
  long long m_train_words;	/**< number of words used for training */
  vw_t *m_vocab;		/**< internal word storage */
  int *m_vocab_hash;		/**< internal mapping from word
				   strings to hash codes */
} vocab_t;

/////////////
// Methods //
/////////////

/**
 * Add a word to the vocabulary.
 *
 * \param a_vocab vocabulary to add the word to
 * \param a_word word to be added
 *
 * \return \c int position of a word in the vocabulary
 */
int add_word2vocab(vocab_t *a_vocab, const char *a_word);

/**
 * Free memory occupied by vocabulary.
 *
 * \param a_vocab vocabulary innstance
 *
 * \return \c void
 */
void free_vocab(vocab_t *a_vocab);

/**
 * Create binary search tree for the vocabulary.
 *
 * \param a_vocab vocabulary instance
 *
 * \return \c void
 */
void create_binary_tree(vocab_t *a_vocab);

/**
 * Initialize vocabulary to an empty dictionary.
 *
 * \param a_vocab vocabulary instance
 *
 * \return \c void
 */
void init_vocab(vocab_t *a_vocab);

/**
 * Initialize a unigram table.
 *
 * \param a_vocab vocabulary with relevant information
 *
 * \return pointer to the initialized table
 */
int *init_unigram_table(vocab_t *a_vocab);

/**
 * Output vocabulary to the specified stream.
 *
 * \param a_ostream output stream
 * \param a_vocab vocabulary instance
 *
 * \return \c void
 */
void output_vocab(FILE *a_ostream, const vocab_t *a_vocab);

/**
 * Reduce vocabulary by removing infrequent tokens.
 *
 * \param a_vocab vocabulary innstance
 * \param a_opts CLI options defining reduce behavior
 *
 * \return \c void
 */
void reduce_vocab(vocab_t *a_vocab, opt_t *a_opts);

/**
 * Look up a word in the vocabulary.
 *
 * \param a_word word to search for
 * \param a_vocab vocabulary to search in
 * \param a_vocab_hash hash of word indices
 *
 * \return position of a word in the vocabulary or \c -1 if the word
 *   is not found
 */
int search_vocab(const char *a_word, const vw_t *a_vocab, const int *a_vocab_hash);

/**
 * Sort vocabulary.
 *
 * \param a_vocab vocabulary to be sorted
 * \param a_min_count minimum required word frequency
 *
 * \return number of words in the vocabulary
 */
int sort_vocab(vocab_t *a_vocab, const int a_min_count);

#endif  /* ifndef __WORD2VEC_VOCAB_H__ */
