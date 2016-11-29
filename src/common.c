//////////////
// Includes //
//////////////
#include "common.h"

#include <stdio.h>

/////////////
// Methods //
/////////////

void reset_opt(opt_t *opt) {
  opt->m_train_file[0] = '\0';
  opt->m_output_file[0] = '\0';

  opt->m_layer1_size = 100;
  opt->m_iter = 5;

  opt->m_alpha = (real) 0.025;
  opt->m_sample = (real) 1e-3;

  opt->m_binary = 0;
  opt->m_cbow = 1;
  opt->m_debug_mode = 2;
  opt->m_hs = 0;
  opt->m_min_count = 5;
  opt->m_min_reduce = 1;
  opt->m_negative = 5;
  opt->m_num_threads = 12;
  opt->m_window = 5;

  opt->m_ts = 0;
  opt->m_ts_w2v = 0;
  opt->m_ts_least_sq = 0;
}
