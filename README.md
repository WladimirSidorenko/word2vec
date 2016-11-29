# word2vec
---

![License](http://img.shields.io/:license-apache-blue.svg?style=flat-square)](http://www.apache.org/licenses/LICENSE-2.0.html)

This projects provides an enhanced version of the original
[word2vec](https://github.com/svn2github/word2vec) code.  In addition
to the normal functionality (i.e., training word vectors based on
their surrounding context), this implementation also provides a
possibility to train word embeddings tweaked to a particular
user-defined task (in addition to or instead of the normal objective).

## Building

In order to build this project, you need to change to the `build`
directory of the checked-out repository and execute the following
command:

```shell
cmake ../
make
```

This will look for the necessary libraries, adjust the compilation
options, and, finally, compile the executables. Currently, this
project depends on the following 3-rd party utils:

 * [CMake](https://cmake.org/) itself with at least one working C compiler;
 * [Threads](https://www.gnu.org/software/hurd/hurd/libthreads.html);
 * and [GSL](https://www.gnu.org/software/gsl/).

## Testing

In order to test the built program, you should run the following
command:

```shell
make test
```

## Running

Finally, you can start using the compiled program.  You can find
examples of input data in the `test/` directory of this repository.

In order to run the normal `word2vec` training, you can execute the
following command (from the `build` directory):

```shell
./bin/word2vec -min-count 0 -train ../tests/test_1.0.in
```

this will train the vanilla `word2vec` embeddings, which, however,
might be slightly different from the original when trained with
multiple threads (the default).

If you, however, want to train embeddings with respect to a particular
task (e.g., whether a sentence conveys a positive or negative
polarity), you can launch:

```shell
./bin/word2vec -ts -min-count 0 -train ../tests/test_2.0.in
```

Then, the resulting word vector will be trained to best fit this
custom task.  The labels for each task should be specified as
contiguous non-negative integers starting from zero (i.e., if a task
has three classes, the labels to use should be `0`, `1`, and `2`) and
separated by a tab character from the main text, e.g.:

```text
Ich fahre morgen nach Hause.\t0
Ich bin sehr froh dich zu sehen.\t1
Schade, dass wir uns nicht getroffen haben.\t2
```

If the label for the task is not known, you should put an underscore
`_` instead of the tag.  In the same way, you can also specify
multiple tags for training, e.g.:

```text
Ich fahre morgen nach Hause.\t0\t1
Ich bin sehr froh dich zu sehen.\t1\t_
Schade, dass wir uns nicht getroffen haben.\t2\t0
```

Besides the `-ts` mode which trains purely task-specific embeddings,
we also provide a couple of in-between solutions:

1.  With the `-ts-w2v` option, you can simultaneously train both
`word2vec` and task-specific objectives, in which case word embeddings
will be shared and updated to match both tasks.

2.  Alternatively, you can also use the `-ts-least-sq` option, in
which case `word2vec` and task-specific embeddings will be trained
independently.  In the final step, however, task-specific embeddings
of words which did not appear in the task-labeled lines will be
computed from their `word2vec` representation using the linear
least-squares method.

## Documentation

To build the documentation for the compiled executable, you need to
install [Doxygen](www.doxygen.org/) prior to executing `cmake` and
then run:

```shell
make doc
```

after the Makefiles have been generated.
