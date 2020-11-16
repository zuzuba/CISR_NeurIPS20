# Curriculum Induction for Safe Reinforcement learning (CISR)

## Overview

This repository contains the implementation of the Curriculum Induction for Safe Reinforcement learning (CISR) framework and the code to reproduce the experiments presented in the paper ["Safe Reinforcement Learning via Curriculum Induction", Matteo Turchetta, Andrey Kolobov, Shital Shah, Andreas Krause and Alekh Agarwal](https://arxiv.org/abs/2006.12136).

## Installation

We recommend Ubuntu 18+ and Python 3.7+ installation using [Anaconda](https://www.anaconda.com/products/individual#downloads).

To install, run:

```bash
git clone https://github.com/zuzuba/SafeCL.git
cd SafeCL
./install.sh
```

This script creates a conda environment and automatically installs all the dependencies necessary to reproduce the experiments in the paper. If you have OpenAI gym environment working already then just run,

```bash
pip install -e .
```

## Running Experiments

For each environment considered in the paper (lunar lander and frozen lake), we provide pre-trained teachers and the data for their evaluation with 10
independent students. Therefore, for each of the environments, it is possible to :

   - **Plot**: Plot or print the table of the comparison of a curriculum induced by a teaching policy optimized with CISR against the baselines. The plot corresponds to figure 3 of the paper and is only available for frozen lake. The values in the tables correspond to tables 1, 6, 7a, 7b and 7c.
   - **Evaluate**: Run the comparison of a teacher that was pre-trained with CISR against against the baselines.
   - **Train**: Train a new teacher from scratch using CISR (~1 hour for frozen lake and 5-7 hours for lunar lander).

For both environments, we provide a compare\_teachers.py script, which
 performs the *plot* and *evaluate* functions and a teacher\_learning.py script, which carries out the *train* function. All the results are stored in the ./results directory.

### Frozen lake

#### Plot

To generate the plots run the compare\_teachers script with the --plot flag. You can use the --teacher\_dir flag to specify which pre-trained teachers you want to use among those saved in ./results/flake/teacher_training. For example, the following command plots the comparison for the default teacher:

```bash
python src/teacher/flake_approx/compare_teachers.py --plot
```

The following command plots the comparison for all the available teachers

```bash
python src/teacher/flake_approx/compare_teachers.py --plot --teacher_dir 03_06_20__11_46_57 03_06_20__12_20_36 02_06_20__11_46_57
```

#### Evaluate

To compare the trained teacher against the baselines, it is sufficient to run the compare\_teachers.py script with the --evaluate flag. Similar to the plotting case, the --teacher_dir flag can be used to specify the teacher to run the comparison for.

#### Train

To train a new teacher, you need to run the teacher\_learning.py script:

```bash
python src/teacher/flake_approx/teacher_learning.py
```

The trained teacher as well as some information about the training process will be stored in ./results/flake/teacher_training in a directory named after the date and time when the training process is performed. If you want to evaluate and/or plot this newly trained teacher, it is sufficient to pass the name of this directory as an argument to the compare\_teachers.py script (see above).

### Lunar Lander

#### Analyze

To generate the tables with the statistics relative to the lunar lander experiments run the compare\_teachers script with the --analyze flag. In this case, a positional argument that specifies the scenario is required:

 - Scenario 0: Two-layered teacher with noiseless observation (Table 7a)
 - Scenario 1: One-layered teacher with noiseless observation  (Table 7c)
 - Scenario 2: Two-layered teacher with noisy observation (Table 7b)

You can use the --teacher\_dir flag to specify which pre-trained teachers you want to use among those saved in ./results/lunar\_lander/teacher_training. For example, the following command prints the table with the comparison for the default teacher for the two-layered student, noiseless scenario:

```bash
python src/teacher/lunar_lander/compare_teachers.py 0
```

The following command plots the comparison for all the available teachers for the one-layered student, noiseless scenario

```bash
python src/teacher/lunar_lander/compare_teachers.py 1 --analyze --teacher_dir 03_06_20__18_24_43 01_06_20__16_10_17 09_06_20__19_21_22
```

#### Evaluate

To compare the trained teacher against the baselines, it is sufficient to run the compare\_teachers.py script with the --evaluate flag. Similar to the plotting case, a scenario must be specified and the --teacher_dir flag can be used to specify the teacher to run the comparison for.

#### Train

To train a new teacher, you need to run the teacher\_learning.py script:

```bash
python src/teacher/lunar_lander/teacher_learning.py
```

The trained teacher as well as some information about the training process will be stored in ./results/flake/teacher_training in a directory named after the date and time when the training process is performed. If you want to evaluate and/or get the evaluation statistics table of this newly trained teacher, it is sufficient to pass the name of this directory as an argument to the compare\_teachers.py script (see above).

## Citation

Please refer to paper [Safe Reinforcement Learning via Curriculum Induction)](https://arxiv.org/abs/2006.12136) for further details. Please cite this as:

```
@inproceedings{cisr2020neurips,
  title={Safe Reinforcement Learning via Curriculum Induction},
  author={Matteo Turchetta and Andrey Kolobov and Shital Shah and Andreas Krause and Alekh Agarwal},
  year={2020},
  eprint={2006.12136},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url = {https://arxiv.org/abs/1705.05065}
}
```

## License

This project is released under the MIT License. Please review the [License file](LICENSE.txt) for more details.

