## Transfetch w/ Data Version Control Integration


```
transfetch_dvc
├─ .dvc
├─ .git
├─ README.md
├─ data
│  ├─ 473.astar-s0.txt
│  ├─ 473.astar-s0.txt.dvc
│  ├─ 473.astar-s0.txt.xz
│  ├─ 473.astar-s0.txt.xz.dvc
│  ├─ 473.astar-s1.txt.xz
│  ├─ 473.astar-s1.txt.xz.dvc
│  ├─ 473.astar-s2.txt.xz
│  ├─ 473.astar-s2.txt.xz.dvc
│  ├─ 602.gcc-s2.txt.xz
│  └─ 602.gcc-s2.txt.xz.dvc
├─ num_tch_1
│  ├─ .dvcignore
│  ├─ dvc.lock
│  ├─ dvc.yaml
│  ├─ model
│  │  └─ teacher_1.pth
│  ├─ processed
│  │  ├─ test_df_1
│  │  ├─ test_loader_1
│  │  └─ train_loader_1
│  ├─ res
│  │  ├─ metrics.json
│  │  ├─ plots
│  │  │  └─ metrics
│  │  │     ├─ train_stu
│  │  │     └─ train_tch_1
│  │  │        ├─ test_loss.tsv
│  │  │        └─ train_loss.tsv
│  │  ├─ report.html
│  │  ├─ student.json
│  │  └─ teacher_1.json
│  └─ src
│     ├─ __pycache__
│     │  ├─ data_loader.cpython-38.pyc
│     │  └─ preprocess.cpython-38.pyc
│     ├─ data_loader.py
│     ├─ models
│     │  ├─ __pycache__
│     │  │  ├─ d.cpython-38.pyc
│     │  │  ├─ r.cpython-38.pyc
│     │  │  └─ v.cpython-38.pyc
│     │  ├─ d.py
│     │  ├─ r.py
│     │  └─ v.py
│     ├─ preprocess.py
│     ├─ train_stu.py
│     ├─ train_tch.py
│     ├─ validate_stu.py
│     └─ validate_tch.py
├─ num_tch_2
│  ├─ dvc.yaml
│  └─ src
│     ├─ __pycache__
│     │  ├─ data_loader.cpython-38.pyc
│     │  └─ preprocess.cpython-38.pyc
│     ├─ data_loader.py
│     ├─ models
│     │  ├─ __pycache__
│     │  │  ├─ d.cpython-38.pyc
│     │  │  ├─ r.cpython-38.pyc
│     │  │  └─ v.cpython-38.pyc
│     │  ├─ d.py
│     │  ├─ r.py
│     │  └─ v.py
│     ├─ preprocess.py
│     ├─ train_stu.py
│     ├─ train_tch.py
│     ├─ validate_stu.py
│     └─ validate_tch.py
└─ params.yaml

```