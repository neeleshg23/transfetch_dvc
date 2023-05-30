## Transfetch w/ Data Version Control Integration

To run: `dvc exp run [-n --queue -S] num_tch_1/dvc.yaml`
To save current set of experiments: `dvc exp save`


```
transfetch_dvc
├─ .dvc
├─ .dvcignore
├─ .tours
│  └─ dvc.tour
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
│  ├─ c.yaml
│  ├─ dvc.lock
│  ├─ dvc.yaml
│  ├─ model
│  │  ├─ student.pth
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
│  │  │     │  ├─ test_loss.tsv
│  │  │     │  └─ train_loss.tsv
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
│  ├─ d.yaml
│  ├─ dvc.lock
│  ├─ dvc.yaml
│  ├─ model
│  │  ├─ student.pth
│  │  ├─ teacher_1.pth
│  │  └─ teacher_2.pth
│  ├─ processed
│  │  ├─ test_df_1
│  │  ├─ test_df_2
│  │  ├─ test_df_stu
│  │  ├─ test_loader_1
│  │  ├─ test_loader_2
│  │  ├─ test_loader_stu
│  │  ├─ train_loader_1
│  │  ├─ train_loader_2
│  │  └─ train_loader_stu
│  ├─ res
│  │  ├─ metrics.json
│  │  ├─ plots
│  │  │  └─ metrics
│  │  │     ├─ train_stu
│  │  │     │  ├─ test_loss.tsv
│  │  │     │  └─ train_loss.tsv
│  │  │     ├─ train_tch_1
│  │  │     │  ├─ test_loss.tsv
│  │  │     │  └─ train_loss.tsv
│  │  │     └─ train_tch_2
│  │  │        ├─ test_loss.tsv
│  │  │        └─ train_loss.tsv
│  │  ├─ report.html
│  │  ├─ student.json
│  │  ├─ teacher_1.json
│  │  └─ teacher_2.json
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
├─ num_tch_3
│  ├─ dvc.lock
│  ├─ dvc.yaml
│  ├─ e.yaml
│  ├─ model
│  │  ├─ student.pth
│  │  ├─ teacher_1.pth
│  │  ├─ teacher_2.pth
│  │  └─ teacher_3.pth
│  ├─ processed
│  │  ├─ test_df_1
│  │  ├─ test_df_2
│  │  ├─ test_df_3
│  │  ├─ test_df_stu
│  │  ├─ test_loader_1
│  │  ├─ test_loader_2
│  │  ├─ test_loader_3
│  │  ├─ test_loader_stu
│  │  ├─ train_loader_1
│  │  ├─ train_loader_2
│  │  ├─ train_loader_3
│  │  └─ train_loader_stu
│  ├─ res
│  │  ├─ metrics.json
│  │  ├─ plots
│  │  │  └─ metrics
│  │  │     ├─ train_stu
│  │  │     │  ├─ test_loss.tsv
│  │  │     │  └─ train_loss.tsv
│  │  │     ├─ train_tch_1
│  │  │     │  ├─ test_loss.tsv
│  │  │     │  └─ train_loss.tsv
│  │  │     ├─ train_tch_2
│  │  │     │  ├─ test_loss.tsv
│  │  │     │  └─ train_loss.tsv
│  │  │     └─ train_tch_3
│  │  │        ├─ test_loss.tsv
│  │  │        └─ train_loss.tsv
│  │  ├─ report.html
│  │  ├─ student.json
│  │  ├─ teacher_1.json
│  │  ├─ teacher_2.json
│  │  └─ teacher_3.json
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
├─ num_tch_4
│  ├─ dvc.lock
│  ├─ dvc.yaml
│  ├─ f.yaml
│  ├─ model
│  │  ├─ student.pth
│  │  ├─ teacher_1.pth
│  │  ├─ teacher_2.pth
│  │  ├─ teacher_3.pth
│  │  └─ teacher_4.pth
│  ├─ processed
│  │  ├─ test_df_1
│  │  ├─ test_df_2
│  │  ├─ test_df_3
│  │  ├─ test_df_4
│  │  ├─ test_df_stu
│  │  ├─ test_loader_1
│  │  ├─ test_loader_2
│  │  ├─ test_loader_3
│  │  ├─ test_loader_4
│  │  ├─ test_loader_stu
│  │  ├─ train_loader_1
│  │  ├─ train_loader_2
│  │  ├─ train_loader_3
│  │  ├─ train_loader_4
│  │  └─ train_loader_stu
│  ├─ res
│  │  ├─ metrics.json
│  │  ├─ plots
│  │  │  └─ metrics
│  │  │     ├─ train_stu
│  │  │     │  ├─ test_loss.tsv
│  │  │     │  └─ train_loss.tsv
│  │  │     ├─ train_tch_1
│  │  │     │  ├─ test_loss.tsv
│  │  │     │  └─ train_loss.tsv
│  │  │     ├─ train_tch_2
│  │  │     │  ├─ test_loss.tsv
│  │  │     │  └─ train_loss.tsv
│  │  │     ├─ train_tch_3
│  │  │     │  ├─ test_loss.tsv
│  │  │     │  └─ train_loss.tsv
│  │  │     └─ train_tch_4
│  │  │        ├─ test_loss.tsv
│  │  │        └─ train_loss.tsv
│  │  ├─ report.html
│  │  ├─ student.json
│  │  ├─ teacher_1.json
│  │  ├─ teacher_2.json
│  │  ├─ teacher_3.json
│  │  └─ teacher_4.json
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
└─ num_tch_5
   ├─ dvc.lock
   ├─ dvc.yaml
   ├─ g.yaml
   ├─ model
   │  ├─ student.pth
   │  ├─ teacher_1.pth
   │  ├─ teacher_2.pth
   │  ├─ teacher_3.pth
   │  ├─ teacher_4.pth
   │  └─ teacher_5.pth
   ├─ processed
   │  ├─ test_df_1
   │  ├─ test_df_2
   │  ├─ test_df_3
   │  ├─ test_df_4
   │  ├─ test_df_5
   │  ├─ test_df_stu
   │  ├─ test_loader_1
   │  ├─ test_loader_2
   │  ├─ test_loader_3
   │  ├─ test_loader_4
   │  ├─ test_loader_5
   │  ├─ test_loader_stu
   │  ├─ train_loader_1
   │  ├─ train_loader_2
   │  ├─ train_loader_3
   │  ├─ train_loader_4
   │  ├─ train_loader_5
   │  └─ train_loader_stu
   ├─ res
   │  ├─ metrics.json
   │  ├─ plots
   │  │  └─ metrics
   │  │     ├─ train_stu
   │  │     │  ├─ test_loss.tsv
   │  │     │  └─ train_loss.tsv
   │  │     ├─ train_tch_1
   │  │     │  ├─ test_loss.tsv
   │  │     │  └─ train_loss.tsv
   │  │     ├─ train_tch_2
   │  │     │  ├─ test_loss.tsv
   │  │     │  └─ train_loss.tsv
   │  │     ├─ train_tch_3
   │  │     │  ├─ test_loss.tsv
   │  │     │  └─ train_loss.tsv
   │  │     ├─ train_tch_4
   │  │     │  ├─ test_loss.tsv
   │  │     │  └─ train_loss.tsv
   │  │     └─ train_tch_5
   │  │        ├─ test_loss.tsv
   │  │        └─ train_loss.tsv
   │  ├─ report.html
   │  ├─ student.json
   │  ├─ teacher_1.json
   │  ├─ teacher_2.json
   │  ├─ teacher_3.json
   │  ├─ teacher_4.json
   │  └─ teacher_5.json
   └─ src
      ├─ __pycache__
      │  ├─ data_loader.cpython-38.pyc
      │  └─ preprocess.cpython-38.pyc
      ├─ data_loader.py
      ├─ models
      │  ├─ __pycache__
      │  │  ├─ d.cpython-38.pyc
      │  │  ├─ r.cpython-38.pyc
      │  │  └─ v.cpython-38.pyc
      │  ├─ d.py
      │  ├─ r.py
      │  └─ v.py
      ├─ preprocess.py
      ├─ train_stu.py
      ├─ train_tch.py
      ├─ validate_stu.py
      └─ validate_tch.py

```