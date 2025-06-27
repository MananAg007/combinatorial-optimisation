# LKH Setup and Usage Guide for FrontierCO

## 1. Create and Activate Conda Environment
```bash
conda env create -f environment.yml
conda activate FrontierCO
```

## 2. Install LKH
Download and build LKH from the official site:
- [LKH-3 Download Page](http://webhotel4.ruc.dk/~keld/research/LKH-3/)

Example:
```bash
wget http://webhotel4.ruc.dk/~keld/research/LKH-3/LKH-3.0.13.tgz
tar -xzf LKH-3.0.13.tgz
cd LKH-3.0.13
make
```

## 3. Generate TSP Instances
```bash
python generate_training_instances.py \
  --min_nodes 1000 \
  --max_nodes 1000 \
  --num_instances 20 \
  --seed 42 \
  --output valid_instances
```

## 4. Create LKH Parameter Files
```bash
bash generate_par.sh valid_instances
```

## 5. Solve TSP Instances
```bash
bash solve.sh \
  --instance-dir valid_instances \
  --lkh-path /home/mananaga/projects/combinatorial-optimisation/FrontierCO/LKH-3.0.13/LKH \
  --parallel 10
```

## 6. Summarize Results
```bash
python summary.py \
  valid_instances \
  valid_instances_sol/summary_results.csv
``` 