# autoamp
AMP discovery with large language models

## Install

On Polaris:
```bash
module use /soft/modulefiles
module load conda; conda activate
conda create -n sdl-amp --clone base
conda activate sdl-amp
pip install -U pip setuptools wheel
pip install -e .
```

## Usage

Split a FASTA file into training and validation sets:
```bash
autoamp --input_fasta /lus/eagle/projects/FoundEpidem/braceal/projects/sdl-amp/data/unique_sequences.fasta --output_dir /lus/eagle/projects/FoundEpidem/braceal/projects/sdl-amp/data/unique_sequences.split --split 0.9
```
| File Name                          | Number of Unique Sequences |
|------------------------------------|----------------------------|
| unique_sequences.split/valid.fasta | 2030                       |
| unique_sequences.split/train.fasta | 18266                      |

Fine-tune a model on Polaris:
```bash
qsub -I -l select=1 -l filesystems=home:eagle -l walltime=1:00:00 -q debug-scaling -A FoundEpidem
module use /soft/modulefiles; module load conda; conda activate
conda activate sdl-amp
python -m autoamp.finetune --config examples/finetune/esm2-8m-finetune.yaml
```

Fine-tune a model on lambda (a standard workstation):
```bash
conda create -n sdl-amp python=3.10
conda activate sdl-amp
pip install -U pip setuptools wheel
pip install -e .
python -m autoamp.finetune --config examples/finetune/esm2-8m-finetune.yaml
```


## Contributing

For development, it is recommended to use a virtual environment. The following
commands will create a virtual environment, install the package in editable
mode, and install the pre-commit hooks.
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
pip install -e '.[dev,docs]'
pre-commit install
```

To test the code, run the following command:
```bash
pre-commit run --all-files
tox -e py310
```
