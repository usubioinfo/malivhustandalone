# Malivhu - Standalone version

## Prerequisites

Blast+ must be installed. Check [this page](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/) to download the version for your system. 

## Optional: Create environment

`python3 -m venv <path>/<to>/<environment>`
  
## Install requirements

`pip3 install -r requirements.txt`

## How to run?

You can see the help running:

`python3 malivhu.py -h` or `python3 malivhu.py --help`

The basic way to run the script is:

`python3 malivhu.py -iv <input virus file path> -o <output directory path> -p <last phase to run>`

If you're running up to the fourth phase, you also need to add the path to the human file:

`python3 malivhu.py -iv <input virus file path> -ih <input human file path> -o <output directory path> -p 4`

You can also run the fourth phase only by adding the `-p4` or `--phase4only` flag and specifying what virus do the virus proteins belong to with the `-v` or `--virus` flag. It accepts 'cov1', 'cov2' and 'mers'.

`python3 malivhu.py -iv <input virus file path> -ih <input human file path> -o <output directory path> -p 4 -p4 -v cov1`
