# xiDOUBLET

Citation: Cleavable Cross-Linkers Redefined by a Novel MS3-Trigger Algorithm; Lars Kolbowski, Lutz Fischer, and Juri Rappsilber, Analytical Chemistry 2023 95 (42), 15461-15464, DOI: https://doi.org/10.1021/acs.analchem.3c01673

## Using xiDOUBLET


### Setup
xiDOUBLET uses [pipenv](https://pipenv.readthedocs.io/en/latest/) to
manage package dependencies.

From the root directory of the project, run
```
pipenv install
```
to install all the required packages into a pipenv.

Commands need to be
prefixed with `pipenv run` to ensure that they are run inside the
pipenv. Alternatively, you can run `pipenv shell`, which changes the
environment to ensure all subsequent commands are executed within the
pipenv.

### Running a search
#### Configuration
To run xiDOUBLET, you first need to setup a config file in JSON
format. An example is given in `example_config.json`. Please have a
look at that file to see what options are available (although that
file doesn't necessarily list all available options at the moment).

#### Searching
To run xiDOUBLET, you need to tell it what spectra to process, what config to use and what directory to write the
results to:
```
pipenv run python -m xidoublet \
-i example.mgf \
-o output_directory \
-c example_config.json \
-n result_name_prefix
```
