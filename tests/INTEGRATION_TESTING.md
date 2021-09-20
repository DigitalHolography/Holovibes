# Integrations CLI Tests

### How to add tests

Just create a folder in the `tests/data` folder. this is the name of your test
You shall put 2 or 3 files in the folder:
* a `input.holo` file as input
* a `ref.holo` file as supposed output
* an optional `holovibes.ini` config file for the parameters

### How to run tests

The tool used to run these tests is `pytest`. Just run this from the root of the project
```sh
$ python -m pytest -v
```