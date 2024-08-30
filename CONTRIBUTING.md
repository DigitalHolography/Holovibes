## Contributing

### Coding style

#### Clang-Format

Make sure to configure your IDE to format source code automatically according to the project `.clang-format`.

For instance, on Visual Studio Code you can install the extension `Clang-format` and enable `Format On Save` in preferences.

If possible, install a pre-commit hook. Install pre-commit with 'pip install pre-commit' and then use 'pre-commit install' at the root of the project.

#### Clang-Format details

- No brackets in one line ifs
- Comments must use the doxygen format like the following: /_! \brief This is an example _/. See `DOCUMENTING.md` for further instructions
- Headers files must have the extention .hh
- Sources files must have the extention .cc
- Templated classes must have the extention \*.hxx
- Getters, Setters and inlinable functions must written in \*.hh files

#### Naming style

- Classes, Structs, Unions, Enums must be named in CamelCase.
- Class Members, Variables, Namespaces, Functions, Files must be named in snake_case.
- A class member getter must be named 'get\_{var}'. Idem for setters.
- Class members must end with '_' as in 'frame_size_'

### Git

- Branch `master` is only used for stable releases.
- Never push trash / generated files.
- Always work on separate branches when developing a new feature or fixing a bug.
- Use `git pull --rebase` if possible to avoids useless merge commits.
- Use a consistent commit messages convention (e.g. [Angular](https://github.com/angular/angular/blob/master/CONTRIBUTING.md#commit))
- Use the pull request feature so that other team members can review your work when you are done.

### Benchmark protocol

- The benchmark protocol is a way to compare the performance of the different versions of the project.
- It is important to run the benchmark before creating a new release.
- Create a "benchmark" folder in the app version AppData folder.
- Make a clean build in debug mode and run the exec with the -b or --benchmark option.
- Once the app is open, open the designated benchmark holo file/camera and do a given list of action (record, tweaking values, etc) for a given amount of time.
- It should make a file in the benchmark folder with the information worker output once the application is closed.
- The generated file can be better visualised using the BenchmarkViewer Jupiter Notebook.
- *You are invited to improve the protocol, the benchmark informations gathering in the Information Worker and the BenchmarkViewer file.*

### Create a new release

1.  Merge all your feature branches on `dev`
2.  Update `CHANGELOG.md`.
3.  Apply the Benchmark protocol (written above, may be updated in the future).
4.  Make a clean build in release mode (`./dev.py clean build -b Release`).
5.  Make sure everything works as intended and run test suite (`./dev.py pytest -b Release`). (`ctest` aren't working and are not to be used for now)
6.  Make sure the git repository has no work in progress
7.  Run the release script with `./dev.py -b Release release {bump-type}` with `bump-type` being either `major`, `minor` or `patch`
    if you want to modify either X, Y, or Z of the version number X.Y.Z
8.  add the updated CMakeLists.txt to git
9.  do a `git push --follow-tags`
10.  Merge master with `dev`
