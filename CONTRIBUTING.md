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

### Create a new release

1.  Merge all your feature branches on `dev`
2.  Update `CHANGELOG.md`.
3.  Make a clean build in release mode (`./dev.py clean build -b Release`).
4.  Make sure everything works as intended and run test suite (`./dev.py pytest -b Release`). (`ctest` aren't working and are not to be used for now)
5.  Make sure the git repository has no work in progress
6.  Run the release script with `./dev.py -b Release release {bump-type}` with `bump-type` being either `major`, `minor` or `patch`
    if you want to modify either X, Y, or Z of the version number X.Y.Z
7.  add the updated CMakeLists.txt to git
8.  do a `git push --follow-tags`
9.  Merge master with `dev`
