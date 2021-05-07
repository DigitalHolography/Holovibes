## Contributing

### Coding style

Make sure to configure your IDE to format source code automatically according to the project `.clang-format`.

For instance, on Visual Studio Code you can install the extension `Clang-format` and enable `Format On Save` in preferences.

### Git

- Branch `master` must be clean and compile.
- Never push trash / generated files.
- Always work on separate branches when developing a new feature or fixing a bug.
- Use `git pull --rebase` to avoids useless merge commits.
- Use a consistent commit messages convention (e.g. [Angular](https://github.com/angular/angular/blob/master/CONTRIBUTING.md#commit))

### Create a new release

1. Make sure all features are on branch `develop`.
2. Change the version number in `compute_descriptor.hh` and `setupCreator.iss`.
3. Update `CHANGELOG.md`.
4. Make a clean build in release mode (`rm -rf build && ./build.py R`).
5. Make sure everything works as intended and run test suite (`./build.py && ./run_tests.py`).
6. Delete the file holovibes.ini from the release build directory
7. Commit your changes, push and merge develop on master (`git checkout master && git merge develop`).
8. Tag your commit and push (`git tag -a "vX.X" -m "vX.X" && git push origin master --tags`).
9. Create setup installer with Inno Setup (`setupCreator.iss`).
10. Run the installer (located in `Output/`) and make sure everything still works as intended.
11. Upload the installer on `ftp.espci.fr` in `incoming/Atlan/holovibes/installer/Holovibes_vX.X.X`.