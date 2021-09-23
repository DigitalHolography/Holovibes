## Contributing

### Coding style

Make sure to configure your IDE to format source code automatically according to the project `.clang-format`.

For instance, on Visual Studio Code you can install the extension `Clang-format` and enable `Format On Save` in preferences.

If possible, install a pre-commit hook. Install pre-commit with 'pip install pre-commit' and then use 'pre-commit install' at the root of the project.

### Git

- Branch `master` is only used for stable releases.
- Never push trash / generated files.
- Always work on separate branches when developing a new feature or fixing a bug.
- Use `git pull --rebase` to avoids useless merge commits.
- Use a consistent commit messages convention (e.g. [Angular](https://github.com/angular/angular/blob/master/CONTRIBUTING.md#commit))
- Use the pull request feature so that other team members can review your work when you are done.

### Create a new release

1. Merge all your feature branches on `develop`
2. Change the version number in `compute_descriptor.hh` and `setupCreator.iss`.
3. Update `CHANGELOG.md`.
4. Make a clean build in release mode (`rm -rf build && ./build.py R`).
5. Make sure everything works as intended and run test suite (`./build.py && ./run_unit_tests.py`).
6. Delete the file holovibes.ini from the release build directory
7. Create a commit for the new version (`git commit -m "vX.X"`).
8. Tag your commit and push (`git tag -a "vX.X" -m "vX.X" && git push origin develop --tags`).
