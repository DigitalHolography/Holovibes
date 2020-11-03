## Contributing

### Git

- Branch `master` must be clean and compile.
- Never push trash / generated files.
- Always work on separate branches when developing a new feature of fixing a bug.
- Use `git pull --rebase` to avoids useless merge commits.
- Use `git merge --squash` when merging on `develop` or `master`.
- Use a consistent commit messages convention (e.g. [Angular](https://github.com/angular/angular/blob/master/CONTRIBUTING.md#commit))

### Create a new release

1. Make sure all features are on branch `develop`.
2. Change the version number in `compute_descriptor.hh`.
3. Make a clean build in release mode (`rm -rf build && ./build.py R`).
4. Make sure everything works as intended.
5. Update `CHANGELOG.md`.
6. Change the version number in `setupCreator.iss`.
7. Commit your changes and `git merge develop --squash` on master (with commit message `vX.X`).
8. Tag your commit and push (`git tag -a "vX.X" -m "vX.X" && git push origin master --tags`).
9. Create setup installer with Inno Setup (`setupCreator.iss`).
10. Run the installer (located in `Output/`) and make sure everything still works as intended.
11. Upload the installer on `ftp.espci.fr` in `incoming/Atlan/holovibes/installer/Holovibes_vX.X.X`.