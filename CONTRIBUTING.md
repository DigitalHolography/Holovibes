## Contributing

### Coding style

* use Allman coding style
* never do bitshift

### Git

If you do not know how to use git, please have a look at the following tutorial.
	* Git - Documentation
	* Bitbucket Git FAQ

### Clone the repository

You can use a GUI tool as SourceTree or use git in command line.
git clone git@bitbucket.org:micatlan/holovibes.git

### Git rules

To let the versioning tool consistent, you have to respect these rules.
* master branch must be clean and compile.
* Never push generated files.
* Use branch for everything. For example to develop a new feature : new/myfeature.
*  Prefer use rebase when pulling changes in your own branch (it avoids merge commits).
* Use merge when pushing your changes to another branch.
* Never commits on master branch directly (without the acknowledge of your team mates).
* Commit messages: use keywords as ‘add, fix, rm, up, change’
* Git rules - Code review
* Git names conventions