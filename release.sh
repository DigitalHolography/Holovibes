##################################
## Release Script for Holovibes ##
##################################

######################
## Config Variables ##
######################
compute_descriptor_path="./Holovibes/includes/compute/compute_descriptor.hh"
setup_creator_path="./setupCreator.iss"
changelog_path="./CHANGELOG.md"

vc_redist_path="./resources/setup_creator_files/vcredist_2019_x64.exe"
vc_redist_url="https://aka.ms/vs/16/release/vc_redist.x64.exe"

# line on which the version is stored
cd_version_line_identifier="const static std::string version = \"v"
iss_version_line_identifier="#define MyAppVersion \""

# Getting the whole line where the version is stored
cd_version_line=`cat $compute_descriptor_path | grep "$cd_version_line_identifier"`
iss_version_line=`cat $setup_creator_path | grep "$iss_version_line_identifier"`

release_branch="bugfixes-936"

####################################################
## Checking if we are on branch '$release_branch' ##
####################################################

# Warning message if not sync with remote
echo "Verifying if current branch is up to date"
is_branch_sync=`git fetch --dry-run 2>&1`
if [ "$is_branch_sync" != "" ]
then
    echo ""
    echo $is_branch_sync
    echo ""
    echo -n "Your branch is not synchronized with the remote. Do you still want to continue ? [Y/n] "
    read response
    if [ "" != "$response" ] && [ "Y" != "$response" ] && [ "y" != "$response" ]
    then
        exit 0
    fi
fi

# Check if we're on the good branch
current_branch=`git rev-parse --abbrev-ref HEAD`
if [ "$current_branch" != "$release_branch" ]
then
    echo "You can only release on branch '$release_branch'"
    echo "Commit your changes, merge if necessary to '$release_branch' and execute again this script"
    exit 1
fi

######################
## Changing Version ##
######################
if [ "$cd_version_line" = "" ]
then
    echo "No version found in $compute_descriptor_path"
    exit 1
elif [ "$iss_version_line" = "" ]
then
    echo "No version found in $setup_creator_path"
    exit 1
fi

# Ectracting version number
cd_current_version=`echo "$cd_version_line" | sed 's/.*\"v\(.*\)\".*/\1/'`
iss_current_version=`echo "$iss_version_line" | sed 's/.*\"\(.*\)\".*/\1/'`
if [ "$cd_current_version" != "$iss_current_version" ]
then
    echo "Version of '$compute_descriptor_path'(version $cd_current_version) and '$setup_creator_path'(version $iss_current_version) does not match"
fi

# Creating new version
echo "Version found: \"$cd_current_version\""
echo -n "New Version: v"
read new_version
echo ""

# Change version in files
sed -i "s/\($cd_version_line_identifier\).*\(\".*\)/\1$new_version\2/" $compute_descriptor_path
sed -i "s/\($iss_version_line_identifier\).*\(\".*\)/\1$new_version\2/" $setup_creator_path

###########################
## Updating CHANGELOG.md ##
###########################
sed -i '2i\ ' $changelog_path
sed -i "3i\### $new_version" $changelog_path
sed -i '4i\ ' $changelog_path

response=""
i=5
echo "Edit changelog: (empty line to end editing)"
while [ 1 ]
do
    read response
    if [ "$response" = "" ]
    then
        break
    fi
    sed -i "${i}i\\* $response" $changelog_path
    i=$(($i + 1))
done

######################
## Rebuild solution ##
######################
rm -rf build/

has_python3=`which python3 > /dev/null; echo $?`
if [ "$has_python3" = "0" ]
then
    python3 build.py r p 
else
    python_version=`python --version | cut -d. -f1`
    if [ "$python_version" != "Python 3" ]
    then
        echo "Python version not supported !"
        exit 1
    fi
    python build.py r p
fi

response=`echo $?`
if [ "$response" != "0" ]
then
    echo "Build failed ! Exiting script..."
    exit 1
fi

####################
## Run unit tests ##
####################

#################################
## Commit Tag and Push version ##
#################################
echo -n "Commit, Tag and Push? [Y/n] "
read response

if [ "" = "$response" ] || [ "Y" = "$response" ] || [ "y" = "$response" ]
then
    git add $changelog_path $compute_descriptor_path $setup_creator_path
    git commit -m "Holovibes v$new_version"
    git tag -a "v$new_version" -m "v$new_version"
    git push origin $release_branch --tags
fi

################
## Inno Setup ##
################
if [ ! -f "$vc_redist_path" ]
then
    mkdir -p `dirname $vc_redist_path`
    wget $vc_redist_url -O $vc_redist_path
fi
iscc.exe $setup_creator_path
./Output/holovibes_setup_$new_version.exe
