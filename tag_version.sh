#!/bin/bash
# Version tagging script for etorotrade

# Display usage information
display_help() {
    echo "Usage: ./tag_version.sh VERSION [MESSAGE]"
    echo "Create a new version tag and push it to the remote repository."
    echo ""
    echo "Arguments:"
    echo "  VERSION     Version number (e.g., 1.0.0)"
    echo "  MESSAGE     Optional tag message (default: 'Version VERSION')"
    echo ""
    echo "Example:"
    echo "  ./tag_version.sh 1.0.0 'Initial stable release'"
    echo ""
}

# Check if version is provided
if [ $# -lt 1 ]; then
    echo "Error: Version number is required."
    display_help
    exit 1
fi

VERSION=$1

# Use default message if not provided
if [ $# -lt 2 ]; then
    MESSAGE="Version $VERSION"
else
    MESSAGE=$2
fi

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Error: Not a git repository."
    exit 1
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "Error: There are uncommitted changes in the repository."
    echo "Commit or stash your changes before tagging a version."
    exit 1
fi

# Verify the version format (optional)
if ! [[ $VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Warning: Version '$VERSION' doesn't follow semantic versioning (x.y.z)."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Run tests before tagging
echo "Running tests before tagging..."
if ! ./run_tests.sh --unit; then
    echo "Error: Tests failed. Fix issues before tagging a version."
    exit 1
fi

# Create and push the tag
echo "Creating tag v$VERSION with message: '$MESSAGE'"
git tag -a "v$VERSION" -m "$MESSAGE"

read -p "Push tag to remote repository? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git push origin "v$VERSION"
    echo "Tag v$VERSION pushed to remote repository."
else
    echo "Tag created locally but not pushed."
    echo "Use 'git push origin v$VERSION' to push it later."
fi

# Create a VERSION file for reference
echo "$VERSION" > VERSION
echo "VERSION file updated with current version."

echo "Version $VERSION successfully tagged!"