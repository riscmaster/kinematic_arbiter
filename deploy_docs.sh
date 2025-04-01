#!/bin/bash

set -e

# Config
BUILD_DIR="docs/build"
SOURCE_DIR="docs/source"
GH_PAGES_BRANCH="gh-pages"
GH_PAGES_DIR=".gh-pages-tmp"

echo "▶ Building Sphinx documentation..."
sphinx-build -b html "$SOURCE_DIR" "$BUILD_DIR"

echo "▶ Preparing Git worktree for $GH_PAGES_BRANCH branch..."
rm -rf "$GH_PAGES_DIR"
git worktree add -B $GH_PAGES_BRANCH "$GH_PAGES_DIR" origin/$GH_PAGES_BRANCH

echo "▶ Copying documentation to $GH_PAGES_DIR..."
rm -rf "$GH_PAGES_DIR"/*
cp -r "$BUILD_DIR"/* "$GH_PAGES_DIR"/

cd "$GH_PAGES_DIR"

echo "▶ Committing and pushing to $GH_PAGES_BRANCH..."
git add .
PRE_COMMIT_ALLOW_NO_CONFIG=1 git commit -m "Deploy updated docs" || echo "Nothing to commit"
git push origin $GH_PAGES_BRANCH

cd ..
echo "▶ Cleaning up..."
git worktree remove "$GH_PAGES_DIR"

echo "✅ Deployment complete. Docs should be live at:"
echo "   https://$(git config --get remote.origin.url | sed 's|.*github.com[:/]||;s/.git$//')/"
