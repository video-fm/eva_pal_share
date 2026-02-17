YOUR_NAME=chris
# Copy the repo excluding gitignored files
rsync -av --exclude-from=eva_tony/.gitignore eva_tony/ eva_${YOUR_NAME}/

# Then replace the .git directory
rm -rf eva_${YOUR_NAME}/.git
# If you want to initialize a new git repo:
cd eva_${YOUR_NAME}
git init