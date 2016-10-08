# How to fork mlpractical to your own github account

## 1. Fork 
Simply click `Fork` button on the upper right corner.

## 2. Configuring a remote for the fork
```
git remote -v
```
The output should be something similar to
```
origin	https://github.com/XingxingZhang/mlpractical (fetch)
origin	https://github.com/XingxingZhang/mlpractical (push)
```
Then, specify a new remote upstream repository that will be synced with the fork.
```
git remote add upstream https://github.com/CSTR-Edinburgh/mlpractical.git
git remote -v
```
Outputs
```
origin	https://github.com/XingxingZhang/mlpractical (fetch)
origin	https://github.com/XingxingZhang/mlpractical (push)
upstream	https://github.com/CSTR-Edinburgh/mlpractical.git (fetch)
upstream	https://github.com/CSTR-Edinburgh/mlpractical.git (push)
```
Configuration done!

## 3. Syncing the fork (need to be done for each lab)
Get everything up-to-date
```
git fetch upstream
```
Create local branch
```
git checkout -b mlp2016-7/lab[n] upstream/mlp2016-7/lab[n]
```
Optionally, you can also merge the branch
```
git merge upstream/mlp2016-7/lab[n]
```

### 4. Work on your own branch (optional)
After you have finished step 3, you can create your own branch and work on it.
```
git checkout -b my-lab[n]-solution
```
Then work on this branch. After you finished everything, then use
```
git status
```
to check which files have you modified. Then add all the modified files with
```
git add [files-you-have-changed]
```
Next, commit the changes
```
git commit -m "a comment for your solution"
```
Finally, push it to your repo.
```
git push origin my-lab[n]-solution
```



