# 这段代码提供了一种自动化的方法来卸载所有用户安装的 Python 包,适合在需要清理环境或重置依赖时使用.

import subprocess

# Get the list of user-installed packages
installed_packages = subprocess.check_output(['pip', 'freeze', '--user']).decode('utf-8')
packages = [pkg.split('==')[0] for pkg in installed_packages.splitlines()]

# Uninstall each package
for package in packages:
    subprocess.call(['pip', 'uninstall', '-y', package])

print("All user-specific packages have been removed.")
