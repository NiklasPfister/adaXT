import pkgutil

# Replace 'your_package_name' with the actual package name you want to inspect
package_name = 'adaXT'

# Use pkgutil to iteratively list all modules within the package
package = __import__(package_name)
for importer, modname, ispkg in pkgutil.walk_packages(package.__path__):
    print(modname)
