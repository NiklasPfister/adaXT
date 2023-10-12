import pydoc

# Replace 'your_package_name' with the actual package name you want to inspect
package_name = 'adaXT'

# Generate documentation for the package
pydoc.writedoc(package_name)

# List the modules in the package
modules = pydoc.modules(package_name)
for module in modules:
    print(module)

