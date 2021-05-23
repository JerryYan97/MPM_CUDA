# Build and compile the project
# NOTE1: Before building and compiling, please recursively clone the original project, which has the submodule projects.
# NOTE2: If there is anything wrong, please take a look at the error message.
import os

if __name__ == '__main__':
    root_dir = os.getcwd()

    # Build the glfw
    print("Building and compiling glfw.")
    os.makedirs('./thirdparties/glfw/build', exist_ok=True)
    os.chdir('./thirdparties/glfw/build')
    os.system('cmake ..')
    os.system('make')
    print("Building and compiling glfw complete.")

    print("Building and compiling complete.")
