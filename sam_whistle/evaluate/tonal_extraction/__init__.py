import os
import jpype
import jpype.imports
from jpype.types import *
import glob


root_dir = os.path.dirname(os.path.abspath(__file__))
java_dist_dir = os.path.join(root_dir, 'jar')
jar_files = glob.glob(os.path.join(java_dist_dir, '*.jar'))
jpype.startJVM(classpath=jar_files) 
# Import and test Java classes
from tonals import *

try:
    tonal()
    ActiveSet()
    print("Java classes loaded successfully.")
except Exception as e:
    print("Could not load Java classes:", e)

# Shutdown JVM when done (optional based on the use case)
# jpype.shutdownJVM()
