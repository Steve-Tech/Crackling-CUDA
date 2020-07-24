import sys

## This class displays, and writes to file, every `print` command    
class Logger(object):
    def __init__(self, outputFile):
        self.terminal = sys.stdout
        self.log = open(outputFile, "w+")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  
        
    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass  
